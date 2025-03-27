import os
import sys
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import lightgbm as lgb
import json
import threading
import logging
import pandas as pd
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model import aggregate_weights, create_shared_model
from common.expert_system import RuleBasedExpertSystem
from common.shap_explanation import ModelExplainer
from common.adaptive_learning import AdaptiveLearningManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow connections from any origin

# Global components
global_model = None
client_updates = {}  # Dictionary to store updates from each client
client_metrics = {}  # Dictionary to store metrics from each client
active_clients = set()  # Set of active client IDs
lock = threading.Lock()
expert_system = RuleBasedExpertSystem()
model_explainer = None
adaptive_manager = AdaptiveLearningManager()

# Aggregation parameters
MIN_CLIENTS_FOR_AGGREGATION = 2  # Minimum number of clients needed for aggregation
AGGREGATION_INTERVAL = 10  # Aggregation every 10 seconds

# Create output directory for visualizations
OUTPUT_DIR = "server_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def aggregate_and_update():
    """
    Periodically aggregates client updates into the global model.
    Runs in a background thread.
    """
    global global_model, client_updates, model_explainer

    while True:
        socketio.sleep(AGGREGATION_INTERVAL)

        with lock:
            if len(client_updates) >= MIN_CLIENTS_FOR_AGGREGATION:
                logging.info(f"Aggregating updates from {len(client_updates)} clients...")
                global_model = aggregate_weights(global_model, list(client_updates.values()))
                
                # Update model explainer with new global model
                model_explainer = ModelExplainer(global_model)
                
                # Generate new SHAP explanations
                if model_explainer:
                    model_explainer.explain_predictions(
                        np.zeros((1, 11)),  # Dummy data for initialization
                        output_dir=OUTPUT_DIR
                    )
                
                # Clear updates after aggregation
                client_updates.clear()
                logging.info("Global model updated successfully!")

@socketio.on("connect")
def handle_connect():
    """Handle new client connection"""
    client_id = request.args.get('client_id')
    if client_id:
        with lock:
            active_clients.add(client_id)
            logging.info(f"Client {client_id} connected via WebSocket")
            # Send acknowledgment to client
            socketio.emit('connection_ack', {'client_id': client_id, 'status': 'connected'})

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.args.get('client_id')
    if client_id:
        with lock:
            active_clients.discard(client_id)
            logging.info(f"Client {client_id} disconnected")

@app.route('/register_client', methods=['POST'])
def register_client():
    """Handle client registration"""
    global global_model, model_explainer
    
    try:
        data = request.get_json()
        if not data or 'client_id' not in data:
            return jsonify({'error': 'Invalid registration data'}), 400
            
        client_id = data['client_id']
        logging.info(f"Received registration request from client {client_id}")
        
        with lock:
            if client_id in active_clients:
                logging.info(f"Client {client_id} already registered")
                return jsonify({'status': 'success', 'message': 'Client already registered'})
            
            active_clients.add(client_id)
            logging.info(f"Client {client_id} registered successfully")
            
            # Initialize global model if this is the first client
            if global_model is None:
                logging.info("Initializing global model...")
                global_model = create_shared_model()
                # Don't create model_explainer until model is trained
                logging.info("Global model initialized successfully")
            
            return jsonify({'status': 'success'})
            
    except Exception as e:
        logging.error(f"Error in client registration: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def load_latest_model():
    """Load the latest saved model if it exists"""
    try:
        model_path = "models/global_model.txt"
        if os.path.exists(model_path):
            logging.info(f"Loading existing model from {model_path}")
            return lgb.Booster(model_file=model_path)
        return None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def save_model(model, path):
    """Save model to file"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save_model(path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")

def federated_average(global_model, client_models):
    """Perform federated averaging of client models"""
    try:
        if not client_models:
            logging.error("No client models to aggregate")
            return global_model
            
        # Get the first client model as base
        base_model = client_models[0]
        
        # Average the model weights
        for i in range(1, len(client_models)):
            client_model = client_models[i]
            # Average the model parameters
            for param_name in ['num_leaves', 'max_depth', 'min_data_in_leaf', 
                             'max_bin', 'learning_rate', 'reg_alpha', 'reg_lambda',
                             'subsample', 'colsample_bytree', 'scale_pos_weight']:
                if hasattr(base_model, param_name) and hasattr(client_model, param_name):
                    base_value = getattr(base_model, param_name)
                    client_value = getattr(client_model, param_name)
                    if isinstance(base_value, (int, float)):
                        setattr(base_model, param_name, (base_value + client_value) / 2)
        
        logging.info("Federated averaging completed successfully")
        return base_model
        
    except Exception as e:
        logging.error(f"Error in federated averaging: {str(e)}", exc_info=True)
        return global_model

@app.route('/update_model', methods=['POST'])
def update_model():
    """Handle model updates from clients"""
    global global_model, model_explainer, last_update_time, global_model_metrics
    
    try:
        data = request.get_json()
        if not data or 'client_id' not in data or 'model_weights' not in data:
            logging.error("Invalid update data received")
            return jsonify({'error': 'Invalid update data'}), 400
            
        client_id = data['client_id']
        model_weights = data['model_weights']
        metrics = data.get('metrics', {})
        
        logging.info(f"Received model update from client {client_id}")
        logging.info(f"Client metrics: {metrics}")
        
        with lock:
            try:
                # Update client metrics
                client_metrics[client_id] = metrics
                
                # Create client model from weights
                logging.info(f"Processing model update from client {client_id}")
                client_model = lgb.Booster(model_str=model_weights)
                
                # Store client update
                client_updates[client_id] = client_model
                logging.info(f"Stored update from client {client_id}")
                
                # Perform federated averaging if we have enough clients
                if len(client_updates) >= MIN_CLIENTS_FOR_AGGREGATION:
                    logging.info(f"Aggregating updates from {len(client_updates)} clients...")
                    global_model = federated_average(global_model, list(client_updates.values()))
                    
                    if global_model is not None:
                        try:
                            # Update model explainer only if model is trained
                            if hasattr(global_model, 'n_classes_'):
                                model_explainer = ModelExplainer(global_model)
                                logging.info("Model explainer updated successfully")
                        except Exception as e:
                            logging.warning(f"Could not create model explainer: {str(e)}")
                        
                        # Clear updates after aggregation
                        client_updates.clear()
                        last_update_time = datetime.now()
                        logging.info("Global model updated successfully!")
                        
                        # Save the updated model
                        save_model(global_model, "models/global_model.txt")
                        
                        # Update global metrics
                        global_model_metrics = {
                            'last_update': last_update_time.isoformat(),
                            'num_clients': len(client_metrics),
                            'average_auc': np.mean([m.get('auc', 0) for m in client_metrics.values()])
                        }
                        
                        # Log global model metrics
                        logging.info("Global model metrics:")
                        for metric, value in global_model_metrics.items():
                            logging.info(f"  {metric}: {value}")
                    else:
                        logging.error("Failed to create global model during aggregation")
                        return jsonify({'error': 'Model aggregation failed'}), 500
                
                return jsonify({'status': 'success'})
                
            except Exception as e:
                logging.error(f"Error processing model update: {str(e)}", exc_info=True)
                return jsonify({'error': f'Model processing error: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"Error in update endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def aggregate_weights(global_model, client_models):
    """Aggregate client model weights"""
    try:
        if not client_models:
            return global_model
            
        # Convert all models to booster format
        boosters = []
        for model in client_models:
            try:
                if isinstance(model, lgb.LGBMClassifier):
                    boosters.append(model.booster_)
                else:
                    boosters.append(model)
            except Exception as e:
                logging.error(f"Error converting model to booster: {e}")
                continue
                
        if not boosters:
            return global_model
            
        # Perform federated averaging
        global_model = federated_average(global_model, boosters[1:])
        return global_model
        
    except Exception as e:
        logging.error(f"Error in model aggregation: {e}")
        return global_model

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with both ML model and expert rules"""
    global global_model, expert_system, model_explainer
    data = request.json

    if not data or "features" not in data:
        logging.error("Invalid prediction request received")
        return jsonify({'error': 'Invalid input format'}), 400

    features = np.array(data['features'])
    logging.info(f"Received prediction request with {len(features)} features")
    
    try:
        # Get ML model prediction
        ml_prediction = global_model.predict_proba(features.reshape(1, -1))[0]
        logging.info(f"ML model prediction: {ml_prediction[1]:.4f}")
        
        # Get expert system score
        expert_scores = expert_system.apply_rules(pd.DataFrame([data['features']]))
        logging.info(f"Expert system score: {expert_scores[0]['score']:.4f}")
        
        # Combine predictions
        ml_weight = 0.7
        expert_weight = 0.3
        combined_score = ml_weight * ml_prediction[1] + expert_weight * expert_scores[0]['score']
        logging.info(f"Combined score: {combined_score:.4f}")
        
        # Get SHAP explanation if available
        explanation = None
        if model_explainer:
            logging.info("Generating SHAP explanation...")
            explanation = model_explainer.explain_single_prediction(
                features.reshape(1, -1),
                feature_names=global_model.feature_name_
            )
            logging.info("SHAP explanation generated successfully")
        
        return jsonify({
            'ml_prediction': float(ml_prediction[1]),
            'expert_score': expert_scores[0]['score'],
            'combined_score': float(combined_score),
            'expert_reasons': expert_scores[0]['reasons'],
            'shap_explanation': explanation
        })
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_explanations', methods=['GET'])
def get_explanations():
    """Get SHAP explanations and expert rules"""
    try:
        logging.info("Retrieving explanations...")
        # Get expert rules
        expert_rules = expert_system.get_rule_explanations()
        logging.info(f"Retrieved {len(expert_rules)} expert rules")
        
        # Get visualization paths
        visualization_paths = {
            'summary': f"{OUTPUT_DIR}/shap_summary.png",
            'bar': f"{OUTPUT_DIR}/shap_bar.png"
        }
        
        # Check if visualizations exist
        for path in visualization_paths.values():
            if not os.path.exists(path):
                logging.warning(f"Visualization file not found: {path}")
        
        logging.info("Successfully retrieved all explanations")
        return jsonify({
            'expert_rules': expert_rules,
            'visualization_paths': visualization_paths
        })
        
    except Exception as e:
        logging.error(f"Error getting explanations: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_system_status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    try:
        with lock:
            status = {
                'active_clients': list(active_clients),
                'client_metrics': client_metrics,
                'global_model_metrics': global_model_metrics if global_model else None,
                'last_update': last_update_time.isoformat() if last_update_time else None
            }
            logging.info(f"System status: {len(active_clients)} active clients")
            return jsonify(status)
    except Exception as e:
        logging.error(f"Error getting system status: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def create_shared_model():
    """Create a shared model instance"""
    # Try to load existing global model
    existing_model = load_latest_model()
    if existing_model:
        logging.info("Loaded existing global model")
        return existing_model
    
    # Create new model if none exists
    logging.info("Creating new global model")
    return lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        num_leaves=31,
        max_depth=7,
        min_data_in_leaf=100,
        max_bin=200,
        learning_rate=0.01,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=770,
        min_child_weight=1,
        min_child_samples=20,
        min_split_gain=0.0,
        verbosity=-1
    )

if __name__ == "__main__":
    # Start background aggregation thread
    threading.Thread(target=aggregate_and_update, daemon=True).start()
    logging.info("Server is running...")
    socketio.run(app, host="0.0.0.0", port=5000)
    logging.info("Server stopped.")
