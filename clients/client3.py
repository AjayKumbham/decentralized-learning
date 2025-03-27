import os
import sys
import numpy as np
import lightgbm as lgb
import requests
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import time
from datetime import datetime
import matplotlib.pyplot as plt
import shap

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.adaptive_learning import AdaptiveLearningManager
from common.expert_system import RuleBasedExpertSystem
from common.shap_explanation import ModelExplainer, generate_shap_explanations
from common.model import load_latest_model, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)

# Create output directory for visualizations
OUTPUT_DIR = "client_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.server_url = 'http://localhost:5000'
        self.adaptive_manager = AdaptiveLearningManager()
        self.expert_system = RuleBasedExpertSystem()
        self.model = None
        self.model_explainer = None
        self.registered = False
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        logging.info(f"Initialized client {client_id} with server URL: {self.server_url}")
        
    def register_with_server(self):
        """Register with the server with retry logic"""
        logging.info(f"Attempting to register client {self.client_id} with server...")
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/register_client",
                    json={"client_id": self.client_id}
                )
                response.raise_for_status()
                data = response.json()
                if data.get('status') == 'success':
                    self.registered = True
                    logging.info(f"Successfully registered with server as {self.client_id}")
                    return True
                else:
                    logging.error(f"Registration failed: {data.get('message', 'Unknown error')}")
                    return False
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logging.warning(f"Registration attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"Failed to register with server after {self.max_retries} attempts: {str(e)}")
                    return False
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the model efficiently"""
        logging.info(f"Client {self.client_id} starting model training...")
        logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        try:
            # Calculate class weights
            n_samples = len(y_train)
            n_positives = sum(y_train)
            n_negatives = n_samples - n_positives
            scale_pos_weight = n_negatives / n_positives if n_positives > 0 else 1.0
            
            # Create LightGBM datasets with free_raw_data=False
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
            
            # Initialize training parameters with adaptive learning rate
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': self.adaptive_manager.learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'scale_pos_weight': scale_pos_weight
            }
            
            # Load existing model if available
            model_path = f"models/{self.client_id}_model.txt"
            if os.path.exists(model_path):
                logging.info(f"Loading existing model from {model_path}")
                self.model = lgb.Booster(model_file=model_path)
                # Update learning rate for incremental training
                self.model.learning_rate = self.adaptive_manager.learning_rate
            
            # Adaptive training loop
            best_model = None
            best_auc = 0
            patience = 5
            patience_counter = 0
            
            for epoch in range(100):  # Maximum 100 epochs
                # Train for a small number of rounds per epoch
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=10,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=5)],
                    init_model=self.model
                )
                
                # Evaluate current model
                y_pred_proba = self.model.predict(X_test)
                current_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Update adaptive parameters with validation data
                self.adaptive_manager.update_learning_rate(current_auc, valid_data)
                params['learning_rate'] = self.adaptive_manager.learning_rate
                
                # Check for improvement
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_model = self.model
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    logging.info(f"Early stopping: validation scores not improving for {patience} rounds")
                    break
                
                logging.info(f"Epoch {epoch + 1}, AUC: {current_auc:.4f}, Learning Rate: {params['learning_rate']:.4f}")
            
            # Use best model
            if best_model is not None:
                self.model = best_model
                logging.info(f"Best model AUC: {best_auc:.4f}")
                
                # Save model with versioning
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version_path = f"models/versions/{self.client_id}_model_v{timestamp}.txt"
                os.makedirs("models/versions", exist_ok=True)
                self.model.save_model(version_path)
                logging.info(f"Saved model version to {version_path}")
                
                # Also save as latest model
                self.model.save_model(model_path)
                logging.info(f"Saved latest model to {model_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}", exc_info=True)
            return False
    
    def generate_explanations(self, X_test):
        """Generate SHAP explanations efficiently"""
        if self.model is None:
            logging.error("Cannot generate explanations: model not trained")
            return None
            
        try:
            logging.info("Starting SHAP explanation generation...")
            # Use a smaller sample for SHAP to speed up computation
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            logging.info(f"Using {sample_size} samples for SHAP analysis")
            
            # Create TreeExplainer
            explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values for the sample
            logging.info("Calculating SHAP values...")
            shap_values = explainer.shap_values(X_sample)
            
            # Handle both single and multi-class cases
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
                logging.info("Using positive class SHAP values")
            
            # Create summary plot
            logging.info("Generating SHAP visualization...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=X_sample.columns,
                show=False,
                plot_type="bar"
            )
            plt.title("Feature Importance (SHAP Values)")
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/shap_summary.png')
            plt.close()
            logging.info(f"SHAP visualization saved to {OUTPUT_DIR}/shap_summary.png")
            
            # Calculate and return feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(X_sample.columns, feature_importance))
            logging.info("Top 5 most important features:")
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
                logging.info(f"  {feature}: {importance:.4f}")
            
            return importance_dict
            
        except Exception as e:
            logging.error(f"Error generating SHAP explanations: {str(e)}", exc_info=True)
            return None
    
    def apply_expert_rules(self, X_test):
        """Apply expert rules efficiently"""
        try:
            logging.info("Starting expert rule application...")
            # Sample data for faster processing
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            logging.info(f"Using {sample_size} samples for expert rules")
            
            # Apply rules to the sample
            expert_scores = self.expert_system.apply_rules(X_sample)
            
            # Log rule application results
            logging.info(f"Applied expert rules to {sample_size} samples")
            logging.info(f"Number of rules triggered: {len(expert_scores)}")
            
            return expert_scores
            
        except Exception as e:
            logging.error(f"Error applying expert rules: {str(e)}", exc_info=True)
            return None
    
    def send_update_to_server(self):
        """Send model update efficiently"""
        try:
            logging.info("Preparing model update for server...")
            # Convert model to string directly
            model_str = self.model.model_to_string()
            
            # Quick metrics
            y_pred_proba = self.model.predict(self.X_test)
            metrics = {
                'auc': roc_auc_score(self.y_test, y_pred_proba),
                'timestamp': datetime.now().isoformat()
            }
            logging.info(f"Model metrics: {metrics}")
            
            # Send update with retry logic
            for attempt in range(self.max_retries):
                try:
                    logging.info(f"Sending update to server (attempt {attempt + 1}/{self.max_retries})...")
                    response = requests.post(
                        f"{self.server_url}/update_model",
                        json={
                            'client_id': self.client_id,
                            'model_weights': model_str,
                            'metrics': metrics
                        },
                        timeout=30  # Add timeout
                    )
                    response.raise_for_status()
                    logging.info(f"Successfully sent model update from {self.client_id}")
                    return True
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        logging.warning(f"Update attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
        except Exception as e:
            logging.error(f"Error sending update: {str(e)}", exc_info=True)
            return False
    
    def get_system_status(self):
        """Get current system status from server with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.server_url}/get_system_status")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.warning(f"Status check attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"Failed to get system status after {self.max_retries} attempts")
                    return None

def main():
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
    else:
        client_id = 'client1'
    
    client = FederatedClient(client_id)
    
    try:
        # Quick registration
        if not client.register_with_server():
            return
        
        # Load and preprocess data efficiently
        df = pd.read_csv('small-paysim.csv')
        df = df.drop(columns=['nameOrig', 'nameDest'])
        df['type'] = df['type'].map({'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'TRANSFER': 3, 'PAYMENT': 4})
        
        # Quick feature engineering
        df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['transaction_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        client.X_test = X_test
        client.y_test = y_test
        
        # Train and evaluate
        if not client.train_model(X_train, y_train, X_test, y_test):
            return
        
        # Generate SHAP explanations
        logging.info("Generating SHAP explanations...")
        importance_dict = client.generate_explanations(X_test)
        if importance_dict:
            logging.info("SHAP explanations generated successfully")
        
        # Apply expert rules
        logging.info("Applying expert rules...")
        expert_scores = client.apply_expert_rules(X_test)
        if expert_scores:
            logging.info("Expert rules applied successfully")
        
        # Send update with retry
        max_update_attempts = 3
        for attempt in range(max_update_attempts):
            if client.send_update_to_server():
                logging.info(f"Client {client_id} completed successfully")
                break
            elif attempt < max_update_attempts - 1:
                logging.warning(f"Update attempt {attempt + 1} failed, retrying...")
                time.sleep(5)
            else:
                logging.error(f"Client {client_id} failed to send update after {max_update_attempts} attempts")
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
