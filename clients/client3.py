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
            n_positives = sum(y_train)
            n_negatives = len(y_train) - n_positives
            scale_pos_weight = n_negatives / n_positives if n_positives > 0 else 1
            logging.info(f"Class distribution - Positives: {n_positives}, Negatives: {n_negatives}, Scale weight: {scale_pos_weight:.2f}")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)
            
            # Simplified parameters for faster training
            params = {
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 7,
                'learning_rate': 0.1,
                'scale_pos_weight': scale_pos_weight,
                'metric': ['auc'],
                'verbosity': -1,
                'num_threads': 4
            }
            logging.info(f"Training parameters: {params}")
            
            # Train model with minimal iterations
            logging.info("Starting model training...")
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[valid_data],
                num_boost_round=50,  # Reduced iterations
                callbacks=[
                    lgb.early_stopping(stopping_rounds=5),
                    lgb.log_evaluation(period=5)
                ]
            )
            
            # Quick evaluation
            y_pred_proba = self.model.predict(X_test)
            current_auc = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"Training completed. Final AUC: {current_auc:.4f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}", exc_info=True)
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
