import lightgbm as lgb # type: ignore
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import json
import logging
import os
from datetime import datetime
import glob

def create_shared_model():
    """Create a LightGBM model with optimized parameters for fraud detection"""
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',  # Changed from goss to gbdt
        objective='binary',
        metric=['auc', 'binary_logloss'],
        num_leaves=31,
        max_depth=7,
        min_data_in_leaf=100,
        max_bin=200,
        learning_rate=0.01,  # Reduced learning rate
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=0.1,      # L2 regularization
        subsample=0.8,       # Subsample ratio
        colsample_bytree=0.8,  # Feature subsample ratio
        is_unbalance=True,   # Handle class imbalance
        scale_pos_weight=770,  # Approximate ratio of negative to positive samples
        min_child_weight=1,    # Minimum sum of instance weight in a leaf
        min_child_samples=20,  # Minimum number of data needed in a leaf
        min_split_gain=0.0,    # Minimum gain to split
        verbosity=-1,
        device_type='cpu'
    )
    return model

def aggregate_weights(global_model, client_weights_list):
    """
    Aggregate client model weights into the global model.
    """
    if global_model is None or not client_weights_list:
        return global_model

    try:
        # Convert JSON weights to LightGBM booster objects
        client_boosters = []
        for w in client_weights_list:
            try:
                client_boosters.append(lgb.Booster(model_str=w))
            except Exception as e:
                logging.error(f"Error loading client model: {e}")
                continue

        if not client_boosters:
            logging.warning("No valid client models for aggregation.")
            return global_model

        # Perform federated averaging
        global_model = federated_average(global_model, client_boosters)

    except Exception as e:
        logging.error(f"Failed during aggregation: {e}")

    return global_model

def federated_average(global_model, client_boosters):
    """
    Averages multiple LightGBM models.
    """
    try:
        global_trees = global_model.booster_.dump_model()["tree_info"]

        for client_model in client_boosters:
            try:
                client_trees = client_model.dump_model()["tree_info"]

                for i in range(len(global_trees)):
                    if i >= len(client_trees):
                        continue  # Skip if client has fewer trees

                    # Average leaf values
                    global_leaves = np.array(global_trees[i]["leaf_value"])
                    client_leaves = np.array(client_trees[i]["leaf_value"])

                    if global_leaves.shape == client_leaves.shape:
                        global_trees[i]["leaf_value"] = ((global_leaves + client_leaves) / 2).tolist()
                    else:
                        logging.warning(f"Skipping tree {i}, structure mismatch.")

            except Exception as e:
                logging.error(f"Error processing a client model: {e}")

        # Convert back to LightGBM booster
        updated_model_str = json.dumps({"tree_info": global_trees})
        updated_model = lgb.Booster(model_str=updated_model_str)
        return updated_model

    except Exception as e:
        logging.error(f"Error in federated averaging: {e}")
        return global_model  # Return previous model if averaging fails

def save_model(model, client_id):
    """Save the model to disk"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'models/model_{client_id}_{timestamp}.txt'
        
        # Handle both LGBMClassifier and Booster objects
        if isinstance(model, lgb.LGBMClassifier):
            model.booster_.save_model(filename)
        else:  # Direct Booster object
            model.save_model(filename)
            
        logging.info(f"Model saved successfully to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return False

def load_latest_model(client_id):
    """Load the latest model for a client"""
    try:
        # Get list of model files for this client
        model_files = glob.glob(f'models/model_{client_id}_*.txt')
        if not model_files:
            logging.info(f"No existing models found for client {client_id}")
            return None
            
        # Get the most recent model file
        latest_model = max(model_files, key=os.path.getctime)
        
        # Load the model
        model = lgb.Booster(model_file=latest_model)
        logging.info(f"Loaded model from {latest_model}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None
