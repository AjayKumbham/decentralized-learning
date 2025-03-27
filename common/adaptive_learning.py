import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import logging
from datetime import datetime

class AdaptiveLearningManager:
    def __init__(self):
        self.learning_rate = 0.1
        self.best_auc = 0
        self.patience = 5
        self.patience_counter = 0
        self.min_lr = 0.001
        self.max_lr = 0.5
        self.adaptation_history = []
        
    def update_learning_rate(self, current_auc, validation_data=None):
        """Update learning rate based on performance"""
        try:
            # Store current state
            self.adaptation_history.append({
                'learning_rate': self.learning_rate,
                'auc': current_auc,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best AUC
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Adjust learning rate based on performance
            if current_auc > self.best_auc:
                # Increase learning rate if performance is improving
                self.learning_rate = min(self.learning_rate * 1.1, self.max_lr)
                logging.info(f"Performance improving. Increased learning rate to {self.learning_rate:.4f}")
            else:
                # Decrease learning rate if performance is not improving
                self.learning_rate = max(self.learning_rate * 0.9, self.min_lr)
                logging.info(f"Performance not improving. Decreased learning rate to {self.learning_rate:.4f}")
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                logging.info(f"Early stopping triggered after {self.patience} rounds without improvement")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error updating learning rate: {str(e)}")
            return False
    
    def get_adaptation_history(self):
        """Get the history of learning rate adaptations"""
        return self.adaptation_history
    
    def get_current_state(self):
        """Get current adaptive learning state"""
        return {
            'learning_rate': self.learning_rate,
            'best_auc': self.best_auc,
            'patience_counter': self.patience_counter,
            'adaptation_history': self.adaptation_history
        }

    def get_model_metrics(self, y_true, y_pred_proba):
        """Calculate model performance metrics"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
        
        return metrics
    
    def get_optimal_threshold(self, 
                            y_true: np.ndarray,
                            y_pred_proba: np.ndarray) -> float:
        """
        Find optimal decision threshold using precision-recall curve.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        return optimal_threshold
    
    def get_model_metrics(self, 
                         y_true: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive model metrics.
        """
        optimal_threshold = self.get_optimal_threshold(y_true, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        return {
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])
        } 