import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import logging

class AdaptiveLearningManager:
    def __init__(self):
        self.learning_rate = 0.01
        self.min_learning_rate = 0.0001
        self.max_learning_rate = 0.1
        self.adaptation_rate = 0.1
        self.metrics_history = []
        self.window_size = 5
        self.performance_threshold = 0.001
        
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
        
        # Store metrics history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        
        return metrics
    
    def update_learning_rate(self, metrics, validation_data):
        """Adaptively update learning rate based on performance"""
        if len(self.metrics_history) < 2:
            return
            
        # Calculate performance change
        current_auc = metrics['roc_auc']
        previous_auc = self.metrics_history[-2]['roc_auc']
        performance_change = current_auc - previous_auc
        
        # Adaptive learning rate update
        if performance_change > self.performance_threshold:
            # Performance improving, increase learning rate
            self.learning_rate = min(
                self.learning_rate * (1 + self.adaptation_rate),
                self.max_learning_rate
            )
        elif performance_change < -self.performance_threshold:
            # Performance degrading, decrease learning rate
            self.learning_rate = max(
                self.learning_rate * (1 - self.adaptation_rate),
                self.min_learning_rate
            )
        
        # Log adaptation
        logging.info(
            f"Learning rate adapted: {self.learning_rate:.4f} "
            f"(change: {performance_change:.4f})"
        )
    
    def get_adaptive_parameters(self):
        """Get current adaptive parameters"""
        return {
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'performance_threshold': self.performance_threshold,
            'metrics_history': self.metrics_history
        }

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