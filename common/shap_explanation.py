import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging

class ModelExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
    
    def explain_predictions(self, X, output_dir: str = None) -> Dict:
        """Generate SHAP explanations for predictions"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For binary classification, use the positive class
            shap_values = shap_values[0]
        
        # Convert to numpy array if needed
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        explanations = {
            'feature_importance': self._get_feature_importance(X, shap_values),
            'summary_plot': self._generate_summary_plot(X, shap_values, output_dir),
            'bar_plot': self._generate_bar_plot(X, shap_values, output_dir),
            'dependence_plots': self._generate_dependence_plots(X, shap_values, output_dir)
        }
        
        return explanations
    
    def explain_single_prediction(self, features: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Generate SHAP explanation for a single prediction"""
        # Reshape features if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Get feature importance
        importance = self._get_feature_importance(
            pd.DataFrame(features, columns=feature_names) if feature_names else features,
            shap_values
        )
        
        return {
            'feature_importance': importance,
            'shap_values': shap_values.tolist()
        }
    
    def _get_feature_importance(self, X, shap_values):
        """Calculate feature importance from SHAP values"""
        importance_dict = {}
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate mean absolute SHAP values for each feature
        for i, feature in enumerate(feature_names):
            importance_dict[feature] = float(np.abs(shap_values[:, i]).mean())
        
        # Sort by absolute importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_summary_plot(self, X, shap_values, output_dir: str = None) -> str:
        """Generate SHAP summary plot"""
        if not output_dir:
            return None
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _generate_bar_plot(self, X, shap_values, output_dir: str = None) -> str:
        """Generate SHAP bar plot"""
        if not output_dir:
            return None
            
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For binary classification, use the positive class
            shap_values = shap_values[0]
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create bar plot
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        y_pos = np.arange(len(feature_names))
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        y_pos = y_pos[sorted_idx]
        feature_names = [feature_names[i] for i in sorted_idx]
        feature_importance = feature_importance[sorted_idx]
        
        # Create bars
        plt.barh(y_pos, feature_importance)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'shap_bar.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _generate_dependence_plots(self, X, shap_values, output_dir: str = None) -> List[str]:
        """Generate SHAP dependence plots for each feature"""
        if not output_dir:
            return []
            
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        plot_paths = []
        
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(i, shap_values, X, show=False)
            plt.title(f"Dependence Plot for {feature}")
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'dependence_{feature}.png')
            plt.savefig(output_path)
            plt.close()
            
            plot_paths.append(output_path)
        
        return plot_paths

def generate_shap_explanations(model, X_test, feature_names):
    """Generate SHAP explanations for the model predictions"""
    try:
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Handle both single and multi-class cases
        if isinstance(shap_values, list):
            # For multi-class, use the positive class (index 1)
            shap_values = shap_values[1]
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False,
            plot_type="bar"
        )
        plt.title("Feature Importance (SHAP Values)")
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        
        # Create detailed feature importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title("Detailed Feature Impact (SHAP Values)")
        plt.tight_layout()
        plt.savefig('shap_detailed.png')
        plt.close()
        
        # Calculate and return feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_dict = dict(zip(feature_names, feature_importance))
        
        return importance_dict
        
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {str(e)}")
        return None
