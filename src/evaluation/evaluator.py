"""
Evaluation Module for Parkinson's Disease Detection
Handles performance metrics, visualization, and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, cohen_kappa_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseEvaluator, BaseModel

class ParkinsonEvaluator(BaseEvaluator):
    """
    Comprehensive evaluator for Parkinson's disease detection models
    Provides clinical metrics, visualization, and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.clinical_metrics = config.get('clinical_metrics', True)
        self.feature_importance = config.get('feature_importance', True)
        self.confusion_matrix = config.get('confusion_matrix', True)
        self.roc_curve = config.get('roc_curve', True)
        self.learning_curves = config.get('learning_curves', False)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating model performance")
        
        try:
            results = {}
            
            # Basic metrics
            results['accuracy'] = accuracy_score(y_true, y_pred)
            results['precision'] = precision_score(y_true, y_pred, average='weighted')
            results['recall'] = recall_score(y_true, y_pred, average='weighted')
            results['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # Additional metrics
            results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            results['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            
            # Probability-based metrics
            if y_prob is not None:
                results['roc_auc'] = roc_auc_score(y_true, y_prob)
                results['average_precision'] = average_precision_score(y_true, y_prob)
            
            # Clinical metrics
            if self.clinical_metrics:
                clinical = self._compute_clinical_metrics(y_true, y_pred)
                results.update(clinical)
            
            # Confusion matrix
            if self.confusion_matrix:
                results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # Classification report
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=['Healthy', 'Parkinson\'s'], output_dict=True
            )
            
            self.results = results
            self.logger.info("Model evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def _compute_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute clinical metrics for medical interpretation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with clinical metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Additional clinical metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                      cv_strategy: Any, groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with CV results
        """
        self.logger.info("Performing cross-validation evaluation")
        
        try:
            cv_results = {}

            # Cross-validation for each metric
            for metric in self.metrics:
                try:
                    scores = cross_val_score(
                        model.model, X, y,
                        groups=groups,  # Use groups parameter for StratifiedGroupKFold
                        cv=cv_strategy,
                        scoring=metric,
                        n_jobs=-1,
                        error_score=np.nan
                    )
                    # Filter out NaN scores (e.g. folds where only one class was present)
                    valid_scores = scores[~np.isnan(scores)]
                    if len(valid_scores) == 0:
                        self.logger.warning(f"All CV folds returned NaN for {metric}, skipping")
                        continue
                    cv_results[metric] = {
                        'scores': valid_scores,
                        'mean': valid_scores.mean(),
                        'std': valid_scores.std(),
                        'min': valid_scores.min(),
                        'max': valid_scores.max()
                    }
                except Exception as metric_error:
                    self.logger.warning(f"Could not compute CV for {metric}: {metric_error}")
                    continue
            
            # Store results
            self.results['cross_validation'] = cv_results
            
            # Log results
            self.logger.info("Cross-validation results:")
            for metric, stats in cv_results.items():
                self.logger.info(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def plot_results(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Generate evaluation plots
        
        Args:
            save_path: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if not self.results:
            self.logger.warning("No results available for plotting")
            return {}
        
        self.logger.info("Generating evaluation plots")
        
        try:
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            
            plot_paths = {}
            
            # Confusion Matrix
            if self.confusion_matrix and 'confusion_matrix' in self.results:
                path = self._plot_confusion_matrix(save_path)
                if path:
                    plot_paths['confusion_matrix'] = path
            
            # ROC Curve
            if self.roc_curve and 'roc_auc' in self.results:
                path = self._plot_roc_curve(save_path)
                if path:
                    plot_paths['roc_curve'] = path
            
            # Precision-Recall Curve
            if 'average_precision' in self.results:
                path = self._plot_precision_recall_curve(save_path)
                if path:
                    plot_paths['precision_recall_curve'] = path
            
            # Feature Importance
            if self.feature_importance and hasattr(self, 'feature_importance_data'):
                path = self._plot_feature_importance(save_path)
                if path:
                    plot_paths['feature_importance'] = path
            
            # Learning Curves
            if self.learning_curves and hasattr(self, 'learning_curve_data'):
                path = self._plot_learning_curves(save_path)
                if path:
                    plot_paths['learning_curves'] = path
            
            self.plots = plot_paths
            self.logger.info(f"Generated {len(plot_paths)} plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            return {}
    
    def _plot_confusion_matrix(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot confusion matrix"""
        try:
            cm = self.results['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Healthy', 'Parkinson\'s'],
                       yticklabels=['Healthy', 'Parkinson\'s'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_path:
                path = Path(save_path) / 'confusion_matrix.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            return None
    
    def _plot_roc_curve(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot ROC curve"""
        try:
            # This would need y_true and y_prob from the evaluation
            # For now, return None as we need to modify the interface
            return None
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            return None
    
    def _plot_precision_recall_curve(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot precision-recall curve"""
        try:
            # Similar to ROC curve, needs y_true and y_prob
            return None
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curve: {str(e)}")
            return None
    
    def _plot_feature_importance(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot feature importance"""
        try:
            if not hasattr(self, 'feature_importance_data'):
                return None
            
            df = self.feature_importance_data.head(15)  # Top 15 features
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            
            if save_path:
                path = Path(save_path) / 'feature_importance.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                return str(path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            return None
    
    def _plot_learning_curves(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot learning curves"""
        try:
            if not hasattr(self, 'learning_curve_data'):
                return None
            
            # Implementation would go here
            return None
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {str(e)}")
            return None
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate evaluation report
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report content or file path
        """
        if not self.results:
            return "No evaluation results available"
        
        self.logger.info("Generating evaluation report")
        
        try:
            report = []
            report.append("# Parkinson's Disease Detection - Model Evaluation Report")
            report.append("=" * 60)
            report.append("")
            
            # Basic Metrics
            report.append("## Performance Metrics")
            report.append("")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in self.results:
                    value = self.results[metric]
                    report.append(f"- **{metric.upper()}**: {value:.3f}")
            report.append("")
            
            # Clinical Metrics
            if self.clinical_metrics:
                report.append("## Clinical Metrics")
                report.append("")
                clinical_metrics = ['sensitivity', 'specificity', 'positive_predictive_value', 'negative_predictive_value']
                for metric in clinical_metrics:
                    if metric in self.results:
                        value = self.results[metric]
                        report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.3f}")
                report.append("")
            
            # Confusion Matrix
            if 'confusion_matrix' in self.results:
                cm = self.results['confusion_matrix']
                report.append("## Confusion Matrix")
                report.append("")
                report.append("| Actual \\ Predicted | Healthy | Parkinson's |")
                report.append("|-------------------|---------|-------------|")
                report.append(f"| Healthy          | {cm[0,0]:8d} | {cm[0,1]:11d} |")
                report.append(f"| Parkinson's      | {cm[1,0]:8d} | {cm[1,1]:11d} |")
                report.append("")
            
            # Cross-Validation Results
            if 'cross_validation' in self.results:
                report.append("## Cross-Validation Results")
                report.append("")
                cv_results = self.results['cross_validation']
                for metric, stats in cv_results.items():
                    report.append(f"- **{metric.upper()}**: {stats['mean']:.3f} ± {stats['std']:.3f}")
                report.append("")
            
            report_content = "\n".join(report)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(report_content)
                self.logger.info(f"Report saved to: {save_path}")
                return save_path
            else:
                return report_content
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def set_feature_importance_data(self, feature_importance_df: pd.DataFrame) -> None:
        """
        Set feature importance data for plotting
        
        Args:
            feature_importance_df: DataFrame with feature importance
        """
        self.feature_importance_data = feature_importance_df
        self.logger.info("Feature importance data set for plotting")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary
        
        Returns:
            Dictionary with evaluation summary
        """
        if not self.results:
            return {"error": "No evaluation results available"}
        
        summary = {
            "evaluation_complete": True,
            "metrics_computed": len([k for k in self.results.keys() if k not in ['confusion_matrix', 'classification_report']]),
            "plots_generated": len(self.plots),
            "clinical_metrics_available": self.clinical_metrics and any(k in self.results for k in ['sensitivity', 'specificity'])
        }
        
        # Add key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        summary['key_metrics'] = {k: self.results.get(k, None) for k in key_metrics}
        
        return summary
