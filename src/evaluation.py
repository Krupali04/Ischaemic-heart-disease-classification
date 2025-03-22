from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                          roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate a model using multiple metrics and visualizations
    
    Args:
        model: Trained model or pipeline
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        model_name: Name of the model for display
    """
    try:
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Try to get probabilities, handle case where model doesn't support it
        try:
            proba = model.predict_proba(X_test)
            # Check if we have probabilities for both classes
            y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        except (AttributeError, IndexError):
            y_proba = y_pred  # Use predictions if probabilities not available
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Print metrics
        print(f"\nResults for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
        
        plt.figure(figsize=(10, 4))
        
        # Plot ROC curve if we have valid probabilities
        plt.subplot(1, 2, 1)
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            metrics['AUC'] = roc_auc
        except Exception as e:
            print(f"Warning: Could not compute ROC curve: {str(e)}")
            plt.text(0.5, 0.5, 'ROC curve not available',
                    horizontalalignment='center',
                    verticalalignment='center')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot confusion matrix
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance if the model supports it
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()