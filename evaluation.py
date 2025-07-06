import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate various classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for the positive class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
    
    return metrics

def print_metrics(metrics):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    if 'average_precision' in metrics:
        print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

def evaluate_threshold_impact(y_true, y_prob, thresholds=None):
    """
    Evaluate the impact of different probability thresholds on model performance
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for the positive class
        thresholds: List of thresholds to evaluate (default: [0.1, 0.2, ..., 0.9])
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold to get predicted labels
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Store results
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find the optimal threshold based on a specified metric
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for the positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', or 'accuracy')
        
    Returns:
        Optimal threshold value
    """
    # Evaluate thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    threshold_metrics = evaluate_threshold_impact(y_true, y_prob, thresholds)
    
    # Find optimal threshold
    optimal_idx = threshold_metrics[metric].idxmax()
    optimal_threshold = threshold_metrics.loc[optimal_idx, 'threshold']
    
    print(f"Optimal threshold for {metric}: {optimal_threshold:.2f}")
    print(f"Resulting {metric} score: {threshold_metrics.loc[optimal_idx, metric]:.4f}")
    
    return optimal_threshold

def calculate_cost_benefit(y_true, y_pred, cost_matrix=None):
    """
    Calculate the cost-benefit of a model based on a cost matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_matrix: 2x2 matrix of costs/benefits:
                    [
                        [TN benefit, FP cost],
                        [FN cost, TP benefit]
                    ]
                    
    Returns:
        Total cost/benefit and breakdown
    """
    # Default cost matrix if not provided
    if cost_matrix is None:
        # Example: 
        # - True Negative: $0 (no action needed)
        # - False Positive: -$10 (investigation cost)
        # - False Negative: -$100 (fraud loss)
        # - True Positive: $90 (fraud prevented - investigation cost)
        cost_matrix = np.array([
            [0, -10],    # [TN, FP]
            [-100, 90]   # [FN, TP]
        ])
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate costs/benefits
    tn_benefit = tn * cost_matrix[0, 0]
    fp_cost = fp * cost_matrix[0, 1]
    fn_cost = fn * cost_matrix[1, 0]
    tp_benefit = tp * cost_matrix[1, 1]
    
    # Calculate total
    total = tn_benefit + fp_cost + fn_cost + tp_benefit
    
    # Return results
    return {
        'total': total,
        'tn_benefit': tn_benefit,
        'fp_cost': fp_cost,
        'fn_cost': fn_cost,
        'tp_benefit': tp_benefit,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }

def compare_models(models, X_test, y_test):
    """
    Compare multiple models on the same test data
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Add model name
        metrics['model'] = name
        
        # Store results
        results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'average_precision']
    df = df[cols]
    
    return df
