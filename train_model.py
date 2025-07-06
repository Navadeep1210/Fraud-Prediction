import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def load_preprocessed_data():
    """
    Load the preprocessed data
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'preprocessed_data.pkl')
        X_train, X_test, y_train, y_test = joblib.load(data_path)
        print("Preprocessed data loaded successfully")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run preprocess.py first.")
        return None, None, None, None

def handle_class_imbalance(X_train, y_train, method='smote', sampling_strategy=0.1, fast_mode=True):
    """
    Handle class imbalance using various techniques
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: Method to handle imbalance ('smote', 'undersampling', 'combined')
        sampling_strategy: Desired ratio of minority to majority class
        
    Returns:
        X_resampled, y_resampled: Resampled training data
    """
    print(f"\nHandling class imbalance using {method.upper()}")
    
    if method == 'smote':
        # Oversample the minority class using SMOTE
        # Use a smaller sampling_strategy in fast mode
        actual_strategy = min(0.05, sampling_strategy) if fast_mode else sampling_strategy
        print(f"Using SMOTE with sampling_strategy={actual_strategy}")
        smote = SMOTE(sampling_strategy=actual_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    elif method == 'undersampling':
        # Undersample the majority class
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    elif method == 'combined':
        # Combine SMOTE and undersampling
        over = SMOTE(sampling_strategy=0.1, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    else:
        # No resampling
        X_resampled, y_resampled = X_train, y_train
    
    # Print class distribution after resampling
    unique, counts = np.unique(y_resampled, return_counts=True)
    print("Class distribution after resampling:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")
    
    return X_resampled, y_resampled

def train_models(X_train, y_train, fast_mode=True):
    """
    Train multiple models for fraud detection
    
    Args:
        X_train: Training features
        y_train: Training labels
        fast_mode: If True, use faster training settings
        
    Returns:
        Dictionary of trained models
    """
    print("\n--- Training Models ---")
    
    # Use fewer estimators and parallel processing in fast mode
    n_estimators = 25 if fast_mode else 100
    n_jobs = -1  # Use all available CPU cores
    
    models = {
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=1000, 
                                                 random_state=42, n_jobs=n_jobs),
        'random_forest': RandomForestClassifier(n_estimators=n_estimators, 
                                              class_weight='balanced', 
                                              random_state=42, 
                                              n_jobs=n_jobs,
                                              max_depth=10),  # Limit tree depth
        'gradient_boosting': GradientBoostingClassifier(n_estimators=n_estimators, 
                                                      random_state=42,
                                                      max_depth=5)  # Limit tree depth
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = pd.Timestamp.now()
        model.fit(X_train, y_train)
        end_time = pd.Timestamp.now()
        training_time = (end_time - start_time).total_seconds()
        trained_models[name] = model
        print(f"{name} trained successfully in {training_time:.2f} seconds")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the trained models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n--- Model Evaluation ---")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        }
    
    return results

def save_models(models, results):
    """
    Save the trained models and evaluation results
    
    Args:
        models: Dictionary of trained models
        results: Dictionary of evaluation metrics
    """
    print("\n--- Saving Models ---")
    
    # Create models directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    print(f"Best model: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})")
    
    # Save all models
    for name, model in models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    # Save the best model separately
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Save model results
    results_path = os.path.join(models_dir, "model_results.pkl")
    joblib.dump(results, results_path)
    print(f"Model results saved to {results_path}")

def plot_results(results):
    """
    Plot evaluation results
    
    Args:
        results: Dictionary of evaluation metrics
    """
    print("\n--- Plotting Results ---")
    
    # Create plots directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result['fpr'], result['tpr'], label=f"{name} (AUC = {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
    
    # Plot metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.subplot(2, 3, i+1)
        sns.barplot(x=model_names, y=values)
        plt.title(metric.upper())
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    
    print(f"Plots saved to {plots_dir}")

def main(fast_mode=True, sample_size=None):
    """Main function to run the model training pipeline
    
    Args:
        fast_mode: If True, use faster training settings
        sample_size: If provided, use only a subset of data for faster testing
    """
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        return
    
    # Use a smaller subset of data if requested (for testing)
    if sample_size is not None and sample_size < len(X_train):
        print(f"\nUsing a subset of {sample_size} samples for faster training")
        # Ensure we maintain class distribution
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        for train_idx, _ in sss.split(X_train, y_train):
            X_train_sample = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_train_sample = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_train, y_train = X_train_sample, y_train_sample
    
    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train, method='smote', fast_mode=fast_mode)
    
    # Train models
    models = train_models(X_resampled, y_resampled, fast_mode=fast_mode)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Save models
    save_models(models, results)
    
    # Plot results
    plot_results(results)
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--fast', action='store_true', help='Use fast mode with reduced parameters')
    parser.add_argument('--sample', type=int, default=None, help='Use a subset of data for faster testing')
    args = parser.parse_args()
    
    # Default to fast mode unless explicitly set to False
    fast_mode = True if args.fast or args.fast is None else False
    
    print(f"Running in {'fast' if fast_mode else 'standard'} mode")
    if args.sample:
        print(f"Using sample size of {args.sample}")
    
    main(fast_mode=fast_mode, sample_size=args.sample)
