import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_class_distribution(y, title='Class Distribution'):
    """
    Plot the distribution of classes
    
    Args:
        y: Target variable (0 for normal, 1 for fraud)
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Count the occurrences of each class
    counts = pd.Series(y).value_counts().sort_index()
    
    # Create a bar plot
    ax = sns.barplot(x=counts.index, y=counts.values)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts.values):
        ax.text(i, count + 0.1, f"{count}", ha='center')
    
    # Add percentage labels
    total = len(y)
    for i, count in enumerate(counts.values):
        percentage = count / total * 100
        ax.text(i, count / 2, f"{percentage:.2f}%", ha='center')
    
    plt.title(title)
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    
    plt.close()

def plot_feature_distributions(X, y, n_features=10):
    """
    Plot the distribution of top features by class
    
    Args:
        X: Feature matrix
        y: Target variable (0 for normal, 1 for fraud)
        n_features: Number of features to plot
    """
    # Create a DataFrame with features and target
    data = X.copy()
    data['Class'] = y
    
    # Select the top n features (excluding 'Class')
    features = X.columns[:n_features]
    
    # Create plots directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot each feature distribution by class
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # Plot the distribution for each class
        sns.histplot(data=data, x=feature, hue='Class', kde=True, element='step')
        
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Class', labels=['Normal', 'Fraud'])
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f'distribution_{feature}.png'))
        plt.close()

def plot_correlation_matrix(X):
    """
    Plot the correlation matrix of features
    
    Args:
        X: Feature matrix
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
    
    plt.close()

def plot_roc_curves(results):
    """
    Plot ROC curves for multiple models
    
    Args:
        results: Dictionary of model evaluation results
    """
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
    
    plt.close()

def plot_precision_recall_curves(models, X_test, y_test):
    """
    Plot Precision-Recall curves for multiple models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'precision_recall_curves.png'))
    
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and get top N
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    
    plt.close()

def plot_pca_visualization(X, y, n_components=2):
    """
    Visualize data using PCA
    
    Args:
        X: Feature matrix
        y: Target variable (0 for normal, 1 for fraud)
        n_components: Number of PCA components
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Class'] = y
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with different colors for each class
    sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, palette=['blue', 'red'], alpha=0.6)
    
    plt.title('PCA Visualization of Credit Card Transactions')
    plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(title='Class', labels=['Normal', 'Fraud'])
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'pca_visualization.png'))
    
    plt.close()
    
    # Return explained variance ratio
    return pca.explained_variance_ratio_

def plot_tsne_visualization(X, y, perplexity=30, n_iter=1000):
    """
    Visualize data using t-SNE
    
    Args:
        X: Feature matrix
        y: Target variable (0 for normal, 1 for fraud)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
    """
    # Apply t-SNE
    # Note: t-SNE can be slow for large datasets, so we might want to sample
    if X.shape[0] > 5000:
        # Sample 5000 points
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]
    else:
        X_sample = X
        y_sample = y
    
    print("Applying t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Class'] = y_sample
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with different colors for each class
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Class', data=tsne_df, palette=['blue', 'red'], alpha=0.6)
    
    plt.title('t-SNE Visualization of Credit Card Transactions')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class', labels=['Normal', 'Fraud'])
    plt.tight_layout()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', '..', 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'tsne_visualization.png'))
    
    plt.close()
