import os
import joblib
import numpy as np
import pandas as pd

def load_model(model_name='best_model'):
    """
    Load a trained model
    
    Args:
        model_name: Name of the model to load (without .pkl extension)
        
    Returns:
        Trained model
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', f"{model_name}.pkl")
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_name}.pkl not found.")
        print("Please train the model first by running train_model.py")
        return None

def load_scaler():
    """
    Load the amount scaler
    
    Returns:
        Trained StandardScaler for the Amount feature
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(current_dir, '..', '..', 'models', 'amount_scaler.pkl')
        scaler = joblib.load(scaler_path)
        print("Amount scaler loaded successfully")
        return scaler
    except FileNotFoundError:
        print("Error: Amount scaler not found.")
        print("Please run preprocess.py first")
        return None

def preprocess_transaction(transaction, scaler):
    """
    Preprocess a single transaction for prediction
    
    Args:
        transaction: Dictionary or Series containing transaction data
        scaler: Trained StandardScaler for the Amount feature
        
    Returns:
        Preprocessed transaction data as a DataFrame
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(transaction, dict):
        transaction = pd.Series(transaction)
    
    # Create a DataFrame with a single row
    df = pd.DataFrame([transaction])
    
    # Handle the 'Time' feature if present
    if 'Time' in df.columns:
        df['Hour'] = df['Time'] / 3600 % 24
        df = df.drop('Time', axis=1)
    
    # Scale the Amount feature if present
    if 'Amount' in df.columns and scaler is not None:
        df['Amount'] = scaler.transform(df[['Amount']])
    
    # Remove the Class column if present (for testing purposes)
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    
    # Ensure features are in the correct order expected by the model
    # Get expected feature names from the model's feature_names_in_ attribute
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', 'best_model.pkl')
        model = joblib.load(model_path)
        
        # If the model has feature_names_in_ attribute, reorder columns accordingly
        if hasattr(model, 'feature_names_in_'):
            # Get available features (intersection of model features and df columns)
            available_features = [f for f in model.feature_names_in_ if f in df.columns]
            
            # Reorder columns to match model's expected order
            if available_features:
                df = df[available_features]
                print(f"Reordered features to match model's expected order: {available_features}")
    except Exception as e:
        print(f"Warning: Could not reorder features: {str(e)}")
    
    return df

def predict_fraud(transaction, model=None, scaler=None, threshold=0.2):
    """
    Predict whether a transaction is fraudulent
    
    Args:
        transaction: Dictionary or Series containing transaction data
        model: Trained model (if None, the best model will be loaded)
        scaler: Trained StandardScaler (if None, the scaler will be loaded)
        threshold: Probability threshold for classifying as fraud (default lowered to 0.2)
        
    Returns:
        Dictionary containing prediction results
    """
    # Load model and scaler if not provided
    if model is None:
        model = load_model()
    
    if scaler is None:
        scaler = load_scaler()
    
    if model is None or scaler is None:
        return {"error": "Failed to load model or scaler"}
    
    # Preprocess the transaction
    X = preprocess_transaction(transaction, scaler)
    
    # Make prediction
    try:
        # Get probability of fraud
        fraud_prob = model.predict_proba(X)[0, 1]
        print(f"Fraud probability: {fraud_prob:.6f}, Threshold: {threshold}")
        
        # Classify as fraud if probability exceeds threshold
        is_fraud = fraud_prob >= threshold
        
        # Get feature importance if available
        feature_importance = {}
        explanation = ""
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models like Random Forest, Gradient Boosting
            importances = model.feature_importances_
            feature_names = X.columns
            
            # Create a dictionary of feature importances
            for name, importance in zip(feature_names, importances):
                feature_importance[name] = float(importance)
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:5]  # Top 5 most important features
            
            # Generate explanation
            if is_fraud:
                explanation = "This transaction was flagged as fraudulent primarily due to unusual patterns in: "
                explanation += ", ".join([f"{name} (importance: {importance:.4f})" for name, importance in top_features])
                explanation += ". These features showed significant deviation from normal transaction patterns."
            else:
                explanation = "This transaction appears legitimate based on its patterns across all features, "
                explanation += "particularly in the most important features: "
                explanation += ", ".join([f"{name}" for name, _ in top_features])
        
        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_prob),
            "threshold": threshold,
            "feature_importance": feature_importance,
            "explanation": explanation
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def predict_batch(transactions, model=None, scaler=None, threshold=0.5):
    """
    Predict fraud for a batch of transactions
    
    Args:
        transactions: DataFrame containing multiple transactions
        model: Trained model (if None, the best model will be loaded)
        scaler: Trained StandardScaler (if None, the scaler will be loaded)
        threshold: Probability threshold for classifying as fraud
        
    Returns:
        DataFrame with original data and prediction results
    """
    # Load model and scaler if not provided
    if model is None:
        model = load_model()
    
    if scaler is None:
        scaler = load_scaler()
    
    if model is None or scaler is None:
        print("Failed to load model or scaler")
        return None
    
    # Create a copy of the transactions
    results = transactions.copy()
    
    # Preprocess the transactions
    X = transactions.copy()
    
    # Handle the 'Time' feature if present
    if 'Time' in X.columns:
        X['Hour'] = X['Time'] / 3600 % 24
        X = X.drop('Time', axis=1)
    
    # Scale the Amount feature if present
    if 'Amount' in X.columns:
        X['Amount'] = scaler.transform(X[['Amount']])
    
    # Remove the Class column if present (for testing purposes)
    if 'Class' in X.columns:
        X = X.drop('Class', axis=1)
    
    # Make predictions
    try:
        # Get probability of fraud
        fraud_probs = model.predict_proba(X)[:, 1]
        
        # Classify as fraud if probability exceeds threshold
        is_fraud = fraud_probs >= threshold
        
        # Add results to the DataFrame
        results['fraud_probability'] = fraud_probs
        results['is_fraud'] = is_fraud
        
        return results
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
        return None

def main():
    """Test the prediction functionality with sample data"""
    print("Testing fraud prediction with sample data...")
    
    # Load the model and scaler
    model = load_model()
    scaler = load_scaler()
    
    if model is None or scaler is None:
        return
    
    # Create a sample transaction
    # Note: This is just an example with random values
    sample_transaction = {
        'Time': 80000,
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 149.62
    }
    
    # Make prediction
    result = predict_fraud(sample_transaction, model, scaler)
    
    # Print result
    print("\nPrediction result:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
