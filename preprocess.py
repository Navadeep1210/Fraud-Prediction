import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(data_path='../../data/creditcard.csv'):
    """
    Load the credit card fraud dataset
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame containing the credit card data
    """
    print(f"Loading data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in the data directory.")
        print("See data/README.md for instructions.")
        return None

def explore_data(data):
    """
    Perform basic exploratory data analysis
    
    Args:
        data: DataFrame containing the credit card data
        
    Returns:
        None (prints summary statistics)
    """
    print("\n--- Data Exploration ---")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\nSummary statistics:")
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values:")
    print(missing_values)
    
    # Check class distribution
    fraud_count = data['Class'].value_counts()
    print("\nClass distribution:")
    print(fraud_count)
    fraud_percentage = fraud_count[1] / len(data) * 100
    print(f"Percentage of fraudulent transactions: {fraud_percentage:.4f}%")

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data for model training
    
    Args:
        data: DataFrame containing the credit card data
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits
    """
    print("\n--- Data Preprocessing ---")
    
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Handle the 'Time' feature
    # Convert Time to a more meaningful feature (hour of day)
    X['Hour'] = X['Time'] / 3600 % 24
    X = X.drop('Time', axis=1)
    
    # Scale the Amount feature
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Save the scaler for later use
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'amount_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Amount scaler saved to {scaler_path}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Save the preprocessed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    data_path = os.path.join(processed_dir, 'preprocessed_data.pkl')
    joblib.dump((X_train, X_test, y_train, y_test), data_path)
    print(f"Preprocessed data saved to {data_path}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to run the preprocessing pipeline"""
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'creditcard.csv')
    
    # Load the data
    data = load_data(data_path)
    
    if data is not None:
        # Explore the data
        explore_data(data)
        
        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()
