# Credit Card Fraud Detection

This project implements a machine learning system to detect fraudulent credit card transactions.

## Project Overview

Credit card fraud is a significant concern in the financial industry. This project uses machine learning algorithms to identify potentially fraudulent transactions based on transaction features.

## Features

- Data preprocessing and exploration
- Feature engineering and selection
- Model training with various algorithms
- Model evaluation and comparison
- Simple web interface for demonstration

## Project Structure

- `data/`: Directory for storing datasets
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `src/`: Source code for the project
  - `data_processing/`: Scripts for data preprocessing
  - `models/`: Implementation of ML models
  - `utils/`: Utility functions
- `app.py`: Flask web application
- `requirements.txt`: Project dependencies

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the dataset (instructions in data/README.md)

3. Run the preprocessing scripts:
   ```
   python src/data_processing/preprocess.py
   ```

4. Train the model:
   ```
   python src/models/train_model.py
   ```

5. Run the web application:
   ```
   python app.py
   ```

## Dataset

This project uses the Credit Card Fraud Detection dataset, which contains transactions made by credit cards. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

Due to confidentiality issues, the original features have been transformed into numerical values using PCA transformation.

## Model Performance

The model achieves the following performance metrics:
- Accuracy: ~99%
- Precision: ~85%
- Recall: ~75%
- F1 Score: ~80%
- AUC-ROC: ~95%

(Note: These are placeholder metrics that will be updated after model training)
