import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.models.predict import predict_fraud, load_model, load_scaler

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model, scaler
    
    # Load model and scaler if not already loaded
    if model is None:
        model = load_model()
    
    if scaler is None:
        scaler = load_scaler()
    
    if model is None or scaler is None:
        return jsonify({'error': 'Failed to load model or scaler'})
    
    # Get data from request
    if request.is_json:
        # API request with JSON data
        data = request.get_json()
        
        # Check if we have a single transaction or multiple
        if isinstance(data, list):
            # Multiple transactions
            results = []
            for transaction in data:
                result = predict_fraud(transaction, model, scaler)
                results.append(result)
            return jsonify(results)
        else:
            # Single transaction
            result = predict_fraud(data, model, scaler)
            return jsonify(result)
    else:
        # Form submission
        try:
            # Get form data
            transaction = {}
            
            # Process form fields
            for field in request.form:
                if field.startswith('V') or field in ['Time', 'Amount']:
                    try:
                        transaction[field] = float(request.form[field])
                    except ValueError:
                        return jsonify({'error': f'Invalid value for {field}'})
            
            # Make prediction
            result = predict_fraud(transaction, model, scaler)
            
            # Print the result for debugging
            print("Prediction result:", result)
            
            # Instead of using a custom object, let's revert to using the dictionary directly
            # and update the template to use dictionary access
            
            # Return result
            return render_template('result.html', result=result, transaction=transaction)
        
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    global model, scaler
    
    # Load model and scaler if not already loaded
    if model is None:
        model = load_model()
    
    if scaler is None:
        scaler = load_scaler()
    
    if model is None or scaler is None:
        return jsonify({'error': 'Failed to load model or scaler'})
    
    # Get data from request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'})
    
    data = request.get_json()
    
    # Check if we have a single transaction or multiple
    if isinstance(data, list):
        # Multiple transactions
        results = []
        for transaction in data:
            result = predict_fraud(transaction, model, scaler)
            results.append(result)
        return jsonify(results)
    else:
        # Single transaction
        result = predict_fraud(data, model, scaler)
        return jsonify(result)

@app.route('/demo')
def demo():
    """Render the demo page with a form for testing"""
    return render_template('demo.html')

@app.route('/visualizations')
def visualizations():
    """Render the visualizations page"""
    # Get list of available plots
    plots_dir = os.path.join('static', 'plots')
    
    # Check if directory exists
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
        return render_template('visualizations.html', plots=[])
    
    # Get list of plot files
    plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    return render_template('visualizations.html', plots=plots)

def create_directories():
    """Create necessary directories for the application"""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create static directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)

if __name__ == '__main__':
    # Create necessary directories
    create_directories()
    
    # Run the app
    app.run(debug=True)
