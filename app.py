from flask import Flask, request, jsonify       #Flask → to create API , request → to get data from frontend,jsonify → to send response in JSON format
from flask_cors import CORS                     #Allows frontend (React/HTML) to call backend API
import pandas as pd                             #For data handling (DataFrame & numerical operations)
import numpy as np
import joblib                                   #To load saved ML models
from preprocessing import CarDataPreprocessor   #CarDataPreprocessor → cleans & converts input data
from ml_model import CarPricePredictor          #CarPricePredictor → ML model prediction logic
import os                                       #For file & folder operations

app = Flask(__name__)                          #Creates Flask app
CORS(app)                                      #Enables CORS for all routes

# Global variables
preprocessor = None                            #preprocessing object
predictor = None                               #trained ML model

def load_models():                             #Function to load trained models when server starts
    """Load pre-trained models and preprocessor""" 
    global preprocessor, predictor             #Allows modifying global variables
    
    try:
        # Load preprocessor
        preprocessor = CarDataPreprocessor()      #Create preprocessing object
        preprocessor.load_preprocessor('models/') #Load saved preprocessing files from models/ folder
        
        # Load predictor
        predictor = CarPricePredictor()           #Create ML predictor object   
        predictor.set_preprocessor(preprocessor)  #Attach preprocessor to ML model
        predictor.load_models('models/')          #Load trained ML model   
        
        print("Models loaded successfully!")      
        return True                               #If everything loaded successfully
    except Exception as e:                        #If error occurs, show error and return False
        print(f"Error loading models: {str(e)}")  #
        return False

@app.route('/')                                  #When user opens http://localhost:5000/
def home():
    return jsonify({                             #Sends API info and available endpoints
        "message": "Car Price Prediction API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Predict car price",
            "/health": "GET - API health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": preprocessor is not None and predictor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        
        # Validate required fields
        required_fields = ['year', 'km_driven', 'fuel', 'seller_type', 
                          'transmission', 'owner', 'engine', 'max_power', 'seats']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "required_fields": required_fields
                }), 400
        
        # Add dummy name for preprocessing
        data['name'] = 'Maruti Suzuki'
        
        # Make prediction
        prediction = predictor.predict_price(data, model_name='random_forest')
        
        # Format response
        response = {
            "predicted_price": round(prediction, 2),
            "formatted_price": f"₹{prediction:,.2f}",
            "status": "success",
            "input_data": data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        cars = data.get('cars', [])
        
        if not cars:
            return jsonify({"error": "No cars provided"}), 400
        
        predictions = []
        for car in cars:
            car['name'] = 'Maruti Suzuki'  # Add dummy name
            prediction = predictor.predict_price(car, model_name='random_forest')
            predictions.append({
                "input": car,
                "predicted_price": round(prediction, 2),
                "formatted_price": f"₹{prediction:,.2f}"
            })
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load models before starting the server
    if load_models():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please train models first.")