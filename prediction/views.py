from django.shortcuts import render
from django.http import JsonResponse
from .models import HouseData, PredictionResult
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load models and preprocessors
def load_models():
    models_dir = 'models'
    models = {}
    
    try:
        # Load polynomial regression model
        poly_model = joblib.load(os.path.join(models_dir, 'polynomial_model.joblib'))
        poly_features = joblib.load(os.path.join(models_dir, 'polynomial_preprocessor.joblib'))
        models['polynomial'] = (poly_model, poly_features)
        
        # Load logistic regression model
        log_model = joblib.load(os.path.join(models_dir, 'logistic_model.joblib'))
        log_scaler = joblib.load(os.path.join(models_dir, 'logistic_preprocessor.joblib'))
        models['logistic'] = (log_model, log_scaler)
        
        # Load k-NN classifier
        knn_model = joblib.load(os.path.join(models_dir, 'knn_model.joblib'))
        knn_scaler = joblib.load(os.path.join(models_dir, 'knn_preprocessor.joblib'))
        models['knn'] = (knn_model, knn_scaler)
        
        return models
    except FileNotFoundError:
        return None

# Load models at startup
models = load_models()

# Load the label encoder for location categories
le = joblib.load('models/location_encoder.joblib')

def home(request):
    return render(request, 'prediction/home.html')

def index(request):
    return render(request, 'prediction/index.html')

def polynomial(request):
    if request.method == 'POST':
        try:
            # Get input values
            square_feet = float(request.POST.get('square_feet'))
            bedrooms = int(request.POST.get('bedrooms'))
            location = request.POST.get('location')
            
            # Encode location
            location_encoded = le.transform([location])[0]
            
            # Create input array
            input_data = np.array([[square_feet, bedrooms, location_encoded]])
            
            # Load the model and preprocessor
            model = joblib.load('models/polynomial_model.joblib')
            poly_features = joblib.load('models/polynomial_preprocessor.joblib')
            
            # Transform input data
            input_poly = poly_features.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_poly)[0]
            
            return JsonResponse({
                'success': True,
                'prediction': f'Predicted Price: ${prediction:,.2f}'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'prediction/polynomial.html')

def logistic(request):
    if request.method == 'POST':
        try:
            # Get input values
            square_feet = float(request.POST.get('square_feet'))
            bedrooms = int(request.POST.get('bedrooms'))
            location = request.POST.get('location')
            
            # Encode location
            location_encoded = le.transform([location])[0]
            
            # Create input array
            input_data = np.array([[square_feet, bedrooms, location_encoded]])
            
            # Load the model and preprocessor
            model = joblib.load('models/logistic_model.joblib')
            scaler = joblib.load('models/logistic_preprocessor.joblib')
            
            # Scale input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            result = "Affordable" if prediction == 1 else "Expensive"
            return JsonResponse({
                'success': True,
                'prediction': f'Prediction: {result}'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'prediction/logistic.html')

def knn(request):
    if request.method == 'POST':
        try:
            # Get input values
            square_feet = float(request.POST.get('square_feet'))
            bedrooms = int(request.POST.get('bedrooms'))
            location = request.POST.get('location')
            
            # Encode location
            location_encoded = le.transform([location])[0]
            
            # Create input array
            input_data = np.array([[square_feet, bedrooms, location_encoded]])
            
            # Load the model and preprocessor
            model = joblib.load('models/knn_model.joblib')
            scaler = joblib.load('models/knn_preprocessor.joblib')
            
            # Scale input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            return JsonResponse({
                'success': True,
                'prediction': f'Predicted Category: {prediction}'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'prediction/knn.html')
