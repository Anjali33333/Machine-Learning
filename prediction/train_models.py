import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(data_path):
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Convert Location_Category to numeric using LabelEncoder
        le = LabelEncoder()
        df['Location_Category'] = le.fit_transform(df['Location_Category'])
        
        # Save the label encoder
        joblib.dump(le, 'models/location_encoder.joblib')
        
        # Prepare features and targets
        X = df[['Square_Feet', 'Bedrooms', 'Location_Category']]
        
        # Different target variables for different models
        y_price = df['House_Price']
        y_affordable = df['Affordable']
        y_category = df['Price_Category']
        
        # Split the data
        X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
        _, _, y_affordable_train, y_affordable_test = train_test_split(X, y_affordable, test_size=0.2, random_state=42)
        _, _, y_category_train, y_category_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
        
        return (X_train, X_test, y_price_train, y_price_test,
                y_affordable_train, y_affordable_test,
                y_category_train, y_category_test)
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def train_polynomial_regression(X_train, y_train, degree=2):
    try:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        return model, poly
    except Exception as e:
        print(f"Error in polynomial regression training: {str(e)}")
        raise

def train_logistic_regression(X_train, y_train):
    try:
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    except Exception as e:
        print(f"Error in logistic regression training: {str(e)}")
        raise

def train_knn_classifier(X_train, y_train, n_neighbors=5):
    try:
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    except Exception as e:
        print(f"Error in k-NN classifier training: {str(e)}")
        raise

def save_models(models_dict, models_dir='models'):
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save each model and its associated preprocessor
        for name, (model, preprocessor) in models_dict.items():
            joblib.dump(model, f'{models_dir}/{name}_model.joblib')
            joblib.dump(preprocessor, f'{models_dir}/{name}_preprocessor.joblib')
            print(f"Saved {name} model and preprocessor")
    except Exception as e:
        print(f"Error saving models: {str(e)}")
        raise

def main():
    try:
        # Path to your dataset
        data_path = 'data/house_data.csv'
        
        print("Starting model training process...")
        
        # Preprocess the data
        print("Preprocessing data...")
        (X_train, X_test, y_price_train, y_price_test,
         y_affordable_train, y_affordable_test,
         y_category_train, y_category_test) = preprocess_data(data_path)
        
        # Train models
        models_dict = {}
        
        # Polynomial Regression for price prediction
        print("Training polynomial regression model...")
        poly_model, poly_features = train_polynomial_regression(X_train, y_price_train)
        models_dict['polynomial'] = (poly_model, poly_features)
        
        # Logistic Regression for affordable/expensive classification
        print("Training logistic regression model...")
        log_model, log_scaler = train_logistic_regression(X_train, y_affordable_train)
        models_dict['logistic'] = (log_model, log_scaler)
        
        # k-NN Classifier for price category
        print("Training k-NN classifier...")
        knn_model, knn_scaler = train_knn_classifier(X_train, y_category_train)
        models_dict['knn'] = (knn_model, knn_scaler)
        
        # Save the models
        print("Saving models...")
        save_models(models_dict)
        
        print("Models trained and saved successfully!")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 