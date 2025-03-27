# House Price Prediction Web Application

This Django web application predicts house prices using three different machine learning techniques:
1. Polynomial Regression - For predicting exact house prices
2. Logistic Regression - For classifying houses as expensive or affordable
3. k-NN Classifier - For categorizing houses into Basic, Standard, or Luxury categories

## Features

- Modern and responsive web interface
- Three different prediction techniques
- Real-time predictions
- Data preprocessing and model training pipeline
- Model persistence using joblib

## Prerequisites

- Python 3.8 or higher
- Django 5.0.2
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd house-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `data` directory and place your house dataset (CSV file) in it:
```bash
mkdir data
# Place your house_data.csv file in the data directory
```

5. Train the models:
```bash
python prediction/train_models.py
```

6. Run database migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

7. Start the development server:
```bash
python manage.py runserver
```

8. Open your web browser and navigate to `http://127.0.0.1:8000/`

## Dataset Format

The application expects a CSV file with the following columns:
- area: Area of the house in square feet
- bedrooms: Number of bedrooms
- bathrooms: Number of bathrooms
- stories: Number of stories
- mainroad: Main road access (yes/no)
- guestroom: Guest room availability (yes/no)
- basement: Basement availability (yes/no)
- hotwaterheating: Hot water heating availability (yes/no)
- airconditioning: Air conditioning availability (yes/no)
- parking: Number of parking spaces
- prefarea: Preferred area (yes/no)
- furnishingstatus: Furnishing status (furnished/semi-furnished/unfurnished)
- price: House price (target variable)

## Usage

1. On the home page, select one of the three prediction techniques
2. Fill in the house details in the form
3. Click the predict button to get the result
4. The prediction will be displayed below the form

## Project Structure

```
house_prediction/
├── data/
│   └── house_data.csv
├── models/
│   ├── polynomial_model.joblib
│   ├── polynomial_preprocessor.joblib
│   ├── logistic_model.joblib
│   ├── logistic_preprocessor.joblib
│   ├── knn_model.joblib
│   └── knn_preprocessor.joblib
├── prediction/
│   ├── templates/
│   │   └── prediction/
│   │       ├── base.html
│   │       ├── home.html
│   │       ├── polynomial_regression.html
│   │       ├── logistic_regression.html
│   │       └── knn_classifier.html
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── train_models.py
├── house_prediction/
│   ├── settings.py
│   └── urls.py
├── manage.py
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 