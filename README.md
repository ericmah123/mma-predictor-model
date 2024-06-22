# MMA Fight Predictor

## Overview

The MMA Fight Predictor is a machine learning application designed to predict the outcomes of MMA fights based on fighter statistics. The project utilizes a Gradient Boosting Classifier to make predictions, incorporating advanced feature engineering and data preprocessing techniques to enhance model performance.

## Technologies

- Python
- Flask: For building the web application.
- Scikit-learn: For machine learning model implementation.
- Pandas: For data manipulation and preprocessing.
- Tkinter: For creating a graphical user interface.
- Joblib: For model serialization and deserialization.
- HTML/CSS: For creating a responsive web interface.
- Matplotlib and Seaborn: For data visualization and model performance evaluation.

## Features

- Machine Learning Model: Utilizes Gradient Boosting Classifier with PolynomialFeatures and StandardScaler for accurate predictions.
- Data Preprocessing: Handles missing values, unit conversions, feature creation, and one-hot encoding to prepare high-quality input data.
- Web Application: Built with Flask, providing a clean and responsive HTML/CSS interface for user interactions.
- Model Persistence: Uses joblib for efficient saving and loading of the trained model.
- Model Evaluation: Visualizes performance using confusion matrices and ROC curves.

## Installation

1. Clone repository
```
git clone https://github.com/ericmah123/mma-predictor-model.git
cd mma-fight-predictor
```

2. Install the dependencies:
```
pip install -r requirements.txt
```

