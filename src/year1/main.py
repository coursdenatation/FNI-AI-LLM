"""
Main application for African Language Models project
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.year1.utils import (
    get_data_path,
    get_model_path,
    create_directories,
    load_config,
    validate_data
)

def load_data(file_path):
    """
    Load and preprocess data
    """
    data = pd.read_csv(file_path)
    validate_data(data)
    return data

def preprocess_data(data):
    """
    Preprocess data for model training
    """
    # Example preprocessing steps
    data = data.dropna()
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Train a simple model
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

def main():
    """
    Main function to run the application
    """
    config = load_config()
    logger = logging.getLogger(__name__)
    
    logger.info("African Language Models Project")
    logger.info("Creating directories...")
    create_directories()
    
    logger.info("Loading data...")
    data = load_data(get_data_path() / 'sample.csv')
    
    logger.info("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    
    logger.info("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()