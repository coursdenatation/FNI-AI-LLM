"""
Test suite for African Language Models project
"""

import pytest
import pandas as pd
from src.year1.main import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """
    Test data loading functionality
    """
    data = load_data('data/sample.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    """
    Test data preprocessing functionality
    """
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

def test_train_model():
    """
    Test model training functionality
    """
    # Create sample data
    X_train = pd.DataFrame({'feature1': [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])
    
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')

def test_evaluate_model():
    """
    Test model evaluation functionality
    """
    # Create sample data
    X_test = pd.DataFrame({'feature1': [1, 2]})
    y_test = pd.Series([0, 1])
    
    # Create a simple model
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_test, y_test)
    
    # Test evaluation doesn't raise errors
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    pytest.main()