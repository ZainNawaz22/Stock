#!/usr/bin/env python3
"""
Test script for ML Predictor functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.ml_predictor import MLPredictor, MLPredictorError, InsufficientDataError
from psx_ai_advisor.data_storage import DataStorage
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer

def create_sample_data(symbol="TEST", days=300):
    """Create sample stock data for testing"""
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock price data
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 100.0
    prices = [base_price]
    
    # Generate price movements with some trend and volatility
    for i in range(1, len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure price stays positive
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = close * 0.02  # 2% intraday volatility
        
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = low + np.random.uniform(0, high - low)
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(10000, 100000))
        
        data.append({
            'Date': date,
            'Symbol': symbol,
            'Company_Name': f'{symbol} Test Company',
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume,
            'Previous_Close': round(prices[i-1] if i > 0 else close, 2),
            'Change': round(close - (prices[i-1] if i > 0 else close), 2)
        })
    
    return pd.DataFrame(data)

def test_ml_predictor():
    """Test ML Predictor functionality"""
    
    print("Testing ML Predictor...")
    
    try:
        # Initialize components
        ml_predictor = MLPredictor()
        data_storage = DataStorage()
        technical_analyzer = TechnicalAnalyzer()
        
        # Create sample data
        print("Creating sample data...")
        sample_data = create_sample_data("TEST", days=300)
        
        # Add technical indicators
        print("Adding technical indicators...")
        sample_data = technical_analyzer.add_all_indicators(sample_data)
        
        # Save sample data
        print("Saving sample data...")
        data_storage.save_stock_data("TEST", sample_data)
        
        # Test feature preparation
        print("Testing feature preparation...")
        X, y = ml_predictor.prepare_features(sample_data)
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution: UP={y.sum()}, DOWN={len(y)-y.sum()}")
        
        # Test model training
        print("Testing model training...")
        training_results = ml_predictor.train_model("TEST")
        print(f"Training results: {training_results}")
        
        # Test model persistence
        print("Testing model persistence...")
        model_info = ml_predictor.get_model_info("TEST")
        print(f"Model info: {model_info}")
        
        # Test prediction
        print("Testing prediction...")
        prediction = ml_predictor.predict_movement("TEST")
        print(f"Prediction: {prediction}")
        
        # Test model evaluation
        print("Testing model evaluation...")
        evaluation = ml_predictor.evaluate_model("TEST")
        print(f"Evaluation: {evaluation}")
        
        # Test with insufficient data
        print("Testing insufficient data handling...")
        small_data = create_sample_data("SMALL", days=50)
        data_storage.save_stock_data("SMALL", small_data)
        
        try:
            ml_predictor.train_model("SMALL")
            print("ERROR: Should have raised InsufficientDataError")
        except InsufficientDataError as e:
            print(f"Correctly caught InsufficientDataError: {e}")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_predictor()
    sys.exit(0 if success else 1)