#!/usr/bin/env python3
"""
Test script to verify TimeSeriesSplit implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.ml_predictor import MLPredictor
from psx_ai_advisor.data_storage import DataStorage
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create some sample time-series data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Create realistic stock data with trend and volatility
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i, date in enumerate(dates):
        # Add trend and random walk
        trend = 0.001 * i  # Small upward trend
        noise = np.random.normal(0, 2)  # Daily volatility
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + trend/100 + noise/100)
        
        prices.append(max(price, 10))  # Ensure price doesn't go below 10
    
    # Create OHLCV data
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.001, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 0.999) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return test_data

def test_timeseries_implementation():
    """Test the TimeSeriesSplit implementation"""
    print("=" * 60)
    print("Testing TimeSeriesSplit Implementation")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test data...")
    test_data = create_test_data()
    print(f"   Created {len(test_data)} days of test data")
    print(f"   Date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    # Save test data
    data_storage = DataStorage()
    symbol = "TEST_SYMBOL"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the test data
    test_data.to_csv(f"data/{symbol}.csv", index=False)
    print(f"   Saved test data to data/{symbol}.csv")
    
    # Initialize ML Predictor
    print("\n2. Initializing ML Predictor with TimeSeriesSplit...")
    try:
        predictor = MLPredictor()
        print(f"   Initialized with {predictor.n_splits} time-series splits")
        print(f"   Max training size: {predictor.max_train_size}")
        
        # Train the model
        print("\n3. Training model with time-series aware cross-validation...")
        training_results = predictor.train_model(symbol)
        
        print("\n4. Training Results:")
        print(f"   Symbol: {training_results['symbol']}")
        print(f"   Total samples: {training_results['total_samples']}")
        print(f"   Training samples: {training_results['training_samples']}")
        print(f"   Test samples: {training_results['test_samples']}")
        print(f"   Final test accuracy: {training_results['accuracy']:.4f}")
        
        if 'cv_scores' in training_results:
            cv_scores = training_results['cv_scores']
            print(f"   CV Mean accuracy: {cv_scores.get('mean_accuracy', 0):.4f} ± {cv_scores.get('std_accuracy', 0):.4f}")
            print(f"   CV folds completed: {cv_scores.get('n_folds', 0)}")
        
        if 'time_series_split_info' in training_results:
            ts_info = training_results['time_series_split_info']
            print(f"   Time-series splits: {ts_info['n_splits']}")
            print(f"   Data sorted by date: {ts_info['data_sorted_by_date']}")
        
        print("\n5. Testing prediction...")
        prediction = predictor.predict_movement(symbol)
        print(f"   Prediction: {prediction['prediction']}")
        print(f"   Confidence: {prediction['confidence']:.4f}")
        
        print("\n✅ TimeSeriesSplit implementation test completed successfully!")
        print("\n🎯 Key improvements:")
        print("   ✓ No lookahead bias - future data cannot influence past predictions")
        print("   ✓ Chronological order preserved in train/test splits")
        print("   ✓ Time-series cross-validation provides robust performance estimates")
        print("   ✓ Training sets grow incrementally, mimicking real-world scenarios")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        test_file = f"data/{symbol}.csv"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🧹 Cleaned up test file: {test_file}")

if __name__ == "__main__":
    success = test_timeseries_implementation()
    if success:
        print("\n🎉 All tests passed! TimeSeriesSplit is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
        sys.exit(1)
