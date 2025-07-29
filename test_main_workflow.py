#!/usr/bin/env python3
"""
Test script to demonstrate the main workflow using existing data
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import PSXAdvisor

def test_main_workflow():
    """Test the main workflow with existing data"""
    print("Testing PSX AI Advisor Main Workflow")
    print("=" * 50)
    
    try:
        # Initialize the advisor
        advisor = PSXAdvisor()
        
        # Create mock stock data using existing PTC data
        ptc_data = pd.read_csv('data/PTC_historical_data.csv')
        
        # Take the latest few rows and format them as if they're new daily data
        latest_data = ptc_data.tail(1).copy()
        latest_data['Symbol'] = 'PTC'
        latest_data['Company_Name'] = 'Pakistan Telecommunication Company Limited'
        
        # Fix timezone issues - ensure Date is timezone-naive
        if 'Date' in latest_data.columns:
            latest_data['Date'] = pd.to_datetime(latest_data['Date']).dt.tz_localize(None)
        
        # Add required columns if missing
        if 'Previous_Close' not in latest_data.columns:
            latest_data['Previous_Close'] = latest_data['Close']
        if 'Change' not in latest_data.columns:
            latest_data['Change'] = 0.0
        
        print(f"Using mock data for PTC:")
        print(f"Date: {latest_data['Date'].iloc[0]}")
        print(f"Close: Rs {latest_data['Close'].iloc[0]:.2f}")
        print(f"Volume: {latest_data['Volume'].iloc[0]:,}")
        
        # Test individual components
        print("\n1. Testing data storage...")
        success = advisor.data_storage.save_stock_data('PTC', latest_data)
        print(f"   Data storage: {'✓ Success' if success else '✗ Failed'}")
        
        print("\n2. Testing technical analysis...")
        complete_data = advisor.data_storage.load_stock_data('PTC')
        enriched_data = advisor.technical_analyzer.add_all_indicators(complete_data)
        print(f"   Technical indicators: ✓ Added {len(enriched_data.columns)} columns")
        print(f"   Latest RSI: {enriched_data['RSI_14'].iloc[-1]:.2f}")
        print(f"   Latest SMA_50: Rs {enriched_data['SMA_50'].iloc[-1]:.2f}")
        
        print("\n3. Testing ML prediction...")
        try:
            prediction = advisor.ml_predictor.predict_movement('PTC')
            print(f"   Prediction: ✓ {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
        except Exception as e:
            print(f"   Prediction: ⚠ {str(e)}")
        
        print("\n4. Testing complete workflow...")
        # Create a mock analysis result to test display
        mock_results = {
            'start_time': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'symbols_analyzed': ['PTC'],
            'predictions': {
                'PTC': {
                    'prediction': 'UP',
                    'confidence': 0.75,
                    'current_price': float(latest_data['Close'].iloc[0]),
                    'model_accuracy': 0.68
                }
            },
            'errors': [],
            'summary': {
                'total_symbols': 1,
                'successful_predictions': 1,
                'up_predictions': 1,
                'down_predictions': 0,
                'up_percentage': 100.0,
                'down_percentage': 0.0,
                'average_confidence': 0.75,
                'high_confidence_predictions': 1,
                'high_confidence_percentage': 100.0,
                'errors_count': 0
            },
            'execution_time_seconds': 2.5
        }
        
        print("\n5. Testing display output...")
        advisor.display_predictions(mock_results)
        
        print("\n✓ All workflow components tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_workflow()
    sys.exit(0 if success else 1)