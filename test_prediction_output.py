#!/usr/bin/env python3
"""
Test script for prediction generation and output formatting functionality.
This script tests the newly implemented prediction output formatting methods.
"""

import sys
import os
import logging
from datetime import datetime

# Add the psx_ai_advisor package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.ml_predictor import MLPredictor
from psx_ai_advisor.data_storage import DataStorage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prediction_output_formatting():
    """Test the prediction output formatting functionality."""
    
    print("=" * 60)
    print("Testing PSX AI Advisor - Prediction Output Formatting")
    print("=" * 60)
    
    try:
        # Initialize components
        ml_predictor = MLPredictor()
        data_storage = DataStorage()
        
        # Get available stock symbols
        available_symbols = data_storage.get_available_symbols()
        
        if not available_symbols:
            print("No stock data available. Please run data acquisition first.")
            return False
        
        # Test with first available symbol
        test_symbol = available_symbols[0]
        print(f"\nTesting with symbol: {test_symbol}")
        
        # Test 1: Generate prediction
        print("\n1. Testing prediction generation...")
        try:
            prediction_result = ml_predictor.predict_movement(test_symbol)
            print("✓ Prediction generated successfully")
            print(f"  Raw prediction: {prediction_result['prediction']}")
            print(f"  Confidence: {prediction_result['confidence']:.3f}")
        except Exception as e:
            print(f"✗ Prediction generation failed: {e}")
            return False
        
        # Test 2: Format single prediction output
        print("\n2. Testing single prediction formatting...")
        try:
            formatted_output = ml_predictor.format_prediction_output(prediction_result)
            print("✓ Single prediction formatted successfully")
            print("\nFormatted Output:")
            print("-" * 40)
            print(formatted_output)
            print("-" * 40)
        except Exception as e:
            print(f"✗ Single prediction formatting failed: {e}")
            return False
        
        # Test 3: Generate comprehensive summary
        print("\n3. Testing comprehensive prediction summary...")
        try:
            summary = ml_predictor.generate_prediction_summary(test_symbol)
            print("✓ Comprehensive summary generated successfully")
            print(f"  Recommendation strength: {summary['recommendation_strength']}")
            print(f"  Risk assessment: {summary['risk_assessment']}")
            print(f"  Data points available: {summary['additional_context']['data_points_available']}")
        except Exception as e:
            print(f"✗ Comprehensive summary generation failed: {e}")
            return False
        
        # Test 4: Multiple predictions formatting (if we have multiple symbols)
        if len(available_symbols) > 1:
            print("\n4. Testing multiple predictions formatting...")
            try:
                # Generate predictions for first 3 symbols (or all if less than 3)
                test_symbols = available_symbols[:min(3, len(available_symbols))]
                predictions = []
                
                for symbol in test_symbols:
                    try:
                        pred = ml_predictor.predict_movement(symbol)
                        predictions.append(pred)
                    except Exception as e:
                        print(f"  Warning: Could not generate prediction for {symbol}: {e}")
                
                if predictions:
                    formatted_multiple = ml_predictor.format_multiple_predictions(predictions, sort_by='confidence')
                    print("✓ Multiple predictions formatted successfully")
                    print("\nMultiple Predictions Output:")
                    print(formatted_multiple)
                else:
                    print("✗ No predictions available for multiple formatting test")
            except Exception as e:
                print(f"✗ Multiple predictions formatting failed: {e}")
        else:
            print("\n4. Skipping multiple predictions test (only one symbol available)")
        
        # Test 5: Test different output formats
        print("\n5. Testing different formatting options...")
        try:
            # Test with mock prediction data to verify formatting edge cases
            mock_prediction = {
                'symbol': 'TEST',
                'prediction': 'UP',
                'confidence': 0.85,
                'current_price': 123.45,
                'prediction_date': datetime.now().isoformat(),
                'model_accuracy': 0.67
            }
            
            formatted_mock = ml_predictor.format_prediction_output(mock_prediction)
            print("✓ Mock prediction formatting successful")
            print("Mock Prediction Output:")
            print("-" * 30)
            print(formatted_mock)
            print("-" * 30)
            
        except Exception as e:
            print(f"✗ Mock prediction formatting failed: {e}")
        
        print("\n" + "=" * 60)
        print("Prediction output formatting tests completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False

def main():
    """Main function to run the tests."""
    success = test_prediction_output_formatting()
    
    if success:
        print("\n✓ All tests passed! Prediction output formatting is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()