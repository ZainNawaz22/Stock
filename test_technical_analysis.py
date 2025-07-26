"""
Test script for TechnicalAnalyzer class

This script tests the technical indicator calculations to ensure they work correctly.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the psx_ai_advisor module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'psx_ai_advisor'))

from technical_analysis import TechnicalAnalyzer


def create_test_data(num_days=300):
    """Create synthetic stock data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    
    # Generate synthetic price data with some trend and volatility
    np.random.seed(42)  # For reproducible results
    base_price = 100
    price_changes = np.random.normal(0, 2, num_days)  # Daily price changes
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = max(prices[-1] + change, 1)  # Ensure price stays positive
        prices.append(new_price)
    
    # Create OHLCV data
    data = {
        'Date': dates,
        'Open': [p * (1 + np.random.uniform(-0.02, 0.02)) for p in prices],
        'High': [p * (1 + abs(np.random.uniform(0, 0.03))) for p in prices],
        'Low': [p * (1 - abs(np.random.uniform(0, 0.03))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, num_days)
    }
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


def test_technical_analyzer():
    """Test the TechnicalAnalyzer class."""
    print("Testing TechnicalAnalyzer...")
    
    # Create test data
    test_df = create_test_data(300)
    print(f"Created test data with {len(test_df)} rows")
    print(f"Date range: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Price range: {test_df['Close'].min():.2f} to {test_df['Close'].max():.2f}")
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer()
    
    # Test individual indicator calculations
    print("\n=== Testing Individual Indicators ===")
    
    # Test SMA calculations
    print("\n1. Testing Simple Moving Averages...")
    sma_50 = analyzer.calculate_sma(test_df['Close'], 50)
    sma_200 = analyzer.calculate_sma(test_df['Close'], 200)
    
    print(f"SMA_50: {sma_50.dropna().iloc[-1]:.2f} (last valid value)")
    print(f"SMA_200: {sma_200.dropna().iloc[-1]:.2f} (last valid value)")
    print(f"SMA_50 non-null values: {sma_50.count()}")
    print(f"SMA_200 non-null values: {sma_200.count()}")
    
    # Test RSI calculation
    print("\n2. Testing RSI...")
    rsi_14 = analyzer.calculate_rsi(test_df['Close'], 14)
    print(f"RSI_14: {rsi_14.dropna().iloc[-1]:.2f} (last valid value)")
    print(f"RSI_14 non-null values: {rsi_14.count()}")
    
    # Test MACD calculation
    print("\n3. Testing MACD...")
    macd_df = analyzer.calculate_macd(test_df['Close'])
    print(f"MACD columns: {list(macd_df.columns)}")
    print(f"MACD: {macd_df['MACD'].dropna().iloc[-1]:.4f} (last valid value)")
    print(f"MACD_Signal: {macd_df['MACD_Signal'].dropna().iloc[-1]:.4f} (last valid value)")
    print(f"MACD_Histogram: {macd_df['MACD_Histogram'].dropna().iloc[-1]:.4f} (last valid value)")
    
    # Test add_all_indicators method
    print("\n=== Testing add_all_indicators Method ===")
    enriched_df = analyzer.add_all_indicators(test_df)
    
    print(f"Original columns: {list(test_df.columns)}")
    print(f"Enriched columns: {list(enriched_df.columns)}")
    
    # Check that all expected indicators were added
    expected_indicators = ['SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram']
    for indicator in expected_indicators:
        if indicator in enriched_df.columns:
            non_null_count = enriched_df[indicator].count()
            last_value = enriched_df[indicator].dropna().iloc[-1] if non_null_count > 0 else "N/A"
            print(f"✓ {indicator}: {non_null_count} values, last = {last_value}")
        else:
            print(f"✗ {indicator}: Missing!")
    
    # Test indicator summary
    print("\n=== Testing Indicator Summary ===")
    summary = analyzer.get_indicator_summary(enriched_df)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test edge cases
    print("\n=== Testing Edge Cases ===")
    
    # Test with insufficient data
    small_df = test_df.head(10)
    try:
        small_enriched = analyzer.add_all_indicators(small_df)
        print("✓ Handled insufficient data gracefully")
        print(f"Small dataset indicators: {[col for col in small_enriched.columns if col not in test_df.columns]}")
    except Exception as e:
        print(f"✗ Error with small dataset: {e}")
    
    # Test with empty DataFrame
    try:
        empty_df = pd.DataFrame()
        empty_enriched = analyzer.add_all_indicators(empty_df)
        print("✓ Handled empty DataFrame gracefully")
    except Exception as e:
        print(f"✗ Error with empty DataFrame: {e}")
    
    print("\n=== Test Complete ===")
    return enriched_df


if __name__ == "__main__":
    # Set up basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        result_df = test_technical_analyzer()
        print(f"\nFinal DataFrame shape: {result_df.shape}")
        print("All tests completed successfully!")
        
        # Save a sample of the results for inspection
        sample_output = result_df.tail(10)
        print(f"\nSample of last 10 rows:")
        print(sample_output.to_string())
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()