"""
Integration test for TechnicalAnalyzer with real PSX data

This script tests the technical indicator calculations with actual stock data.
"""

import pandas as pd
import sys
import os

# Add the psx_ai_advisor module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'psx_ai_advisor'))

from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
from psx_ai_advisor.data_storage import DataStorage


def test_with_real_data():
    """Test TechnicalAnalyzer with real stock data."""
    print("Testing TechnicalAnalyzer with real PSX data...")
    
    # Initialize components
    analyzer = TechnicalAnalyzer()
    storage = DataStorage()
    
    # Get available stock symbols
    symbols = storage.get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    if not symbols:
        print("No stock data available. Please run data acquisition first.")
        return
    
    # Test with the first available symbol
    test_symbol = symbols[0]
    print(f"\nTesting with symbol: {test_symbol}")
    
    # Load stock data
    stock_data = storage.load_stock_data(test_symbol)
    print(f"Loaded {len(stock_data)} rows of data")
    print(f"Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
    print(f"Columns: {list(stock_data.columns)}")
    
    # Display sample of original data
    print(f"\nOriginal data sample:")
    print(stock_data.head().to_string())
    
    # Add technical indicators
    print(f"\nAdding technical indicators...")
    enriched_data = analyzer.add_all_indicators(stock_data)
    
    print(f"Enriched data columns: {list(enriched_data.columns)}")
    
    # Display sample of enriched data
    print(f"\nEnriched data sample (last 5 rows):")
    print(enriched_data.tail().to_string())
    
    # Get indicator summary
    print(f"\nIndicator summary for {test_symbol}:")
    summary = analyzer.get_indicator_summary(enriched_data)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test saving enriched data
    print(f"\nTesting data storage with indicators...")
    success = storage.save_stock_data(f"{test_symbol}_with_indicators", enriched_data)
    if success:
        print(f"✓ Successfully saved enriched data for {test_symbol}")
        
        # Verify we can load it back
        loaded_enriched = storage.load_stock_data(f"{test_symbol}_with_indicators")
        print(f"✓ Successfully loaded enriched data: {len(loaded_enriched)} rows")
        print(f"  Columns: {list(loaded_enriched.columns)}")
    else:
        print(f"✗ Failed to save enriched data for {test_symbol}")
    
    return enriched_data


if __name__ == "__main__":
    # Set up basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        result = test_with_real_data()
        if result is not None:
            print(f"\nIntegration test completed successfully!")
            print(f"Final enriched data shape: {result.shape}")
        else:
            print("Integration test completed but no data was processed.")
            
    except Exception as e:
        print(f"Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()