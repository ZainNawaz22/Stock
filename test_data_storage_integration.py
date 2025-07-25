#!/usr/bin/env python3
"""
Integration test for DataStorage with actual data from data acquisition
"""

import os
import sys
import tempfile
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_storage import DataStorage
from psx_ai_advisor.data_acquisition import PSXDataAcquisition


def test_integration_with_real_data():
    """Test DataStorage with real data from data acquisition"""
    print("Testing DataStorage integration with real data...")
    
    # Check if we have any existing PDF files to test with
    data_dir = "data"
    pdf_files = []
    
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found for integration testing. Skipping...")
        return True
    
    print(f"Found {len(pdf_files)} PDF files for testing")
    
    # Use the most recent PDF file
    pdf_file = sorted(pdf_files)[-1]
    pdf_path = os.path.join(data_dir, pdf_file)
    
    print(f"Testing with PDF: {pdf_file}")
    
    try:
        # Initialize data acquisition
        acquisition = PSXDataAcquisition()
        
        # Extract data from PDF
        stock_data = acquisition.extract_stock_data(pdf_path)
        print(f"Extracted data for {len(stock_data)} stocks")
        
        if stock_data.empty:
            print("No stock data extracted. Skipping integration test.")
            return True
        
        # Test with temporary storage
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(data_dir=temp_dir)
            
            # Test saving data for a few stocks
            test_symbols = stock_data['Symbol'].head(3).tolist()
            
            for symbol in test_symbols:
                symbol_data = stock_data[stock_data['Symbol'] == symbol]
                
                # Save data
                result = storage.save_stock_data(symbol, symbol_data)
                assert result == True, f"Failed to save data for {symbol}"
                
                # Load and verify
                loaded_data = storage.load_stock_data(symbol)
                assert len(loaded_data) == len(symbol_data), f"Data length mismatch for {symbol}"
                
                # Convert symbols to string for comparison (handle numeric symbols)
                loaded_symbol = str(loaded_data['Symbol'].iloc[0])
                expected_symbol = str(symbol)
                assert loaded_symbol == expected_symbol, f"Symbol mismatch: expected '{expected_symbol}', got '{loaded_symbol}'"
                
                # Validate data integrity
                is_valid = storage.validate_data_integrity(symbol)
                assert is_valid == True, f"Data integrity validation failed for {symbol}"
                
                print(f"✓ Successfully processed {symbol}")
            
            # Test storage statistics
            stats = storage.get_storage_stats()
            assert stats['total_symbols'] == len(test_symbols), "Symbol count mismatch in stats"
            
            print("✓ Integration test with real data passed")
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format_compatibility():
    """Test that DataStorage can handle the exact format from data acquisition"""
    print("Testing data format compatibility...")
    
    # Create sample data in the exact format that data acquisition produces
    sample_data = pd.DataFrame([
        {
            'Date': pd.to_datetime('2025-07-26'),
            'Symbol': 'ENGRO',
            'Company_Name': 'Engro Corporation Limited',
            'Open': 250.50,
            'High': 255.75,
            'Low': 248.25,
            'Close': 253.00,
            'Volume': 125000,
            'Previous_Close': 251.00,
            'Change': 2.00
        },
        {
            'Date': pd.to_datetime('2025-07-25'),
            'Symbol': 'ENGRO',
            'Company_Name': 'Engro Corporation Limited',
            'Open': 249.00,
            'High': 252.50,
            'Low': 247.75,
            'Close': 251.00,
            'Volume': 98000,
            'Previous_Close': 248.50,
            'Change': 2.50
        }
    ])
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DataStorage(data_dir=temp_dir)
            
            # Test save
            result = storage.save_stock_data("ENGRO", sample_data)
            assert result == True, "Failed to save sample data"
            
            # Test load
            loaded_data = storage.load_stock_data("ENGRO")
            assert len(loaded_data) == 2, f"Expected 2 records, got {len(loaded_data)}"
            
            # Verify all columns are present
            expected_columns = ['Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 
                              'Close', 'Volume', 'Previous_Close', 'Change']
            
            for col in expected_columns:
                assert col in loaded_data.columns, f"Missing column: {col}"
            
            # Verify data types
            assert pd.api.types.is_datetime64_any_dtype(loaded_data['Date']), "Date should be datetime"
            assert pd.api.types.is_numeric_dtype(loaded_data['Open']), "Open should be numeric"
            assert pd.api.types.is_numeric_dtype(loaded_data['Close']), "Close should be numeric"
            assert pd.api.types.is_numeric_dtype(loaded_data['Volume']), "Volume should be numeric"
            
            print("✓ Data format compatibility test passed")
            return True
            
    except Exception as e:
        print(f"❌ Data format compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests"""
    print("Running DataStorage integration tests...\n")
    
    try:
        success1 = test_data_format_compatibility()
        success2 = test_integration_with_real_data()
        
        if success1 and success2:
            print("\n✅ All DataStorage integration tests passed!")
            return True
        else:
            print("\n❌ Some integration tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)