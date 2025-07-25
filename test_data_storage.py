#!/usr/bin/env python3
"""
Test script for DataStorage class functionality
"""

import os
import sys
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_storage import DataStorage, DataStorageError, DataIntegrityError


def create_sample_data(symbol: str = "ENGRO", num_days: int = 5) -> pd.DataFrame:
    """Create sample stock data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(num_days-1, -1, -1)]
    
    data = []
    base_price = 100.0
    
    for i, date in enumerate(dates):
        price = base_price + i * 2  # Gradually increasing price
        data.append({
            'Date': date,
            'Symbol': symbol,
            'Company_Name': f'{symbol} Limited',
            'Open': price - 1,
            'High': price + 2,
            'Low': price - 2,
            'Close': price,
            'Volume': 1000 + i * 100,
            'Previous_Close': price - 1 if i > 0 else price,
            'Change': 1.0 if i > 0 else 0.0
        })
    
    return pd.DataFrame(data)


def test_basic_functionality():
    """Test basic save and load functionality"""
    print("Testing basic save and load functionality...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Create sample data
        sample_data = create_sample_data("ENGRO", 5)
        
        # Test save
        result = storage.save_stock_data("ENGRO", sample_data)
        assert result == True, "Save operation should return True"
        
        # Test load
        loaded_data = storage.load_stock_data("ENGRO")
        assert len(loaded_data) == 5, f"Expected 5 records, got {len(loaded_data)}"
        assert loaded_data['Symbol'].iloc[0] == "ENGRO", "Symbol should match"
        
        print("✓ Basic save and load functionality works")


def test_append_functionality():
    """Test appending new data without overwriting"""
    print("Testing append functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Save initial data
        initial_data = create_sample_data("ENGRO", 3)
        storage.save_stock_data("ENGRO", initial_data)
        
        # Create new data with different dates (past dates to avoid future date validation error)
        base_date = datetime.now() - timedelta(days=10)
        new_dates = [base_date + timedelta(days=i) for i in range(1, 4)]
        new_data = []
        for i, date in enumerate(new_dates):
            price = 110.0 + i * 2
            new_data.append({
                'Date': date,
                'Symbol': "ENGRO",
                'Company_Name': 'ENGRO Limited',
                'Open': price - 1,
                'High': price + 2,
                'Low': price - 2,
                'Close': price,
                'Volume': 1500 + i * 100,
                'Previous_Close': price - 1,
                'Change': 1.0
            })
        
        new_df = pd.DataFrame(new_data)
        
        # Append new data
        storage.save_stock_data("ENGRO", new_df)
        
        # Load and verify
        all_data = storage.load_stock_data("ENGRO")
        assert len(all_data) == 6, f"Expected 6 records after append, got {len(all_data)}"
        
        # Verify chronological order
        dates = pd.to_datetime(all_data['Date'])
        assert dates.is_monotonic_increasing, "Data should be in chronological order"
        
        print("✓ Append functionality works correctly")


def test_duplicate_prevention():
    """Test duplicate prevention"""
    print("Testing duplicate prevention...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Save initial data
        initial_data = create_sample_data("ENGRO", 3)
        storage.save_stock_data("ENGRO", initial_data)
        
        # Try to save the same data again (should remove duplicates)
        storage.save_stock_data("ENGRO", initial_data)
        
        # Load and verify no duplicates
        loaded_data = storage.load_stock_data("ENGRO")
        assert len(loaded_data) == 3, f"Expected 3 records (no duplicates), got {len(loaded_data)}"
        
        # Check for duplicate dates
        dates = loaded_data['Date'].dt.date
        unique_dates = dates.nunique()
        assert unique_dates == len(loaded_data), "All dates should be unique"
        
        print("✓ Duplicate prevention works correctly")


def test_data_validation():
    """Test data validation"""
    print("Testing data validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Test with invalid data (negative prices)
        invalid_data = create_sample_data("ENGRO", 2)
        invalid_data.loc[0, 'Close'] = -10.0  # Invalid negative price
        
        try:
            storage.save_stock_data("ENGRO", invalid_data)
            assert False, "Should have raised DataIntegrityError for negative price"
        except DataIntegrityError:
            print("✓ Correctly rejected negative prices")
        
        # Test with missing columns
        incomplete_data = create_sample_data("ENGRO", 2)
        incomplete_data = incomplete_data.drop(columns=['Close'])
        
        try:
            storage.save_stock_data("ENGRO", incomplete_data)
            assert False, "Should have raised DataIntegrityError for missing columns"
        except DataIntegrityError:
            print("✓ Correctly rejected missing columns")
        
        # Test with High < Low
        invalid_range_data = create_sample_data("ENGRO", 2)
        invalid_range_data.loc[0, 'High'] = 50.0
        invalid_range_data.loc[0, 'Low'] = 60.0  # Low > High
        
        try:
            storage.save_stock_data("ENGRO", invalid_range_data)
            assert False, "Should have raised DataIntegrityError for High < Low"
        except DataIntegrityError:
            print("✓ Correctly rejected High < Low")


def test_file_management():
    """Test file naming and directory management"""
    print("Testing file management...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Test multiple symbols
        symbols = ["ENGRO", "HBL", "UBL"]
        
        for symbol in symbols:
            data = create_sample_data(symbol, 3)
            storage.save_stock_data(symbol, data)
        
        # Check that files were created with correct names
        for symbol in symbols:
            expected_path = os.path.join(temp_dir, f"{symbol}.csv")
            assert os.path.exists(expected_path), f"File should exist for {symbol}"
        
        # Test get_available_symbols
        available_symbols = storage.get_available_symbols()
        assert set(available_symbols) == set(symbols), "Available symbols should match saved symbols"
        
        print("✓ File management works correctly")


def test_data_integrity_validation():
    """Test data integrity validation"""
    print("Testing data integrity validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Save valid data
        valid_data = create_sample_data("ENGRO", 5)
        storage.save_stock_data("ENGRO", valid_data)
        
        # Test integrity validation
        is_valid = storage.validate_data_integrity("ENGRO")
        assert is_valid == True, "Valid data should pass integrity check"
        
        # Test with non-existent symbol
        is_valid = storage.validate_data_integrity("NONEXISTENT")
        assert is_valid == False, "Non-existent symbol should fail integrity check"
        
        print("✓ Data integrity validation works correctly")


def test_data_summary():
    """Test data summary functionality"""
    print("Testing data summary...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Save sample data
        sample_data = create_sample_data("ENGRO", 10)
        storage.save_stock_data("ENGRO", sample_data)
        
        # Get summary
        summary = storage.get_data_summary("ENGRO")
        
        assert summary['exists'] == True, "Summary should show file exists"
        assert summary['record_count'] == 10, f"Expected 10 records, got {summary['record_count']}"
        assert 'date_range' in summary, "Summary should include date range"
        assert 'price_range' in summary, "Summary should include price range"
        
        # Test summary for non-existent symbol
        summary_nonexistent = storage.get_data_summary("NONEXISTENT")
        assert summary_nonexistent['exists'] == False, "Non-existent symbol should show exists=False"
        
        print("✓ Data summary works correctly")


def test_storage_stats():
    """Test storage statistics"""
    print("Testing storage statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DataStorage(data_dir=temp_dir)
        
        # Save data for multiple symbols
        symbols = ["ENGRO", "HBL", "UBL"]
        for symbol in symbols:
            data = create_sample_data(symbol, 5)
            storage.save_stock_data(symbol, data)
        
        # Get storage stats
        stats = storage.get_storage_stats()
        
        assert stats['total_symbols'] == 3, f"Expected 3 symbols, got {stats['total_symbols']}"
        assert stats['total_records'] == 15, f"Expected 15 total records, got {stats['total_records']}"
        assert stats['total_size_mb'] >= 0, f"Total size should be >= 0, got {stats['total_size_mb']}"
        
        print("✓ Storage statistics work correctly")


def run_all_tests():
    """Run all tests"""
    print("Running DataStorage tests...\n")
    
    try:
        test_basic_functionality()
        test_append_functionality()
        test_duplicate_prevention()
        test_data_validation()
        test_file_management()
        test_data_integrity_validation()
        test_data_summary()
        test_storage_stats()
        
        print("\n✅ All DataStorage tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)