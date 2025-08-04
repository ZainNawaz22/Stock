"""
Test script for PSX Data Loader functionality

This script demonstrates how to use the new data loading features
for updating stock data from Yahoo Finance.
"""

import sys
import os
from datetime import datetime

# Add the Stock directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_loader import PSXDataLoader


def test_single_stock_update():
    """Test updating a single stock."""
    print("=== Testing Single Stock Update ===")
    
    loader = PSXDataLoader()
    symbol = "HBL"  # Habib Bank Limited
    
    print(f"Updating {symbol}...")
    success = loader.update_single_stock(symbol, period="1y")
    
    if success:
        print(f"✅ Successfully updated {symbol}")
    else:
        print(f"❌ Failed to update {symbol}")
    
    return success


def test_multiple_stocks_update():
    """Test updating multiple stocks."""
    print("\n=== Testing Multiple Stocks Update ===")
    
    loader = PSXDataLoader()
    test_symbols = ["HBL", "UBL", "MCB", "NBP", "BAFL"]  # Major banks
    
    print(f"Updating {len(test_symbols)} stocks: {test_symbols}")
    results = loader.update_multiple_stocks(test_symbols, period="1y", max_workers=2)
    
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    print(f"Results: {successful} successful, {failed} failed")
    for symbol, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {symbol}")
    
    return results


def test_data_validation():
    """Test data validation functionality."""
    print("\n=== Testing Data Validation ===")
    
    loader = PSXDataLoader()
    symbol = "HBL"
    
    # Download data
    data = loader.download_single_stock(symbol, period="1y")
    
    if data is not None:
        print(f"Downloaded {len(data)} records for {symbol}")
        print("Data columns:", list(data.columns))
        print("Date range:", data['Date'].min(), "to", data['Date'].max())
        
        # Validate data
        is_valid = loader.validate_data(data, symbol)
        print(f"Data validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        return is_valid
    else:
        print(f"❌ Failed to download data for {symbol}")
        return False


def test_kse100_info():
    """Test KSE-100 information."""
    print("\n=== Testing KSE-100 Information ===")
    
    loader = PSXDataLoader()
    
    print(f"KSE-100 symbols count: {len(loader.KSE_100_SYMBOLS)}")
    print("First 10 symbols:", loader.KSE_100_SYMBOLS[:10])
    print("Last 10 symbols:", loader.KSE_100_SYMBOLS[-10:])
    
    # Test Yahoo symbol conversion
    test_symbol = "HBL"
    yahoo_symbol = loader.get_yahoo_symbol(test_symbol)
    print(f"PSX symbol '{test_symbol}' -> Yahoo symbol '{yahoo_symbol}'")
    
    return True


def test_backup_functionality():
    """Test backup functionality."""
    print("\n=== Testing Backup Functionality ===")
    
    loader = PSXDataLoader()
    
    # Get summary
    summary = loader.get_update_summary()
    print("Backup directory:", summary["session_backup_dir"])
    print("Data directory:", summary["data_directory"])
    print("KSE-100 symbols count:", summary["kse100_symbols_count"])
    
    return True


def main():
    """Run all tests."""
    print("PSX Data Loader Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("KSE-100 Info", test_kse100_info),
        ("Backup Functionality", test_backup_functionality),
        ("Data Validation", test_data_validation),
        ("Single Stock Update", test_single_stock_update),
        ("Multiple Stocks Update", test_multiple_stocks_update),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    print(f"Test completed at: {datetime.now()}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
