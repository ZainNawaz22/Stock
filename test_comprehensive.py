#!/usr/bin/env python3
"""
Comprehensive test script for PDF parsing functionality
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition, PDFParsingError

def test_comprehensive():
    """Comprehensive test of PDF parsing functionality"""
    print("Running comprehensive PDF parsing tests...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test 1: Valid PDF parsing
        print("\n1. Testing valid PDF parsing...")
        pdf_path = "data/2025-07-25.pdf"
        stock_data = data_acq.extract_stock_data(pdf_path)
        print(f"✓ Extracted {len(stock_data)} stocks")
        
        # Test 2: Data structure validation
        print("\n2. Testing data structure...")
        expected_columns = ['Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
        if list(stock_data.columns) == expected_columns:
            print("✓ All expected columns present")
        else:
            print(f"✗ Column mismatch. Expected: {expected_columns}, Got: {list(stock_data.columns)}")
            return False
        
        # Test 3: Data types validation
        print("\n3. Testing data types...")
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
        for col in numeric_cols:
            if stock_data[col].dtype in ['float64', 'int64']:
                print(f"✓ {col} has correct numeric type")
            else:
                print(f"✗ {col} has incorrect type: {stock_data[col].dtype}")
                return False
        
        # Test 4: Business logic validation
        print("\n4. Testing business logic...")
        
        # Check High >= Low
        invalid_high_low = stock_data[stock_data['High'] < stock_data['Low']]
        if len(invalid_high_low) == 0:
            print("✓ All stocks have High >= Low")
        else:
            print(f"✗ Found {len(invalid_high_low)} stocks with High < Low")
            return False
        
        # Check positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            negative_prices = stock_data[stock_data[col] <= 0]
            if len(negative_prices) == 0:
                print(f"✓ All {col} prices are positive")
            else:
                print(f"✗ Found {len(negative_prices)} stocks with non-positive {col}")
                return False
        
        # Check non-negative volume
        negative_volume = stock_data[stock_data['Volume'] < 0]
        if len(negative_volume) == 0:
            print("✓ All volumes are non-negative")
        else:
            print(f"✗ Found {len(negative_volume)} stocks with negative volume")
            return False
        
        # Test 5: Symbol uniqueness
        print("\n5. Testing symbol uniqueness...")
        duplicate_symbols = stock_data[stock_data.duplicated('Symbol')]
        if len(duplicate_symbols) == 0:
            print("✓ All symbols are unique")
        else:
            print(f"⚠ Found {len(duplicate_symbols)} duplicate symbols")
            print("Duplicate symbols:", duplicate_symbols['Symbol'].tolist())
        
        # Test 6: Sample data inspection
        print("\n6. Sample data inspection...")
        sample_stocks = stock_data.head(3)
        for idx, row in sample_stocks.iterrows():
            print(f"Stock {idx + 1}: {row['Symbol']} - {row['Company_Name']}")
            print(f"  OHLC: {row['Open']:.2f}, {row['High']:.2f}, {row['Low']:.2f}, {row['Close']:.2f}")
            print(f"  Volume: {row['Volume']:,}")
            print(f"  Change: {row['Change']:+.2f}")
        
        # Test 7: Error handling
        print("\n7. Testing error handling...")
        try:
            data_acq.extract_stock_data("nonexistent.pdf")
            print("✗ Should have raised FileNotFoundError")
            return False
        except FileNotFoundError:
            print("✓ Correctly handles missing file")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
        
        # Test 8: Performance check
        print("\n8. Performance check...")
        import time
        start_time = time.time()
        stock_data_2 = data_acq.extract_stock_data(pdf_path)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"✓ Processing time: {processing_time:.2f} seconds for {len(stock_data_2)} stocks")
        
        if processing_time > 30:  # Should process within 30 seconds
            print("⚠ Processing time is quite long")
        else:
            print("✓ Processing time is acceptable")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    if success:
        print("\n🎉 All comprehensive tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    sys.exit(0 if success else 1)