#!/usr/bin/env python3
"""
Test script to verify PDF parsing functionality
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_pdf_parsing():
    """Test PDF parsing functionality"""
    print("Testing PDF parsing functionality...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test with existing PDF
        pdf_path = "data/2025-07-25.pdf"
        if not os.path.exists(pdf_path):
            print(f"✗ PDF file not found: {pdf_path}")
            return False
        
        print(f"Testing PDF parsing with: {pdf_path}")
        
        # Extract stock data
        stock_data = data_acq.extract_stock_data(pdf_path)
        print(f"✓ Successfully extracted data for {len(stock_data)} stocks")
        
        # Display basic info about the data
        print(f"✓ DataFrame shape: {stock_data.shape}")
        print(f"✓ Columns: {list(stock_data.columns)}")
        
        # Show first few rows
        print("\nFirst 5 stocks:")
        print(stock_data.head())
        
        # Show some statistics
        print(f"\nData statistics:")
        print(f"- Total stocks: {len(stock_data)}")
        print(f"- Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        print(f"- Volume range: {stock_data['Volume'].min():,} to {stock_data['Volume'].max():,}")
        print(f"- Price range: {stock_data['Close'].min():.2f} to {stock_data['Close'].max():.2f}")
        
        # Test data validation
        print(f"\nData validation:")
        print(f"- No null values in critical columns: {stock_data[['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum() == 0}")
        print(f"- High >= Low for all stocks: {(stock_data['High'] >= stock_data['Low']).all()}")
        print(f"- All prices > 0: {(stock_data[['Open', 'High', 'Low', 'Close']] > 0).all().all()}")
        print(f"- All volumes >= 0: {(stock_data['Volume'] >= 0).all()}")
        
        # Test get_all_stock_data method
        print("\nTesting get_all_stock_data method...")
        all_data = data_acq.get_all_stock_data("2025-07-25")
        print(f"✓ get_all_stock_data returned {len(all_data)} stocks")
        
        # Verify both methods return same data
        if len(stock_data) == len(all_data):
            print("✓ Both methods return same number of stocks")
        else:
            print(f"⚠ Methods return different counts: {len(stock_data)} vs {len(all_data)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_parsing()
    if success:
        print("\n🎉 PDF parsing is working correctly!")
    else:
        print("\n❌ PDF parsing failed")
    
    sys.exit(0 if success else 1)