#!/usr/bin/env python3
"""
Test complete workflow: download + extraction
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_complete_workflow():
    """Test complete workflow from download to data extraction"""
    print("Testing complete workflow...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test get_all_stock_data which combines download + extraction
        print("\nTesting complete workflow with get_all_stock_data...")
        test_date = "2025-07-25"
        
        stock_data = data_acq.get_all_stock_data(test_date)
        print(f"✓ Successfully processed complete workflow for {len(stock_data)} stocks")
        
        # Verify data quality
        print(f"\nData quality check:")
        print(f"- Total stocks: {len(stock_data)}")
        print(f"- Columns: {list(stock_data.columns)}")
        print(f"- Date: {stock_data['Date'].iloc[0]}")
        print(f"- Symbol range: {stock_data['Symbol'].min()} to {stock_data['Symbol'].max()}")
        print(f"- Volume range: {stock_data['Volume'].min():,} to {stock_data['Volume'].max():,}")
        
        # Show top 10 by volume
        print(f"\nTop 10 stocks by volume:")
        top_volume = stock_data.nlargest(10, 'Volume')[['Symbol', 'Company_Name', 'Volume', 'Close', 'Change']]
        for idx, row in top_volume.iterrows():
            print(f"  {row['Symbol']}: {row['Volume']:,} shares, Close: {row['Close']:.2f}, Change: {row['Change']:+.2f}")
        
        # Show top 10 by price
        print(f"\nTop 10 stocks by price:")
        top_price = stock_data.nlargest(10, 'Close')[['Symbol', 'Company_Name', 'Close', 'Volume', 'Change']]
        for idx, row in top_price.iterrows():
            print(f"  {row['Symbol']}: {row['Close']:.2f}, Volume: {row['Volume']:,}, Change: {row['Change']:+.2f}")
        
        # Test data consistency
        print(f"\nData consistency checks:")
        
        # Check for any data anomalies
        high_volume_stocks = stock_data[stock_data['Volume'] > 1000000]
        print(f"- Stocks with volume > 1M: {len(high_volume_stocks)}")
        
        high_price_stocks = stock_data[stock_data['Close'] > 1000]
        print(f"- Stocks with price > 1000: {len(high_price_stocks)}")
        
        big_movers = stock_data[abs(stock_data['Change']) > 10]
        print(f"- Stocks with |change| > 10: {len(big_movers)}")
        
        if len(big_movers) > 0:
            print("  Big movers:")
            for idx, row in big_movers.head().iterrows():
                print(f"    {row['Symbol']}: {row['Change']:+.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    if success:
        print("\n🎉 Complete workflow test passed!")
    else:
        print("\n❌ Complete workflow test failed")
    
    sys.exit(0 if success else 1)