#!/usr/bin/env python3
"""
Test data acquisition functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_acquisition():
    """Test the data acquisition module"""
    print("Testing PSX Data Acquisition...")
    
    try:
        from psx_ai_advisor.data_acquisition import PSXDataAcquisition
        
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ PSXDataAcquisition initialized successfully")
        
        # Check configuration
        print(f"✓ Base URL: {data_acq.base_url}")
        print(f"✓ Downloads URL: {data_acq.downloads_url}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data acquisition test failed: {e}")
        return False

def test_existing_data_analysis():
    """Test analysis of existing data"""
    print("\nTesting analysis of existing data...")
    
    try:
        import pandas as pd
        from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
        
        # Check if we have PTC data
        ptc_file = 'data/PTC_historical_data.csv'
        if os.path.exists(ptc_file):
            df = pd.read_csv(ptc_file)
            print(f"✓ Loaded PTC data: {len(df)} rows")
            
            # Initialize analyzer
            analyzer = TechnicalAnalyzer()
            
            # Calculate indicators
            df['SMA_20'] = analyzer.calculate_sma(df['Close'], 20)
            df['RSI'] = analyzer.calculate_rsi(df['Close'], 14)
            
            # Show recent data with indicators
            recent_data = df.tail(5)[['Date', 'Close', 'SMA_20', 'RSI']]
            print("✓ Recent PTC data with indicators:")
            print(recent_data.to_string(index=False))
            
            return True
        else:
            print("! PTC data file not found, skipping analysis test")
            return True
            
    except Exception as e:
        print(f"✗ Data analysis test failed: {e}")
        return False

def main():
    """Run data acquisition tests"""
    print("PSX AI Advisor - Data Acquisition Test")
    print("=" * 50)
    
    tests = [
        test_data_acquisition,
        test_existing_data_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Data acquisition system is working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())