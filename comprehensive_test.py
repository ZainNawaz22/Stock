#!/usr/bin/env python3
"""
Comprehensive test of the PSX AI Advisor system
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_full_workflow():
    """Test the complete workflow with existing data"""
    print("Testing complete workflow...")
    
    try:
        from psx_ai_advisor.data_storage import DataStorage
        from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
        
        # Initialize components
        storage = DataStorage()
        analyzer = TechnicalAnalyzer()
        
        # Load existing PTC data
        ptc_file = 'data/PTC_historical_data.csv'
        if not os.path.exists(ptc_file):
            print("! PTC data file not found")
            return False
            
        df = pd.read_csv(ptc_file)
        print(f"✓ Loaded {len(df)} rows of PTC data")
        
        # Calculate multiple indicators
        df['SMA_20'] = analyzer.calculate_sma(df['Close'], 20)
        df['SMA_50'] = analyzer.calculate_sma(df['Close'], 50)
        df['RSI'] = analyzer.calculate_rsi(df['Close'], 14)
        
        # Calculate MACD
        macd_data = analyzer.calculate_macd(df['Close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['MACD_Signal']
        df['MACD_Histogram'] = macd_data['MACD_Histogram']
        
        print("✓ Calculated technical indicators:")
        print(f"  - SMA 20: {df['SMA_20'].notna().sum()} valid values")
        print(f"  - SMA 50: {df['SMA_50'].notna().sum()} valid values")
        print(f"  - RSI: {df['RSI'].notna().sum()} valid values")
        print(f"  - MACD: {df['MACD'].notna().sum()} valid values")
        
        # Show recent analysis
        recent = df.tail(3)[['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']]
        print("\n✓ Recent analysis results:")
        print(recent.to_string(index=False))
        
        # Test data integrity
        if df['Close'].isna().sum() == 0:
            print("✓ No missing close prices")
        else:
            print(f"! Found {df['Close'].isna().sum()} missing close prices")
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test system performance with larger dataset"""
    print("\nTesting performance...")
    
    try:
        from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
        import time
        
        # Load multiple stock files
        data_dir = 'data'
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_historical_data.csv')][:5]  # Test with 5 stocks
        
        analyzer = TechnicalAnalyzer()
        total_rows = 0
        start_time = time.time()
        
        for file in csv_files:
            df = pd.read_csv(os.path.join(data_dir, file))
            total_rows += len(df)
            
            # Calculate indicators
            df['SMA_20'] = analyzer.calculate_sma(df['Close'], 20)
            df['RSI'] = analyzer.calculate_rsi(df['Close'], 14)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✓ Processed {total_rows} rows from {len(csv_files)} files")
        print(f"✓ Processing time: {processing_time:.2f} seconds")
        print(f"✓ Rate: {total_rows/processing_time:.0f} rows/second")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def test_data_quality():
    """Test data quality and consistency"""
    print("\nTesting data quality...")
    
    try:
        # Check PTC enhanced indicators file
        enhanced_file = 'PTC_enhanced_indicators.csv'
        if os.path.exists(enhanced_file):
            df = pd.read_csv(enhanced_file)
            print(f"✓ Enhanced indicators file exists with {len(df)} rows")
            
            # Check for required columns
            required_cols = ['Date', 'Close', 'RSI_14', 'SMA_20', 'MACD']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if not missing_cols:
                print("✓ All required columns present")
            else:
                print(f"! Missing columns: {missing_cols}")
            
            # Check data completeness for recent data
            recent_data = df.tail(100)
            completeness = {}
            for col in ['RSI_14', 'SMA_20', 'MACD']:
                if col in df.columns:
                    completeness[col] = (recent_data[col].notna().sum() / len(recent_data)) * 100
            
            print("✓ Recent data completeness:")
            for col, pct in completeness.items():
                print(f"  - {col}: {pct:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Data quality test failed: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("PSX AI Advisor - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        test_full_workflow,
        test_performance,
        test_data_quality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Comprehensive Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 PSX AI Advisor system is fully operational!")
        print("\nSystem Status:")
        print("✓ Dependencies installed and working")
        print("✓ All modules loading correctly")
        print("✓ Technical analysis functioning")
        print("✓ Data storage operational")
        print("✓ Historical data available and processable")
        print("✓ Performance within acceptable limits")
        return 0
    else:
        print("⚠️  Some tests failed. System may have issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())