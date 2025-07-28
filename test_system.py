#!/usr/bin/env python3
"""
Test script to verify PSX AI Advisor system functionality
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported successfully"""
    print("Testing imports...")
    try:
        from psx_ai_advisor.data_acquisition import PSXDataAcquisition
        from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
        from psx_ai_advisor.data_storage import DataStorage
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from psx_ai_advisor.config_loader import get_section, get_value
        config = get_section('data_sources')
        print(f"✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis with sample data"""
    print("\nTesting technical analysis...")
    try:
        from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Close': [100 + i + (i % 10) for i in range(50)],
            'High': [105 + i + (i % 10) for i in range(50)],
            'Low': [95 + i + (i % 10) for i in range(50)],
            'Volume': [1000000 + i * 10000 for i in range(50)]
        })
        
        analyzer = TechnicalAnalyzer()
        
        # Test SMA calculation
        sma_20 = analyzer.calculate_sma(sample_data['Close'], 20)
        print(f"✓ SMA calculation successful, got {len(sma_20)} values")
        
        # Test RSI calculation
        rsi = analyzer.calculate_rsi(sample_data['Close'], 14)
        print(f"✓ RSI calculation successful, got {len(rsi)} values")
        
        return True
    except Exception as e:
        print(f"✗ Technical analysis error: {e}")
        return False

def test_data_storage():
    """Test data storage functionality"""
    print("\nTesting data storage...")
    try:
        from psx_ai_advisor.data_storage import DataStorage
        
        storage = DataStorage()
        print("✓ DataStorage initialized successfully")
        
        # Test if data directory exists
        if os.path.exists(storage.data_dir):
            print(f"✓ Data directory exists: {storage.data_dir}")
        else:
            print(f"! Data directory will be created: {storage.data_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Data storage error: {e}")
        return False

def test_existing_data():
    """Test if existing data files can be read"""
    print("\nTesting existing data files...")
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print("! No data directory found")
            return True
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("! No CSV files found in data directory")
            return True
        
        # Test reading a sample file
        sample_file = csv_files[0]
        df = pd.read_csv(os.path.join(data_dir, sample_file))
        print(f"✓ Successfully read {sample_file} with {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        print(f"✗ Data reading error: {e}")
        return False

def main():
    """Run all tests"""
    print("PSX AI Advisor System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_technical_analysis,
        test_data_storage,
        test_existing_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())