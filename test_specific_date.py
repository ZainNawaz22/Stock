#!/usr/bin/env python3
"""
Test script to check CSV downloading for a specific date
"""

import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_specific_date_download():
    """Test downloading for July 25th, 2025 (Friday)"""
    print("Testing PSX stock data download for 2025-07-25...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test with July 25th (Friday - should be available)
        test_date = "2025-07-25"
        print(f"Attempting to download data for: {test_date}")
        
        # Try to download the CSV
        csv_path = data_acq.download_daily_csv(test_date)
        print(f"✓ CSV downloaded/found successfully: {csv_path}")
        
        # Verify the CSV
        if data_acq.verify_csv_download(csv_path):
            print("✓ CSV verification successful")
            
            # Check file size
            file_size = os.path.getsize(csv_path)
            print(f"✓ CSV file size: {file_size} bytes")
            
            return True
        else:
            print("✗ CSV verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

if __name__ == "__main__":
    success = test_specific_date_download()
    if success:
        print("\n🎉 CSV downloading is working correctly!")
    else:
        print("\n❌ CSV downloading failed")
    
    sys.exit(0 if success else 1)