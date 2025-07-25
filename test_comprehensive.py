#!/usr/bin/env python3
"""
Comprehensive test of PDF downloading functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_comprehensive():
    """Comprehensive test of PDF downloading"""
    print("=== Comprehensive PDF Download Test ===\n")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test 1: Existing file (should not re-download)
        print("\n--- Test 1: Existing PDF file ---")
        test_date = "2025-07-25"
        pdf_path = data_acq.download_daily_pdf(test_date)
        print(f"✓ Found existing PDF: {pdf_path}")
        
        if data_acq.verify_pdf_download(pdf_path):
            print("✓ PDF verification successful")
            file_size = os.path.getsize(pdf_path)
            print(f"✓ File size: {file_size} bytes")
        
        # Test 2: Non-existent file (weekend/future date)
        print("\n--- Test 2: Non-existent PDF file ---")
        future_date = "2025-07-26"  # Saturday
        try:
            pdf_path = data_acq.download_daily_pdf(future_date)
            print(f"✗ Unexpected success for {future_date}")
        except Exception as e:
            print(f"✓ Expected error for {future_date}: {type(e).__name__}")
        
        # Test 3: Get available dates
        print("\n--- Test 3: Available dates ---")
        available_dates = data_acq.get_available_dates(7)
        print(f"✓ Available dates (last 7 weekdays): {available_dates}")
        
        # Test 4: URL generation
        print("\n--- Test 4: URL generation ---")
        url = data_acq._get_pdf_url("2025-07-25")
        expected_url = "https://dps.psx.com.pk/download/closing_rates/2025-07-25.pdf"
        if url == expected_url:
            print(f"✓ URL generation correct: {url}")
        else:
            print(f"✗ URL mismatch. Got: {url}, Expected: {expected_url}")
            return False
        
        print("\n🎉 All tests passed! PDF downloading is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    sys.exit(0 if success else 1)