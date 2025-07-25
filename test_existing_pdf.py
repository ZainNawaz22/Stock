#!/usr/bin/env python3
"""
Test script to verify existing PDF file handling
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_existing_pdf():
    """Test that existing PDF files are properly handled"""
    print("Testing existing PDF file handling...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Check if the existing file is recognized
        test_date = "2025-07-25"
        print(f"Checking for existing PDF for: {test_date}")
        
        # This should find the existing file without downloading
        pdf_path = data_acq.download_daily_pdf(test_date)
        print(f"✓ PDF found: {pdf_path}")
        
        # Verify the PDF
        if data_acq.verify_pdf_download(pdf_path):
            print("✓ PDF verification successful")
            
            # Check file size
            file_size = os.path.getsize(pdf_path)
            print(f"✓ PDF file size: {file_size} bytes")
            
            # Check if file exists in expected location
            expected_path = os.path.join("data", "2025-07-25.pdf")
            if os.path.exists(expected_path):
                print(f"✓ PDF exists at expected location: {expected_path}")
            
            return True
        else:
            print("✗ PDF verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_existing_pdf()
    if success:
        print("\n🎉 Existing PDF handling is working correctly!")
    else:
        print("\n❌ Existing PDF handling failed")
    
    sys.exit(0 if success else 1)