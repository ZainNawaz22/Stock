#!/usr/bin/env python3
"""
Test script to check if stock downloading is working
"""

import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition

def test_stock_download():
    """Test the stock downloading functionality"""
    print("Testing PSX stock data download...")
    
    try:
        # Initialize data acquisition
        data_acq = PSXDataAcquisition()
        print("✓ Data acquisition initialized successfully")
        
        # Test with today's date
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"Attempting to download data for: {today}")
        
        # Try to download today's PDF
        pdf_path = data_acq.download_daily_pdf(today)
        print(f"✓ PDF downloaded successfully: {pdf_path}")
        
        # Verify the PDF
        if data_acq.verify_pdf_download(pdf_path):
            print("✓ PDF verification successful")
            return True
        else:
            print("✗ PDF verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

if __name__ == "__main__":
    success = test_stock_download()
    if success:
        print("\n🎉 Stock downloading is working correctly!")
    else:
        print("\n❌ Stock downloading failed - will need to use Firecrawl fallback")
    
    sys.exit(0 if success else 1)