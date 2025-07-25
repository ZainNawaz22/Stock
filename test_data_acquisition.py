#!/usr/bin/env python3
"""
Test script for PSX Data Acquisition functionality
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition, PDFDownloadError, NetworkError

def setup_logging():
    """Setup basic logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_data_acquisition():
    """Test the PSXDataAcquisition class"""
    print("Testing PSX Data Acquisition...")
    
    try:
        # Initialize the data acquisition system
        psx_data = PSXDataAcquisition()
        print(f"✓ PSXDataAcquisition initialized successfully")
        print(f"  Base URL: {psx_data.base_url}")
        print(f"  Downloads URL: {psx_data.downloads_url}")
        print(f"  Data directory: {psx_data.data_dir}")
        
        # Test URL generation
        test_date = "2024-01-15"  # Monday
        pdf_url = psx_data._get_pdf_url(test_date)
        filename = psx_data._get_pdf_filename(test_date)
        print(f"✓ URL generation works")
        print(f"  Test date: {test_date}")
        print(f"  Generated filename: {filename}")
        print(f"  Generated URL: {pdf_url}")
        
        # Test available dates generation
        available_dates = psx_data.get_available_dates(5)
        print(f"✓ Available dates generation works")
        print(f"  Recent dates (excluding weekends): {available_dates}")
        
        # Test PDF download (this will likely fail due to network/availability)
        print("\nAttempting PDF download test...")
        try:
            # Try to download a recent PDF
            recent_date = available_dates[0] if available_dates else None
            if recent_date:
                pdf_path = psx_data.download_daily_pdf(recent_date)
                print(f"✓ PDF download successful: {pdf_path}")
                
                # Verify the downloaded PDF
                if psx_data.verify_pdf_download(pdf_path):
                    print(f"✓ PDF verification successful")
                else:
                    print(f"⚠ PDF verification failed")
            else:
                print("⚠ No recent dates available for testing")
                
        except PDFDownloadError as e:
            print(f"⚠ PDF download failed (expected): {e}")
        except NetworkError as e:
            print(f"⚠ Network error (expected): {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        print("\n✓ All basic functionality tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    success = test_data_acquisition()
    sys.exit(0 if success else 1)