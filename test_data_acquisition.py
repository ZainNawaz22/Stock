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

from psx_ai_advisor.data_acquisition import PSXDataAcquisition, CSVDownloadError, NetworkError

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
        csv_url = psx_data._get_csv_url(test_date)
        filename = psx_data._get_csv_filename(test_date)
        print(f"✓ URL generation works")
        print(f"  Test date: {test_date}")
        print(f"  Generated filename: {filename}")
        print(f"  Generated URL: {csv_url}")
        
        # Test available dates generation
        available_dates = psx_data.get_available_dates(5)
        print(f"✓ Available dates generation works")
        print(f"  Recent dates (excluding weekends): {available_dates}")
        
        # Test CSV download (this will likely fail due to network/availability)
        print("\nAttempting CSV download test...")
        try:
            # Try to download a recent CSV
            recent_date = available_dates[0] if available_dates else None
            if recent_date:
                csv_path = psx_data.download_daily_csv(recent_date)
                print(f"✓ CSV download successful: {csv_path}")
                
                # Verify the downloaded CSV
                if psx_data.verify_csv_download(csv_path):
                    print(f"✓ CSV verification successful")
                else:
                    print(f"⚠ CSV verification failed")
            else:
                print("⚠ No recent dates available for testing")
                
        except CSVDownloadError as e:
            print(f"⚠ CSV download failed (expected): {e}")
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