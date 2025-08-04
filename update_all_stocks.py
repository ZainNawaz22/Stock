#!/usr/bin/env python3
"""
Full Stock Data Update Script

This script updates data for all KSE-100 stocks using the PSX Data Loader.
It provides progress tracking and handles errors gracefully.
"""

import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_loader import PSXDataLoader
from psx_ai_advisor.logging_config import get_logger

logger = get_logger(__name__)

def main():
    """Update all KSE-100 stocks with progress tracking"""
    print("=" * 60)
    print("PSX Stock Data Full Update (10 Years)")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Initialize data loader
    loader = PSXDataLoader()
    
    # Get all KSE-100 symbols
    kse100_symbols = loader.KSE_100_SYMBOLS
    total_stocks = len(kse100_symbols)
    
    print(f"Updating data for {total_stocks} KSE-100 stocks...")
    print("Fetching 10 years of historical data - this may take several minutes...")
    print()
    
    # Update all stocks with progress tracking
    successful_updates = []
    failed_updates = []
    
    try:
        results = loader.update_kse100_stocks(
            period="10y",
            max_workers=5  # Limit concurrent requests to avoid overwhelming the server
        )
        
        # Process results
        for symbol, success in results.items():
            if success:
                successful_updates.append(symbol)
            else:
                failed_updates.append(symbol)
        
        print("\n" + "=" * 60)
        print("UPDATE SUMMARY")
        print("=" * 60)
        print(f"Total stocks: {total_stocks}")
        print(f"Successful updates: {len(successful_updates)}")
        print(f"Failed updates: {len(failed_updates)}")
        print(f"Success rate: {len(successful_updates)/total_stocks*100:.1f}%")
        print(f"Data period: 10 years of historical data")
        
        if failed_updates:
            print(f"\nFailed stocks ({len(failed_updates)}):")
            for symbol in failed_updates:
                print(f"  ❌ {symbol}")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Return appropriate exit code
        return 0 if len(failed_updates) == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nUpdate interrupted by user.")
        return 2
    except Exception as e:
        print(f"\nError during update: {e}")
        logger.error(f"Full update failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)