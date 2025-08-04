#!/usr/bin/env python3
"""
Enhanced Stock Data Update Script

This script provides advanced options for updating stock data including:
- Data merging with existing data
- Complete data replacement
- Data cleaning and optimization
- Flexible period selection
"""

import sys
import os
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_loader import PSXDataLoader
from psx_ai_advisor.logging_config import get_logger

logger = get_logger(__name__)

def main():
    """Enhanced stock data update with multiple options"""
    parser = argparse.ArgumentParser(description='Enhanced PSX Stock Data Update')
    parser.add_argument('--period', default='10y', choices=['1y', '2y', '5y', '10y', 'max'],
                       help='Data period to download (default: 10y)')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of concurrent workers (default: 5)')
    parser.add_argument('--no-merge', action='store_true',
                       help='Replace existing data instead of merging')
    parser.add_argument('--clean-only', action='store_true',
                       help='Only clean and optimize existing data')
    parser.add_argument('--symbols', nargs='+',
                       help='Specific symbols to update (default: all KSE-100)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Enhanced PSX Stock Data Update")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print(f"Period: {args.period}")
    print(f"Workers: {args.workers}")
    print(f"Merge with existing: {not args.no_merge}")
    print()
    
    # Initialize data loader
    loader = PSXDataLoader()
    
    if args.clean_only:
        print("🧹 Cleaning and optimizing existing data...")
        results = loader.clean_all_data()
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n✅ Cleaned {successful}/{total} data files")
        return 0 if successful == total else 1
    
    # Determine symbols to update
    if args.symbols:
        symbols = args.symbols
        print(f"Updating {len(symbols)} specified symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    else:
        symbols = loader.KSE_100_SYMBOLS
        print(f"Updating all {len(symbols)} KSE-100 symbols")
    
    print("This may take several minutes depending on network speed and data period...")
    print()
    
    # Update stocks
    try:
        merge_with_existing = not args.no_merge
        
        if args.symbols:
            # Update specific symbols
            results = loader.update_multiple_stocks(
                symbols=symbols,
                period=args.period,
                max_workers=args.workers,
                merge_with_existing=merge_with_existing
            )
        else:
            # Update all KSE-100 stocks
            results = loader.update_kse100_stocks(
                period=args.period,
                max_workers=args.workers,
                merge_with_existing=merge_with_existing
            )
        
        # Process results
        successful_updates = [symbol for symbol, success in results.items() if success]
        failed_updates = [symbol for symbol, success in results.items() if not success]
        
        print("\n" + "=" * 70)
        print("UPDATE SUMMARY")
        print("=" * 70)
        print(f"Total stocks: {len(symbols)}")
        print(f"Successful updates: {len(successful_updates)}")
        print(f"Failed updates: {len(failed_updates)}")
        print(f"Success rate: {len(successful_updates)/len(symbols)*100:.1f}%")
        print(f"Data period: {args.period}")
        print(f"Merge strategy: {'Merge with existing' if merge_with_existing else 'Replace existing'}")
        
        if failed_updates:
            print(f"\nFailed stocks ({len(failed_updates)}):")
            for symbol in failed_updates:
                print(f"  ❌ {symbol}")
        
        # Offer to clean data
        if successful_updates and not args.no_merge:
            print(f"\n🧹 Cleaning and optimizing updated data...")
            clean_results = {}
            for symbol in successful_updates[:10]:  # Clean first 10 as example
                clean_results[symbol] = loader.clean_and_optimize_data(symbol)
            
            cleaned = sum(1 for success in clean_results.values() if success)
            print(f"Cleaned {cleaned}/{len(clean_results)} sample files")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Return appropriate exit code
        return 0 if len(failed_updates) == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nUpdate interrupted by user.")
        return 2
    except Exception as e:
        print(f"\nError during update: {e}")
        logger.error(f"Enhanced update failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)