"""
Data Loader Module for PSX AI Advisor

This module provides functionality to download and update stock data from Yahoo Finance
for KSE-100 stocks with fail-safe mechanisms to ensure data integrity.
"""

import os
import shutil
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .logging_config import get_logger
from .exceptions import DataScrapingError, NetworkError
from .data_storage import DataStorage

logger = get_logger(__name__)

class PSXDataLoader:
    """
    Data loader for Pakistan Stock Exchange (PSX) stocks using Yahoo Finance.
    Implements fail-safe mechanisms to ensure data integrity during updates.
    """
    
    # KSE-100 stock symbols (major constituents)
    KSE_100_SYMBOLS = [
        'ABL', 'ABOT', 'AGIL', 'AGP', 'AICL', 'AIRLINK', 'AKBL', 'ANL', 'APL', 'ARPL',
        'ATLH', 'ATRL', 'BAFL', 'BAHL', 'BNWM', 'BOP', 'CHCC', 'COLG', 'DCR', 'DGKC',
        'EFERT', 'EFUG', 'EPCL', 'FABL', 'FATIMA', 'FCCL', 'FFBL', 'FFC', 'FHAM', 'FML',
        'GATM', 'GHGL', 'GLAXO', 'HASCOL', 'HBL', 'HCAR', 'HGFA', 'HINOON', 'HMB', 'HUBC',
        'IBFL', 'IDYM', 'ILP', 'INDU', 'INIL', 'ISL', 'JDWS', 'JLICL', 'KAPCO', 'KEL',
        'KOHC', 'KTML', 'LOTCHEM', 'LUCK', 'MARI', 'MCB', 'MEBL', 'MLCF', 'MTL', 'MUREB',
        'NATF', 'NBP', 'NCL', 'NESTLE', 'NML', 'OGDC', 'OLPL', 'PAEL', 'PAKT', 'PIBTL',
        'PIOC', 'PKGS', 'PMPK', 'POL', 'PPL', 'PSEL', 'PSMC', 'PSO', 'PSX', 'PTC',
        'SCBPL', 'SEARL', 'SHEL', 'SHFA', 'SNGP', 'SPWL', 'SRVI', 'SSGC', 'SYS', 'THALL',
        'TRG', 'UBL', 'UNITY', 'YOUW',
        # Additional major stocks
        'CNERGY', 'NRL', 'KASB', 'DAWH', 'SILK', 'UPFL', 'WAFI', 'WAVES'
    ]
    
    def __init__(self, data_dir: str = "data", backup_dir: str = "backups"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory where stock data files are stored
            backup_dir: Directory for backup files
        """
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        self.data_storage = DataStorage()
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Create timestamped backup directory for this session
        self.session_backup_dir = os.path.join(
            self.backup_dir, 
            f"data_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_backup_dir, exist_ok=True)
    
    def get_yahoo_symbol(self, psx_symbol: str) -> str:
        """
        Convert PSX symbol to Yahoo Finance symbol by adding .KA suffix.
        
        Args:
            psx_symbol: PSX stock symbol
            
        Returns:
            Yahoo Finance symbol with .KA suffix
        """
        return f"{psx_symbol}.KA"
    
    def download_single_stock(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Download data for a single stock from Yahoo Finance.
        
        Args:
            symbol: PSX stock symbol
            period: Period for data download (1y, 2y, 5y, 10y, max)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        yahoo_symbol = self.get_yahoo_symbol(symbol)
        
        try:
            logger.info(f"Downloading data for {symbol} ({yahoo_symbol})")
            
            # Download data using yfinance
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Ensure we have the required columns matching the existing format
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in data.columns:
                    if col == 'Dividends':
                        data[col] = 0.0
                    elif col == 'Stock Splits':
                        data[col] = 0.0
                    elif col == 'Capital Gains':
                        data[col] = 0.0
            
            # Add Capital Gains column if not present (from existing format)
            if 'Capital Gains' not in data.columns:
                data['Capital Gains'] = 0.0
            
            # Ensure proper column order
            final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Capital Gains']
            data = data[final_columns]
            
            # Sort by date
            data = data.sort_values('Date')
            
            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate downloaded data for completeness and quality.
        
        Args:
            data: Downloaded stock data
            symbol: Stock symbol
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.error(f"Data validation failed for {symbol}: Empty dataset")
            return False
        
        # Check required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Data validation failed for {symbol}: Missing columns {missing_columns}")
            return False
        
        # Check for recent data (within last 30 days)
        latest_date = data['Date'].max()
        if pd.isna(latest_date):
            logger.error(f"Data validation failed for {symbol}: Invalid latest date")
            return False
        
        # Convert to datetime if it's not already
        if not isinstance(latest_date, datetime):
            try:
                latest_date = pd.to_datetime(latest_date)
            except:
                logger.error(f"Data validation failed for {symbol}: Cannot parse latest date")
                return False
        
        days_old = (datetime.now() - latest_date.replace(tzinfo=None)).days
        if days_old > 30:
            logger.warning(f"Data for {symbol} is {days_old} days old, but accepting it")
        
        # Check for reasonable data ranges
        if data['Close'].isna().all():
            logger.error(f"Data validation failed for {symbol}: All Close prices are NaN")
            return False
        
        logger.info(f"Data validation passed for {symbol}: {len(data)} records, latest: {latest_date}")
        return True
    
    def backup_existing_file(self, symbol: str) -> bool:
        """
        Create backup of existing data file.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if backup successful, False otherwise
        """
        source_file = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        
        if not os.path.exists(source_file):
            logger.info(f"No existing file to backup for {symbol}")
            return True
        
        try:
            backup_file = os.path.join(self.session_backup_dir, f"{symbol}_historical_data.csv")
            shutil.copy2(source_file, backup_file)
            logger.info(f"Backed up existing data for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup data for {symbol}: {e}")
            return False
    
    def save_data_safely(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Save data with fail-safe mechanism.
        
        Args:
            data: Stock data to save
            symbol: Stock symbol
            
        Returns:
            True if save successful, False otherwise
        """
        # Create temporary file first
        temp_file = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tf:
                temp_file = tf.name
                data.to_csv(tf, index=False)
            
            # Validate the temporary file
            try:
                test_data = pd.read_csv(temp_file)
                if test_data.empty or len(test_data) != len(data):
                    raise ValueError("Temporary file validation failed")
            except Exception as e:
                logger.error(f"Temporary file validation failed for {symbol}: {e}")
                return False
            
            # Backup existing file
            if not self.backup_existing_file(symbol):
                logger.error(f"Failed to backup existing file for {symbol}")
                return False
            
            # Move temporary file to final location
            final_file = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
            shutil.move(temp_file, final_file)
            
            logger.info(f"Successfully saved data for {symbol} to {final_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return False
    
    def update_single_stock(self, symbol: str, period: str = "5y") -> bool:
        """
        Update data for a single stock with full fail-safe mechanism.
        
        Args:
            symbol: Stock symbol
            period: Period for data download
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            logger.info(f"Starting update for {symbol}")
            
            # Download new data
            new_data = self.download_single_stock(symbol, period)
            if new_data is None:
                logger.error(f"Failed to download data for {symbol}")
                return False
            
            # Validate data
            if not self.validate_data(new_data, symbol):
                logger.error(f"Data validation failed for {symbol}")
                return False
            
            # Save data safely
            if not self.save_data_safely(new_data, symbol):
                logger.error(f"Failed to save data for {symbol}")
                return False
            
            logger.info(f"Successfully updated data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error updating {symbol}: {e}")
            return False
    
    def update_multiple_stocks(self, symbols: List[str], period: str = "5y", max_workers: int = 4) -> Dict[str, bool]:
        """
        Update data for multiple stocks in parallel.
        
        Args:
            symbols: List of stock symbols
            period: Period for data download
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping symbol to success status
        """
        results = {}
        
        logger.info(f"Starting parallel update for {len(symbols)} stocks")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(self.update_single_stock, symbol, period): symbol
                for symbol in symbols
            }
            
            # Collect results with progress logging
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    results[symbol] = success
                    completed += 1
                    
                    status = "SUCCESS" if success else "FAILED"
                    logger.info(f"[{completed}/{len(symbols)}] {symbol}: {status}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results[symbol] = False
                    completed += 1
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        logger.info(f"Update complete: {successful} successful, {failed} failed")
        
        if failed > 0:
            failed_symbols = [symbol for symbol, success in results.items() if not success]
            logger.warning(f"Failed symbols: {failed_symbols}")
        
        return results
    
    def update_kse100_stocks(self, period: str = "10y", max_workers: int = 4) -> Dict[str, bool]:
        """
        Update all KSE-100 stocks.
        
        Args:
            period: Period for data download
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping symbol to success status
        """
        logger.info("Starting KSE-100 stocks update")
        return self.update_multiple_stocks(self.KSE_100_SYMBOLS, period, max_workers)
    
    def update_existing_stocks(self, period: str = "5y", max_workers: int = 4) -> Dict[str, bool]:
        """
        Update all stocks that already have data files.
        
        Args:
            period: Period for data download
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping symbol to success status
        """
        existing_symbols = []
        
        # Find all existing data files
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_historical_data.csv'):
                    symbol = filename.replace('_historical_data.csv', '')
                    existing_symbols.append(symbol)
        
        if not existing_symbols:
            logger.warning("No existing stock data files found")
            return {}
        
        logger.info(f"Found {len(existing_symbols)} existing stocks to update")
        return self.update_multiple_stocks(existing_symbols, period, max_workers)
    
    def restore_from_backup(self, symbol: str) -> bool:
        """
        Restore a stock's data from the most recent backup.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            backup_file = os.path.join(self.session_backup_dir, f"{symbol}_historical_data.csv")
            target_file = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
            
            if not os.path.exists(backup_file):
                logger.error(f"No backup file found for {symbol}")
                return False
            
            shutil.copy2(backup_file, target_file)
            logger.info(f"Restored data for {symbol} from backup")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup for {symbol}: {e}")
            return False
    
    def get_update_summary(self) -> Dict[str, any]:
        """
        Get summary of the current update session.
        
        Returns:
            Dictionary with update session summary
        """
        backup_files = []
        if os.path.exists(self.session_backup_dir):
            backup_files = [f for f in os.listdir(self.session_backup_dir) if f.endswith('.csv')]
        
        return {
            "session_backup_dir": self.session_backup_dir,
            "backup_count": len(backup_files),
            "kse100_symbols_count": len(self.KSE_100_SYMBOLS),
            "data_directory": self.data_dir,
            "timestamp": datetime.now().isoformat()
        }


# Convenience functions for easy usage
def update_all_kse100_data(period: str = "5y", max_workers: int = 4) -> Dict[str, bool]:
    """
    Convenience function to update all KSE-100 stock data.
    
    Args:
        period: Period for data download (1y, 2y, 5y, 10y, max)
        max_workers: Maximum number of concurrent downloads
        
    Returns:
        Dictionary mapping symbol to success status
    """
    loader = PSXDataLoader()
    return loader.update_kse100_stocks(period, max_workers)


def update_existing_data(period: str = "2y", max_workers: int = 4) -> Dict[str, bool]:
    """
    Convenience function to update all existing stock data.
    
    Args:
        period: Period for data download (1y, 2y, 5y, 10y, max)
        max_workers: Maximum number of concurrent downloads
        
    Returns:
        Dictionary mapping symbol to success status
    """
    loader = PSXDataLoader()
    return loader.update_existing_stocks(period, max_workers)


def update_single_stock_data(symbol: str, period: str = "5y") -> bool:
    """
    Convenience function to update a single stock's data.
    
    Args:
        symbol: Stock symbol
        period: Period for data download
        
    Returns:
        True if successful, False otherwise
    """
    loader = PSXDataLoader()
    return loader.update_single_stock(symbol, period)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "kse100":
            print("Updating all KSE-100 stocks...")
            results = update_all_kse100_data()
            print(f"Results: {results}")
        elif sys.argv[1] == "existing":
            print("Updating all existing stocks...")
            results = update_existing_data()
            print(f"Results: {results}")
        else:
            symbol = sys.argv[1].upper()
            print(f"Updating {symbol}...")
            success = update_single_stock_data(symbol)
            print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    else:
        print("Usage:")
        print("  python data_loader.py kse100     # Update all KSE-100 stocks")
        print("  python data_loader.py existing   # Update all existing stocks")
        print("  python data_loader.py SYMBOL     # Update specific stock")
