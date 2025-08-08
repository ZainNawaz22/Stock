"""
Data storage module for PSX AI Advisor
Handles CSV file operations for persistent stock data storage
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from .config_loader import get_section, get_value
from .exceptions import (
    DataStorageError, DataIntegrityError, ValidationError,
    create_error_context
)
from .logging_config import get_logger, log_exception, create_operation_logger


class DataStorage:
    """
    Manages persistent storage of stock data in CSV format
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data storage system
        
        Args:
            data_dir (str, optional): Directory for data storage. Defaults to config value.
        """
        # Load configuration
        self.storage_config = get_section('storage')
        
        # Set data directory
        self.data_dir = data_dir or self.storage_config.get('data_directory', 'data')
        self.backup_dir = self.storage_config.get('backup_directory', 'backups')
        self.max_file_age_days = self.storage_config.get('max_file_age_days', 365)
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Expected columns for data validation
        self.expected_columns = [
            'Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 
            'Close', 'Volume', 'Previous_Close', 'Change'
        ]
    
    def _ensure_directories(self) -> None:
        """Create data and backup directories if they don't exist"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)
            self.logger.debug(f"Ensured directories exist: {self.data_dir}, {self.backup_dir}")
        except OSError as e:
            raise DataStorageError(f"Failed to create directories: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to use historical data naming convention.
        Ensures uppercase and appends _HISTORICAL_DATA suffix if missing.
        Also strips common market suffixes like .KS/.KA from inputs.
        """
        if symbol is None:
            return ""
        s = str(symbol).strip().upper()
        if s.endswith('.KS') or s.endswith('.KA'):
            s = s.split('.')[0]
        if not s.endswith('_HISTORICAL_DATA'):
            s = f"{s}_HISTORICAL_DATA"
        return s
    
    def _get_csv_path(self, symbol: str) -> str:
        """
        Get the CSV file path for a given stock symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            str: Full path to CSV file
        """
        # Normalize to historical data naming and clean for filename
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.data_dir, f"{clean_symbol}.csv")

    def _get_unsuffixed_csv_path(self, symbol: str) -> str:
        """Get legacy CSV path without the historical suffix for backward compatibility."""
        if symbol is None:
            base = ""
        else:
            s = str(symbol).strip().upper()
            if s.endswith('_HISTORICAL_DATA'):
                base = s.replace('_HISTORICAL_DATA', '')
            else:
                base = s
            if base.endswith('.KS') or base.endswith('.KA'):
                base = base.split('.')[0]
        clean_base = "".join(c for c in base if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.data_dir, f"{clean_base}.csv")
    
    def _get_backup_path(self, symbol: str, timestamp: Optional[str] = None) -> str:
        """
        Get the backup file path for a given stock symbol
        
        Args:
            symbol (str): Stock symbol
            timestamp (str, optional): Timestamp for backup file. Defaults to current time.
            
        Returns:
            str: Full path to backup file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.backup_dir, f"{clean_symbol}_{timestamp}.csv")
    
    def _validate_data_format(self, data: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has expected format and columns
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Returns:
            bool: True if data format is valid
            
        Raises:
            DataIntegrityError: If data format is invalid
        """
        if data.empty:
            raise DataIntegrityError("Data is empty")
        
        # Check required columns
        missing_columns = set(self.expected_columns) - set(data.columns)
        if missing_columns:
            raise DataIntegrityError(f"Missing required columns: {missing_columns}")
        
        # Check data types for numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise DataIntegrityError(f"Column {col} must be numeric")
        
        # Check Date column
        if 'Date' in data.columns:
            try:
                pd.to_datetime(data['Date'])
            except Exception:
                raise DataIntegrityError("Date column must be valid datetime")
        
        # Validate data ranges
        if 'Open' in data.columns and (data['Open'] <= 0).any():
            raise DataIntegrityError("Open prices must be positive")
        
        if 'High' in data.columns and (data['High'] <= 0).any():
            raise DataIntegrityError("High prices must be positive")
        
        if 'Low' in data.columns and (data['Low'] <= 0).any():
            raise DataIntegrityError("Low prices must be positive")
        
        if 'Close' in data.columns and (data['Close'] <= 0).any():
            raise DataIntegrityError("Close prices must be positive")
        
        if 'Volume' in data.columns and (data['Volume'] < 0).any():
            raise DataIntegrityError("Volume must be non-negative")
        
        # Check High >= Low
        if 'High' in data.columns and 'Low' in data.columns:
            if (data['High'] < data['Low']).any():
                raise DataIntegrityError("High prices must be >= Low prices")
        
        return True
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate entries based on Date and Symbol
        
        Args:
            data (pd.DataFrame): Data to deduplicate
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        initial_count = len(data)
        
        # Remove duplicates based on Date and Symbol, keeping the last occurrence
        data = data.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        
        final_count = len(data)
        if initial_count != final_count:
            self.logger.info(f"Removed {initial_count - final_count} duplicate entries")
        
        return data
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save/append stock data to CSV file without overwriting history
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Stock data to save
            
        Returns:
            bool: True if save successful
            
        Raises:
            DataStorageError: If save operation fails
            DataIntegrityError: If data validation fails
        """
        try:
            # Validate input data
            if data.empty:
                self.logger.warning(f"No data to save for symbol: {symbol}")
                return False
            
            # Ensure proper data types
            if 'Date' in data.columns:
                data = data.copy()
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Ensure Symbol column is present and normalized
            normalized_symbol = self._normalize_symbol(symbol)
            if 'Symbol' in data.columns:
                data['Symbol'] = str(normalized_symbol)
            else:
                data = data.copy()
                data['Symbol'] = str(normalized_symbol)
            data['Symbol'] = data['Symbol'].astype(str)
            
            # Validate data format
            self._validate_data_format(data)
            
            # Get file paths
            csv_path = self._get_csv_path(symbol)
            legacy_csv_path = self._get_unsuffixed_csv_path(symbol)
            
            # Load existing data if file exists
            if os.path.exists(csv_path) or os.path.exists(legacy_csv_path):
                try:
                    read_path = csv_path if os.path.exists(csv_path) else legacy_csv_path
                    existing_data = pd.read_csv(read_path, parse_dates=['Date'])
                    
                    # Combine existing and new data
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    
                    # Remove duplicates
                    combined_data = self._remove_duplicates(combined_data)
                    
                    # Sort by date
                    combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                    
                    self.logger.info(f"Appending {len(data)} new records to existing {len(existing_data)} records for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error reading existing data for {symbol}: {e}")
                    # Create backup of corrupted file
                    backup_path = self._get_backup_path(symbol, f"corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    try:
                        if os.path.exists(csv_path):
                            os.rename(csv_path, backup_path)
                        elif os.path.exists(legacy_csv_path):
                            os.rename(legacy_csv_path, backup_path)
                        self.logger.info(f"Backed up corrupted file to: {backup_path}")
                    except Exception:
                        pass
                    
                    # Use only new data
                    combined_data = self._remove_duplicates(data.copy())
                    combined_data = combined_data.sort_values('Date').reset_index(drop=True)
            else:
                # No existing file, use new data only
                combined_data = self._remove_duplicates(data.copy())
                combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                self.logger.info(f"Creating new file for {symbol} with {len(combined_data)} records")
            
            # Create backup before saving (if file exists)
            if os.path.exists(csv_path):
                backup_path = self._get_backup_path(symbol)
                try:
                    pd.read_csv(csv_path).to_csv(backup_path, index=False)
                    self.logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create backup for {symbol}: {e}")
            
            # Save combined data
            combined_data.to_csv(csv_path, index=False)
            
            self.logger.info(f"Successfully saved {len(combined_data)} records for {symbol} to {csv_path}")
            
            # Validate saved file
            if not self.validate_data_integrity(symbol):
                raise DataStorageError(f"Data integrity validation failed after saving {symbol}")
            
            return True
            
        except DataIntegrityError:
            raise
        except Exception as e:
            raise DataStorageError(f"Error saving data for {symbol}: {e}")    

    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Load historical stock data from CSV file
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Historical stock data
            
        Raises:
            DataStorageError: If load operation fails
            FileNotFoundError: If CSV file doesn't exist
        """
        try:
            csv_path = self._get_csv_path(symbol)
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"No data file found for symbol: {symbol}")
            
            # Load data
            data = pd.read_csv(csv_path, parse_dates=['Date'])
            
            # Validate loaded data
            if data.empty:
                self.logger.warning(f"Empty data file for symbol: {symbol}")
                return pd.DataFrame()
            
            # Sort by date to ensure chronological order
            data = data.sort_values('Date').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(data)} records for {symbol} from {csv_path}")
            
            return data
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DataStorageError(f"Error loading data for {symbol}: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of stock symbols that have data files
        
        Returns:
            List[str]: List of available stock symbols
        """
        try:
            symbols = []
            
            if not os.path.exists(self.data_dir):
                return symbols
            
            # Find all CSV files in data directory
            all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

            # Prefer only files that follow the _HISTORICAL_DATA convention
            hist_files = [f for f in all_files if f[:-4].upper().endswith('_HISTORICAL_DATA')]
            files_to_use = hist_files if hist_files else all_files

            for filename in files_to_use:
                symbol = filename[:-4]
                symbols.append(symbol.upper())
            
            symbols.sort()
            self.logger.debug(f"Found {len(symbols)} symbols with data files")
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def validate_data_integrity(self, symbol: str) -> bool:
        """
        Validate data integrity for a specific symbol
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            bool: True if data integrity is valid
        """
        try:
            csv_path = self._get_csv_path(symbol)
            
            if not os.path.exists(csv_path):
                self.logger.warning(f"No data file found for validation: {symbol}")
                return False
            
            # Load and validate data
            data = pd.read_csv(csv_path, parse_dates=['Date'])
            
            # Check if file is empty
            if data.empty:
                self.logger.error(f"Data file is empty: {symbol}")
                return False
            
            # Validate data format
            try:
                self._validate_data_format(data)
            except DataIntegrityError as e:
                self.logger.error(f"Data integrity validation failed for {symbol}: {e}")
                return False
            
            # Check for chronological order
            if not data['Date'].is_monotonic_increasing:
                self.logger.warning(f"Data is not in chronological order for {symbol}")
                # This is not a critical error, just a warning
            
            # Check for reasonable date range
            min_date = data['Date'].min()
            max_date = data['Date'].max()
            today = pd.Timestamp.now().normalize()
            
            if max_date > today + timedelta(days=2):  # Allow up to 2 days in future for timezone tolerance
                self.logger.error(f"Future dates found in data for {symbol}: {max_date}")
                return False
            
            if min_date < today - timedelta(days=self.max_file_age_days * 2):
                self.logger.warning(f"Very old data found for {symbol}: {min_date}")
                # This is not a critical error, just a warning
            
            # Check for duplicate dates
            duplicate_dates = data[data.duplicated(subset=['Date'], keep=False)]
            if not duplicate_dates.empty:
                self.logger.error(f"Duplicate dates found for {symbol}: {len(duplicate_dates)} records")
                return False
            
            self.logger.debug(f"Data integrity validation passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data integrity for {symbol}: {e}")
            return False
    
    def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get summary information about stored data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Summary information including record count, date range, etc.
        """
        try:
            csv_path = self._get_csv_path(symbol)
            
            if not os.path.exists(csv_path):
                return {
                    'symbol': symbol,
                    'exists': False,
                    'record_count': 0,
                    'file_size': 0
                }
            
            # Get file info
            file_size = os.path.getsize(csv_path)
            
            # Load data for analysis
            data = pd.read_csv(csv_path, parse_dates=['Date'])
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'exists': True,
                    'record_count': 0,
                    'file_size': file_size,
                    'is_empty': True
                }
            
            # Calculate summary statistics
            summary = {
                'symbol': symbol,
                'exists': True,
                'record_count': len(data),
                'file_size': file_size,
                'is_empty': False,
                'date_range': {
                    'start': data['Date'].min().strftime('%Y-%m-%d'),
                    'end': data['Date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min_close': float(data['Close'].min()),
                    'max_close': float(data['Close'].max()),
                    'current_close': float(data['Close'].iloc[-1])
                },
                'volume_stats': {
                    'avg_volume': int(data['Volume'].mean()),
                    'max_volume': int(data['Volume'].max())
                },
                'last_updated': data['Date'].max().strftime('%Y-%m-%d')
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'exists': False,
                'error': str(e)
            }
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """
        Clean up old backup files
        
        Args:
            days_to_keep (int): Number of days to keep backup files
            
        Returns:
            int: Number of files deleted
        """
        try:
            if not os.path.exists(self.backup_dir):
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(self.backup_dir, filename)
                    
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            self.logger.debug(f"Deleted old backup: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Failed to delete backup {filename}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old backup files")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get overall storage statistics
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        try:
            stats = {
                'data_directory': self.data_dir,
                'backup_directory': self.backup_dir,
                'total_symbols': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'oldest_data': None,
                'newest_data': None
            }
            
            if not os.path.exists(self.data_dir):
                return stats
            
            total_size = 0
            total_records = 0
            oldest_date = None
            newest_date = None
            
            symbols = self.get_available_symbols()
            stats['total_symbols'] = len(symbols)
            
            for symbol in symbols:
                csv_path = self._get_csv_path(symbol)
                
                # Add file size
                total_size += os.path.getsize(csv_path)
                
                # Get record count and date range
                try:
                    data = pd.read_csv(csv_path, parse_dates=['Date'])
                    if not data.empty:
                        total_records += len(data)
                        
                        min_date = data['Date'].min()
                        max_date = data['Date'].max()
                        
                        if oldest_date is None or min_date < oldest_date:
                            oldest_date = min_date
                        
                        if newest_date is None or max_date > newest_date:
                            newest_date = max_date
                
                except Exception as e:
                    self.logger.warning(f"Error reading {symbol} for stats: {e}")
            
            stats['total_records'] = total_records
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            if oldest_date:
                stats['oldest_data'] = oldest_date.strftime('%Y-%m-%d')
            if newest_date:
                stats['newest_data'] = newest_date.strftime('%Y-%m-%d')
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}