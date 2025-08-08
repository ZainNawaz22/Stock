"""
Data acquisition module for PSX AI Advisor
Handles downloading and parsing stock data from PSX Closing Rate Summary CSV files
"""

import requests
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from .config_loader import get_section, get_value
from .exceptions import (
    DataScrapingError, NetworkError, CSVDownloadError, CSVParsingError,
    create_error_context
)
from .logging_config import get_logger, log_exception, create_operation_logger


class PSXDataAcquisition:
    """
    Handles downloading daily stock data from PSX Closing Rate Summary CSV files
    """
    
    def __init__(self):
        """Initialize the data acquisition system with configuration"""
        self.config = get_section('data_sources')
        self.performance_config = get_section('performance')
        
        self.base_url = self.config['psx_base_url']
        self.downloads_endpoint = "/download/closing_rates"  # Correct endpoint
        self.downloads_url = f"{self.base_url}{self.downloads_endpoint}"
        
        # Performance settings
        self.request_timeout = self.performance_config.get('request_timeout', 30)
        self.retry_attempts = self.performance_config.get('retry_attempts', 3)
        self.retry_delay = self.performance_config.get('retry_delay', 2)
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Create data directory if it doesn't exist
        self.data_dir = get_value('storage', 'data_directory', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_csv_filename(self, date: Optional[str] = None) -> str:
        """
        Generate CSV filename for a given date
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: CSV filename
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # PSX uses YYYY-MM-DD.csv format for closing rates
        return f"{date}.csv"
    
    def _get_csv_url(self, date: Optional[str] = None) -> str:
        """
        Generate full URL for CSV download
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: Full CSV download URL
        """
        filename = self._get_csv_filename(date)
        return f"{self.downloads_url}/{filename}"
    
    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry mechanism
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            NetworkError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except (requests.RequestException, requests.Timeout) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.retry_attempts} attempts failed")
        
        raise NetworkError(f"Network operation failed after {self.retry_attempts} attempts: {last_exception}")
    
    def _download_file(self, url: str, local_path: str) -> bool:
        """
        Download file from URL to local path
        
        Args:
            url (str): URL to download from
            local_path (str): Local file path to save to
            
        Returns:
            bool: True if download successful
            
        Raises:
            requests.RequestException: If download fails
        """
        self.logger.info(f"Downloading from: {url}")
        
        response = requests.get(
            url,
            timeout=self.request_timeout,
            stream=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        response.raise_for_status()
        
        # Check if response contains CSV content
        content_type = response.headers.get('content-type', '').lower()
        if 'csv' not in content_type and 'text' not in content_type and 'application/octet-stream' not in content_type:
            raise CSVDownloadError(f"Expected CSV content, got: {content_type}")
        
        # Write file in chunks to handle large files
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        # Verify file was created and has content
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise CSVDownloadError(f"Downloaded file is empty or doesn't exist: {local_path}")
        
        self.logger.info(f"Successfully downloaded: {local_path} ({os.path.getsize(local_path)} bytes)")
        return True
    
    def download_daily_csv(self, date: Optional[str] = None) -> str:
        """
        Download the daily Closing Rate Summary CSV from PSX
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: Path to downloaded CSV file
            
        Raises:
            CSVDownloadError: If CSV download fails
            NetworkError: If network operations fail after retries
        """
        operation_name = "download_daily_csv"
        context = create_error_context(operation_name, date=date)
        
        with create_operation_logger(operation_name) as op_logger:
            op_logger.add_context(date=date)
            
            try:
                # Generate URL and local file path
                csv_url = self._get_csv_url(date)
                filename = self._get_csv_filename(date)
                local_path = os.path.join(self.data_dir, filename)
                
                op_logger.add_context(csv_url=csv_url, local_path=local_path)
                
                # Check if file already exists
                if os.path.exists(local_path):
                    self.logger.info(f"CSV already exists: {local_path}")
                    op_logger.log_progress("File already exists, skipping download")
                    return local_path
                
                op_logger.log_progress("Starting CSV download with retry mechanism")
                
                # Download with retry mechanism
                self._retry_with_backoff(self._download_file, csv_url, local_path)
                
                op_logger.log_progress("CSV download completed successfully")
                return local_path
                
            except requests.HTTPError as e:
                error_context = {**context, 'http_status': e.response.status_code if e.response else None}
                if e.response and e.response.status_code == 404:
                    error_msg = f"CSV not found for date {date}. It may not be available yet."
                    exc = CSVDownloadError(error_msg, 'CSV_NOT_FOUND', error_context)
                else:
                    error_msg = f"HTTP error downloading CSV: {e}"
                    exc = CSVDownloadError(error_msg, 'HTTP_ERROR', error_context)
                log_exception(self.logger, exc, error_context, operation_name)
                raise exc
            except requests.RequestException as e:
                error_context = {**context, 'request_error': str(e)}
                exc = NetworkError(f"Network error downloading CSV: {e}", 'NETWORK_ERROR', error_context)
                log_exception(self.logger, exc, error_context, operation_name)
                raise exc
            except Exception as e:
                error_context = {**context, 'unexpected_error': str(e)}
                exc = CSVDownloadError(f"Unexpected error downloading CSV: {e}", 'UNEXPECTED_ERROR', error_context)
                log_exception(self.logger, exc, error_context, operation_name)
                raise exc
    
    def get_available_dates(self, days_back: int = 7) -> list:
        """
        Get list of dates for which CSVs might be available
        
        Args:
            days_back (int): Number of days to look back from today
            
        Returns:
            list: List of date strings in YYYY-MM-DD format
        """
        dates = []
        today = datetime.now()
        
        for i in range(days_back):
            date = today - timedelta(days=i)
            # Skip weekends (PSX is closed)
            if date.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(date.strftime('%Y-%m-%d'))
        
        return dates
    
    def verify_csv_download(self, csv_path: str) -> bool:
        """
        Verify that downloaded CSV is valid and contains expected content
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            bool: True if CSV is valid
        """
        try:
            if not os.path.exists(csv_path):
                return False
            
            # Check file size (should be at least a few KB for a valid CSV)
            file_size = os.path.getsize(csv_path)
            if file_size < 100:  # Less than 100 bytes is suspicious
                self.logger.warning(f"CSV file seems too small: {file_size} bytes")
                return False
            
            # Try to read the first few lines to verify CSV format
            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip()
                    if not first_line:
                        self.logger.error("CSV file appears to be empty")
                        return False
                    
                    # Check if it looks like CSV (has commas)
                    if ',' not in first_line:
                        self.logger.warning("File doesn't appear to be comma-separated")
                        # Could still be valid with different separator
                
            except UnicodeDecodeError:
                # Try with different encoding
                with open(csv_path, 'r', encoding='latin-1') as file:
                    first_line = file.readline().strip()
                    if not first_line:
                        self.logger.error("CSV file appears to be empty")
                        return False
            
            self.logger.info(f"CSV verification successful: {csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying CSV: {e}")
            return False
    
    def _validate_csv_format(self, csv_path: str) -> bool:
        """
        Validate CSV format and structure for the expected format:
        Date,Open,High,Low,Close,Volume,Dividends,Stock Splits,Capital Gains
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            bool: True if CSV format is valid
        """
        try:
            # Try to read the CSV with pandas to validate structure
            df = pd.read_csv(csv_path, nrows=5)  # Read just first 5 rows for validation
            
            # Check if we have the expected OHLCV columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"CSV missing required columns: {missing_columns}")
                self.logger.info(f"Available columns: {list(df.columns)}")
                return False
            
            # Check if we have at least some data
            if len(df) == 0:
                self.logger.error("CSV file contains no data rows")
                return False
            
            # Verify Date column can be parsed
            try:
                pd.to_datetime(df['Date'].iloc[0])
            except Exception as e:
                self.logger.error(f"Date column cannot be parsed: {e}")
                return False
            
            # Verify numeric columns contain numeric data
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                try:
                    pd.to_numeric(df[col].iloc[0])
                except Exception as e:
                    self.logger.error(f"Column {col} contains non-numeric data: {e}")
                    return False
            
            self.logger.info(f"CSV validation successful: {len(df.columns)} columns, {len(df)} sample rows")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV validation failed: {e}")
            return False
    
    def parse_stock_data(self, csv_path: str) -> pd.DataFrame:
        """
        Parse OHLCV data for all stocks from the CSV file
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing stock data with columns:
                         Symbol, Company_Name, Open, High, Low, Close, Volume, Previous_Close, Change
                         
        Raises:
            CSVParsingError: If CSV parsing fails
            FileNotFoundError: If CSV file doesn't exist
        """
        operation_name = "parse_stock_data"
        context = create_error_context(operation_name, csv_path=csv_path)
        
        if not os.path.exists(csv_path):
            exc = FileNotFoundError(f"CSV file not found: {csv_path}")
            log_exception(self.logger, exc, context, operation_name)
            raise exc
        
        with create_operation_logger(operation_name) as op_logger:
            op_logger.add_context(csv_path=csv_path)
            
            try:
                self.logger.info(f"Parsing stock data from: {csv_path}")
                op_logger.log_progress("Starting CSV parsing with multiple encoding attempts")
                
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding)
                        successful_encoding = encoding
                        self.logger.info(f"Successfully read CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to read CSV with {encoding}: {e}")
                        continue
                
                if df is None:
                    error_context = {**context, 'attempted_encodings': encodings}
                    exc = CSVParsingError("Could not read CSV file with any supported encoding", 'ENCODING_ERROR', error_context)
                    log_exception(self.logger, exc, error_context, operation_name)
                    raise exc
                
                if len(df) == 0:
                    error_context = {**context, 'encoding_used': successful_encoding}
                    exc = CSVParsingError("CSV file contains no data", 'EMPTY_FILE', error_context)
                    log_exception(self.logger, exc, error_context, operation_name)
                    raise exc
                
                op_logger.log_progress(f"CSV read successfully", rows=len(df), columns=len(df.columns), encoding=successful_encoding)
                self.logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns")
                self.logger.info(f"Columns: {list(df.columns)}")
                
                # Check if this is the expected format: Date,Open,High,Low,Close,Volume,Dividends,Stock Splits,Capital Gains
                expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Verify we have the required OHLCV columns
                missing_columns = [col for col in expected_columns if col not in df.columns]
                if missing_columns:
                    error_context = {**context, 'missing_columns': missing_columns, 'found_columns': list(df.columns)}
                    exc = CSVParsingError(f"CSV missing required columns: {missing_columns}. Found columns: {list(df.columns)}", 'MISSING_COLUMNS', error_context)
                    log_exception(self.logger, exc, error_context, operation_name)
                    raise exc
                
                op_logger.log_progress("Column validation passed, processing data")
                
                # Keep only the columns we need for technical analysis (ignore Dividends, Stock Splits, Capital Gains)
                columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = df[columns_to_keep].copy()
                
                # Convert Date column to datetime and remove timezone info for consistency
                try:
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                except Exception as e:
                    error_context = {**context, 'date_conversion_error': str(e)}
                    exc = CSVParsingError(f"Error converting Date column: {e}", 'DATE_CONVERSION_ERROR', error_context)
                    log_exception(self.logger, exc, error_context, operation_name)
                    raise exc
                
                # Data validation and type conversion for OHLCV columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove rows with invalid data
                initial_count = len(df)
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                final_count = len(df)
                
                if initial_count != final_count:
                    removed_count = initial_count - final_count
                    self.logger.warning(f"Removed {removed_count} rows with invalid data")
                    op_logger.log_progress(f"Data cleaning completed", removed_invalid_rows=removed_count)
                
                # Additional data validation
                # Remove rows where High < Low or Close <= 0
                validation_filter = (
                    (df['High'] >= df['Low']) & 
                    (df['Close'] > 0) & 
                    (df['Open'] > 0) & 
                    (df['Volume'] >= 0)
                )
                
                pre_validation_count = len(df)
                df = df[validation_filter]
                post_validation_count = len(df)
                
                if pre_validation_count != post_validation_count:
                    validation_removed = pre_validation_count - post_validation_count
                    self.logger.warning(f"Removed {validation_removed} rows failing business logic validation")
                    op_logger.log_progress("Business logic validation completed", removed_invalid_rows=validation_removed)
                
                filename = os.path.basename(csv_path)
                if filename.lower().endswith('.csv'):
                    symbol = filename[:-4].upper()
                else:
                    import re
                    symbol_match = re.search(r'([A-Z]+)', filename)
                    symbol = symbol_match.group(1).upper() if symbol_match else 'UNKNOWN'
                
                op_logger.add_context(symbol=symbol)
                
                # Add Symbol column
                df['Symbol'] = symbol
                
                # Add Company_Name (use Symbol as placeholder)
                df['Company_Name'] = symbol
                
                # Calculate Previous_Close and Change
                df = df.sort_values('Date').reset_index(drop=True)
                df['Previous_Close'] = df['Close'].shift(1)
                df['Change'] = df['Close'] - df['Previous_Close']
                
                # Fill first row's Previous_Close with Close value
                df.loc[0, 'Previous_Close'] = df.loc[0, 'Close']
                df.loc[0, 'Change'] = 0.0
                
                # Reorder columns
                column_order = ['Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
                df = df[column_order]
                
                # Sort by date (most recent first)
                df = df.sort_values('Date', ascending=False).reset_index(drop=True)
                
                date_range = f"{df['Date'].min()} to {df['Date'].max()}"
                self.logger.info(f"Successfully parsed {len(df)} records for stock {symbol}")
                self.logger.info(f"Date range: {date_range}")
                
                op_logger.log_progress("Parsing completed successfully", 
                                     final_record_count=len(df), 
                                     symbol=symbol, 
                                     date_range=date_range)
                
                return df
                
            except Exception as e:
                if isinstance(e, (CSVParsingError, FileNotFoundError)):
                    raise
                else:
                    error_context = {**context, 'unexpected_error': str(e)}
                    exc = CSVParsingError(f"Error parsing stock data from CSV: {e}", 'UNEXPECTED_PARSING_ERROR', error_context)
                    log_exception(self.logger, exc, error_context, operation_name)
                    raise exc
    
    def get_all_stock_data(self, date: Optional[str] = None, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Download CSV and parse all stock data for a given date, or parse from a specific CSV file
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            csv_path (str, optional): Direct path to CSV file to parse. If provided, date is ignored.
            
        Returns:
            pd.DataFrame: DataFrame containing all stock data
            
        Raises:
            CSVDownloadError: If CSV download fails
            CSVParsingError: If CSV parsing fails
        """
        try:
            if csv_path:
                # Use provided CSV file path
                if not os.path.exists(csv_path):
                    raise CSVParsingError(f"CSV file not found: {csv_path}")
                file_path = csv_path
            else:
                # Download the CSV
                file_path = self.download_daily_csv(date)
            
            # Parse stock data
            stock_data = self.parse_stock_data(file_path)
            
            self.logger.info(f"Successfully processed {len(stock_data)} records from {file_path}")
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error getting stock data: {e}")
            raise