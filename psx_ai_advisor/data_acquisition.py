"""
Data acquisition module for PSX AI Advisor
Handles downloading and extracting stock data from PSX Closing Rate Summary PDFs
"""

import requests
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .config_loader import get_section, get_value


class PSXDataAcquisitionError(Exception):
    """Base exception for data acquisition errors"""
    pass


class NetworkError(PSXDataAcquisitionError):
    """Raised when network operations fail"""
    pass


class PDFDownloadError(PSXDataAcquisitionError):
    """Raised when PDF download fails"""
    pass


class PSXDataAcquisition:
    """
    Handles downloading daily stock data from PSX Closing Rate Summary PDFs
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
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        self.data_dir = get_value('storage', 'data_directory', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_pdf_filename(self, date: Optional[str] = None) -> str:
        """
        Generate PDF filename for a given date
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: PDF filename
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # PSX uses YYYY-MM-DD.pdf format for closing rates
        return f"{date}.pdf"
    
    def _get_pdf_url(self, date: Optional[str] = None) -> str:
        """
        Generate full URL for PDF download
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: Full PDF download URL
        """
        filename = self._get_pdf_filename(date)
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
        
        # Check if response contains PDF content
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
            raise PDFDownloadError(f"Expected PDF content, got: {content_type}")
        
        # Write file in chunks to handle large files
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        # Verify file was created and has content
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise PDFDownloadError(f"Downloaded file is empty or doesn't exist: {local_path}")
        
        self.logger.info(f"Successfully downloaded: {local_path} ({os.path.getsize(local_path)} bytes)")
        return True
    
    def download_daily_pdf(self, date: Optional[str] = None) -> str:
        """
        Download the daily Closing Rate Summary PDF from PSX
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            str: Path to downloaded PDF file
            
        Raises:
            PDFDownloadError: If PDF download fails
            NetworkError: If network operations fail after retries
        """
        try:
            # Generate URL and local file path
            pdf_url = self._get_pdf_url(date)
            filename = self._get_pdf_filename(date)
            local_path = os.path.join(self.data_dir, filename)
            
            # Check if file already exists
            if os.path.exists(local_path):
                self.logger.info(f"PDF already exists: {local_path}")
                return local_path
            
            # Download with retry mechanism
            self._retry_with_backoff(self._download_file, pdf_url, local_path)
            
            return local_path
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise PDFDownloadError(f"PDF not found for date {date}. It may not be available yet.")
            else:
                raise PDFDownloadError(f"HTTP error downloading PDF: {e}")
        except requests.RequestException as e:
            raise NetworkError(f"Network error downloading PDF: {e}")
        except Exception as e:
            raise PDFDownloadError(f"Unexpected error downloading PDF: {e}")
    
    def get_available_dates(self, days_back: int = 7) -> list:
        """
        Get list of dates for which PDFs might be available
        
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
    
    def verify_pdf_download(self, pdf_path: str) -> bool:
        """
        Verify that downloaded PDF is valid and contains expected content
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            bool: True if PDF is valid
        """
        try:
            if not os.path.exists(pdf_path):
                return False
            
            # Check file size (should be at least a few KB for a valid PDF)
            file_size = os.path.getsize(pdf_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                self.logger.warning(f"PDF file seems too small: {file_size} bytes")
                return False
            
            # Check PDF header
            with open(pdf_path, 'rb') as file:
                header = file.read(4)
                if header != b'%PDF':
                    self.logger.error("File does not have valid PDF header")
                    return False
            
            self.logger.info(f"PDF verification successful: {pdf_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying PDF: {e}")
            return False