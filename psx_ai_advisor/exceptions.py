"""
Custom exception classes for PSX AI Advisor

This module defines a hierarchy of custom exceptions for different types of errors
that can occur in the PSX AI Advisor system. These exceptions provide better
error handling and debugging capabilities.
"""


class PSXAdvisorError(Exception):
    """
    Base exception class for all PSX AI Advisor errors.
    
    This is the root exception that all other custom exceptions inherit from.
    It provides a common interface for handling all application-specific errors.
    """
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize PSXAdvisorError
        
        Args:
            message (str): Human-readable error message
            error_code (str, optional): Machine-readable error code
            details (dict, optional): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def __str__(self):
        """Return string representation of the error"""
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"
    
    def to_dict(self):
        """Convert exception to dictionary for logging/serialization"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class DataScrapingError(PSXAdvisorError):
    """
    Raised when data scraping/acquisition operations fail.
    
    This includes CSV download failures, parsing errors, and data validation issues.
    """
    pass


class NetworkError(DataScrapingError):
    """
    Raised when network operations fail.
    
    This includes connection timeouts, DNS resolution failures, and HTTP errors.
    """
    pass


class CSVDownloadError(DataScrapingError):
    """
    Raised when CSV download operations fail.
    
    This includes file not found errors, invalid content types, and download interruptions.
    """
    pass


class CSVParsingError(DataScrapingError):
    """
    Raised when CSV parsing operations fail.
    
    This includes malformed CSV files, encoding issues, and data format problems.
    """
    pass


class DataStorageError(PSXAdvisorError):
    """
    Raised when data storage operations fail.
    
    This includes file I/O errors, permission issues, and disk space problems.
    """
    pass


class DataIntegrityError(DataStorageError):
    """
    Raised when data integrity validation fails.
    
    This includes duplicate data, invalid data ranges, and consistency check failures.
    """
    pass


class TechnicalAnalysisError(PSXAdvisorError):
    """
    Raised when technical analysis calculations fail.
    
    This includes insufficient data for indicators and calculation errors.
    """
    pass


class InsufficientDataError(TechnicalAnalysisError):
    """
    Raised when there's insufficient data for analysis or model training.
    
    This includes cases where minimum data requirements are not met.
    """
    pass


class MLPredictorError(PSXAdvisorError):
    """
    Raised when machine learning operations fail.
    
    This includes model training failures, prediction errors, and model persistence issues.
    """
    pass


class ModelTrainingError(MLPredictorError):
    """
    Raised when model training operations fail.
    
    This includes feature preparation errors and algorithm-specific failures.
    """
    pass


class ModelPersistenceError(MLPredictorError):
    """
    Raised when model save/load operations fail.
    
    This includes file corruption, serialization errors, and version compatibility issues.
    """
    pass


class ConfigurationError(PSXAdvisorError):
    """
    Raised when configuration-related errors occur.
    
    This includes missing config files, invalid settings, and environment issues.
    """
    pass


class ValidationError(PSXAdvisorError):
    """
    Raised when input validation fails.
    
    This includes invalid parameters, malformed data, and constraint violations.
    """
    pass


class SystemError(PSXAdvisorError):
    """
    Raised when system-level errors occur.
    
    This includes resource exhaustion, permission issues, and environment problems.
    """
    pass


class APIError(PSXAdvisorError):
    """
    Raised when API-related errors occur.
    
    This includes request validation, response formatting, and service availability issues.
    """
    pass


class TimeoutError(PSXAdvisorError):
    """
    Raised when operations exceed their timeout limits.
    
    This includes long-running processes and unresponsive external services.
    """
    pass


# Exception mapping for backward compatibility with existing code
EXCEPTION_MAPPING = {
    'PSXDataAcquisitionError': DataScrapingError,
    'DataStorageError': DataStorageError,
    'DataIntegrityError': DataIntegrityError,
    'MLPredictorError': MLPredictorError,
    'InsufficientDataError': InsufficientDataError,
    'ModelTrainingError': ModelTrainingError,
    'NetworkError': NetworkError,
    'CSVDownloadError': CSVDownloadError,
    'CSVParsingError': CSVParsingError,
}


def get_exception_class(exception_name: str) -> type:
    """
    Get exception class by name for backward compatibility.
    
    Args:
        exception_name (str): Name of the exception class
        
    Returns:
        type: Exception class
        
    Raises:
        ValueError: If exception name is not found
    """
    if exception_name in EXCEPTION_MAPPING:
        return EXCEPTION_MAPPING[exception_name]
    
    # Try to get from globals
    if exception_name in globals():
        return globals()[exception_name]
    
    raise ValueError(f"Unknown exception class: {exception_name}")


def create_error_context(operation: str, symbol: str = None, **kwargs) -> dict:
    """
    Create standardized error context for logging and debugging.
    
    Args:
        operation (str): Operation being performed when error occurred
        symbol (str, optional): Stock symbol if applicable
        **kwargs: Additional context information
        
    Returns:
        dict: Error context dictionary
    """
    context = {
        'operation': operation,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }
    
    if symbol:
        context['symbol'] = symbol
    
    context.update(kwargs)
    return context