"""
Logging configuration module for PSX AI Advisor

This module provides centralized logging configuration with file output,
rotation, and different log levels for various components.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from .config_loader import get_section, get_value


class PSXLoggerFormatter(logging.Formatter):
    """
    Custom formatter for PSX AI Advisor logs with enhanced formatting.
    """
    
    def __init__(self):
        super().__init__()
        
        # Color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # Format strings
        self.file_format = (
            '%(asctime)s | %(levelname)-8s | %(name)-25s | '
            '%(funcName)-20s:%(lineno)-4d | %(message)s'
        )
        
        self.console_format = (
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        
        self.detailed_format = (
            '%(asctime)s | %(levelname)-8s | %(name)-25s | '
            '%(funcName)-20s:%(lineno)-4d | PID:%(process)d | %(message)s'
        )
    
    def format(self, record):
        """Format log record with appropriate format and colors"""
        # Choose format based on handler type
        if hasattr(record, 'handler_type'):
            if record.handler_type == 'console':
                format_str = self.console_format
            elif record.handler_type == 'detailed':
                format_str = self.detailed_format
            else:
                format_str = self.file_format
        else:
            format_str = self.file_format
        
        # Create formatter with chosen format
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Format the record
        formatted = formatter.format(record)
        
        # Add colors for console output
        if hasattr(record, 'handler_type') and record.handler_type == 'console':
            level_name = record.levelname
            if level_name in self.colors:
                color = self.colors[level_name]
                reset = self.colors['RESET']
                formatted = f"{color}{formatted}{reset}"
        
        return formatted


class LoggingManager:
    """
    Centralized logging manager for PSX AI Advisor.
    
    Provides configuration and management of logging across all modules.
    """
    
    def __init__(self):
        """Initialize the logging manager"""
        self.config = get_section('logging')
        self.is_configured = False
        self.loggers = {}
        self.handlers = {}
        
        # Default configuration
        self.default_config = {
            'level': 'INFO',
            'file': 'psx_advisor.log',
            'max_size_mb': 10,
            'backup_count': 5,
            'console_output': True,
            'detailed_errors': True,
            'log_directory': 'logs'
        }
        
        # Merge with user configuration
        self.effective_config = {**self.default_config, **self.config}
    
    def setup_logging(self, force_reconfigure: bool = False) -> None:
        """
        Set up logging configuration for the entire application.
        
        Args:
            force_reconfigure (bool): Force reconfiguration even if already configured
        """
        if self.is_configured and not force_reconfigure:
            return
        
        try:
            # Create logs directory
            log_dir = self.effective_config['log_directory']
            os.makedirs(log_dir, exist_ok=True)
            
            # Clear existing handlers to avoid duplicates
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Set root logger level
            log_level = getattr(logging, self.effective_config['level'].upper(), logging.INFO)
            root_logger.setLevel(log_level)
            
            # Create custom formatter
            formatter = PSXLoggerFormatter()
            
            # Setup file handler with rotation
            self._setup_file_handler(formatter, log_dir)
            
            # Setup console handler if enabled
            if self.effective_config.get('console_output', True):
                self._setup_console_handler(formatter)
            
            # Setup error file handler for detailed error logging
            if self.effective_config.get('detailed_errors', True):
                self._setup_error_handler(formatter, log_dir)
            
            # Configure specific loggers
            self._configure_module_loggers()
            
            self.is_configured = True
            
            # Log successful configuration
            logger = logging.getLogger(__name__)
            logger.info("Logging system configured successfully")
            logger.info(f"Log level: {self.effective_config['level']}")
            logger.info(f"Log directory: {log_dir}")
            logger.info(f"Console output: {self.effective_config.get('console_output', True)}")
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.error(f"Failed to configure logging: {e}")
    
    def _setup_file_handler(self, formatter: PSXLoggerFormatter, log_dir: str) -> None:
        """Setup rotating file handler for general logs"""
        log_file = os.path.join(log_dir, self.effective_config['file'])
        max_bytes = self.effective_config['max_size_mb'] * 1024 * 1024
        backup_count = self.effective_config['backup_count']
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handler type for formatter
        file_handler.addFilter(lambda record: setattr(record, 'handler_type', 'file') or True)
        
        logging.getLogger().addHandler(file_handler)
        self.handlers['file'] = file_handler
    
    def _setup_console_handler(self, formatter: PSXLoggerFormatter) -> None:
        """Setup console handler for stdout output"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Console shows INFO and above by default
        console_level = getattr(logging, self.effective_config.get('console_level', 'INFO').upper(), logging.INFO)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        
        # Add handler type for formatter
        console_handler.addFilter(lambda record: setattr(record, 'handler_type', 'console') or True)
        
        logging.getLogger().addHandler(console_handler)
        self.handlers['console'] = console_handler
    
    def _setup_error_handler(self, formatter: PSXLoggerFormatter, log_dir: str) -> None:
        """Setup separate handler for error-level logs"""
        error_log_file = os.path.join(log_dir, 'errors.log')
        max_bytes = self.effective_config['max_size_mb'] * 1024 * 1024
        backup_count = self.effective_config['backup_count']
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Add handler type for formatter
        error_handler.addFilter(lambda record: setattr(record, 'handler_type', 'detailed') or True)
        
        logging.getLogger().addHandler(error_handler)
        self.handlers['error'] = error_handler
    
    def _configure_module_loggers(self) -> None:
        """Configure loggers for specific modules"""
        # PSX AI Advisor modules
        psx_modules = [
            'psx_ai_advisor.data_acquisition',
            'psx_ai_advisor.data_storage',
            'psx_ai_advisor.technical_analysis',
            'psx_ai_advisor.ml_predictor',
            'psx_ai_advisor.config_loader'
        ]
        
        for module_name in psx_modules:
            logger = logging.getLogger(module_name)
            logger.setLevel(logging.DEBUG)
            self.loggers[module_name] = logger
        
        # External library loggers (reduce verbosity)
        external_loggers = {
            'requests': logging.WARNING,
            'urllib3': logging.WARNING,
            'sklearn': logging.WARNING,
            'pandas': logging.WARNING,
            'numpy': logging.WARNING
        }
        
        for logger_name, level in external_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            self.loggers[logger_name] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger for a specific module.
        
        Args:
            name (str): Logger name (usually __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if not self.is_configured:
            self.setup_logging()
        
        return logging.getLogger(name)
    
    def log_exception(self, logger: logging.Logger, exception: Exception, 
                     context: Dict[str, Any] = None, operation: str = None) -> None:
        """
        Log an exception with full context and stack trace.
        
        Args:
            logger (logging.Logger): Logger to use
            exception (Exception): Exception to log
            context (dict, optional): Additional context information
            operation (str, optional): Operation being performed when exception occurred
        """
        error_info = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'operation': operation or 'unknown'
        }
        
        if context:
            error_info.update(context)
        
        # Log error with context
        logger.error(
            f"Exception in {error_info['operation']}: {error_info['exception_type']} - {error_info['exception_message']}",
            extra={'error_context': error_info},
            exc_info=True
        )
    
    def log_performance(self, logger: logging.Logger, operation: str, 
                       duration: float, details: Dict[str, Any] = None) -> None:
        """
        Log performance metrics for operations.
        
        Args:
            logger (logging.Logger): Logger to use
            operation (str): Operation name
            duration (float): Duration in seconds
            details (dict, optional): Additional performance details
        """
        perf_info = {
            'operation': operation,
            'duration_seconds': round(duration, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            perf_info.update(details)
        
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra={'performance': perf_info})
    
    def create_operation_logger(self, operation: str, symbol: str = None) -> 'OperationLogger':
        """
        Create a context manager for logging operations.
        
        Args:
            operation (str): Operation name
            symbol (str, optional): Stock symbol if applicable
            
        Returns:
            OperationLogger: Context manager for operation logging
        """
        return OperationLogger(self, operation, symbol)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logging activity.
        
        Returns:
            dict: Logging statistics
        """
        stats = {
            'configured': self.is_configured,
            'handlers': list(self.handlers.keys()),
            'loggers': list(self.loggers.keys()),
            'config': self.effective_config
        }
        
        # Get file sizes if handlers exist
        if 'file' in self.handlers:
            try:
                handler = self.handlers['file']
                if hasattr(handler, 'baseFilename'):
                    log_file = handler.baseFilename
                    if os.path.exists(log_file):
                        stats['main_log_size_mb'] = round(os.path.getsize(log_file) / (1024 * 1024), 2)
            except Exception:
                pass
        
        if 'error' in self.handlers:
            try:
                handler = self.handlers['error']
                if hasattr(handler, 'baseFilename'):
                    error_file = handler.baseFilename
                    if os.path.exists(error_file):
                        stats['error_log_size_mb'] = round(os.path.getsize(error_file) / (1024 * 1024), 2)
            except Exception:
                pass
        
        return stats


class OperationLogger:
    """
    Context manager for logging operations with automatic timing and error handling.
    """
    
    def __init__(self, logging_manager: LoggingManager, operation: str, symbol: str = None):
        """
        Initialize operation logger.
        
        Args:
            logging_manager (LoggingManager): Logging manager instance
            operation (str): Operation name
            symbol (str, optional): Stock symbol if applicable
        """
        self.logging_manager = logging_manager
        self.operation = operation
        self.symbol = symbol
        self.logger = logging_manager.get_logger('psx_ai_advisor.operations')
        self.start_time = None
        self.context = {'operation': operation}
        
        if symbol:
            self.context['symbol'] = symbol
    
    def __enter__(self):
        """Start operation logging"""
        self.start_time = __import__('time').time()
        
        log_message = f"Starting operation: {self.operation}"
        if self.symbol:
            log_message += f" for symbol: {self.symbol}"
        
        self.logger.info(log_message, extra={'operation_context': self.context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End operation logging with timing and error handling"""
        duration = __import__('time').time() - self.start_time
        
        if exc_type is None:
            # Success
            log_message = f"Completed operation: {self.operation} in {duration:.3f}s"
            if self.symbol:
                log_message += f" for symbol: {self.symbol}"
            
            self.logger.info(log_message, extra={'operation_context': self.context})
            self.logging_manager.log_performance(self.logger, self.operation, duration, self.context)
        else:
            # Error occurred
            error_context = {**self.context, 'duration_seconds': round(duration, 3)}
            self.logging_manager.log_exception(self.logger, exc_val, error_context, self.operation)
        
        return False  # Don't suppress exceptions
    
    def add_context(self, **kwargs):
        """Add additional context to the operation"""
        self.context.update(kwargs)
    
    def log_progress(self, message: str, **kwargs):
        """Log progress during the operation"""
        progress_context = {**self.context, **kwargs}
        self.logger.info(f"{self.operation}: {message}", extra={'progress_context': progress_context})


# Global logging manager instance
_logging_manager = None


def get_logging_manager() -> LoggingManager:
    """
    Get the global logging manager instance.
    
    Returns:
        LoggingManager: Global logging manager
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def setup_logging(force_reconfigure: bool = False) -> None:
    """
    Setup logging for the entire application.
    
    Args:
        force_reconfigure (bool): Force reconfiguration even if already configured
    """
    manager = get_logging_manager()
    manager.setup_logging(force_reconfigure)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    manager = get_logging_manager()
    return manager.get_logger(name)


def log_exception(logger: logging.Logger, exception: Exception, 
                 context: Dict[str, Any] = None, operation: str = None) -> None:
    """
    Log an exception with full context.
    
    Args:
        logger (logging.Logger): Logger to use
        exception (Exception): Exception to log
        context (dict, optional): Additional context information
        operation (str, optional): Operation being performed
    """
    manager = get_logging_manager()
    manager.log_exception(logger, exception, context, operation)


def create_operation_logger(operation: str, symbol: str = None) -> OperationLogger:
    """
    Create a context manager for logging operations.
    
    Args:
        operation (str): Operation name
        symbol (str, optional): Stock symbol if applicable
        
    Returns:
        OperationLogger: Context manager for operation logging
    """
    manager = get_logging_manager()
    return manager.create_operation_logger(operation, symbol)