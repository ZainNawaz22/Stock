"""
API monitoring and performance tracking utilities.

This module provides logging, performance monitoring, and request tracking
functionality for the PSX AI Advisor FastAPI application.
"""

import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Performance metrics storage
_performance_metrics = {
    'request_count': 0,
    'total_response_time': 0.0,
    'endpoint_metrics': {},
    'error_count': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'start_time': datetime.now()
}


class APIMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests and responses.
    
    Tracks request timing, response codes, and logs API usage.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and track performance metrics.
        
        Args:
            request (Request): FastAPI request object
            call_next (Callable): Next middleware/endpoint
            
        Returns:
            Response: FastAPI response object
        """
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        # Log request start
        logger.info(f"API Request: {endpoint} from {request.client.host if request.client else 'unknown'}")
        
        # Track request count
        _performance_metrics['request_count'] += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            _performance_metrics['total_response_time'] += response_time
            
            if endpoint not in _performance_metrics['endpoint_metrics']:
                _performance_metrics['endpoint_metrics'][endpoint] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'error_count': 0
                }
            
            endpoint_metrics = _performance_metrics['endpoint_metrics'][endpoint]
            endpoint_metrics['count'] += 1
            endpoint_metrics['total_time'] += response_time
            endpoint_metrics['avg_time'] = endpoint_metrics['total_time'] / endpoint_metrics['count']
            endpoint_metrics['min_time'] = min(endpoint_metrics['min_time'], response_time)
            endpoint_metrics['max_time'] = max(endpoint_metrics['max_time'], response_time)
            
            # Track errors
            if response.status_code >= 400:
                _performance_metrics['error_count'] += 1
                endpoint_metrics['error_count'] += 1
                logger.warning(f"API Error: {endpoint} returned {response.status_code} in {response_time:.3f}s")
            else:
                logger.info(f"API Success: {endpoint} returned {response.status_code} in {response_time:.3f}s")
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-ID"] = str(_performance_metrics['request_count'])
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            _performance_metrics['error_count'] += 1
            
            logger.error(f"API Exception: {endpoint} failed after {response_time:.3f}s - {str(e)}")
            raise


def track_cache_hit():
    """Track a cache hit for performance metrics."""
    _performance_metrics['cache_hits'] += 1


def track_cache_miss():
    """Track a cache miss for performance metrics."""
    _performance_metrics['cache_misses'] += 1


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get current performance metrics.
    
    Returns:
        Dict[str, Any]: Performance metrics data
    """
    uptime = datetime.now() - _performance_metrics['start_time']
    total_requests = _performance_metrics['request_count']
    
    metrics = {
        'uptime_seconds': uptime.total_seconds(),
        'uptime_formatted': str(uptime),
        'total_requests': total_requests,
        'error_count': _performance_metrics['error_count'],
        'error_rate': _performance_metrics['error_count'] / max(total_requests, 1),
        'average_response_time': _performance_metrics['total_response_time'] / max(total_requests, 1),
        'cache_hits': _performance_metrics['cache_hits'],
        'cache_misses': _performance_metrics['cache_misses'],
        'cache_hit_rate': _performance_metrics['cache_hits'] / max(
            _performance_metrics['cache_hits'] + _performance_metrics['cache_misses'], 1
        ),
        'requests_per_minute': total_requests / max(uptime.total_seconds() / 60, 1),
        'endpoint_metrics': _performance_metrics['endpoint_metrics']
    }
    
    return metrics


def log_api_performance(func: Callable) -> Callable:
    """
    Decorator to log API endpoint performance.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            logger.debug(f"Starting {function_name}")
            result = await func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(f"Completed {function_name} in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {function_name} after {execution_time:.3f}s - {str(e)}")
            raise
    
    return wrapper


def log_data_operation(operation: str, symbol: str = None, record_count: int = None):
    """
    Log data operations for monitoring.
    
    Args:
        operation (str): Type of operation (load, save, process, etc.)
        symbol (str, optional): Stock symbol involved
        record_count (int, optional): Number of records processed
    """
    message = f"Data operation: {operation}"
    if symbol:
        message += f" for {symbol}"
    if record_count:
        message += f" ({record_count} records)"
    
    logger.info(message)


def log_ml_operation(operation: str, symbol: str = None, accuracy: float = None, 
                    feature_count: int = None, execution_time: float = None):
    """
    Log ML operations for monitoring.
    
    Args:
        operation (str): Type of ML operation (train, predict, evaluate, etc.)
        symbol (str, optional): Stock symbol involved
        accuracy (float, optional): Model accuracy
        feature_count (int, optional): Number of features used
        execution_time (float, optional): Execution time in seconds
    """
    message = f"ML operation: {operation}"
    if symbol:
        message += f" for {symbol}"
    if accuracy is not None:
        message += f" (accuracy: {accuracy:.3f})"
    if feature_count:
        message += f" (features: {feature_count})"
    if execution_time:
        message += f" (time: {execution_time:.3f}s)"
    
    logger.info(message)


def log_cache_operation(operation: str, key: str = None, hit: bool = None):
    """
    Log cache operations for monitoring.
    
    Args:
        operation (str): Type of cache operation (get, set, clear, etc.)
        key (str, optional): Cache key
        hit (bool, optional): Whether it was a cache hit
    """
    message = f"Cache operation: {operation}"
    if key:
        message += f" for {key}"
    if hit is not None:
        message += f" ({'HIT' if hit else 'MISS'})"
        
        # Track cache metrics
        if hit:
            track_cache_hit()
        else:
            track_cache_miss()
    
    logger.debug(message)


def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """
    Log errors with additional context information.
    
    Args:
        error (Exception): Exception that occurred
        context (Dict[str, Any], optional): Additional context information
    """
    error_message = f"Error: {type(error).__name__}: {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_message += f" (Context: {context_str})"
    
    logger.error(error_message, exc_info=True)


class PerformanceTimer:
    """
    Context manager for timing operations.
    
    Usage:
        with PerformanceTimer("operation_name") as timer:
            # do something
            pass
        print(f"Operation took {timer.elapsed_time:.3f} seconds")
    """
    
    def __init__(self, operation_name: str, log_result: bool = True):
        """
        Initialize performance timer.
        
        Args:
            operation_name (str): Name of the operation being timed
            log_result (bool): Whether to log the result automatically
        """
        self.operation_name = operation_name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        logger.debug(f"Starting timer for {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and optionally log result."""
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.log_result:
            if exc_type is None:
                logger.info(f"Completed {self.operation_name} in {self.elapsed_time:.3f}s")
            else:
                logger.error(f"Failed {self.operation_name} after {self.elapsed_time:.3f}s")


def create_health_check_logger():
    """
    Create a separate logger for health check operations.
    
    Returns:
        logging.Logger: Health check logger
    """
    health_logger = logging.getLogger('psx_advisor.health')
    health_logger.setLevel(logging.INFO)
    
    # Create handler if it doesn't exist
    if not health_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        health_logger.addHandler(handler)
    
    return health_logger


# Health check logger instance
health_logger = create_health_check_logger()


def log_health_check(component: str, status: str, details: Dict[str, Any] = None):
    """
    Log health check results.
    
    Args:
        component (str): Component being checked (data, models, cache, etc.)
        status (str): Health status (healthy, degraded, unhealthy)
        details (Dict[str, Any], optional): Additional health details
    """
    message = f"Health check - {component}: {status}"
    
    if details:
        details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        message += f" ({details_str})"
    
    if status == 'healthy':
        health_logger.info(message)
    elif status == 'degraded':
        health_logger.warning(message)
    else:
        health_logger.error(message)


def reset_performance_metrics():
    """Reset all performance metrics to initial state."""
    global _performance_metrics
    _performance_metrics = {
        'request_count': 0,
        'total_response_time': 0.0,
        'endpoint_metrics': {},
        'error_count': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'start_time': datetime.now()
    }
    logger.info("Performance metrics reset")