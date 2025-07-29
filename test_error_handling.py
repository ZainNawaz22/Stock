#!/usr/bin/env python3
"""
Test script for PSX AI Advisor error handling and logging system

This script tests the comprehensive error handling and logging functionality
to ensure all components work correctly.
"""

import sys
import os
import traceback
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.exceptions import *
from psx_ai_advisor.logging_config import setup_logging, get_logger, log_exception, create_operation_logger
from psx_ai_advisor.fallback_mechanisms import execute_with_fallback, get_fallback_stats, register_fallback


def test_logging_setup():
    """Test logging configuration"""
    print("Testing logging setup...")
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("✓ Logging setup test completed")


def test_custom_exceptions():
    """Test custom exception classes"""
    print("Testing custom exceptions...")
    
    logger = get_logger(__name__)
    
    # Test different exception types
    exceptions_to_test = [
        (PSXAdvisorError, "Base PSX Advisor error"),
        (DataScrapingError, "Data scraping failed"),
        (NetworkError, "Network connection failed"),
        (CSVDownloadError, "CSV download failed"),
        (CSVParsingError, "CSV parsing failed"),
        (DataStorageError, "Data storage failed"),
        (DataIntegrityError, "Data integrity check failed"),
        (TechnicalAnalysisError, "Technical analysis failed"),
        (InsufficientDataError, "Insufficient data for analysis"),
        (MLPredictorError, "ML prediction failed"),
        (ModelTrainingError, "Model training failed"),
        (ModelPersistenceError, "Model persistence failed"),
        (ConfigurationError, "Configuration error"),
        (ValidationError, "Validation failed"),
        (SystemError, "System error occurred"),
        (APIError, "API error occurred"),
        (TimeoutError, "Operation timed out")
    ]
    
    for exc_class, message in exceptions_to_test:
        try:
            context = create_error_context("test_operation", test_param="test_value")
            exc = exc_class(message, f"TEST_{exc_class.__name__.upper()}", context)
            raise exc
        except Exception as e:
            log_exception(logger, e, context, "test_custom_exceptions")
            print(f"✓ {exc_class.__name__}: {e}")
    
    print("✓ Custom exceptions test completed")


def test_operation_logger():
    """Test operation logger context manager"""
    print("Testing operation logger...")
    
    # Test successful operation
    with create_operation_logger("test_successful_operation", "TEST_SYMBOL") as op_logger:
        op_logger.add_context(test_param="test_value")
        op_logger.log_progress("Operation in progress")
        # Simulate some work
        import time
        time.sleep(0.1)
        op_logger.log_progress("Operation completed successfully")
    
    # Test operation with error
    try:
        with create_operation_logger("test_failed_operation", "TEST_SYMBOL") as op_logger:
            op_logger.add_context(test_param="test_value")
            op_logger.log_progress("Operation starting")
            # Simulate error
            raise ValueError("Simulated error for testing")
    except ValueError:
        pass  # Expected
    
    print("✓ Operation logger test completed")


def test_fallback_mechanisms():
    """Test fallback mechanisms"""
    print("Testing fallback mechanisms...")
    
    logger = get_logger(__name__)
    
    # Test function that always fails
    def failing_function(param1, param2=None):
        raise ValueError(f"Function failed with param1={param1}, param2={param2}")
    
    # Test function that succeeds
    def succeeding_function(param1, param2=None):
        return f"Success with param1={param1}, param2={param2}"
    
    # Register a custom fallback
    def custom_fallback(primary_error, operation_context, fallback_data, args, kwargs):
        logger.info("Custom fallback executed")
        return f"Fallback result for args={args}, kwargs={kwargs}"
    
    register_fallback('test_operation', custom_fallback)
    
    # Test successful operation (no fallback needed)
    try:
        result = execute_with_fallback(
            'test_operation',
            succeeding_function,
            "test_param",
            param2="test_param2"
        )
        print(f"✓ Successful operation: {result}")
    except Exception as e:
        print(f"✗ Unexpected error in successful operation: {e}")
    
    # Test failed operation with fallback
    try:
        result = execute_with_fallback(
            'test_operation',
            failing_function,
            "test_param",
            param2="test_param2",
            fallback_data={'test_data': 'test_value'}
        )
        print(f"✓ Failed operation with fallback: {result}")
    except Exception as e:
        print(f"✗ Fallback also failed: {e}")
    
    # Test operation without fallback
    try:
        result = execute_with_fallback(
            'no_fallback_operation',
            failing_function,
            "test_param"
        )
        print(f"✗ Should have failed: {result}")
    except ValueError as e:
        print(f"✓ Operation failed as expected (no fallback): {e}")
    
    # Get fallback statistics
    stats = get_fallback_stats()
    print(f"✓ Fallback statistics: {stats}")
    
    print("✓ Fallback mechanisms test completed")


def test_error_context():
    """Test error context creation"""
    print("Testing error context creation...")
    
    # Test basic context
    context1 = create_error_context("test_operation")
    print(f"✓ Basic context: {context1}")
    
    # Test context with symbol
    context2 = create_error_context("test_operation", "TEST_SYMBOL")
    print(f"✓ Context with symbol: {context2}")
    
    # Test context with additional parameters
    context3 = create_error_context(
        "test_operation", 
        "TEST_SYMBOL",
        param1="value1",
        param2=123,
        param3=True
    )
    print(f"✓ Context with additional params: {context3}")
    
    print("✓ Error context test completed")


def test_exception_serialization():
    """Test exception serialization to dictionary"""
    print("Testing exception serialization...")
    
    context = create_error_context("test_operation", "TEST_SYMBOL", test_param="test_value")
    
    # Create and serialize different exceptions
    exceptions_to_test = [
        PSXAdvisorError("Test error", "TEST_ERROR", context),
        DataScrapingError("Data scraping failed", "SCRAPING_ERROR", context),
        NetworkError("Network failed", "NETWORK_ERROR", context)
    ]
    
    for exc in exceptions_to_test:
        exc_dict = exc.to_dict()
        print(f"✓ {exc.__class__.__name__} serialized: {exc_dict}")
    
    print("✓ Exception serialization test completed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("PSX AI Advisor Error Handling and Logging Tests")
    print("=" * 60)
    
    tests = [
        test_logging_setup,
        test_custom_exceptions,
        test_operation_logger,
        test_fallback_mechanisms,
        test_error_context,
        test_exception_serialization
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n{'-' * 40}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            print(traceback.format_exc())
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    # Check if log files were created
    log_files = ['logs/psx_advisor.log', 'logs/errors.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"✓ Log file created: {log_file} ({size} bytes)")
        else:
            print(f"✗ Log file not found: {log_file}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)