#!/usr/bin/env python3
"""
Test script for API models validation and serialization.

This script tests the Pydantic models and serialization utilities
to ensure they work correctly with the PSX AI Advisor API.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from psx_ai_advisor.api_models import (
    StockSummary, StocksListResponse, StockDataResponse, PredictionResult,
    PredictionsResponse, SystemStatus, serialize_dataframe_simple,
    create_error_response
)


def test_stock_summary():
    """Test StockSummary model."""
    print("Testing StockSummary model...")
    
    # Valid stock summary
    stock = StockSummary(
        symbol="PTC",
        company_name="Pakistan Telecommunication Company",
        record_count=1000,
        current_price=285.50,
        last_updated="2025-01-29",
        date_range={"start": "2020-01-01", "end": "2025-01-29"},
        has_data=True
    )
    
    print(f"✓ StockSummary created: {stock.symbol} - {stock.current_price}")
    
    # Stock with error
    error_stock = StockSummary(
        symbol="INVALID",
        company_name="Invalid Stock",
        has_data=False,
        error="No data available"
    )
    
    print(f"✓ Error stock created: {error_stock.symbol} - {error_stock.error}")


def test_stocks_list_response():
    """Test StocksListResponse model."""
    print("\nTesting StocksListResponse model...")
    
    stocks = [
        StockSummary(symbol="PTC", company_name="PTC", has_data=True),
        StockSummary(symbol="ENGRO", company_name="ENGRO", has_data=True)
    ]
    
    response = StocksListResponse(
        stocks=stocks,
        total_count=100,
        returned_count=2,
        has_more=True,
        processing_time_optimized=True
    )
    
    print(f"✓ StocksListResponse created with {len(response.stocks)} stocks")


def test_prediction_result():
    """Test PredictionResult model."""
    print("\nTesting PredictionResult model...")
    
    # Valid prediction
    prediction = PredictionResult(
        symbol="PTC",
        prediction="UP",
        confidence=0.75,
        current_price=285.50,
        prediction_date=datetime.now().isoformat(),
        model_accuracy=0.68,
        model_type="RandomForest"
    )
    
    print(f"✓ PredictionResult created: {prediction.symbol} - {prediction.prediction} ({prediction.confidence})")
    
    # Test validation
    try:
        invalid_prediction = PredictionResult(
            symbol="TEST",
            prediction="INVALID",  # Should fail validation
            confidence=1.5  # Should fail validation
        )
        print("✗ Validation should have failed")
    except ValueError as e:
        print(f"✓ Validation correctly failed: {e}")


def test_dataframe_serialization():
    """Test DataFrame serialization."""
    print("\nTesting DataFrame serialization...")
    
    # Create test DataFrame with various data types
    df = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=3),
        'Symbol': ['PTC', 'PTC', 'PTC'],
        'Close': [285.50, 286.00, np.nan],  # Include NaN
        'Volume': [100000, 150000, 120000],
        'RSI': [65.2, np.nan, 58.1]  # Include NaN
    })
    
    # Test serialization
    serialized = serialize_dataframe_simple(df)
    
    print(f"✓ DataFrame serialized: {len(serialized)} records")
    print(f"  Sample record: {serialized[0] if serialized else 'None'}")
    
    # Check NaN handling
    if len(serialized) > 2:
        nan_record = serialized[2]
        print(f"  NaN handling: Close={nan_record.get('Close')}, RSI={nan_record.get('RSI')}")


def test_system_status():
    """Test SystemStatus model."""
    print("\nTesting SystemStatus model...")
    
    status = SystemStatus(
        status="operational",
        health={
            "score": 95,
            "status": "excellent",
            "issues": []
        },
        data={
            "total_symbols": 50,
            "total_records": 125000,
            "storage_size_mb": 45.2
        },
        models={
            "available_count": 35,
            "sample_checked": 10
        },
        cache={
            "predictions_cached": 25,
            "cache_expiry_minutes": 30
        },
        uptime={},
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )
    
    print(f"✓ SystemStatus created: {status.status} - Health: {status.health.status}")


def test_error_response():
    """Test error response creation."""
    print("\nTesting error response...")
    
    error = create_error_response("Test error message", "TEST_ERROR")
    
    print(f"✓ Error response created: {error.detail}")
    print(f"  Error code: {error.error_code}")
    print(f"  Timestamp: {error.timestamp}")


def main():
    """Run all tests."""
    print("=== Testing PSX AI Advisor API Models ===\n")
    
    try:
        test_stock_summary()
        test_stocks_list_response()
        test_prediction_result()
        test_dataframe_serialization()
        test_system_status()
        test_error_response()
        
        print("\n=== All tests passed! ===")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()