"""
Pydantic models for API request/response validation and serialization.

This module defines the data models used for API validation, serialization,
and documentation in the PSX AI Advisor FastAPI application.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np


class StockSummary(BaseModel):
    """Basic stock information summary."""
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field(..., description="Company name")
    current_price: float = Field(..., description="Current stock price")
    change: float = Field(..., description="Price change from previous close")
    change_percent: float = Field(..., description="Percentage change from previous close")
    volume: int = Field(..., description="Trading volume")
    last_updated: str = Field(..., description="Last update date")
    has_data: bool = Field(..., description="Whether stock has valid data")
    prediction: Optional[Dict[str, Any]] = Field(None, description="ML prediction result")
    error: Optional[str] = Field(None, description="Error message if any")


class StocksListResponse(BaseModel):
    """Response model for stocks list endpoint."""
    stocks: List[StockSummary] = Field(..., description="List of stock summaries")
    total_count: int = Field(..., description="Total number of available stocks")
    returned_count: int = Field(..., description="Number of stocks returned in this response")
    has_more: bool = Field(..., description="Whether more stocks are available")
    processing_time_optimized: Optional[bool] = Field(None, description="Whether response was optimized for performance")


class CurrentValues(BaseModel):
    """Current stock price values."""
    date: Optional[str] = Field(None, description="Date of the values")
    open: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Highest price")
    low: Optional[float] = Field(None, description="Lowest price")
    close: Optional[float] = Field(None, description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    change: Optional[float] = Field(None, description="Price change")
    previous_close: Optional[float] = Field(None, description="Previous closing price")


class TechnicalIndicators(BaseModel):
    """Technical indicators summary."""
    RSI_14: Optional[float] = Field(None, description="14-period Relative Strength Index")
    MACD: Optional[float] = Field(None, description="MACD value")
    MACD_Signal: Optional[float] = Field(None, description="MACD Signal line")
    SMA_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    SMA_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    BB_Upper: Optional[float] = Field(None, description="Bollinger Band Upper")
    BB_Lower: Optional[float] = Field(None, description="Bollinger Band Lower")
    price_above_sma_20: Optional[bool] = Field(None, description="Price above 20-day SMA")
    price_above_sma_50: Optional[bool] = Field(None, description="Price above 50-day SMA")
    rsi_bullish: Optional[bool] = Field(None, description="RSI indicates bullish trend")
    macd_bullish: Optional[bool] = Field(None, description="MACD indicates bullish trend")
    status: Optional[str] = Field(None, description="Indicator calculation status")
    summary: Optional[str] = Field(None, description="Indicator summary")


class DateRange(BaseModel):
    """Date range information."""
    start: Optional[str] = Field(None, description="Start date")
    end: Optional[str] = Field(None, description="End date")


class StockDataPoint(BaseModel):
    """Individual stock data point."""
    Date: Optional[str] = Field(None, description="Date of the data point")
    Symbol: Optional[str] = Field(None, description="Stock symbol")
    Company_Name: Optional[str] = Field(None, description="Company name")
    Open: Optional[float] = Field(None, description="Opening price")
    High: Optional[float] = Field(None, description="Highest price")
    Low: Optional[float] = Field(None, description="Lowest price")
    Close: Optional[float] = Field(None, description="Closing price")
    Volume: Optional[int] = Field(None, description="Trading volume")
    Previous_Close: Optional[float] = Field(None, description="Previous closing price")
    Change: Optional[float] = Field(None, description="Price change")
    
    # Technical indicators
    RSI_14: Optional[float] = Field(None, description="14-period RSI")
    MACD: Optional[float] = Field(None, description="MACD value")
    MACD_Signal: Optional[float] = Field(None, description="MACD Signal")
    MACD_Histogram: Optional[float] = Field(None, description="MACD Histogram")
    SMA_20: Optional[float] = Field(None, description="20-day SMA")
    SMA_50: Optional[float] = Field(None, description="50-day SMA")
    BB_Upper: Optional[float] = Field(None, description="Bollinger Band Upper")
    BB_Middle: Optional[float] = Field(None, description="Bollinger Band Middle")
    BB_Lower: Optional[float] = Field(None, description="Bollinger Band Lower")


class StockDataResponse(BaseModel):
    """Response model for stock data endpoint."""
    symbol: str = Field(..., description="Stock symbol")
    data_points: int = Field(..., description="Number of data points returned")
    date_range: DateRange = Field(..., description="Date range of the data")
    current_values: CurrentValues = Field(..., description="Current stock values")
    technical_indicators: Optional[TechnicalIndicators] = Field(None, description="Technical indicators summary")
    data: List[StockDataPoint] = Field(default_factory=list, description="Historical stock data points")
    indicators_included: bool = Field(..., description="Whether technical indicators are included")
    data_limited: Optional[bool] = Field(None, description="Whether data was limited for performance")


class PredictionProbabilities(BaseModel):
    """Prediction probabilities for different outcomes."""
    DOWN: Optional[float] = Field(None, description="Probability of price going down")
    UP: Optional[float] = Field(None, description="Probability of price going up")


class PredictionResult(BaseModel):
    """Individual prediction result."""
    symbol: str = Field(..., description="Stock symbol")
    prediction: Optional[str] = Field(None, description="Prediction (UP/DOWN)")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    prediction_probabilities: Optional[PredictionProbabilities] = Field(None, description="Prediction probabilities")
    current_price: Optional[float] = Field(None, description="Current stock price")
    prediction_date: Optional[str] = Field(None, description="When prediction was made")
    data_date: Optional[str] = Field(None, description="Date of the data used for prediction")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy score")
    model_type: Optional[str] = Field(None, description="Type of ML model used")
    feature_count: Optional[int] = Field(None, description="Number of features used")
    error: Optional[str] = Field(None, description="Error type if prediction failed")
    message: Optional[str] = Field(None, description="Error message if prediction failed")

    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Confidence must be between 0 and 1')
        return v

    @validator('prediction')
    def validate_prediction(cls, v):
        """Validate prediction is UP or DOWN."""
        if v is not None and v not in ['UP', 'DOWN']:
            raise ValueError('Prediction must be UP or DOWN')
        return v


class PredictionsResponse(BaseModel):
    """Response model for predictions endpoint."""
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    successful_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    processing_time_optimized: Optional[bool] = Field(None, description="Whether processing was optimized")
    parallel_processing: Optional[bool] = Field(None, description="Whether parallel processing was used")
    cache_used: Optional[bool] = Field(None, description="Whether cached data was used")
    last_updated: Optional[str] = Field(None, description="When predictions were last updated")


class HealthInfo(BaseModel):
    """System health information."""
    score: int = Field(..., description="Health score (0-100)")
    status: str = Field(..., description="Health status (excellent/good/fair/poor)")
    issues: List[str] = Field(default_factory=list, description="List of health issues")

    @validator('score')
    def validate_score(cls, v):
        """Validate health score is between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError('Health score must be between 0 and 100')
        return v

    @validator('status')
    def validate_status(cls, v):
        """Validate health status."""
        valid_statuses = ['excellent', 'good', 'fair', 'poor']
        if v not in valid_statuses:
            raise ValueError(f'Health status must be one of: {valid_statuses}')
        return v


class DataInfo(BaseModel):
    """System data information."""
    total_symbols: int = Field(..., description="Total number of stock symbols")
    total_records: int = Field(..., description="Total number of data records")
    storage_size_mb: float = Field(..., description="Storage size in MB")
    latest_data_date: Optional[str] = Field(None, description="Date of latest data")
    oldest_data_date: Optional[str] = Field(None, description="Date of oldest data")
    data_directory: Optional[str] = Field(None, description="Data directory path")
    sample_size: Optional[int] = Field(None, description="Sample size used for status check")


class ModelsInfo(BaseModel):
    """ML models information."""
    available_count: int = Field(..., description="Number of available trained models")
    sample_checked: int = Field(..., description="Number of models checked in sample")
    estimated_total: Optional[int] = Field(None, description="Estimated total models available")


class CacheInfo(BaseModel):
    """Cache information."""
    predictions_cached: int = Field(..., description="Number of cached predictions")
    last_updated: Optional[str] = Field(None, description="When cache was last updated")
    cache_expiry_minutes: int = Field(..., description="Cache expiry time in minutes")


class UptimeInfo(BaseModel):
    """System uptime information."""
    api_started: Optional[str] = Field(None, description="When API was started")
    cache_last_updated: Optional[str] = Field(None, description="When cache was last updated")


class PerformanceInfo(BaseModel):
    """Performance configuration information."""
    optimized: bool = Field(..., description="Whether performance optimizations are enabled")
    parallel_processing: bool = Field(..., description="Whether parallel processing is enabled")
    caching_enabled: bool = Field(..., description="Whether caching is enabled")
    max_symbols_per_request: int = Field(..., description="Maximum symbols per request")
    max_data_points_default: int = Field(..., description="Default maximum data points")


class SystemStatus(BaseModel):
    """Complete system status information."""
    status: str = Field(..., description="Overall system status")
    health: HealthInfo = Field(..., description="System health information")
    data: DataInfo = Field(..., description="Data information")
    models: ModelsInfo = Field(..., description="ML models information")
    cache: CacheInfo = Field(..., description="Cache information")
    performance: PerformanceInfo = Field(..., description="Performance information")
    api_started: Optional[str] = Field(None, description="When API was started")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Status timestamp")

    @validator('status')
    def validate_status(cls, v):
        """Validate system status."""
        valid_statuses = ['operational', 'degraded', 'down']
        if v not in valid_statuses:
            raise ValueError(f'System status must be one of: {valid_statuses}')
        return v


class APIError(BaseModel):
    """Standard API error response."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


class CacheClearResponse(BaseModel):
    """Response for cache clear operation."""
    message: str = Field(..., description="Success message")
    timestamp: str = Field(..., description="Operation timestamp")


class PredictionWarmupResponse(BaseModel):
    """Response for prediction warmup operation."""
    message: str = Field(..., description="Operation message")
    symbols_count: int = Field(..., description="Number of symbols to warm up")
    estimated_time_minutes: int = Field(..., description="Estimated time in minutes")
    status: str = Field(..., description="Operation status")


# Utility functions for data serialization

def serialize_dataframe_to_model(df: pd.DataFrame, model_class: BaseModel) -> List[Dict[str, Any]]:
    """
    Convert pandas DataFrame to list of Pydantic model instances.
    
    Args:
        df (pd.DataFrame): DataFrame to serialize
        model_class (BaseModel): Pydantic model class to use
        
    Returns:
        List[Dict[str, Any]]: List of serialized model instances
    """
    if df.empty:
        return []
    
    # Convert DataFrame to dict records and handle NaN values
    records = df.to_dict('records')
    
    # Clean and validate each record
    validated_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned_record[key] = None
            elif isinstance(value, pd.Timestamp):
                cleaned_record[key] = value.isoformat()
            elif isinstance(value, np.integer):
                cleaned_record[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_record[key] = float(value)
            elif hasattr(value, 'item'):  # Other numpy types
                try:
                    cleaned_record[key] = value.item()
                except (ValueError, TypeError):
                    cleaned_record[key] = str(value)
            else:
                cleaned_record[key] = value
        
        # Validate using Pydantic model
        try:
            validated_record = model_class(**cleaned_record)
            validated_records.append(validated_record.dict())
        except Exception as e:
            # Log validation error but continue with cleaned record
            print(f"Validation error for record: {e}")
            validated_records.append(cleaned_record)
    
    return validated_records


def serialize_dataframe_simple(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Simple DataFrame serialization without model validation.
    
    Args:
        df (pd.DataFrame): DataFrame to serialize
        
    Returns:
        List[Dict[str, Any]]: JSON-serializable list of records
    """
    if df.empty:
        return []
    
    # Convert DataFrame to dict records and handle NaN values
    records = df.to_dict('records')
    
    # Replace NaN values and convert numpy types for JSON serialization
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                record[key] = float(value)
            elif hasattr(value, 'item'):  # Other numpy types
                try:
                    record[key] = value.item()
                except (ValueError, TypeError):
                    record[key] = str(value)
    
    return records


def clean_nan_values(data: Any) -> Any:
    """
    Recursively clean NaN values from any data structure.
    
    Args:
        data: Any data structure that may contain NaN values
        
    Returns:
        Any: Cleaned data structure with NaN values replaced by None
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned[key] = clean_nan_values(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif pd.isna(data):
        return None
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        if np.isnan(data):
            return None
        return float(data)
    elif hasattr(data, 'item'):  # Other numpy types
        try:
            value = data.item()
            if isinstance(value, float) and np.isnan(value):
                return None
            return value
        except (ValueError, TypeError):
            return str(data)
    else:
        return data


def create_error_response(message: str, error_code: str = None) -> APIError:
    """
    Create a standardized error response.
    
    Args:
        message (str): Error message
        error_code (str, optional): Error code
        
    Returns:
        APIError: Standardized error response
    """
    return APIError(
        detail=message,
        error_code=error_code,
        timestamp=datetime.now().isoformat()
    )