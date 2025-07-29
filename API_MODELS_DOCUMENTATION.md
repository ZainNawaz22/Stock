# PSX AI Advisor API Models and Monitoring

## Overview

This document describes the Pydantic models and monitoring features implemented for the PSX AI Advisor API. These enhancements provide robust data validation, serialization, error handling, and performance monitoring capabilities.

## 🚀 New Features

### 1. Pydantic Data Models
- **Type Safety**: All API responses are now validated using Pydantic models
- **Automatic Documentation**: OpenAPI schema generation with detailed field descriptions
- **Data Validation**: Input/output validation with meaningful error messages
- **JSON Serialization**: Proper handling of pandas DataFrames and numpy types

### 2. API Monitoring & Performance Tracking
- **Request Monitoring**: Automatic tracking of all API requests and responses
- **Performance Metrics**: Response time tracking, error rates, and throughput
- **Cache Monitoring**: Cache hit/miss rates and performance optimization
- **Health Checks**: Comprehensive system health monitoring

### 3. Enhanced Error Handling
- **Structured Errors**: Consistent error response format across all endpoints
- **HTTP Status Codes**: Appropriate status codes for different error types
- **Error Context**: Detailed error logging with contextual information
- **Graceful Degradation**: Fallback responses when components fail

## 📋 API Models

### Core Response Models

#### `StocksListResponse`
```python
{
    "stocks": [StockSummary],
    "total_count": int,
    "returned_count": int,
    "has_more": bool,
    "processing_time_optimized": bool
}
```

#### `StockDataResponse`
```python
{
    "symbol": str,
    "data_points": int,
    "date_range": DateRange,
    "current_values": CurrentValues,
    "technical_indicators": TechnicalIndicators,
    "data": [StockDataPoint],
    "indicators_included": bool,
    "data_limited": bool
}
```

#### `PredictionsResponse`
```python
{
    "predictions": [PredictionResult],
    "total_count": int,
    "successful_count": int,
    "failed_count": int,
    "processing_time_optimized": bool,
    "parallel_processing": bool,
    "last_updated": str
}
```

#### `SystemStatus`
```python
{
    "status": str,
    "health": HealthInfo,
    "data": DataInfo,
    "models": ModelsInfo,
    "cache": CacheInfo,
    "uptime": UptimeInfo,
    "version": str,
    "timestamp": str
}
```

### Data Models

#### `StockSummary`
```python
{
    "symbol": str,
    "company_name": str,
    "record_count": int | None,
    "current_price": float | None,
    "last_updated": str | None,
    "date_range": dict | None,
    "has_data": bool,
    "error": str | None
}
```

#### `PredictionResult`
```python
{
    "symbol": str,
    "prediction": "UP" | "DOWN" | None,
    "confidence": float,  # 0.0 to 1.0
    "prediction_probabilities": PredictionProbabilities,
    "current_price": float | None,
    "prediction_date": str | None,
    "data_date": str | None,
    "model_accuracy": float | None,
    "model_type": str | None,
    "feature_count": int | None,
    "error": str | None,
    "message": str | None
}
```

#### `CurrentValues`
```python
{
    "date": str | None,
    "open": float | None,
    "high": float | None,
    "low": float | None,
    "close": float | None,
    "volume": int | None,
    "change": float | None,
    "previous_close": float | None
}
```

#### `TechnicalIndicators`
```python
{
    "RSI_14": float | None,
    "MACD": float | None,
    "MACD_Signal": float | None,
    "SMA_20": float | None,
    "SMA_50": float | None,
    "BB_Upper": float | None,
    "BB_Lower": float | None,
    "price_above_sma_20": bool | None,
    "price_above_sma_50": bool | None,
    "rsi_bullish": bool | None,
    "macd_bullish": bool | None,
    "status": str | None,
    "summary": str | None
}
```

## 🔧 Monitoring Features

### API Monitoring Middleware

The `APIMonitoringMiddleware` automatically tracks:

- **Request Count**: Total number of API requests
- **Response Times**: Min, max, average response times per endpoint
- **Error Rates**: HTTP error counts and rates
- **Cache Performance**: Hit/miss ratios
- **Endpoint Metrics**: Per-endpoint performance statistics

### Performance Metrics

Access performance data via `/api/performance/metrics`:

```python
{
    "performance_metrics": {
        "uptime_seconds": float,
        "total_requests": int,
        "error_count": int,
        "error_rate": float,
        "average_response_time": float,
        "cache_hits": int,
        "cache_misses": int,
        "cache_hit_rate": float,
        "requests_per_minute": float,
        "endpoint_metrics": {
            "GET /api/stocks": {
                "count": int,
                "avg_time": float,
                "min_time": float,
                "max_time": float,
                "error_count": int
            }
        }
    }
}
```

### Logging Features

#### Operation Logging
- `log_data_operation()`: Track data loading/saving operations
- `log_ml_operation()`: Track ML training/prediction operations
- `log_cache_operation()`: Track cache hits/misses
- `log_error_with_context()`: Enhanced error logging with context

#### Performance Timing
```python
with PerformanceTimer("operation_name") as timer:
    # Your operation here
    pass
# Automatically logs execution time
```

## 🛠️ Data Serialization

### DataFrame Serialization

The `serialize_dataframe_simple()` function handles:

- **NaN Values**: Converts to `null` in JSON
- **Pandas Timestamps**: Converts to ISO format strings
- **NumPy Types**: Converts to native Python types
- **Type Safety**: Ensures JSON serialization compatibility

### Usage Example

```python
from psx_ai_advisor.api_models import serialize_dataframe_simple

# Convert DataFrame to JSON-safe format
df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=3)})
serialized = serialize_dataframe_simple(df)
# Result: [{"Date": "2025-01-01T00:00:00"}, ...]
```

## 🚨 Error Handling

### Structured Error Responses

All errors now return consistent format:

```python
{
    "detail": str,
    "error_code": str | None,
    "timestamp": str | None
}
```

### Error Types

- **404**: Resource not found (invalid stock symbol)
- **400**: Bad request (insufficient data, invalid parameters)
- **500**: Internal server error (processing failures)
- **503**: Service unavailable (network issues)

### Error Context Logging

Errors are logged with additional context:

```python
log_error_with_context(
    error=exception,
    context={"operation": "get_stock_data", "symbol": "PTC"}
)
```

## 📊 Cache Management

### Cache Operations

- **GET /api/cache/clear**: Clear all cached data
- **Automatic Expiry**: Different expiry times for different data types
- **Cache Monitoring**: Track hit/miss rates and performance

### Cache Types

- **Predictions**: 30-minute expiry
- **Stock Summaries**: 15-minute expiry
- **Technical Indicators**: 60-minute expiry

## 🔍 Health Monitoring

### Health Check Endpoint

**GET /health**: Simple health check for load balancers

```python
{
    "status": "healthy",
    "timestamp": "2025-01-29T10:30:00",
    "version": "1.0.0"
}
```

### System Status

**GET /api/system/status**: Comprehensive system health

- **Health Score**: 0-100 based on system condition
- **Data Freshness**: Check for recent data updates
- **Model Availability**: Track trained ML models
- **Performance Metrics**: System performance indicators

## 🎯 Validation Features

### Input Validation

- **Query Parameters**: Automatic validation of API parameters
- **Path Parameters**: Stock symbol validation
- **Request Bodies**: JSON schema validation

### Output Validation

- **Response Models**: All responses validated before sending
- **Type Conversion**: Automatic type conversion and validation
- **Error Prevention**: Catch serialization errors before response

## 📈 Performance Optimizations

### Response Time Improvements

- **Parallel Processing**: Multi-threaded operations
- **Intelligent Caching**: Multi-layer caching system
- **Request Limiting**: Prevent timeout scenarios
- **Background Tasks**: Non-blocking operations

### Memory Optimization

- **Efficient Serialization**: Optimized DataFrame conversion
- **Cache Management**: Automatic cleanup of expired data
- **Resource Monitoring**: Track memory and CPU usage

## 🔧 Development Tools

### Testing

Run the test suite to verify models:

```bash
python test_api_models.py
```

### API Documentation

Enhanced OpenAPI documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Debugging

Enable detailed logging for debugging:

```python
import logging
logging.getLogger('psx_advisor').setLevel(logging.DEBUG)
```

## 🚀 Migration Guide

### For Existing Code

1. **Import New Models**: Update imports to use Pydantic models
2. **Response Handling**: Update response parsing for new structure
3. **Error Handling**: Update error handling for new error format
4. **Type Hints**: Add type hints using the new models

### Example Migration

**Before:**
```python
response = requests.get("/api/stocks")
data = response.json()
stocks = data["stocks"]
```

**After:**
```python
from psx_ai_advisor.api_models import StocksListResponse

response = requests.get("/api/stocks")
stocks_response = StocksListResponse(**response.json())
stocks = stocks_response.stocks
```

## 📝 Best Practices

### API Usage

1. **Use Type Hints**: Leverage Pydantic models for type safety
2. **Handle Errors**: Check for error fields in responses
3. **Cache Awareness**: Consider cache expiry times
4. **Performance**: Use appropriate limits and filters

### Development

1. **Model Validation**: Always validate data with Pydantic models
2. **Error Logging**: Use structured logging with context
3. **Performance Monitoring**: Monitor response times and cache rates
4. **Testing**: Test with various data scenarios

## 🔮 Future Enhancements

### Planned Features

- **Rate Limiting**: API rate limiting and throttling
- **Authentication**: API key authentication
- **WebSocket Support**: Real-time data streaming
- **Advanced Metrics**: More detailed performance analytics

### Extensibility

The model system is designed for easy extension:

- **New Models**: Add new Pydantic models as needed
- **Custom Validation**: Add custom validators for specific requirements
- **Monitoring Extensions**: Add new monitoring metrics
- **Serialization**: Support for additional data formats

---

This implementation provides a robust foundation for the PSX AI Advisor API with comprehensive validation, monitoring, and error handling capabilities.