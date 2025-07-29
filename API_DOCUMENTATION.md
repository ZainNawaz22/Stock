# PSX AI Advisor API Documentation

## Overview

The PSX AI Advisor API is a REST API that provides access to Pakistan Stock Exchange (PSX) stock data, technical indicators, and machine learning predictions. The API is built with FastAPI and provides comprehensive endpoints for stock analysis.

## 🚀 Performance Optimizations (v1.0.0)

This API has been optimized for production use with significant performance improvements:

### Key Performance Features:
- **Parallel Processing**: Multi-threaded operations using ThreadPoolExecutor for faster response times
- **Intelligent Caching**: Multi-layer caching system with automatic expiration:
  - Predictions cached for 30 minutes
  - Stock summaries cached for 15 minutes  
  - Technical indicators cached for 60 minutes
- **Request Limiting**: Automatic limits to prevent timeouts:
  - Max 20 symbols per request
  - Max 100 data points by default
  - Sample-based system status checks
- **Background Tasks**: Non-blocking operations for expensive computations
- **Memory Optimization**: Efficient data handling and automatic cleanup

### Performance Improvements:
- **Stocks endpoint**: Now processes symbols in parallel with caching
- **Stock data endpoint**: Limited data points with cached technical indicators
- **Predictions endpoint**: Parallel processing with intelligent caching
- **System status**: Sample-based checks instead of full system scan

### Expected Response Times:
- Health check: <2 seconds
- Stocks list: <10 seconds (was >60 seconds)
- Stock data: <20 seconds (was timeout)
- Predictions: <45 seconds (was timeout)
- System status: <15 seconds (was timeout)

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. The API is designed for personal use.

## CORS Configuration

The API is configured with CORS to allow frontend access from any origin. In production, this should be restricted to specific domains.

## Endpoints

### 1. Root Endpoint

**GET /**

Returns basic API information and available endpoints.

**Response:**
```json
{
  "message": "PSX AI Advisor API",
  "version": "1.0.0",
  "description": "REST API for Pakistan Stock Exchange AI-powered stock analysis",
  "endpoints": {
    "stocks": "/api/stocks",
    "stock_data": "/api/stocks/{symbol}/data",
    "predictions": "/api/predictions",
    "system_status": "/api/system/status",
    "documentation": "/docs"
  }
}
```

### 2. Health Check

**GET /health**

Simple health check endpoint for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-29T10:30:00"
}
```

### 3. Get Stocks List

**GET /api/stocks**

Returns a list of available stocks with basic information.

**Query Parameters:**
- `limit` (optional): Maximum number of stocks to return
- `search` (optional): Search term to filter stocks by symbol

**Example Requests:**
```bash
# Get all stocks
curl http://localhost:8000/api/stocks

# Get first 10 stocks
curl http://localhost:8000/api/stocks?limit=10

# Search for stocks containing "PTC"
curl http://localhost:8000/api/stocks?search=PTC
```

**Response:**
```json
{
  "stocks": [
    {
      "symbol": "PTC",
      "company_name": "PTC",
      "record_count": 1250,
      "current_price": 285.50,
      "last_updated": "2025-01-29",
      "date_range": {
        "start": "2020-01-01",
        "end": "2025-01-29"
      },
      "has_data": true
    }
  ],
  "total_count": 1,
  "returned_count": 1,
  "has_more": false
}
```

### 4. Get Stock Data

**GET /api/stocks/{symbol}/data**

Returns OHLCV data with technical indicators for a specific stock.

**Path Parameters:**
- `symbol`: Stock symbol (e.g., "PTC", "ENGRO")

**Query Parameters:**
- `days` (optional): Number of recent days to return
- `include_indicators` (optional, default: true): Include technical indicators

**Example Requests:**
```bash
# Get all data for PTC with indicators
curl http://localhost:8000/api/stocks/PTC/data

# Get last 30 days of data
curl http://localhost:8000/api/stocks/PTC/data?days=30

# Get data without technical indicators
curl http://localhost:8000/api/stocks/PTC/data?include_indicators=false
```

**Response:**
```json
{
  "symbol": "PTC",
  "data_points": 100,
  "date_range": {
    "start": "2024-09-01T00:00:00",
    "end": "2025-01-29T00:00:00"
  },
  "current_values": {
    "date": "2025-01-29T00:00:00",
    "open": 284.00,
    "high": 287.50,
    "low": 283.00,
    "close": 285.50,
    "volume": 125000,
    "change": 1.50,
    "previous_close": 284.00
  },
  "technical_indicators": {
    "RSI_14": 65.2,
    "MACD": 2.1,
    "MACD_Signal": 1.8,
    "SMA_20": 280.5,
    "SMA_50": 275.8,
    "BB_Upper": 290.0,
    "BB_Lower": 270.0,
    "price_above_sma_20": true,
    "price_above_sma_50": true,
    "rsi_bullish": true,
    "macd_bullish": true
  },
  "data": [
    {
      "Date": "2025-01-29T00:00:00",
      "Symbol": "PTC",
      "Company_Name": "PTC",
      "Open": 284.00,
      "High": 287.50,
      "Low": 283.00,
      "Close": 285.50,
      "Volume": 125000,
      "Previous_Close": 284.00,
      "Change": 1.50,
      "RSI_14": 65.2,
      "MACD": 2.1,
      "MACD_Signal": 1.8,
      "MACD_Histogram": 0.3,
      "SMA_20": 280.5,
      "SMA_50": 275.8,
      "BB_Upper": 290.0,
      "BB_Middle": 280.0,
      "BB_Lower": 270.0
    }
  ]
}
```

### 5. Get Predictions

**GET /api/predictions**

Returns ML predictions for stock price movements.

**Query Parameters:**
- `symbols` (optional): Comma-separated list of symbols to get predictions for
- `force_refresh` (optional, default: false): Force refresh of cached predictions

**Example Requests:**
```bash
# Get predictions for all stocks
curl http://localhost:8000/api/predictions

# Get predictions for specific stocks
curl http://localhost:8000/api/predictions?symbols=PTC,ENGRO,HBL

# Force refresh predictions
curl http://localhost:8000/api/predictions?force_refresh=true
```

**Response:**
```json
{
  "predictions": [
    {
      "symbol": "PTC",
      "prediction": "UP",
      "confidence": 0.78,
      "prediction_probabilities": {
        "DOWN": 0.22,
        "UP": 0.78
      },
      "current_price": 285.50,
      "prediction_date": "2025-01-29T10:30:00",
      "data_date": "2025-01-29T00:00:00",
      "model_accuracy": 0.65,
      "model_type": "RandomForest",
      "feature_count": 16
    }
  ],
  "total_count": 1,
  "successful_count": 1,
  "failed_count": 0,
  "cache_used": false,
  "last_updated": "2025-01-29T10:30:00"
}
```

### 6. Get System Status

**GET /api/system/status**

Returns comprehensive system health and status information.

**Example Request:**
```bash
curl http://localhost:8000/api/system/status
```

**Response:**
```json
{
  "status": "operational",
  "health": {
    "score": 95,
    "status": "excellent",
    "issues": []
  },
  "data": {
    "total_symbols": 50,
    "total_records": 125000,
    "storage_size_mb": 45.2,
    "latest_data_date": "2025-01-29T00:00:00",
    "oldest_data_date": "2020-01-01T00:00:00",
    "data_directory": "data"
  },
  "models": {
    "available_count": 35,
    "sample_checked": 10
  },
  "cache": {
    "predictions_cached": 25,
    "last_updated": "2025-01-29T10:30:00",
    "cache_expiry_minutes": 30
  },
  "uptime": {
    "api_started": "2025-01-29T08:00:00",
    "cache_last_updated": "2025-01-29T10:30:00"
  },
  "version": "1.0.0",
  "timestamp": "2025-01-29T10:35:00"
}
```

## Technical Indicators

The API provides the following technical indicators when `include_indicators=true`:

### Core Indicators
- **RSI_14**: 14-period Relative Strength Index (0-100)
- **MACD**: Moving Average Convergence Divergence
- **MACD_Signal**: MACD Signal Line
- **MACD_Histogram**: MACD Histogram
- **ROC_12**: 12-period Price Rate of Change

### Moving Averages & Bollinger Bands
- **SMA_20**: 20-day Simple Moving Average
- **SMA_50**: 50-day Simple Moving Average
- **BB_Upper**: Bollinger Band Upper Line
- **BB_Middle**: Bollinger Band Middle Line (20-day SMA)
- **BB_Lower**: Bollinger Band Lower Line

### Volume Indicators
- **Volume_MA_20**: 20-day Volume Moving Average
- **OBV**: On-Balance Volume

### Derived Features
- **Return_1d**: 1-day return percentage
- **Return_2d**: 2-day return percentage
- **Return_5d**: 5-day return percentage
- **Volatility_20d**: 20-day rolling volatility

### Indicator Insights
- **price_above_sma_20**: Boolean - Price above 20-day SMA
- **price_above_sma_50**: Boolean - Price above 50-day SMA
- **sma_20_above_sma_50**: Boolean - Short-term bullish signal
- **rsi_overbought**: Boolean - RSI > 70
- **rsi_oversold**: Boolean - RSI < 30
- **rsi_bullish**: Boolean - RSI > 50
- **macd_bullish**: Boolean - MACD above signal line
- **volume_above_average**: Boolean - Volume above 20-day average

### 6. Cache Management

**POST /api/cache/clear**

Clear all cached data to force fresh calculations.

**Response:**
```json
{
  "message": "All caches cleared successfully",
  "timestamp": "2025-07-30T12:00:00"
}
```

### 7. Prediction Warmup

**POST /api/predictions/warmup**

Start background task to warm up prediction cache for faster subsequent requests.

**Parameters:**
- `symbols` (optional): Comma-separated list of symbols to warm up

**Response:**
```json
{
  "message": "Prediction warmup started",
  "symbols_count": 10,
  "estimated_time_minutes": 5,
  "status": "started"
}
```

## Error Handling

The API uses standard HTTP status codes with enhanced error handling:

- **200**: Success
- **400**: Bad Request (e.g., insufficient data)
- **404**: Not Found (e.g., stock symbol not found)
- **500**: Internal Server Error
- **503**: Service Unavailable (e.g., network issues)

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Caching

- Predictions are cached for 30 minutes to improve performance
- Use `force_refresh=true` to bypass cache
- Cache status is included in prediction responses

## Rate Limiting

Currently, no rate limiting is implemented. This should be added for production use.

## Data Requirements

- Stock data must be downloaded using the main workflow (`python main.py`)
- ML models are trained automatically when predictions are requested
- Minimum 200 data points required for model training

## Starting the Server

### Development Mode
```bash
python start_api_server.py
```

### Production Mode
```bash
python api_server.py
```

### Using Uvicorn Directly
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

Run the test suite:
```bash
python test_api_server.py
```

Note: The server must be running for tests to work.

## Frontend Integration

The API is designed to work with modern frontend frameworks:

### JavaScript/Fetch Example
```javascript
// Get all stocks
const response = await fetch('http://localhost:8000/api/stocks');
const data = await response.json();

// Get stock data with indicators
const stockData = await fetch('http://localhost:8000/api/stocks/PTC/data?days=30');
const ptcData = await stockData.json();

// Get predictions
const predictions = await fetch('http://localhost:8000/api/predictions');
const predictionData = await predictions.json();
```

### React Hook Example
```javascript
import { useState, useEffect } from 'react';

function useStockData(symbol) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch(`http://localhost:8000/api/stocks/${symbol}/data`)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, [symbol]);
  
  return { data, loading };
}
```

## Security Considerations

- **CORS**: Currently allows all origins - restrict in production
- **Authentication**: No authentication implemented - add for production
- **Rate Limiting**: Not implemented - add for production
- **Input Validation**: Basic validation implemented
- **Error Handling**: Comprehensive error handling with appropriate HTTP codes

## Performance

- **Caching**: Predictions cached for 30 minutes
- **Async**: All endpoints are async for better performance
- **Pagination**: Stocks endpoint supports limit parameter
- **Streaming**: Large responses use efficient serialization

## Monitoring

- Health check endpoint at `/health`
- Comprehensive system status at `/api/system/status`
- Detailed logging throughout the application
- Error tracking and reporting