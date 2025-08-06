"""
FastAPI Backend Server for PSX AI Advisor

This module implements a REST API server that provides endpoints for accessing
stock data, technical indicators, ML predictions, and system status information.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_storage import DataStorage
from psx_ai_advisor.ml_predictor import MLPredictor
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
from psx_ai_advisor.data_acquisition import PSXDataAcquisition
from psx_ai_advisor.data_loader import PSXDataLoader
from psx_ai_advisor.config_loader import get_section, get_value
from psx_ai_advisor.logging_config import get_logger
from psx_ai_advisor.exceptions import (
    PSXAdvisorError, DataStorageError, MLPredictorError, 
    InsufficientDataError, NetworkError
)
from psx_ai_advisor.api_models import (
    StocksListResponse, StockDataResponse, PredictionsResponse, SystemStatus,
    StockSummary, StockDataPoint, PredictionResult, APIError,
    serialize_dataframe_simple, create_error_response, clean_nan_values
)
from psx_ai_advisor.api_monitoring import (
    APIMonitoringMiddleware, log_api_performance, log_data_operation,
    log_ml_operation, log_cache_operation, log_error_with_context,
    PerformanceTimer, get_performance_metrics
)

# Initialize logging
logger = get_logger(__name__)

# Custom JSON encoder that handles NaN values
class NanSafeJSONResponse(JSONResponse):
    """Custom JSONResponse that converts NaN values to null for JSON serialization."""
    def render(self, content: Any) -> bytes:
        # Clean the content before JSON encoding
        cleaned_content = clean_nan_values(content)
        return super().render(cleaned_content)

# Initialize FastAPI app with custom response class
app = FastAPI(
    title="PSX AI Advisor API",
    description="REST API for Pakistan Stock Exchange AI-powered stock analysis and predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=NanSafeJSONResponse
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add API monitoring middleware
app.add_middleware(APIMonitoringMiddleware)

# Initialize components
data_storage = DataStorage()
ml_predictor = MLPredictor()
technical_analyzer = TechnicalAnalyzer()
data_acquisition = PSXDataAcquisition()

# Global variables for caching and performance optimization
_api_start_time = None
_last_system_update = None
_cached_predictions = {}
_cached_stock_summaries = {}
_cached_technical_indicators = {}
_cache_expiry = timedelta(minutes=30)  # Cache predictions for 30 minutes
_cache_lock = threading.Lock()
_background_tasks_running = set()

# Thread pool for CPU-intensive operations
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Performance settings
MAX_SYMBOLS_PER_REQUEST = 200  # Increased to show all stocks
MAX_DATA_POINTS_DEFAULT = 100
SYSTEM_STATUS_SAMPLE_SIZE = 5


def get_cached_data(cache_dict: Dict, key: str, expiry_minutes: int = 30) -> Optional[Any]:
    """Get cached data if it exists and is not expired."""
    with _cache_lock:
        if key in cache_dict:
            data, timestamp = cache_dict[key]
            if datetime.now() - timestamp < timedelta(minutes=expiry_minutes):
                log_cache_operation("get", key, hit=True)
                return data
            else:
                # Remove expired data
                del cache_dict[key]
                log_cache_operation("get", key, hit=False)
        else:
            log_cache_operation("get", key, hit=False)
    return None


def set_cached_data(cache_dict: Dict, key: str, data: Any) -> None:
    """Set cached data with timestamp."""
    with _cache_lock:
        cache_dict[key] = (data, datetime.now())
        log_cache_operation("set", key)


def clear_symbol_caches(symbols: List[str]) -> int:
    """Clear cached data for specific symbols."""
    cleared_count = 0
    with _cache_lock:
        for symbol in symbols:
            # Clear prediction cache
            if symbol in _cached_predictions:
                del _cached_predictions[symbol]
                cleared_count += 1
                log_cache_operation("clear", f"prediction_{symbol}")
            
            # Clear stock summary cache
            if symbol in _cached_stock_summaries:
                del _cached_stock_summaries[symbol]
                cleared_count += 1
                log_cache_operation("clear", f"summary_{symbol}")
            
            # Clear technical indicators cache
            if symbol in _cached_technical_indicators:
                del _cached_technical_indicators[symbol]
                cleared_count += 1
                log_cache_operation("clear", f"indicators_{symbol}")
    
    return cleared_count


def clear_expired_cache() -> None:
    """Clear expired cache entries."""
    with _cache_lock:
        current_time = datetime.now()
        
        # Clear expired predictions
        expired_keys = [
            key for key, (_, timestamp) in _cached_predictions.items()
            if current_time - timestamp > _cache_expiry
        ]
        for key in expired_keys:
            del _cached_predictions[key]
        
        # Clear expired stock summaries (cache for 15 minutes)
        expired_keys = [
            key for key, (_, timestamp) in _cached_stock_summaries.items()
            if current_time - timestamp > timedelta(minutes=15)
        ]
        for key in expired_keys:
            del _cached_stock_summaries[key]
        
        # Clear expired technical indicators (cache for 60 minutes)
        expired_keys = [
            key for key, (_, timestamp) in _cached_technical_indicators.items()
            if current_time - timestamp > timedelta(minutes=60)
        ]
        for key in expired_keys:
            del _cached_technical_indicators[key]


def process_symbol_summary(symbol: str, include_prediction: bool = True) -> Dict[str, Any]:
    """Process a single symbol summary in a thread-safe way."""
    try:
        # Check cache first
        cache_key = f"{symbol}_{'with_pred' if include_prediction else 'no_pred'}"
        cached_summary = get_cached_data(_cached_stock_summaries, cache_key, 15)
        if cached_summary:
            return cached_summary
        
        # Load actual stock data to get company name and current values
        try:
            stock_data = data_storage.load_stock_data(symbol)
            
            if stock_data.empty:
                stock_info = {
                    "symbol": symbol,
                    "name": symbol,
                    "current_price": 0.0,
                    "change": 0.0,
                    "change_percent": 0.0,
                    "volume": 0,
                    "last_updated": datetime.now().isoformat(),
                    "has_data": False,
                    "error": "No data available",
                    "prediction": None
                }
            else:
                # Get the latest record
                latest_record = stock_data.iloc[-1]
                
                # Extract company name (fallback to symbol if not available)
                company_name = latest_record.get('Company_Name', symbol)
                if pd.isna(company_name) or company_name == '':
                    company_name = symbol
                
                # Calculate change and change percent
                current_price_raw = latest_record.get('Close', 0)
                previous_close_raw = latest_record.get('Previous_Close', current_price_raw)
                volume_raw = latest_record.get('Volume', 0)
                
                # Clean NaN values and convert to appropriate types
                current_price = float(current_price_raw) if pd.notna(current_price_raw) else 0.0
                previous_close = float(previous_close_raw) if pd.notna(previous_close_raw) else current_price
                volume = int(volume_raw) if pd.notna(volume_raw) else 0
                
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close != 0 else 0.0
                
                # Ensure no NaN values in the final calculations
                if np.isnan(change):
                    change = 0.0
                if np.isnan(change_percent):
                    change_percent = 0.0
                
                stock_info = {
                    "symbol": symbol,
                    "name": company_name,
                    "current_price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": volume,
                    "last_updated": latest_record.get('Date', datetime.now()).strftime('%Y-%m-%dT%H:%M:%S') if hasattr(latest_record.get('Date'), 'strftime') else str(latest_record.get('Date', datetime.now().isoformat())),
                    "has_data": True,
                    "prediction": None
                }
                
                # Add prediction if requested
                if include_prediction:
                    try:
                        # Check if prediction is cached
                        cached_prediction = get_cached_data(_cached_predictions, symbol, 30)
                        if cached_prediction:
                            stock_info["prediction"] = cached_prediction
                        else:
                            # Generate new prediction
                            log_ml_operation("predict", symbol)
                            prediction = ml_predictor.predict_movement(symbol)
                            stock_info["prediction"] = prediction
                            # Cache the prediction
                            set_cached_data(_cached_predictions, symbol, prediction)
                        
                        logger.debug(f"Added prediction for {symbol}: {stock_info['prediction'].get('prediction', 'N/A')}")
                        
                    except InsufficientDataError as e:
                        logger.warning(f"Insufficient data for prediction on {symbol}: {e}")
                        stock_info["prediction"] = {
                            "symbol": symbol,
                            "error": "insufficient_data",
                            "message": str(e),
                            "prediction": None,
                            "confidence": None
                        }
                    except Exception as e:
                        logger.warning(f"Error generating prediction for {symbol}: {e}")
                        stock_info["prediction"] = {
                            "symbol": symbol,
                            "error": "prediction_failed",
                            "message": str(e),
                            "prediction": None,
                            "confidence": None
                        }
                
        except FileNotFoundError:
            stock_info = {
                "symbol": symbol,
                "name": symbol,
                "current_price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "last_updated": datetime.now().isoformat(),
                "has_data": False,
                "error": "No data file found",
                "prediction": None
            }
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}")
            stock_info = {
                "symbol": symbol,
                "name": symbol,
                "current_price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "last_updated": datetime.now().isoformat(),
                "has_data": False,
                "error": f"Data loading error: {str(e)}",
                "prediction": None
            }
        
        # Cache the result and clean NaN values before returning
        stock_info = clean_nan_values(stock_info)
        set_cached_data(_cached_stock_summaries, cache_key, stock_info)
        return stock_info
        
    except Exception as e:
        logger.warning(f"Error processing summary for {symbol}: {e}")
        error_info = {
            "symbol": symbol,
            "name": symbol,
            "current_price": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0,
            "last_updated": datetime.now().isoformat(),
            "has_data": False,
            "error": f"Processing error: {str(e)}",
            "prediction": None
        }
        return clean_nan_values(error_info)


def process_single_prediction(symbol: str) -> Dict[str, Any]:
    """Process a single prediction in a thread-safe way."""
    try:
        # Check cache first
        cached_prediction = get_cached_data(_cached_predictions, symbol, 30)
        if cached_prediction:
            return clean_nan_values(cached_prediction)
        
        # Generate new prediction
        log_ml_operation("predict", symbol)
        prediction = ml_predictor.predict_movement(symbol)
        
        # Clean NaN values from prediction
        cleaned_prediction = clean_nan_values(prediction)
        
        # Cache the result
        set_cached_data(_cached_predictions, symbol, cleaned_prediction)
        return cleaned_prediction
        
    except InsufficientDataError as e:
        error_prediction = {
            "symbol": symbol,
            "error": "insufficient_data",
            "message": str(e),
            "prediction": None,
            "confidence": None
        }
        return clean_nan_values(error_prediction)
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        error_prediction = {
            "symbol": symbol,
            "error": "prediction_failed",
            "message": str(e),
            "prediction": None,
            "confidence": None
        }
        return clean_nan_values(error_prediction)


# Use the serialize_dataframe_simple function from api_models instead
# This function is now deprecated in favor of the api_models version


def handle_api_error(error: Exception, operation: str) -> HTTPException:
    """
    Convert internal exceptions to appropriate HTTP exceptions.
    
    Args:
        error (Exception): Internal exception
        operation (str): Operation that failed
        
    Returns:
        HTTPException: HTTP exception with appropriate status code
    """
    # Log error with context
    log_error_with_context(error, {"operation": operation})
    
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=f"Resource not found: {str(error)}")
    elif isinstance(error, InsufficientDataError):
        return HTTPException(status_code=400, detail=f"Insufficient data: {str(error)}")
    elif isinstance(error, NetworkError):
        return HTTPException(status_code=503, detail=f"Service unavailable: {str(error)}")
    elif isinstance(error, (DataStorageError, MLPredictorError)):
        return HTTPException(status_code=500, detail=f"Internal server error: {str(error)}")
    else:
        return HTTPException(status_code=500, detail=f"Unexpected error in {operation}: {str(error)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
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


@app.get("/api/stocks", response_model=StocksListResponse)
@log_api_performance
async def get_stocks(
    limit: Optional[int] = Query(None, description="Maximum number of stocks to return"),
    search: Optional[str] = Query(None, description="Search term to filter stocks by symbol"),
    include_predictions: bool = Query(True, description="Include ML predictions for each stock")
) -> StocksListResponse:
    """
    Get list of available stocks with basic information.
    
    Args:
        limit: Maximum number of stocks to return
        search: Search term to filter stocks by symbol
        
    Returns:
        Dict containing list of stocks with basic info
    """
    try:
        logger.info(f"Getting stocks list (limit: {limit}, search: {search}, include_predictions: {include_predictions})")
        
        # Clear expired cache entries
        clear_expired_cache()
        
        # Get available symbols
        symbols = data_storage.get_available_symbols()
        
        if not symbols:
            return {
                "stocks": [],
                "total_count": 0,
                "message": "No stock data available"
            }
        
        # Apply search filter if provided
        if search:
            search_term = search.upper()
            symbols = [s for s in symbols if search_term in s.upper()]
        
        # Apply reasonable limit to prevent timeouts only if a limit is explicitly requested
        if limit is not None and limit > MAX_SYMBOLS_PER_REQUEST:
            limit = MAX_SYMBOLS_PER_REQUEST
            logger.info(f"Applied maximum limit of {MAX_SYMBOLS_PER_REQUEST} symbols for performance")
        elif limit is None:
            # No limit specified, process all available symbols
            logger.info("No limit specified, processing all available symbols")
        
        # Process symbols in parallel using thread pool
        symbols_to_process = symbols[:limit] if limit else symbols
        
        # Adjust timeout based on whether predictions are included and number of symbols
        base_timeout = 30 if include_predictions else 10
        timeout_seconds = base_timeout + (len(symbols_to_process) // 10)  # Add time for more symbols
        max_workers = 3 if include_predictions else 6  # Adjust workers for better performance
        
        # Use ThreadPoolExecutor for parallel processing
        stocks_info = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_symbol_summary, symbol, include_predictions): symbol 
                for symbol in symbols_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol, timeout=timeout_seconds):
                try:
                    stock_info = future.result()
                    if stock_info.get('has_data', False):
                        stocks_info.append(stock_info)
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.warning(f"Error processing {symbol}: {e}")
                    stocks_info.append({
                        "symbol": symbol,
                        "name": symbol,
                        "current_price": 0.0,
                        "change": 0.0,
                        "change_percent": 0.0,
                        "volume": 0,
                        "last_updated": datetime.now().isoformat(),
                        "has_data": False,
                        "error": "Processing timeout or error",
                        "prediction": None
                    })
        
        # Sort by symbol
        stocks_info.sort(key=lambda x: x['symbol'])
        
        total_count = len(symbols)
        returned_count = len(stocks_info)
        
        logger.info(f"Returning {returned_count} stocks (total available: {total_count})")
        
        # Filter out stocks without data for the main response
        valid_stocks = [stock for stock in stocks_info if stock.get('has_data', False)]
        
        return StocksListResponse(
            stocks=[StockSummary(**stock) for stock in valid_stocks],
            total_count=total_count,
            returned_count=len(valid_stocks),
            has_more=total_count > len(valid_stocks),
            processing_time_optimized=True
        )
        
    except Exception as e:
        raise handle_api_error(e, "get_stocks")


@app.get("/api/stocks/{symbol}/data", response_model=StockDataResponse)
@log_api_performance
async def get_stock_data(
    symbol: str = Path(..., description="Stock symbol"),
    days: Optional[int] = Query(None, description="Number of recent days to return"),
    include_indicators: bool = Query(True, description="Include technical indicators")
) -> StockDataResponse:
    """
    Get OHLCV data with technical indicators for a specific stock.
    
    Args:
        symbol: Stock symbol
        days: Number of recent days to return (optional)
        include_indicators: Whether to include technical indicators
        
    Returns:
        Dict containing stock data with technical indicators
    """
    try:
        symbol_upper = symbol.upper()
        logger.info(f"Getting data for {symbol_upper} (days: {days}, indicators: {include_indicators})")
        
        # Apply reasonable limits to prevent timeouts
        if days is None or days > MAX_DATA_POINTS_DEFAULT:
            days = MAX_DATA_POINTS_DEFAULT
            logger.info(f"Applied default limit of {MAX_DATA_POINTS_DEFAULT} data points for performance")
        
        # Check cache for technical indicators if requested
        cache_key = f"{symbol_upper}_{days}_{include_indicators}"
        cached_data = get_cached_data(_cached_technical_indicators, cache_key, 60) if include_indicators else None
        
        if cached_data:
            logger.info(f"Returning cached data for {symbol_upper}")
            return cached_data
        
        # Load stock data
        try:
            stock_data = data_storage.load_stock_data(symbol_upper)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for stock symbol: {symbol}"
            )
        
        if stock_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for stock symbol: {symbol}"
            )
        
        # Sort by date (newest first) and limit data points early
        if 'Date' in stock_data.columns:
            stock_data = stock_data.sort_values('Date', ascending=False)
        
        # Apply limit early to reduce processing time
        if days and days > 0:
            stock_data = stock_data.head(days)
        
        # Add technical indicators if requested (in background thread for large datasets)
        if include_indicators and len(stock_data) > 50:
            # Use thread pool for expensive indicator calculations
            def calculate_indicators():
                return technical_analyzer.add_all_indicators(stock_data)
            
            try:
                # Run in thread pool with timeout
                future = _thread_pool.submit(calculate_indicators)
                stock_data = future.result(timeout=15)  # 15 second timeout
                logger.info(f"Added technical indicators for {symbol_upper}")
            except Exception as e:
                logger.warning(f"Error adding indicators for {symbol_upper}: {e}")
                # Continue without indicators rather than failing
                include_indicators = False
        elif include_indicators:
            # For small datasets, calculate directly
            try:
                stock_data = technical_analyzer.add_all_indicators(stock_data)
                logger.info(f"Added technical indicators for {symbol_upper}")
            except Exception as e:
                logger.warning(f"Error adding indicators for {symbol_upper}: {e}")
                include_indicators = False
        
        # Get latest values for summary
        latest_row = stock_data.iloc[0] if not stock_data.empty else None
        
        # Prepare response
        response_data = {
            "symbol": symbol_upper,
            "data_points": len(stock_data),
            "date_range": {
                "start": stock_data['Date'].min().isoformat() if 'Date' in stock_data.columns else None,
                "end": stock_data['Date'].max().isoformat() if 'Date' in stock_data.columns else None
            },
            "current_values": {},
            "data": [],  # Data serialization temporarily disabled due to numpy compatibility
            "indicators_included": include_indicators,
            "data_limited": days == MAX_DATA_POINTS_DEFAULT
        }
        
        # Add current values summary
        if latest_row is not None:
            response_data["current_values"] = {
                "date": latest_row['Date'].isoformat() if 'Date' in stock_data.columns else None,
                "open": float(latest_row.get('Open', 0)) if pd.notna(latest_row.get('Open')) else None,
                "high": float(latest_row.get('High', 0)) if pd.notna(latest_row.get('High')) else None,
                "low": float(latest_row.get('Low', 0)) if pd.notna(latest_row.get('Low')) else None,
                "close": float(latest_row.get('Close', 0)) if pd.notna(latest_row.get('Close')) else None,
                "volume": int(latest_row.get('Volume', 0)) if pd.notna(latest_row.get('Volume')) else None,
                "change": float(latest_row.get('Change', 0)) if pd.notna(latest_row.get('Change')) else None,
                "previous_close": float(latest_row.get('Previous_Close', 0)) if pd.notna(latest_row.get('Previous_Close')) else None
            }
            
            # Add technical indicator summary if available
            if include_indicators:
                try:
                    # Temporarily disabled due to numpy serialization issues
                    response_data["technical_indicators"] = {"status": "indicators_calculated", "summary": "available"}
                except Exception as e:
                    logger.warning(f"Error getting indicator summary for {symbol_upper}: {e}")
        
        # Cache the result if indicators were calculated
        if include_indicators:
            set_cached_data(_cached_technical_indicators, cache_key, response_data)
        
        # Convert to Pydantic model
        try:
            # Serialize stock data using the new utility
            serialized_data = serialize_dataframe_simple(stock_data) if not stock_data.empty else []
            
            # Create the response using Pydantic model
            stock_response = StockDataResponse(
                symbol=response_data["symbol"],
                data_points=response_data["data_points"],
                date_range=response_data["date_range"],
                current_values=response_data["current_values"],
                technical_indicators=response_data.get("technical_indicators"),
                data=serialized_data,
                indicators_included=response_data["indicators_included"],
                data_limited=response_data.get("data_limited")
            )
            
            logger.info(f"Returning {len(stock_data)} data points for {symbol_upper}")
            return stock_response
            
        except Exception as e:
            logger.error(f"Error creating response model for {symbol_upper}: {e}")
            # Fallback to basic response
            return StockDataResponse(
                symbol=symbol_upper,
                data_points=len(stock_data),
                date_range=response_data["date_range"],
                current_values=response_data["current_values"],
                data=[],
                indicators_included=include_indicators
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, f"get_stock_data for {symbol}")


@app.get("/api/predictions", response_model=PredictionsResponse)
@log_api_performance
async def get_predictions(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols (optional)"),
    force_refresh: bool = Query(False, description="Force refresh of cached predictions"),
    limit: Optional[int] = Query(10, description="Maximum number of predictions to return")
) -> PredictionsResponse:
    """
    Get current ML predictions for all stocks or specified symbols.
    
    Args:
        symbols: Comma-separated list of symbols to get predictions for
        force_refresh: Force refresh of cached predictions
        limit: Maximum number of predictions to return
        
    Returns:
        Dict containing ML predictions for stocks
    """
    global _cached_predictions, _last_system_update
    
    try:
        logger.info(f"Getting predictions (symbols: {symbols}, force_refresh: {force_refresh}, limit: {limit})")
        
        # Clear expired cache
        clear_expired_cache()
        
        # Determine which symbols to process
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        else:
            all_symbols = data_storage.get_available_symbols()
            # Limit to prevent timeouts - prioritize symbols with existing models
            symbol_list = all_symbols[:limit] if limit else all_symbols[:10]
        
        if not symbol_list:
            return {
                "predictions": [],
                "total_count": 0,
                "message": "No symbols available for predictions"
            }
        
        # Apply reasonable limit
        if limit and len(symbol_list) > limit:
            symbol_list = symbol_list[:limit]
            logger.info(f"Limited predictions to {limit} symbols for performance")
        
        # Process predictions in parallel using thread pool
        predictions = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all prediction tasks
            future_to_symbol = {
                executor.submit(process_single_prediction, symbol): symbol 
                for symbol in symbol_list
            }
            
            # Collect results as they complete with timeout
            for future in as_completed(future_to_symbol, timeout=30):  # 30 second timeout
                try:
                    prediction = future.result()
                    predictions.append(prediction)
                    
                    if prediction.get('prediction') is not None:
                        successful_predictions += 1
                    else:
                        failed_predictions += 1
                        
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error(f"Prediction timeout/error for {symbol}: {e}")
                    predictions.append({
                        "symbol": symbol,
                        "error": "prediction_timeout",
                        "message": "Prediction processing timed out",
                        "prediction": None,
                        "confidence": None
                    })
                    failed_predictions += 1
        
        # Update cache timestamp
        _last_system_update = datetime.now()
        
        # Sort predictions by confidence (successful ones first)
        predictions.sort(key=lambda x: (
            x.get('confidence', 0) if x.get('prediction') else -1
        ), reverse=True)
        
        logger.info(f"Generated predictions for {len(symbol_list)} symbols "
                   f"(successful: {successful_predictions}, failed: {failed_predictions})")
        
        return PredictionsResponse(
            predictions=[PredictionResult(**pred) for pred in predictions],
            total_count=len(predictions),
            successful_count=successful_predictions,
            failed_count=failed_predictions,
            processing_time_optimized=True,
            parallel_processing=True,
            last_updated=_last_system_update.isoformat() if _last_system_update else None
        )
        
    except Exception as e:
        raise handle_api_error(e, "get_predictions")


@app.post("/api/predictions/regenerate", response_model=PredictionsResponse)
@log_api_performance
async def regenerate_predictions(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to regenerate (optional - regenerates all if not provided)"),
    retrain_models: bool = Query(False, description="Whether to retrain models before generating predictions"),
    limit: Optional[int] = Query(20, description="Maximum number of predictions to regenerate")
) -> PredictionsResponse:
    """
    Regenerate predictions for specified symbols or all symbols.
    This clears cache and generates fresh predictions.
    """
    try:
        start_time = time.time()
        
        # Determine which symbols to regenerate
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        else:
            all_symbols = data_storage.get_available_symbols()
            # Apply reasonable limit to prevent timeouts
            symbol_list = all_symbols[:limit] if limit else all_symbols[:20]
        
        if not symbol_list:
            return PredictionsResponse(
                predictions=[],
                total_count=0,
                successful_count=0,
                failed_count=0,
                processing_time_optimized=True,
                parallel_processing=False,
                cache_used=False,
                last_updated=datetime.now().isoformat()
            )
        
        # Apply reasonable limit
        if limit and len(symbol_list) > limit:
            symbol_list = symbol_list[:limit]
            logger.info(f"Limited regeneration to {limit} symbols for performance")
        
        logger.info(f"Regenerating predictions for {len(symbol_list)} symbols (retrain: {retrain_models})")
        
        # Clear existing cache for these symbols
        cleared_count = clear_symbol_caches(symbol_list)
        logger.info(f"Cleared cache for {cleared_count} items")
        
        # Define function to process single prediction with optional retraining
        def process_regeneration(symbol: str) -> Dict[str, Any]:
            try:
                # Optionally retrain model
                if retrain_models:
                    try:
                        logger.info(f"Retraining model for {symbol}")
                        training_result = ml_predictor.train_model(symbol)
                        logger.info(f"Model retrained for {symbol} with accuracy: {training_result.get('accuracy', 'N/A')}")
                    except Exception as e:
                        logger.warning(f"Model retraining failed for {symbol}: {e}")
                
                # Generate fresh prediction (cache already cleared)
                log_ml_operation("regenerate", symbol)
                prediction = ml_predictor.predict_movement(symbol)
                
                # Set new cached data
                set_cached_data(_cached_predictions, symbol, prediction)
                return prediction
                
            except InsufficientDataError as e:
                return {
                    "symbol": symbol,
                    "error": "insufficient_data",
                    "message": str(e),
                    "prediction": None,
                    "confidence": None
                }
            except Exception as e:
                logger.error(f"Error regenerating prediction for {symbol}: {e}")
                return {
                    "symbol": symbol,
                    "error": "regeneration_failed",
                    "message": str(e),
                    "prediction": None,
                    "confidence": None
                }
        
        # Process predictions in parallel
        predictions = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all regeneration tasks
            future_to_symbol = {
                executor.submit(process_regeneration, symbol): symbol 
                for symbol in symbol_list
            }
            
            # Collect results as they complete with timeout
            for future in as_completed(future_to_symbol, timeout=45):  # 45 second timeout for regeneration
                try:
                    prediction = future.result()
                    predictions.append(prediction)
                    
                    if prediction.get('prediction') is not None:
                        successful_predictions += 1
                    else:
                        failed_predictions += 1
                        
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error(f"Prediction regeneration timeout/error for {symbol}: {e}")
                    predictions.append({
                        "symbol": symbol,
                        "error": "regeneration_timeout",
                        "message": "Prediction regeneration processing timed out",
                        "prediction": None,
                        "confidence": None
                    })
                    failed_predictions += 1
        
        # Update cache timestamp
        global _last_system_update
        _last_system_update = datetime.now()
        
        # Sort predictions by confidence (successful ones first)
        predictions.sort(key=lambda x: (
            x.get('confidence', 0) if x.get('prediction') else -1
        ), reverse=True)
        
        processing_time = time.time() - start_time
        logger.info(f"Regenerated predictions for {len(symbol_list)} symbols in {processing_time:.2f}s "
                   f"(successful: {successful_predictions}, failed: {failed_predictions})")
        
        return PredictionsResponse(
            predictions=[PredictionResult(**pred) for pred in predictions],
            total_count=len(predictions),
            successful_count=successful_predictions,
            failed_count=failed_predictions,
            processing_time_optimized=True,
            parallel_processing=True,
            cache_used=False,  # Fresh predictions, no cache used
            last_updated=_last_system_update.isoformat()
        )
        
    except Exception as e:
        raise handle_api_error(e, "regenerate_predictions")


@app.post("/api/predictions/{symbol}/regenerate", response_model=PredictionResult)
@log_api_performance
async def regenerate_single_prediction(
    symbol: str = Path(..., description="Stock symbol to regenerate"),
    retrain_model: bool = Query(False, description="Whether to retrain the model before predicting")
) -> PredictionResult:
    """
    Regenerate prediction for a single stock symbol.
    Clears the cache for this symbol and regenerates its prediction.
    Optionally retrains the model first.
    """
    try:
        symbol_upper = symbol.upper()
        # clear caches for this symbol
        clear_symbol_caches([symbol_upper])

        # optional retraining
        if retrain_model:
            try:
                logger.info(f"Retraining model for {symbol_upper} (single regenerate)")
                ml_predictor.train_model(symbol_upper)
            except Exception as e:
                logger.warning(f"Retraining failed for {symbol_upper}: {e}")

        # generate fresh prediction
        log_ml_operation("regenerate_single", symbol_upper)
        prediction = ml_predictor.predict_movement(symbol_upper)

        # cache fresh prediction
        set_cached_data(_cached_predictions, symbol_upper, prediction)

        # update last update time
        global _last_system_update
        _last_system_update = datetime.now()

        return PredictionResult(**prediction)
    except Exception as e:
        raise handle_api_error(e, f"regenerate_single_prediction for {symbol}")


@app.get("/api/system/status", response_model=SystemStatus)
@log_api_performance
async def get_system_status() -> SystemStatus:
    """
    Get system health and last update information.
    
    Returns:
        Dict containing system status information
    """
    try:
        logger.info("Getting system status")
        
        # Use cached system status if available (cache for 5 minutes)
        cached_status = get_cached_data(_cached_stock_summaries, "system_status", 5)
        if cached_status:
            logger.info("Returning cached system status")
            return cached_status
        
        # Simplified system status - avoid expensive operations that can hang
        try:
            # Quick check - just count files in data directory
            import os
            data_dir = "data"
            if os.path.exists(data_dir):
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('_historical_data.csv')]
                total_symbols = len(csv_files)
            else:
                total_symbols = 0
        except Exception as e:
            logger.warning(f"Error counting data files: {e}")
            total_symbols = 0
        
        # Basic health assessment
        health_score = 100
        health_issues = []
        
        if total_symbols == 0:
            health_score = 50
            health_issues.append("No stock data files found")
        elif total_symbols < 10:
            health_score = 70
            health_issues.append("Limited stock data available")
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 70:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        # Prepare simplified status info
        status_info = {
            "status": "operational",
            "health": {
                "score": health_score,
                "status": health_status,
                "issues": health_issues
            },
            "data": {
                "total_symbols": total_symbols,
                "total_records": total_symbols * 100,  # Rough estimate
                "storage_size_mb": 0.0,  # Skip expensive calculation
                "latest_data_date": None,  # Skip expensive date check
                "oldest_data_date": None,  # Skip expensive date check
                "data_directory": "data",
                "sample_size": 0  # No sampling needed
            },
            "models": {
                "available_count": 0,  # Skip expensive model check
                "sample_checked": 0,
                "estimated_total": 0
            },
            "cache": {
                "predictions_cached": len(_cached_predictions),
                "summaries_cached": len(_cached_stock_summaries),
                "indicators_cached": len(_cached_technical_indicators),
                "last_updated": _last_system_update.isoformat() if _last_system_update else None,
                "cache_expiry_minutes": int(_cache_expiry.total_seconds() / 60)
            },
            "performance": {
                "optimized": True,
                "parallel_processing": True,
                "caching_enabled": True,
                "max_symbols_per_request": MAX_SYMBOLS_PER_REQUEST,
                "max_data_points_default": MAX_DATA_POINTS_DEFAULT
            },
            "api_started": _api_start_time.isoformat() if _api_start_time else None,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the status for 5 minutes
        set_cached_data(_cached_stock_summaries, "system_status", status_info)
        
        # Convert to Pydantic model
        try:
            system_status = SystemStatus(
                status=status_info["status"],
                health=status_info["health"],
                data=status_info["data"],
                models=status_info["models"],
                cache=status_info["cache"],
                performance=status_info["performance"],
                api_started=status_info.get("api_started"),
                version=status_info["version"],
                timestamp=status_info["timestamp"]
            )
            
            logger.info(f"System status: {health_status} (score: {health_score})")
            return system_status
            
        except Exception as e:
            logger.error(f"Error creating system status model: {e}")
            # Fallback to basic status
            return SystemStatus(
                status="degraded",
                health={"score": 0, "status": "error", "issues": [f"Status model error: {str(e)}"]},
                data={"total_symbols": 0, "total_records": 0, "storage_size_mb": 0.0},
                models={"available_count": 0, "sample_checked": 0},
                cache={"predictions_cached": 0, "cache_expiry_minutes": 30},
                performance={"optimized": False, "parallel_processing": False, "caching_enabled": False, "max_symbols_per_request": 20, "max_data_points_default": 100},
                api_started=None,
                version="1.0.0",
                timestamp=datetime.now().isoformat()
            )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        # Return degraded status instead of failing completely
        return SystemStatus(
            status="degraded",
            health={
                "score": 0,
                "status": "error",
                "issues": [f"System status check failed: {str(e)}"]
            },
            data={"total_symbols": 0, "total_records": 0, "storage_size_mb": 0.0},
            models={"available_count": 0, "sample_checked": 0},
            cache={"predictions_cached": 0, "cache_expiry_minutes": 30},
            performance={"optimized": False, "parallel_processing": False, "caching_enabled": False, "max_symbols_per_request": 20, "max_data_points_default": 100},
            api_started=None,
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )


# Background task for warming up predictions
@app.post("/api/predictions/warmup")
async def warmup_predictions(background_tasks: BackgroundTasks, symbols: Optional[str] = Query(None)):
    """
    Start background task to warm up prediction cache.
    
    Args:
        symbols: Comma-separated list of symbols to warm up (optional)
        
    Returns:
        Dict with task status
    """
    def warmup_task(symbol_list: List[str]):
        """Background task to warm up predictions."""
        task_id = f"warmup_{int(time.time())}"
        _background_tasks_running.add(task_id)
        
        try:
            logger.info(f"Starting prediction warmup for {len(symbol_list)} symbols")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_single_prediction, symbol) for symbol in symbol_list]
                
                completed = 0
                for future in as_completed(futures, timeout=300):  # 5 minute timeout
                    try:
                        future.result()
                        completed += 1
                    except Exception as e:
                        logger.warning(f"Warmup prediction failed: {e}")
            
            logger.info(f"Prediction warmup completed: {completed}/{len(symbol_list)} successful")
            
        except Exception as e:
            logger.error(f"Prediction warmup task failed: {e}")
        finally:
            _background_tasks_running.discard(task_id)
    
    try:
        # Determine symbols to warm up
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        else:
            all_symbols = data_storage.get_available_symbols()
            symbol_list = all_symbols[:10]  # Warm up top 10 symbols
        
        # Start background task
        background_tasks.add_task(warmup_task, symbol_list)
        
        return {
            "message": "Prediction warmup started",
            "symbols_count": len(symbol_list),
            "estimated_time_minutes": len(symbol_list) // 2,  # Rough estimate
            "status": "started"
        }
        
    except Exception as e:
        raise handle_api_error(e, "warmup_predictions")


# Cache management endpoint
@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    try:
        with _cache_lock:
            _cached_predictions.clear()
            _cached_stock_summaries.clear()
            _cached_technical_indicators.clear()
        
        logger.info("All caches cleared")
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise handle_api_error(e, "clear_cache")


# Health check endpoint for load balancers
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "cache_entries": {
            "predictions": len(_cached_predictions),
            "summaries": len(_cached_stock_summaries),
            "indicators": len(_cached_technical_indicators)
        },
        "background_tasks": len(_background_tasks_running)
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": "The requested resource was not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global _api_start_time
    _api_start_time = datetime.now()
    
    logger.info("PSX AI Advisor API starting up...")
    logger.info("Performance optimizations enabled:")
    logger.info(f"- Max symbols per request: {MAX_SYMBOLS_PER_REQUEST}")
    logger.info(f"- Max data points default: {MAX_DATA_POINTS_DEFAULT}")
    logger.info(f"- Thread pool workers: {_thread_pool._max_workers}")
    logger.info(f"- Cache expiry: {_cache_expiry.total_seconds()/60} minutes")
    
    # Start background cache cleanup task
    asyncio.create_task(periodic_cache_cleanup())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("PSX AI Advisor API shutting down...")
    
    # Shutdown thread pool
    _thread_pool.shutdown(wait=True)
    
    # Clear caches
    with _cache_lock:
        _cached_predictions.clear()
        _cached_stock_summaries.clear()
        _cached_technical_indicators.clear()
    
    logger.info("Cleanup completed")


async def periodic_cache_cleanup():
    """Periodic task to clean up expired cache entries."""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            clear_expired_cache()
            logger.debug("Periodic cache cleanup completed")
        except Exception as e:
            logger.error(f"Error in periodic cache cleanup: {e}")


# Additional API endpoints for cache management and monitoring

@app.post("/api/cache/clear")
@log_api_performance
async def clear_cache():
    """Clear all cached data to force fresh calculations."""
    try:
        with _cache_lock:
            _cached_predictions.clear()
            _cached_stock_summaries.clear()
            _cached_technical_indicators.clear()
        
        log_cache_operation("clear_all")
        logger.info("All caches cleared successfully")
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise handle_api_error(e, "clear_cache")


@app.get("/api/performance/metrics")
@log_api_performance
async def get_performance_metrics():
    """Get API performance metrics."""
    try:
        metrics = get_performance_metrics()
        return {
            "performance_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise handle_api_error(e, "get_performance_metrics")


# Background task storage for tracking data loading operations
data_loading_tasks = {}
data_loading_lock = threading.Lock()


def background_data_update(task_id: str, update_type: str, symbols: Optional[List[str]] = None, period: str = "5y"):
    """
    Background task for updating stock data.
    
    Args:
        task_id: Unique task identifier
        update_type: Type of update (kse100, existing, single, custom)
        symbols: List of symbols for custom updates
        period: Period for data download
    """
    try:
        with data_loading_lock:
            data_loading_tasks[task_id]["status"] = "running"
            data_loading_tasks[task_id]["start_time"] = datetime.now()
        
        loader = PSXDataLoader()
        
        if update_type == "kse100":
            results = loader.update_kse100_stocks(period=period)
        elif update_type == "existing":
            results = loader.update_existing_stocks(period=period)
        elif update_type == "single" and symbols:
            results = {}
            for symbol in symbols:
                results[symbol] = loader.update_single_stock(symbol, period)
        elif update_type == "custom" and symbols:
            results = loader.update_multiple_stocks(symbols, period)
        else:
            raise ValueError(f"Invalid update type: {update_type}")
        
        # Calculate summary
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        with data_loading_lock:
            data_loading_tasks[task_id].update({
                "status": "completed",
                "end_time": datetime.now(),
                "results": results,
                "summary": {
                    "total_stocks": len(results),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": successful / len(results) if results else 0
                }
            })
        
        logger.info(f"Data update task {task_id} completed: {successful}/{len(results)} successful")
        
    except Exception as e:
        logger.error(f"Data update task {task_id} failed: {e}")
        with data_loading_lock:
            data_loading_tasks[task_id].update({
                "status": "failed",
                "end_time": datetime.now(),
                "error": str(e)
            })


@app.post("/data/update/kse100")
async def update_kse100_data(
    background_tasks: BackgroundTasks,
    period: str = Query("5y", description="Period for data download (1y, 2y, 5y, 10y, max)")
):
    """
    Start background update of all KSE-100 stock data.
    
    Args:
        period: Period for data download
        
    Returns:
        Task information for tracking update progress
    """
    try:
        task_id = f"kse100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with data_loading_lock:
            data_loading_tasks[task_id] = {
                "task_id": task_id,
                "type": "kse100",
                "status": "queued",
                "created_time": datetime.now(),
                "period": period,
                "description": f"Update all KSE-100 stocks with {period} data"
            }
        
        background_tasks.add_task(background_data_update, task_id, "kse100", None, period)
        
        logger.info(f"KSE-100 data update task {task_id} queued")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "KSE-100 data update started in background",
            "period": period,
            "stocks_count": len(PSXDataLoader.KSE_100_SYMBOLS),
            "tracking_url": f"/data/update/status/{task_id}"
        }
        
    except Exception as e:
        raise handle_api_error(e, "update_kse100_data")


@app.post("/data/update/existing")
async def update_existing_data(
    background_tasks: BackgroundTasks,
    period: str = Query("2y", description="Period for data download (1y, 2y, 5y, 10y, max)")
):
    """
    Start background update of all existing stock data.
    
    Args:
        period: Period for data download
        
    Returns:
        Task information for tracking update progress
    """
    try:
        task_id = f"existing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Count existing stocks
        data_dir = "data"
        existing_count = 0
        if os.path.exists(data_dir):
            existing_count = len([f for f in os.listdir(data_dir) if f.endswith('_historical_data.csv')])
        
        with data_loading_lock:
            data_loading_tasks[task_id] = {
                "task_id": task_id,
                "type": "existing",
                "status": "queued",
                "created_time": datetime.now(),
                "period": period,
                "description": f"Update all existing stocks with {period} data"
            }
        
        background_tasks.add_task(background_data_update, task_id, "existing", None, period)
        
        logger.info(f"Existing data update task {task_id} queued for {existing_count} stocks")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Existing stocks data update started in background",
            "period": period,
            "existing_stocks_count": existing_count,
            "tracking_url": f"/data/update/status/{task_id}"
        }
        
    except Exception as e:
        raise handle_api_error(e, "update_existing_data")


@app.post("/data/update/stocks")
async def update_custom_stocks(
    background_tasks: BackgroundTasks,
    symbols: List[str] = Query(..., description="List of stock symbols to update"),
    period: str = Query("5y", description="Period for data download (1y, 2y, 5y, 10y, max)")
):
    """
    Start background update of specific stock symbols.
    
    Args:
        symbols: List of stock symbols to update
        period: Period for data download
        
    Returns:
        Task information for tracking update progress
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol must be provided")
        
        # Validate and clean symbols
        clean_symbols = [symbol.upper().strip() for symbol in symbols if symbol.strip()]
        if not clean_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        task_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with data_loading_lock:
            data_loading_tasks[task_id] = {
                "task_id": task_id,
                "type": "custom",
                "status": "queued",
                "created_time": datetime.now(),
                "period": period,
                "symbols": clean_symbols,
                "description": f"Update {len(clean_symbols)} custom stocks with {period} data"
            }
        
        background_tasks.add_task(background_data_update, task_id, "custom", clean_symbols, period)
        
        logger.info(f"Custom stocks update task {task_id} queued for symbols: {clean_symbols}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": f"Custom stocks data update started in background",
            "period": period,
            "symbols": clean_symbols,
            "symbols_count": len(clean_symbols),
            "tracking_url": f"/data/update/status/{task_id}"
        }
        
    except Exception as e:
        raise handle_api_error(e, "update_custom_stocks")


@app.get("/data/update/status/{task_id}")
async def get_update_status(task_id: str):
    """
    Get status of a data update task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status and progress information
    """
    try:
        with data_loading_lock:
            if task_id not in data_loading_tasks:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            task_info = data_loading_tasks[task_id].copy()
        
        # Calculate runtime if applicable
        if task_info["status"] == "running" and "start_time" in task_info:
            task_info["runtime_seconds"] = (datetime.now() - task_info["start_time"]).total_seconds()
        elif task_info["status"] == "completed" and "start_time" in task_info and "end_time" in task_info:
            task_info["runtime_seconds"] = (task_info["end_time"] - task_info["start_time"]).total_seconds()
        
        # Format datetime objects for JSON response
        for key in ["created_time", "start_time", "end_time"]:
            if key in task_info and task_info[key]:
                task_info[key] = task_info[key].isoformat()
        
        return task_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "get_update_status")


@app.get("/data/update/tasks")
async def list_update_tasks():
    """
    List all data update tasks.
    
    Returns:
        List of all update tasks with their status
    """
    try:
        with data_loading_lock:
            tasks = []
            for task_id, task_info in data_loading_tasks.items():
                task_copy = task_info.copy()
                
                # Format datetime objects
                for key in ["created_time", "start_time", "end_time"]:
                    if key in task_copy and task_copy[key]:
                        task_copy[key] = task_copy[key].isoformat()
                
                # Remove large results data for list view
                if "results" in task_copy:
                    task_copy["has_results"] = True
                    del task_copy["results"]
                
                tasks.append(task_copy)
        
        return {
            "tasks": tasks,
            "total_tasks": len(tasks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise handle_api_error(e, "list_update_tasks")


@app.delete("/data/update/tasks/{task_id}")
async def delete_update_task(task_id: str):
    """
    Delete a completed or failed update task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        with data_loading_lock:
            if task_id not in data_loading_tasks:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            task = data_loading_tasks[task_id]
            if task["status"] == "running":
                raise HTTPException(status_code=400, detail="Cannot delete running task")
            
            del data_loading_tasks[task_id]
        
        logger.info(f"Deleted data update task {task_id}")
        
        return {
            "message": f"Task {task_id} deleted successfully",
            "deleted_task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "delete_update_task")


@app.get("/data/loader/info")
async def get_loader_info():
    """
    Get information about the data loader capabilities.
    
    Returns:
        Data loader configuration and capabilities
    """
    try:
        loader = PSXDataLoader()
        summary = loader.get_update_summary()
        
        return {
            "kse100_symbols": PSXDataLoader.KSE_100_SYMBOLS,
            "kse100_count": len(PSXDataLoader.KSE_100_SYMBOLS),
            "supported_periods": ["1y", "2y", "5y", "10y", "max"],
            "default_period": "5y",
            "backup_directory": summary["session_backup_dir"],
            "data_directory": summary["data_directory"],
            "fail_safe_enabled": True,
            "parallel_downloads": True,
            "max_workers": 4,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise handle_api_error(e, "get_loader_info")


@app.get("/health")
async def health_check():
    """Simple health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting PSX AI Advisor API server on {host}:{port}")
    logger.info("Performance optimizations enabled - API should respond faster")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run server with default settings
    run_server(reload=True)
