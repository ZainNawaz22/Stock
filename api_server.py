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
from psx_ai_advisor.config_loader import get_section, get_value
from psx_ai_advisor.logging_config import get_logger
from psx_ai_advisor.exceptions import (
    PSXAdvisorError, DataStorageError, MLPredictorError, 
    InsufficientDataError, NetworkError
)

# Initialize logging
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PSX AI Advisor API",
    description="REST API for Pakistan Stock Exchange AI-powered stock analysis and predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize components
data_storage = DataStorage()
ml_predictor = MLPredictor()
technical_analyzer = TechnicalAnalyzer()
data_acquisition = PSXDataAcquisition()

# Global variables for caching and performance optimization
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
MAX_SYMBOLS_PER_REQUEST = 20
MAX_DATA_POINTS_DEFAULT = 100
SYSTEM_STATUS_SAMPLE_SIZE = 5


def get_cached_data(cache_dict: Dict, key: str, expiry_minutes: int = 30) -> Optional[Any]:
    """Get cached data if it exists and is not expired."""
    with _cache_lock:
        if key in cache_dict:
            data, timestamp = cache_dict[key]
            if datetime.now() - timestamp < timedelta(minutes=expiry_minutes):
                return data
            else:
                # Remove expired data
                del cache_dict[key]
    return None


def set_cached_data(cache_dict: Dict, key: str, data: Any) -> None:
    """Set cached data with timestamp."""
    with _cache_lock:
        cache_dict[key] = (data, datetime.now())


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


def process_symbol_summary(symbol: str) -> Dict[str, Any]:
    """Process a single symbol summary in a thread-safe way."""
    try:
        # Check cache first
        cached_summary = get_cached_data(_cached_stock_summaries, symbol, 15)
        if cached_summary:
            return cached_summary
        
        # Get data summary
        summary = data_storage.get_data_summary(symbol)
        
        if summary.get('exists', False) and not summary.get('is_empty', True):
            stock_info = {
                "symbol": symbol,
                "company_name": symbol,  # Using symbol as placeholder
                "record_count": summary.get('record_count', 0),
                "current_price": summary.get('price_range', {}).get('current_close'),
                "last_updated": summary.get('last_updated'),
                "date_range": summary.get('date_range', {}),
                "has_data": True
            }
        else:
            stock_info = {
                "symbol": symbol,
                "company_name": symbol,
                "has_data": False,
                "error": "No valid data available"
            }
        
        # Cache the result
        set_cached_data(_cached_stock_summaries, symbol, stock_info)
        return stock_info
        
    except Exception as e:
        logger.warning(f"Error processing summary for {symbol}: {e}")
        return {
            "symbol": symbol,
            "company_name": symbol,
            "has_data": False,
            "error": f"Processing error: {str(e)}"
        }


def process_single_prediction(symbol: str) -> Dict[str, Any]:
    """Process a single prediction in a thread-safe way."""
    try:
        # Check cache first
        cached_prediction = get_cached_data(_cached_predictions, symbol, 30)
        if cached_prediction:
            return cached_prediction
        
        # Generate new prediction
        prediction = ml_predictor.predict_movement(symbol)
        
        # Cache the result
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
        logger.error(f"Error generating prediction for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": "prediction_failed",
            "message": str(e),
            "prediction": None,
            "confidence": None
        }


def serialize_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert pandas DataFrame to JSON-serializable format.
    
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


def handle_api_error(error: Exception, operation: str) -> HTTPException:
    """
    Convert internal exceptions to appropriate HTTP exceptions.
    
    Args:
        error (Exception): Internal exception
        operation (str): Operation that failed
        
    Returns:
        HTTPException: HTTP exception with appropriate status code
    """
    logger.error(f"API error in {operation}: {str(error)}")
    
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


@app.get("/api/stocks")
async def get_stocks(
    limit: Optional[int] = Query(None, description="Maximum number of stocks to return"),
    search: Optional[str] = Query(None, description="Search term to filter stocks by symbol")
) -> Dict[str, Any]:
    """
    Get list of available stocks with basic information.
    
    Args:
        limit: Maximum number of stocks to return
        search: Search term to filter stocks by symbol
        
    Returns:
        Dict containing list of stocks with basic info
    """
    try:
        logger.info(f"Getting stocks list (limit: {limit}, search: {search})")
        
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
        
        # Apply reasonable limit to prevent timeouts
        if limit is None or limit > MAX_SYMBOLS_PER_REQUEST:
            limit = MAX_SYMBOLS_PER_REQUEST
            logger.info(f"Applied default limit of {MAX_SYMBOLS_PER_REQUEST} symbols for performance")
        
        # Process symbols in parallel using thread pool
        symbols_to_process = symbols[:limit] if limit else symbols
        
        # Use ThreadPoolExecutor for parallel processing
        stocks_info = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_symbol_summary, symbol): symbol 
                for symbol in symbols_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol, timeout=10):  # 10 second timeout
                try:
                    stock_info = future.result()
                    if stock_info.get('has_data', False):
                        stocks_info.append(stock_info)
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.warning(f"Error processing {symbol}: {e}")
                    stocks_info.append({
                        "symbol": symbol,
                        "company_name": symbol,
                        "has_data": False,
                        "error": "Processing timeout or error"
                    })
        
        # Sort by symbol
        stocks_info.sort(key=lambda x: x['symbol'])
        
        total_count = len(symbols)
        returned_count = len(stocks_info)
        
        logger.info(f"Returning {returned_count} stocks (total available: {total_count})")
        
        return {
            "stocks": stocks_info,
            "total_count": total_count,
            "returned_count": returned_count,
            "has_more": total_count > returned_count,
            "processing_time_optimized": True
        }
        
    except Exception as e:
        raise handle_api_error(e, "get_stocks")


@app.get("/api/stocks/{symbol}/data")
async def get_stock_data(
    symbol: str = Path(..., description="Stock symbol"),
    days: Optional[int] = Query(None, description="Number of recent days to return"),
    include_indicators: bool = Query(True, description="Include technical indicators")
) -> Dict[str, Any]:
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
        
        logger.info(f"Returning {len(stock_data)} data points for {symbol_upper}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, f"get_stock_data for {symbol}")


@app.get("/api/predictions")
async def get_predictions(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols (optional)"),
    force_refresh: bool = Query(False, description="Force refresh of cached predictions"),
    limit: Optional[int] = Query(10, description="Maximum number of predictions to return")
) -> Dict[str, Any]:
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
        
        return {
            "predictions": predictions,
            "total_count": len(predictions),
            "successful_count": successful_predictions,
            "failed_count": failed_predictions,
            "processing_time_optimized": True,
            "parallel_processing": True,
            "last_updated": _last_system_update.isoformat() if _last_system_update else None
        }
        
    except Exception as e:
        raise handle_api_error(e, "get_predictions")


@app.get("/api/system/status")
async def get_system_status() -> Dict[str, Any]:
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
        
        # Get available symbols count (fast operation)
        available_symbols = data_storage.get_available_symbols()
        
        # Get basic storage stats without heavy processing
        try:
            storage_stats = data_storage.get_storage_stats()
        except Exception as e:
            logger.warning(f"Error getting storage stats: {e}")
            storage_stats = {"error": "Unable to calculate storage stats"}
        
        # Check data freshness with limited sample
        latest_data_date = None
        oldest_data_date = None
        
        if available_symbols:
            # Sample only a few symbols to check data freshness (performance optimization)
            sample_symbols = available_symbols[:SYSTEM_STATUS_SAMPLE_SIZE]
            dates = []
            
            # Use thread pool for parallel processing of samples
            def check_symbol_freshness(symbol):
                try:
                    summary = data_storage.get_data_summary(symbol)
                    if summary.get('exists') and summary.get('date_range'):
                        end_date = summary['date_range'].get('end')
                        start_date = summary['date_range'].get('start')
                        return {
                            'end_date': pd.to_datetime(end_date) if end_date else None,
                            'start_date': pd.to_datetime(start_date) if start_date else None
                        }
                except Exception:
                    pass
                return None
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(check_symbol_freshness, symbol) for symbol in sample_symbols]
                
                for future in as_completed(futures, timeout=5):  # 5 second timeout
                    try:
                        result = future.result()
                        if result:
                            if result['end_date']:
                                dates.append(result['end_date'])
                            if result['start_date'] and (not oldest_data_date or result['start_date'] < oldest_data_date):
                                oldest_data_date = result['start_date']
                    except Exception:
                        pass
            
            if dates:
                latest_data_date = max(dates)
        
        # Calculate system health score
        health_score = 100
        health_issues = []
        
        # Check if we have data
        if not available_symbols:
            health_score -= 50
            health_issues.append("No stock data available")
        
        # Check data freshness (should be within last 7 days for healthy system)
        if latest_data_date:
            days_since_update = (datetime.now() - latest_data_date.to_pydatetime()).days
            if days_since_update > 7:
                health_score -= 30
                health_issues.append(f"Data is {days_since_update} days old")
            elif days_since_update > 3:
                health_score -= 15
                health_issues.append(f"Data is {days_since_update} days old")
        
        # Quick model availability check (limited sample)
        models_available = 0
        sample_for_models = available_symbols[:5]  # Check only 5 symbols for performance
        
        for symbol in sample_for_models:
            try:
                model_info = ml_predictor.get_model_info(symbol)
                if model_info.get('model_exists', False):
                    models_available += 1
            except Exception:
                pass
        
        if available_symbols and models_available == 0:
            health_score -= 20
            health_issues.append("No trained models available")
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 70:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        # Prepare status info
        status_info = {
            "status": "operational",
            "health": {
                "score": health_score,
                "status": health_status,
                "issues": health_issues
            },
            "data": {
                "total_symbols": len(available_symbols),
                "total_records": storage_stats.get('total_records', 0),
                "storage_size_mb": storage_stats.get('total_size_mb', 0),
                "latest_data_date": latest_data_date.isoformat() if latest_data_date else None,
                "oldest_data_date": oldest_data_date.isoformat() if oldest_data_date else None,
                "data_directory": storage_stats.get('data_directory'),
                "sample_size": SYSTEM_STATUS_SAMPLE_SIZE
            },
            "models": {
                "available_count": models_available,
                "sample_checked": len(sample_for_models),
                "estimated_total": int(models_available * len(available_symbols) / len(sample_for_models)) if sample_for_models else 0
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
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the status for 5 minutes
        set_cached_data(_cached_stock_summaries, "system_status", status_info)
        
        logger.info(f"System status: {health_status} (score: {health_score})")
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        # Return degraded status instead of failing completely
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "health": {
                "score": 0,
                "status": "error",
                "issues": [f"System status check failed: {str(e)}"]
            }
        }


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