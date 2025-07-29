"""
Fallback mechanisms for PSX AI Advisor

This module provides fallback strategies for critical system failures,
ensuring the system can continue operating even when some components fail.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from .exceptions import PSXAdvisorError, create_error_context
from .logging_config import get_logger, log_exception, create_operation_logger

logger = get_logger(__name__)


class FallbackManager:
    """
    Manages fallback strategies for various system components.
    
    This class provides methods to handle failures gracefully and maintain
    system functionality even when primary operations fail.
    """
    
    def __init__(self):
        """Initialize the fallback manager"""
        self.fallback_strategies = {}
        self.fallback_history = []
        self.max_history_size = 1000
        
        # Register default fallback strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default fallback strategies for common operations"""
        
        # Data acquisition fallbacks
        self.register_fallback('csv_download', self._fallback_csv_download)
        self.register_fallback('csv_parsing', self._fallback_csv_parsing)
        
        # Data storage fallbacks
        self.register_fallback('data_save', self._fallback_data_save)
        self.register_fallback('data_load', self._fallback_data_load)
        
        # Technical analysis fallbacks
        self.register_fallback('technical_indicators', self._fallback_technical_indicators)
        
        # ML prediction fallbacks
        self.register_fallback('model_training', self._fallback_model_training)
        self.register_fallback('prediction', self._fallback_prediction)
        
        logger.info("Default fallback strategies registered")
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """
        Register a fallback function for a specific operation.
        
        Args:
            operation (str): Operation name
            fallback_func (Callable): Fallback function to execute
        """
        self.fallback_strategies[operation] = fallback_func
        logger.debug(f"Registered fallback strategy for operation: {operation}")
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, 
                            *args, fallback_data: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Execute a function with fallback support.
        
        Args:
            operation (str): Operation name
            primary_func (Callable): Primary function to execute
            *args: Arguments for primary function
            fallback_data (dict, optional): Additional data for fallback
            **kwargs: Keyword arguments for primary function
            
        Returns:
            Any: Result from primary function or fallback
        """
        operation_context = create_error_context(operation)
        
        try:
            # Try primary function first
            result = primary_func(*args, **kwargs)
            return result
            
        except Exception as primary_error:
            logger.warning(f"Primary operation '{operation}' failed: {primary_error}")
            
            # Record failure
            self._record_fallback_event(operation, str(primary_error), 'primary_failure')
            
            # Try fallback if available
            if operation in self.fallback_strategies:
                try:
                    logger.info(f"Executing fallback strategy for operation: {operation}")
                    
                    fallback_func = self.fallback_strategies[operation]
                    fallback_result = fallback_func(
                        primary_error=primary_error,
                        operation_context=operation_context,
                        fallback_data=fallback_data or {},
                        args=args,
                        kwargs=kwargs
                    )
                    
                    self._record_fallback_event(operation, "Fallback successful", 'fallback_success')
                    logger.info(f"Fallback strategy succeeded for operation: {operation}")
                    
                    return fallback_result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed for operation '{operation}': {fallback_error}")
                    self._record_fallback_event(operation, str(fallback_error), 'fallback_failure')
                    
                    # Re-raise original error if fallback also fails
                    raise primary_error
            else:
                logger.error(f"No fallback strategy available for operation: {operation}")
                self._record_fallback_event(operation, "No fallback available", 'no_fallback')
                raise primary_error
    
    def _record_fallback_event(self, operation: str, message: str, event_type: str):
        """Record a fallback event for monitoring and analysis"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'event_type': event_type,
            'message': message
        }
        
        self.fallback_history.append(event)
        
        # Limit history size
        if len(self.fallback_history) > self.max_history_size:
            self.fallback_history = self.fallback_history[-self.max_history_size:]
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics about fallback usage"""
        if not self.fallback_history:
            return {'total_events': 0}
        
        stats = {
            'total_events': len(self.fallback_history),
            'events_by_type': {},
            'events_by_operation': {},
            'recent_events': self.fallback_history[-10:] if self.fallback_history else []
        }
        
        for event in self.fallback_history:
            event_type = event['event_type']
            operation = event['operation']
            
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
            stats['events_by_operation'][operation] = stats['events_by_operation'].get(operation, 0) + 1
        
        return stats
    
    # Default fallback strategies
    
    def _fallback_csv_download(self, primary_error: Exception, operation_context: Dict[str, Any], 
                              fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> str:
        """
        Fallback strategy for CSV download failures.
        
        Tries to use cached/backup files or alternative dates.
        """
        logger.info("Attempting CSV download fallback strategy")
        
        # Try to find existing CSV files from recent dates
        data_dir = fallback_data.get('data_dir', 'data')
        target_date = fallback_data.get('date')
        
        if target_date:
            # Try previous business days
            for days_back in range(1, 8):  # Try up to 7 days back
                try:
                    fallback_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    fallback_filename = f"{fallback_date}.csv"
                    fallback_path = os.path.join(data_dir, fallback_filename)
                    
                    if os.path.exists(fallback_path):
                        logger.info(f"Using fallback CSV from {fallback_date}")
                        return fallback_path
                        
                except Exception as e:
                    continue
        
        # If no fallback file found, raise original error
        raise primary_error
    
    def _fallback_csv_parsing(self, primary_error: Exception, operation_context: Dict[str, Any], 
                             fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> pd.DataFrame:
        """
        Fallback strategy for CSV parsing failures.
        
        Returns a minimal DataFrame structure or uses cached data.
        """
        logger.info("Attempting CSV parsing fallback strategy")
        
        csv_path = args[0] if args else fallback_data.get('csv_path')
        
        # Try to create a minimal DataFrame structure
        try:
            # Check if we can at least read the file as text
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                # Try to create a basic structure
                columns = ['Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
                df = pd.DataFrame(columns=columns)
                
                # Add a single row with placeholder data
                df.loc[0] = [
                    datetime.now().strftime('%Y-%m-%d'),
                    'UNKNOWN',
                    'UNKNOWN',
                    0.0, 0.0, 0.0, 0.0, 0,
                    0.0, 0.0
                ]
                
                logger.warning("Created minimal DataFrame structure as fallback")
                return df
                
        except Exception as e:
            logger.error(f"Fallback CSV parsing also failed: {e}")
        
        # If all fallback attempts fail, raise original error
        raise primary_error
    
    def _fallback_data_save(self, primary_error: Exception, operation_context: Dict[str, Any], 
                           fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> bool:
        """
        Fallback strategy for data save failures.
        
        Tries alternative storage locations or formats.
        """
        logger.info("Attempting data save fallback strategy")
        
        symbol = args[0] if args else fallback_data.get('symbol', 'UNKNOWN')
        data = args[1] if len(args) > 1 else fallback_data.get('data')
        
        if data is None or data.empty:
            logger.warning("No data to save in fallback")
            return False
        
        # Try saving to backup directory
        try:
            backup_dir = fallback_data.get('backup_dir', 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"{symbol}_{timestamp}.csv")
            
            data.to_csv(backup_path, index=False)
            logger.info(f"Data saved to backup location: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup save also failed: {e}")
        
        # If backup save fails, return False but don't raise exception
        return False
    
    def _fallback_data_load(self, primary_error: Exception, operation_context: Dict[str, Any], 
                           fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> pd.DataFrame:
        """
        Fallback strategy for data load failures.
        
        Tries backup locations or returns empty DataFrame.
        """
        logger.info("Attempting data load fallback strategy")
        
        symbol = args[0] if args else fallback_data.get('symbol', 'UNKNOWN')
        
        # Try loading from backup directory
        backup_dir = fallback_data.get('backup_dir', 'backups')
        
        if os.path.exists(backup_dir):
            try:
                # Find most recent backup for this symbol
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith(symbol) and f.endswith('.csv')]
                
                if backup_files:
                    # Sort by filename (which includes timestamp)
                    backup_files.sort(reverse=True)
                    latest_backup = backup_files[0]
                    backup_path = os.path.join(backup_dir, latest_backup)
                    
                    data = pd.read_csv(backup_path, parse_dates=['Date'])
                    logger.info(f"Loaded data from backup: {backup_path}")
                    return data
                    
            except Exception as e:
                logger.error(f"Backup load also failed: {e}")
        
        # Return empty DataFrame as last resort
        columns = ['Date', 'Symbol', 'Company_Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'Change']
        empty_df = pd.DataFrame(columns=columns)
        logger.warning(f"Returning empty DataFrame for symbol: {symbol}")
        return empty_df
    
    def _fallback_technical_indicators(self, primary_error: Exception, operation_context: Dict[str, Any], 
                                     fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> pd.DataFrame:
        """
        Fallback strategy for technical indicator calculation failures.
        
        Returns DataFrame with NaN values for indicators.
        """
        logger.info("Attempting technical indicators fallback strategy")
        
        df = args[0] if args else fallback_data.get('dataframe')
        
        if df is None or df.empty:
            raise primary_error
        
        # Add indicator columns with NaN values
        indicator_columns = [
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_12',
            'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_MA_20', 'OBV', 'Return_1d', 'Return_2d', 'Return_5d', 'Volatility_20d'
        ]
        
        result_df = df.copy()
        for col in indicator_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan
        
        logger.warning("Added technical indicator columns with NaN values as fallback")
        return result_df
    
    def _fallback_model_training(self, primary_error: Exception, operation_context: Dict[str, Any], 
                                fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Fallback strategy for model training failures.
        
        Returns a simple rule-based prediction model.
        """
        logger.info("Attempting model training fallback strategy")
        
        symbol = args[0] if args else fallback_data.get('symbol', 'UNKNOWN')
        
        # Create a simple fallback training result
        fallback_result = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'model_type': 'RuleBased_Fallback',
            'training_samples': 0,
            'test_samples': 0,
            'total_samples': 0,
            'accuracy': 0.5,  # Random guess accuracy
            'precision': 0.5,
            'recall': 0.5,
            'cv_scores': {'mean_accuracy': 0.5, 'std_accuracy': 0.0},
            'feature_count': 0,
            'top_features': [],
            'class_distribution': {'UP': 0, 'DOWN': 0},
            'model_parameters': {'type': 'rule_based_fallback'},
            'fallback_used': True,
            'fallback_reason': str(primary_error)
        }
        
        logger.warning(f"Using rule-based fallback model for {symbol}")
        return fallback_result
    
    def _fallback_prediction(self, primary_error: Exception, operation_context: Dict[str, Any], 
                           fallback_data: Dict[str, Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Fallback strategy for prediction failures.
        
        Returns a simple rule-based prediction.
        """
        logger.info("Attempting prediction fallback strategy")
        
        symbol = args[0] if args else fallback_data.get('symbol', 'UNKNOWN')
        
        # Simple rule-based prediction (random for fallback)
        import random
        prediction_direction = random.choice(['UP', 'DOWN'])
        confidence = 0.5  # Low confidence for fallback
        
        fallback_result = {
            'symbol': symbol,
            'prediction': prediction_direction,
            'confidence': confidence,
            'prediction_probabilities': {
                'DOWN': 0.5,
                'UP': 0.5
            },
            'current_price': 0.0,
            'prediction_date': datetime.now().isoformat(),
            'data_date': datetime.now().isoformat(),
            'model_accuracy': 0.5,
            'model_type': 'RuleBased_Fallback',
            'feature_count': 0,
            'fallback_used': True,
            'fallback_reason': str(primary_error)
        }
        
        logger.warning(f"Using rule-based fallback prediction for {symbol}: {prediction_direction}")
        return fallback_result


# Global fallback manager instance
_fallback_manager = None


def get_fallback_manager() -> FallbackManager:
    """
    Get the global fallback manager instance.
    
    Returns:
        FallbackManager: Global fallback manager
    """
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager()
    return _fallback_manager


def execute_with_fallback(operation: str, primary_func: Callable, *args, 
                         fallback_data: Dict[str, Any] = None, **kwargs) -> Any:
    """
    Execute a function with fallback support.
    
    Args:
        operation (str): Operation name
        primary_func (Callable): Primary function to execute
        *args: Arguments for primary function
        fallback_data (dict, optional): Additional data for fallback
        **kwargs: Keyword arguments for primary function
        
    Returns:
        Any: Result from primary function or fallback
    """
    manager = get_fallback_manager()
    return manager.execute_with_fallback(operation, primary_func, *args, 
                                       fallback_data=fallback_data, **kwargs)


def register_fallback(operation: str, fallback_func: Callable):
    """
    Register a custom fallback function for an operation.
    
    Args:
        operation (str): Operation name
        fallback_func (Callable): Fallback function to execute
    """
    manager = get_fallback_manager()
    manager.register_fallback(operation, fallback_func)


def get_fallback_stats() -> Dict[str, Any]:
    """
    Get statistics about fallback usage.
    
    Returns:
        dict: Fallback usage statistics
    """
    manager = get_fallback_manager()
    return manager.get_fallback_stats()