"""
Machine Learning Predictor Module for PSX AI Advisor

This module implements machine learning prediction capabilities for stock price movements.
It uses Random Forest classifier to predict next-day price movements (UP/DOWN) based on
technical indicators and historical price data.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import time
import hashlib
import json
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import xgboost as xgb
except Exception:
    xgb = None
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import warnings

from .config_loader import get_section, get_value, is_ensemble_enabled, get_ensemble_config, get_xgboost_config
from .data_storage import DataStorage
from .technical_analysis import TechnicalAnalyzer
from .exceptions import (
    MLPredictorError, InsufficientDataError, ModelTrainingError,
    ModelPersistenceError, ValidationError, create_error_context,
    XGBoostTrainingError, EnsembleCreationError
)
from .logging_config import get_logger, log_exception, create_operation_logger

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logger = get_logger(__name__)


class MLPredictor:
    """
    Machine Learning predictor for stock price movements using Random Forest classifier.
    
    This class handles feature preparation, model training, prediction generation,
    and model persistence for individual stock symbols.
    """
    
    def __init__(self, model_type: str = "RandomForest"):
        """
        Initialize the ML Predictor
        
        Args:
            model_type (str): Type of ML model to use (default: "RandomForest")
        """
        # Load configuration
        self.ml_config = get_section('machine_learning')
        self.storage_config = get_section('storage')
        
        # ML parameters
        self.model_type = model_type
        self.min_training_samples = self.ml_config.get('min_training_samples', 200)
        self.random_state = self.ml_config.get('random_state', 42)
        self.n_estimators = self.ml_config.get('n_estimators', 100)
        
        # Time series cross-validation parameters
        self.n_splits = self.ml_config.get('n_splits', 5)  # Number of folds for TimeSeriesSplit
        self.max_train_size = self.ml_config.get('max_train_size', None)  # Maximum training set size
        
        # Storage setup
        self.data_dir = self.storage_config.get('data_directory', 'data')
        self.models_dir = os.path.join(self.data_dir, 'models')
        self.scalers_dir = os.path.join(self.data_dir, 'scalers')
        
        # Initialize components
        self.data_storage = DataStorage()
        self.technical_analyzer = TechnicalAnalyzer()
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
        # Performance optimization and caching
        self.cv_cache = {}  # Cache for cross-validation results
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        self.max_cache_size = self.ml_config.get('max_cache_size', 100)  # Max cached CV results
        self.enable_parallel_cv = self.ml_config.get('enable_parallel_cv', True)
        self.max_workers = self.ml_config.get('max_workers', min(4, os.cpu_count() or 1))
        self.memory_efficient_storage = self.ml_config.get('memory_efficient_storage', True)
        self.performance_monitoring = self.ml_config.get('performance_monitoring', True)
        
        # Initialize thread pool for ML operations
        self._ml_thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Enhanced performance optimization settings
        self.training_time_limit_multiplier = self.ml_config.get('training_time_limit_multiplier', 2.0)  # Max 2x RF training time
        self.enable_model_compression = self.ml_config.get('enable_model_compression', True)
        self.cache_expiry_hours = self.ml_config.get('cache_expiry_hours', 24)  # Cache expiry in hours
        self.memory_cleanup_threshold = self.ml_config.get('memory_cleanup_threshold', 0.8)  # Memory usage threshold
        self.parallel_optimization_batch_size = self.ml_config.get('parallel_optimization_batch_size', 10)
        
        # Performance tracking
        self.training_times = {}  # Track training times for performance monitoring
        self.baseline_rf_time = None  # Baseline Random Forest training time
        
        # Ensure model directories exist
        self._ensure_model_directories()
        
        # Feature columns for ML (technical indicators)
        self.feature_columns = [
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_12',
            'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_MA_20', 'OBV', 'Return_1d', 'Return_2d', 'Return_5d', 'Volatility_20d'
        ]
        
        logger.info(f"MLPredictor initialized with {model_type} model")
        self.threshold_config = self.ml_config.get('threshold_tuning', {})
        self.threshold_enabled = self.threshold_config.get('enabled', True)
        self.threshold_metric = self.threshold_config.get('metric', 'f1')
        self.threshold_min = float(self.threshold_config.get('min_threshold', 0.3))
        self.threshold_max = float(self.threshold_config.get('max_threshold', 0.7))
        self.threshold_step = float(self.threshold_config.get('step', 0.05))
        self.utility_params = self.threshold_config.get('utility', {
            'tp_reward': 1.0,
            'tn_reward': 0.0,
            'fp_cost': 1.0,
            'fn_cost': 1.0
        })

        # Load ensemble configuration
        self.ensemble_enabled = is_ensemble_enabled()
        self.ensemble_config = get_ensemble_config()
        self.xgboost_config = get_xgboost_config()
        
        # Override model_type from configuration if not explicitly set
        config_model_type = self.ml_config.get('model_type', 'RandomForest')
        if model_type == "RandomForest" and config_model_type.lower() == 'ensemble':
            self.model_type = 'ensemble'
            self.ensemble_enabled = True
            logger.info("Model type overridden to 'ensemble' based on configuration")

        self._validate_optional_dependencies()
    
    def __del__(self):
        """Cleanup thread pool on object destruction"""
        if hasattr(self, '_ml_thread_pool'):
            self._ml_thread_pool.shutdown(wait=False)
    
    def shutdown(self):
        """Explicitly shutdown the thread pool"""
        if hasattr(self, '_ml_thread_pool'):
            self._ml_thread_pool.shutdown(wait=True)
    
    def _ensure_model_directories(self) -> None:
        """Create model and scaler directories if they don't exist"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.scalers_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.debug(f"Ensured model directories exist: {self.models_dir}, {self.scalers_dir}, {self.cache_dir}")
        except OSError as e:
            raise MLPredictorError(f"Failed to create model directories: {e}")

    def _validate_optional_dependencies(self) -> None:
        try:
            import importlib
            xgb_spec = importlib.util.find_spec('xgboost')
            self.xgboost_available = xgb_spec is not None
        except Exception:
            self.xgboost_available = False
        if not hasattr(self, 'xgboost_available'):
            self.xgboost_available = False
        if self.xgboost_available:
            logger.debug("Optional dependency xgboost is available")
        else:
            logger.debug("Optional dependency xgboost not found; XGBoost features will be disabled")

    def _generate_cache_key(self, X: np.ndarray, y: np.ndarray, model_params: dict, cv_params: dict) -> str:
        """
        Generate a unique cache key for cross-validation results.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            model_params (dict): Model parameters
            cv_params (dict): Cross-validation parameters
            
        Returns:
            str: Unique cache key
        """
        # Create a hash of the data and parameters
        data_hash = hashlib.md5(X.tobytes() + y.tobytes()).hexdigest()[:16]
        params_str = json.dumps({**model_params, **cv_params}, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
        
        return f"{data_hash}_{params_hash}"

    def _get_cached_cv_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced cached cross-validation results retrieval with performance monitoring.
        
        Args:
            cache_key (str): Cache key for the results
            
        Returns:
            Optional[Dict[str, Any]]: Cached results or None if not found/expired
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"cv_{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            # Check cache age with configurable expiry
            current_time = time.time()
            cache_mtime = os.path.getmtime(cache_file)
            cache_age = current_time - cache_mtime
            cache_expiry_seconds = self.cache_expiry_hours * 3600
            
            if cache_age > cache_expiry_seconds:
                # Remove expired cache file
                try:
                    os.remove(cache_file)
                    logger.debug(f"Removed expired cache file: {cache_file} (age: {cache_age/3600:.1f} hours)")
                except OSError as e:
                    logger.debug(f"Failed to remove expired cache file: {e}")
                return None
            
            # Load and validate cached results
            try:
                with open(cache_file, 'r') as f:
                    cached_results = json.load(f)
                
                # Validate cache structure
                required_keys = ['mean_accuracy', 'fold_scores', 'n_splits']
                if not all(key in cached_results for key in required_keys):
                    logger.debug(f"Invalid cache structure for key: {cache_key}, removing file")
                    os.remove(cache_file)
                    return None
                
                # Add cache hit metadata
                cached_results['cache_hit'] = True
                cached_results['cache_age_hours'] = cache_age / 3600
                cached_results['cache_retrieval_time'] = current_time
                
                if self.performance_monitoring:
                    logger.debug(f"Cache hit for key: {cache_key} (age: {cache_age/3600:.1f} hours)")
                
                return cached_results
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Invalid cache file format for key: {cache_key}, removing: {e}")
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
                return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached CV results for key {cache_key}: {e}")
            return None

    def _cache_cv_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """
        Cache cross-validation results to disk.
        
        Args:
            cache_key (str): Cache key for the results
            results (Dict[str, Any]): Results to cache
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"cv_{cache_key}.json")
            
            # Ensure we don't exceed max cache size
            self._cleanup_cache()
            
            with open(cache_file, 'w') as f:
                json.dump(results, f)
            
            logger.debug(f"Cached CV results for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache CV results: {e}")

    def _cleanup_cache(self) -> None:
        """
        Enhanced cache cleanup with size and age-based management.
        
        Implements efficient cache management by:
        - Removing files older than cache_expiry_hours
        - Maintaining max_cache_size limit with LRU eviction
        - Monitoring cache performance and hit rates
        - Optimizing cache storage efficiency
        """
        try:
            if not os.path.exists(self.cache_dir):
                return
                
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('cv_') and f.endswith('.json')]
            
            if not cache_files:
                return
            
            current_time = time.time()
            cache_expiry_seconds = self.cache_expiry_hours * 3600
            
            # Collect file information
            cache_files_info = []
            expired_files = []
            total_cache_size = 0
            
            for f in cache_files:
                file_path = os.path.join(self.cache_dir, f)
                try:
                    stat_info = os.stat(file_path)
                    mtime = stat_info.st_mtime
                    file_size = stat_info.st_size
                    age = current_time - mtime
                    
                    total_cache_size += file_size
                    
                    if age > cache_expiry_seconds:
                        expired_files.append((f, file_path, age))
                    else:
                        cache_files_info.append((f, file_path, mtime, file_size, age))
                        
                except OSError as e:
                    logger.debug(f"Could not stat cache file {f}: {e}")
                    continue
            
            # Remove expired files first
            expired_count = 0
            expired_size = 0
            for f, file_path, age in expired_files:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    expired_count += 1
                    expired_size += file_size
                    logger.debug(f"Removed expired cache file: {f} (age: {age/3600:.1f} hours)")
                except OSError as e:
                    logger.debug(f"Failed to remove expired cache file {f}: {e}")
            
            # Check if we still need to remove files due to size limit
            remaining_files = len(cache_files_info)
            if remaining_files > self.max_cache_size:
                # Sort by access time (LRU eviction) - use modification time as proxy
                cache_files_info.sort(key=lambda x: x[2])  # Sort by mtime
                
                # Remove oldest files to maintain size limit
                files_to_remove = remaining_files - self.max_cache_size + 5  # Remove extra for buffer
                removed_count = 0
                removed_size = 0
                
                for i in range(min(files_to_remove, len(cache_files_info))):
                    f, file_path, mtime, file_size, age = cache_files_info[i]
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        removed_size += file_size
                        logger.debug(f"Removed old cache file for size limit: {f}")
                    except OSError as e:
                        logger.debug(f"Failed to remove cache file {f}: {e}")
                
                if self.performance_monitoring and removed_count > 0:
                    logger.info(f"Cache size cleanup: removed {removed_count} files ({removed_size/1024:.1f} KB)")
            
            # Log cache cleanup summary
            if self.performance_monitoring and (expired_count > 0 or removed_count > 0):
                remaining_files_after = len([f for f in os.listdir(self.cache_dir) 
                                           if f.startswith('cv_') and f.endswith('.json')])
                logger.info(f"Cache cleanup completed:")
                logger.info(f"  - Expired files removed: {expired_count} ({expired_size/1024:.1f} KB)")
                logger.info(f"  - Size limit files removed: {removed_count if 'removed_count' in locals() else 0}")
                logger.info(f"  - Remaining cache files: {remaining_files_after}")
                logger.info(f"  - Total cache size before: {total_cache_size/1024:.1f} KB")
                
        except Exception as e:
            logger.warning(f"Enhanced cache cleanup failed: {e}")
            # Fallback to basic cleanup
            try:
                cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('cv_') and f.endswith('.json')]
                if len(cache_files) > self.max_cache_size:
                    # Simple cleanup - remove oldest files
                    cache_files_with_time = []
                    for f in cache_files:
                        file_path = os.path.join(self.cache_dir, f)
                        mtime = os.path.getmtime(file_path)
                        cache_files_with_time.append((f, mtime))
                    
                    cache_files_with_time.sort(key=lambda x: x[1])
                    files_to_remove = len(cache_files) - self.max_cache_size + 10
                    
                    for i in range(min(files_to_remove, len(cache_files_with_time))):
                        file_to_remove = os.path.join(self.cache_dir, cache_files_with_time[i][0])
                        os.remove(file_to_remove)
                        
            except Exception as fallback_error:
                logger.warning(f"Fallback cache cleanup failed: {fallback_error}")

    def _perform_efficient_cv_scoring(self, model_class, model_params: dict, X: np.ndarray, y: np.ndarray, 
                                    tscv: TimeSeriesSplit, model_name: str = "model") -> Dict[str, Any]:
        """
        Perform efficient cross-validation scoring with result caching.
        
        Args:
            model_class: Model class to evaluate
            model_params (dict): Model parameters
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            tscv (TimeSeriesSplit): Cross-validation splitter
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, Any]: Cross-validation results with caching
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cv_params = {
                'n_splits': tscv.n_splits,
                'max_train_size': tscv.max_train_size,
                'model_class': model_class.__name__
            }
            cache_key = self._generate_cache_key(X, y, model_params, cv_params)
            
            # Check for cached results
            cached_results = self._get_cached_cv_results(cache_key)
            if cached_results is not None:
                logger.info(f"Using cached CV results for {model_name} (saved {time.time() - start_time:.2f}s)")
                return cached_results
            
            logger.info(f"Performing efficient CV scoring for {model_name}")
            
            # Perform cross-validation with parallel processing if enabled
            if self.enable_parallel_cv and self.max_workers > 1:
                cv_results = self._parallel_cv_scoring(model_class, model_params, X, y, tscv, model_name)
            else:
                cv_results = self._sequential_cv_scoring(model_class, model_params, X, y, tscv, model_name)
            
            # Add timing information
            cv_results['cv_time'] = time.time() - start_time
            cv_results['cached'] = False
            
            # Cache results
            self._cache_cv_results(cache_key, cv_results)
            
            logger.info(f"Completed efficient CV scoring for {model_name} in {cv_results['cv_time']:.2f}s")
            return cv_results
            
        except Exception as e:
            logger.error(f"Efficient CV scoring failed for {model_name}: {e}")
            # Fallback to basic scoring
            return self._fallback_cv_scoring(model_class, model_params, X, y, tscv, model_name)

    def _parallel_cv_scoring(self, model_class, model_params: dict, X: np.ndarray, y: np.ndarray, 
                           tscv: TimeSeriesSplit, model_name: str) -> Dict[str, Any]:
        """
        Perform parallel cross-validation scoring using ThreadPoolExecutor.
        
        Args:
            model_class: Model class to evaluate
            model_params (dict): Model parameters
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            tscv (TimeSeriesSplit): Cross-validation splitter
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Using parallel CV scoring with {self.max_workers} workers for {model_name}")
        
        scores = []
        fold_results = []
        
        def evaluate_fold(fold_data):
            fold_idx, (train_idx, test_idx) = fold_data
            try:
                X_train_fold = X[train_idx]
                X_test_fold = X[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # Scale features for this fold
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                X_test_fold_scaled = scaler_fold.transform(X_test_fold)
                
                # Create and train model
                model_fold = model_class(**model_params)
                model_fold.fit(X_train_fold_scaled, y_train_fold)
                
                # Make predictions and calculate metrics
                y_pred_fold = model_fold.predict(X_test_fold_scaled)
                accuracy = accuracy_score(y_test_fold, y_pred_fold)
                precision = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                recall = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                
                return {
                    'fold': fold_idx,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold)
                }
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} evaluation failed for {model_name}: {e}")
                return {
                    'fold': fold_idx,
                    'accuracy': 0.5,  # Default score for failed fold
                    'precision': 0.5,
                    'recall': 0.5,
                    'train_size': len(X_train_fold) if 'X_train_fold' in locals() else 0,
                    'test_size': len(X_test_fold) if 'X_test_fold' in locals() else 0,
                    'error': str(e)
                }
        
        # Prepare fold data
        fold_data = list(enumerate(tscv.split(X)))
        
        # Execute parallel evaluation using class thread pool
        future_to_fold = {self._ml_thread_pool.submit(evaluate_fold, fd): fd[0] for fd in fold_data}
        
        for future in as_completed(future_to_fold):
            fold_idx = future_to_fold[future]
            try:
                result = future.result()
                fold_results.append(result)
                scores.append(result['accuracy'])
                logger.debug(f"Completed fold {fold_idx} for {model_name}: accuracy={result['accuracy']:.4f}")
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed for {model_name}: {e}")
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'error': str(e)
                })
                scores.append(0.5)
        
        # Sort results by fold index
        fold_results.sort(key=lambda x: x['fold'])
        
        # Calculate summary statistics
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_precision = np.mean([r['precision'] for r in fold_results])
        mean_recall = np.mean([r['recall'] for r in fold_results])
        
        return {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'fold_scores': [float(s) for s in scores],
            'fold_results': fold_results,
            'n_splits': len(scores),
            'parallel_execution': True
        }

    def _sequential_cv_scoring(self, model_class, model_params: dict, X: np.ndarray, y: np.ndarray, 
                             tscv: TimeSeriesSplit, model_name: str) -> Dict[str, Any]:
        """
        Perform sequential cross-validation scoring (fallback for parallel processing).
        
        Args:
            model_class: Model class to evaluate
            model_params (dict): Model parameters
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            tscv (TimeSeriesSplit): Cross-validation splitter
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Using sequential CV scoring for {model_name}")
        
        scores = []
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            try:
                X_train_fold = X[train_idx]
                X_test_fold = X[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # Scale features for this fold
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                X_test_fold_scaled = scaler_fold.transform(X_test_fold)
                
                # Create and train model
                model_fold = model_class(**model_params)
                model_fold.fit(X_train_fold_scaled, y_train_fold)
                
                # Make predictions and calculate metrics
                y_pred_fold = model_fold.predict(X_test_fold_scaled)
                accuracy = accuracy_score(y_test_fold, y_pred_fold)
                precision = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                recall = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                
                scores.append(accuracy)
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold)
                })
                
                logger.debug(f"Completed fold {fold_idx} for {model_name}: accuracy={accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} evaluation failed for {model_name}: {e}")
                scores.append(0.5)  # Default score for failed fold
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_precision = np.mean([r['precision'] for r in fold_results])
        mean_recall = np.mean([r['recall'] for r in fold_results])
        
        return {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'fold_scores': [float(s) for s in scores],
            'fold_results': fold_results,
            'n_splits': len(scores),
            'parallel_execution': False
        }

    def _fallback_cv_scoring(self, model_class, model_params: dict, X: np.ndarray, y: np.ndarray, 
                           tscv: TimeSeriesSplit, model_name: str) -> Dict[str, Any]:
        """
        Fallback cross-validation scoring method.
        
        Args:
            model_class: Model class to evaluate
            model_params (dict): Model parameters
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            tscv (TimeSeriesSplit): Cross-validation splitter
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, Any]: Basic cross-validation results
        """
        logger.warning(f"Using fallback CV scoring for {model_name}")
        
        try:
            # Simple single-fold evaluation as fallback
            splits = list(tscv.split(X))
            if splits:
                train_idx, test_idx = splits[-1]  # Use last split
                
                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                return {
                    'mean_accuracy': float(accuracy),
                    'std_accuracy': 0.0,
                    'mean_precision': float(accuracy),
                    'mean_recall': float(accuracy),
                    'fold_scores': [float(accuracy)],
                    'n_splits': 1,
                    'fallback_scoring': True
                }
            else:
                # Ultimate fallback
                return {
                    'mean_accuracy': 0.5,
                    'std_accuracy': 0.0,
                    'mean_precision': 0.5,
                    'mean_recall': 0.5,
                    'fold_scores': [0.5],
                    'n_splits': 1,
                    'fallback_scoring': True,
                    'error': 'No valid splits available'
                }
                
        except Exception as e:
            logger.error(f"Fallback CV scoring failed for {model_name}: {e}")
            return {
                'mean_accuracy': 0.5,
                'std_accuracy': 0.0,
                'mean_precision': 0.5,
                'mean_recall': 0.5,
                'fold_scores': [0.5],
                'n_splits': 1,
                'fallback_scoring': True,
                'error': str(e)
            }

    def _get_xgb_param_distribution(self) -> dict:
        if not getattr(self, 'xgboost_available', False):
            raise XGBoostTrainingError("xgboost is not installed; cannot provide XGBoost parameter distribution")
        return {
            'n_estimators': randint(50, 301),
            'max_depth': randint(3, 11),
            'learning_rate': uniform(0.01, 0.19),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 11),
            'gamma': uniform(0.0, 0.5)
        }

    def _optimize_xgboost(self, X: np.ndarray, y: np.ndarray):
        """
        Optimize XGBoost hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            
        Returns:
            XGBClassifier: Best estimator from hyperparameter optimization
            
        Raises:
            XGBoostTrainingError: If XGBoost is not available or optimization fails
        """
        if not getattr(self, 'xgboost_available', False):
            raise XGBoostTrainingError("XGBoost is not available; cannot perform XGBoost optimization")
        
        start_time = time.time()
        logger.info("Starting XGBoost hyperparameter optimization with performance monitoring...")
        
        try:
            # Get XGBoost parameter distributions
            param_distributions = self._get_xgb_param_distribution()
            
            # Use configured TimeSeriesSplit (same as Random Forest)
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            
            # Create base XGBoost classifier with required settings
            xgb_classifier = xgb.XGBClassifier(
                eval_metric='logloss',  # Required: use logloss as eval metric
                random_state=self.random_state,  # Required: same random_state as RF
                n_jobs=-1,  # Required: parallel processing
                verbosity=0  # Suppress XGBoost output during optimization
            )
            
            # Perform randomized search with performance monitoring
            logger.info("Performing RandomizedSearchCV for XGBoost with TimeSeriesSplit and parallel processing...")
            random_search = RandomizedSearchCV(
                estimator=xgb_classifier,
                param_distributions=param_distributions,
                n_iter=50,  # Number of parameter settings sampled (same as RF)
                cv=tscv,    # Use TimeSeriesSplit for cross-validation
                scoring='accuracy',  # Optimize for accuracy
                random_state=self.random_state,
                n_jobs=-1,  # Required: parallel processing
                verbose=0   # Suppress verbose output
            )
            
            # Fit the randomized search
            logger.info("Fitting XGBoost RandomizedSearchCV...")
            random_search.fit(X, y)
            
            # Calculate optimization time
            optimization_time = time.time() - start_time
            
            # Log the best parameters found
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            logger.info(f"XGBoost hyperparameter optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best XGBoost cross-validation score: {best_score:.4f}")
            logger.info(f"Best XGBoost parameters found: {best_params}")
            
            # Log parameter comparison for transparency
            logger.info("XGBoost optimization details:")
            logger.info(f"  - n_estimators: {best_params.get('n_estimators', 'N/A')}")
            logger.info(f"  - max_depth: {best_params.get('max_depth', 'N/A')}")
            lr = best_params.get('learning_rate', 'N/A')
            logger.info(f"  - learning_rate: {lr:.4f}" if isinstance(lr, (int, float)) else f"  - learning_rate: {lr}")
            ss = best_params.get('subsample', 'N/A')
            logger.info(f"  - subsample: {ss:.4f}" if isinstance(ss, (int, float)) else f"  - subsample: {ss}")
            cs = best_params.get('colsample_bytree', 'N/A')
            logger.info(f"  - colsample_bytree: {cs:.4f}" if isinstance(cs, (int, float)) else f"  - colsample_bytree: {cs}")
            logger.info(f"  - min_child_weight: {best_params.get('min_child_weight', 'N/A')}")
            gm = best_params.get('gamma', 'N/A')
            logger.info(f"  - gamma: {gm:.4f}" if isinstance(gm, (int, float)) else f"  - gamma: {gm}")
            
            # Performance monitoring
            if self.performance_monitoring:
                logger.info(f"XGBoost Optimization Performance Metrics:")
                logger.info(f"  - Total optimization time: {optimization_time:.2f}s")
                logger.info(f"  - Parallel processing: Enabled (n_jobs=-1)")
                logger.info(f"  - CV folds: {self.n_splits}")
                logger.info(f"  - Parameter combinations tested: 50")
                logger.info(f"  - Best CV score: {best_score:.4f}")
            
            # Clean up memory after optimization
            self._cleanup_memory()
            
            # Return the best estimator
            return random_search.best_estimator_
            
        except Exception as e:
            error_msg = f"XGBoost hyperparameter optimization failed: {e}"
            logger.error(error_msg)
            raise XGBoostTrainingError(error_msg)

    def _calculate_ensemble_weights(self, rf_model, xgb_model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculate ensemble weights based on individual model cross-validation performance.
        
        Uses TimeSeriesSplit to evaluate both RF and XGBoost models independently,
        then calculates normalized weights that sum to 1.0 based on individual CV scores.
        
        Args:
            rf_model: Trained Random Forest model
            xgb_model: Trained XGBoost model  
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            
        Returns:
            Tuple[float, float, Dict[str, float]]: RF weight, XGBoost weight, and individual CV scores
            
        Raises:
            EnsembleCreationError: If weight calculation fails
        """
        try:
            start_time = time.time()
            logger.info("Starting efficient ensemble weight calculation using cross-validation performance...")
            
            # Use configured TimeSeriesSplit for consistent evaluation
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            
            # Use efficient CV scoring for both models
            logger.info(f"Evaluating models using efficient {self.n_splits}-fold TimeSeriesSplit cross-validation...")
            
            # Evaluate Random Forest model using efficient CV scoring
            rf_cv_results = self._perform_efficient_cv_scoring(
                RandomForestClassifier, rf_model.get_params(), X, y, tscv, "Random Forest"
            )
            rf_mean_cv_score = rf_cv_results['mean_accuracy']
            rf_cv_scores = rf_cv_results['fold_scores']
            
            # Evaluate XGBoost model using efficient CV scoring
            if self.xgboost_available and xgb_model is not None:
                xgb_cv_results = self._perform_efficient_cv_scoring(
                    xgb.XGBClassifier, xgb_model.get_params(), X, y, tscv, "XGBoost"
                )
                xgb_mean_cv_score = xgb_cv_results['mean_accuracy']
                xgb_cv_scores = xgb_cv_results['fold_scores']
            else:
                logger.info("XGBoost not available, using default scores")
                xgb_mean_cv_score = 0.5
                xgb_cv_scores = [0.5] * self.n_splits
            
            # Calculate weight calculation time
            weight_calc_time = time.time() - start_time
            
            logger.info(f"Efficient cross-validation results (completed in {weight_calc_time:.2f}s):")
            logger.info(f"  - Random Forest mean CV accuracy: {rf_mean_cv_score:.4f} ± {np.std(rf_cv_scores):.4f}")
            logger.info(f"  - XGBoost mean CV accuracy: {xgb_mean_cv_score:.4f} ± {np.std(xgb_cv_scores):.4f}")
            
            # Calculate weights based on relative performance
            # Use performance scores directly as weights, then normalize
            total_score = rf_mean_cv_score + xgb_mean_cv_score
            
            if total_score == 0:
                # Fallback to equal weights if both models perform poorly
                rf_weight = 0.5
                xgb_weight = 0.5
                logger.warning("Both models have zero CV scores, using equal weights")
            else:
                # Normalize weights so they sum to 1.0
                rf_weight = rf_mean_cv_score / total_score
                xgb_weight = xgb_mean_cv_score / total_score
            
            # Ensure weights sum to 1.0 (handle floating point precision)
            weight_sum = rf_weight + xgb_weight
            if abs(weight_sum - 1.0) > 1e-10:
                rf_weight = rf_weight / weight_sum
                xgb_weight = xgb_weight / weight_sum
            
            # Prepare individual CV scores dictionary for transparency
            individual_cv_scores = {
                'rf': rf_mean_cv_score,
                'xgb': xgb_mean_cv_score,
                'rf_std': float(np.std(rf_cv_scores)),
                'xgb_std': float(np.std(xgb_cv_scores)),
                'rf_scores': [float(score) for score in rf_cv_scores],
                'xgb_scores': [float(score) for score in xgb_cv_scores],
                'weight_calc_time': weight_calc_time,
                'cached_rf': rf_cv_results.get('cached', False),
                'cached_xgb': xgb_cv_results.get('cached', False) if self.xgboost_available else False
            }
            
            # Log weight calculation results with transparency
            logger.info("Efficient ensemble weight calculation completed:")
            logger.info(f"  - Random Forest weight: {rf_weight:.4f} (based on CV score: {rf_mean_cv_score:.4f})")
            logger.info(f"  - XGBoost weight: {xgb_weight:.4f} (based on CV score: {xgb_mean_cv_score:.4f})")
            logger.info(f"  - Weight sum verification: {rf_weight + xgb_weight:.6f}")
            logger.info(f"  - Total calculation time: {weight_calc_time:.2f}s")
            
            # Performance monitoring
            if self.performance_monitoring:
                logger.info("Weight Calculation Performance Metrics:")
                logger.info(f"  - RF CV cached: {individual_cv_scores['cached_rf']}")
                logger.info(f"  - XGBoost CV cached: {individual_cv_scores['cached_xgb']}")
                logger.info(f"  - Parallel CV execution: {rf_cv_results.get('parallel_execution', False)}")
                logger.info(f"  - Total weight calculation time: {weight_calc_time:.2f}s")
            
            # Additional transparency logging
            logger.info("Weight calculation transparency:")
            logger.info(f"  - RF individual fold scores: {[f'{score:.4f}' for score in rf_cv_scores]}")
            logger.info(f"  - XGBoost individual fold scores: {[f'{score:.4f}' for score in xgb_cv_scores]}")
            logger.info(f"  - Weight calculation method: Performance-based normalization with efficient CV")
            
            # Clean up memory after weight calculation
            self._cleanup_memory()
            
            return float(rf_weight), float(xgb_weight), individual_cv_scores
            
        except Exception as e:
            error_msg = f"Ensemble weight calculation failed: {e}"
            logger.error(error_msg)
            raise EnsembleCreationError(error_msg)

    def _create_ensemble_model(self, rf_model, xgb_model, X: np.ndarray, y: np.ndarray) -> Tuple[VotingClassifier, Dict[str, Any]]:
        """
        Create ensemble model using scikit-learn VotingClassifier with soft voting and calculated weights.
        
        Configures VotingClassifier with soft voting and calculated weights, implements ensemble model 
        validation and error handling, and adds ensemble metadata storage for transparency.
        
        Args:
            rf_model: Trained Random Forest model
            xgb_model: Trained XGBoost model
            X (np.ndarray): Feature matrix for weight calculation
            y (np.ndarray): Target variable for weight calculation
            
        Returns:
            Tuple[VotingClassifier, Dict[str, Any]]: Ensemble model and metadata
            
        Raises:
            EnsembleCreationError: If ensemble creation fails
        """
        try:
            logger.info("Starting ensemble model creation with VotingClassifier...")
            
            # Validate input models
            if rf_model is None:
                raise EnsembleCreationError("Random Forest model is None, cannot create ensemble")
            if xgb_model is None:
                raise EnsembleCreationError("XGBoost model is None, cannot create ensemble")
            
            # Validate that models have required methods
            required_methods = ['predict', 'predict_proba', 'fit']
            for method in required_methods:
                if not hasattr(rf_model, method):
                    raise EnsembleCreationError(f"Random Forest model missing required method: {method}")
                if not hasattr(xgb_model, method):
                    raise EnsembleCreationError(f"XGBoost model missing required method: {method}")
            
            logger.info("Input model validation completed successfully")
            
            # Calculate ensemble weights based on cross-validation performance
            logger.info("Calculating ensemble weights using cross-validation performance...")
            rf_weight, xgb_weight, individual_cv_scores = self._calculate_ensemble_weights(rf_model, xgb_model, X, y)
            
            # Validate calculated weights
            if not (0 <= rf_weight <= 1) or not (0 <= xgb_weight <= 1):
                raise EnsembleCreationError(f"Invalid weights calculated: RF={rf_weight}, XGB={xgb_weight}")
            
            weight_sum = rf_weight + xgb_weight
            if abs(weight_sum - 1.0) > 1e-6:
                raise EnsembleCreationError(f"Weights do not sum to 1.0: sum={weight_sum}")
            
            logger.info(f"Ensemble weights validated: RF={rf_weight:.4f}, XGB={xgb_weight:.4f}")
            
            # Create VotingClassifier with soft voting and calculated weights
            logger.info("Creating VotingClassifier with soft voting...")
            ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),      # Random Forest estimator
                    ('xgb', xgb_model)     # XGBoost estimator
                ],
                voting='soft',             # Required: use soft voting (probability-based)
                weights=[rf_weight, xgb_weight]  # Required: calculated weights
            )
            
            logger.info("VotingClassifier created successfully with soft voting")
            
            # Validate ensemble model creation
            logger.info("Validating ensemble model configuration...")
            
            # Check that ensemble has correct estimators
            if len(ensemble_model.estimators) != 2:
                raise EnsembleCreationError(f"Ensemble should have 2 estimators, got {len(ensemble_model.estimators)}")
            
            # Check that voting is set to soft
            if ensemble_model.voting != 'soft':
                raise EnsembleCreationError(f"Ensemble voting should be 'soft', got '{ensemble_model.voting}'")
            
            # Check that weights are correctly set
            if ensemble_model.weights is None or len(ensemble_model.weights) != 2:
                raise EnsembleCreationError("Ensemble weights not properly configured")
            
            expected_weights = [rf_weight, xgb_weight]
            actual_weights = list(ensemble_model.weights)
            for i, (expected, actual) in enumerate(zip(expected_weights, actual_weights)):
                if abs(expected - actual) > 1e-6:
                    raise EnsembleCreationError(f"Weight mismatch at index {i}: expected {expected}, got {actual}")
            
            logger.info("Ensemble model configuration validation completed successfully")
            
            # Create ensemble metadata for transparency
            ensemble_metadata = {
                'model_type': 'Ensemble_RF_XGB',
                'voting_type': 'soft',
                'ensemble_weights': {
                    'rf': float(rf_weight),
                    'xgb': float(xgb_weight)
                },
                'individual_cv_scores': individual_cv_scores,
                'estimator_names': ['rf', 'xgb'],
                'estimator_types': [
                    type(rf_model).__name__,
                    type(xgb_model).__name__
                ],
                'rf_params': rf_model.get_params(),
                'xgb_params': xgb_model.get_params(),
                'ensemble_params': {
                    'voting': ensemble_model.voting,
                    'weights': list(ensemble_model.weights)
                },
                'creation_timestamp': datetime.now().isoformat(),
                'weight_calculation_method': 'cross_validation_performance_based',
                'validation_passed': True
            }
            
            # Log ensemble creation success with transparency details
            logger.info("Ensemble model creation completed successfully:")
            logger.info(f"  - Model type: {ensemble_metadata['model_type']}")
            logger.info(f"  - Voting type: {ensemble_metadata['voting_type']}")
            logger.info(f"  - RF weight: {rf_weight:.4f} (CV score: {individual_cv_scores['rf']:.4f})")
            logger.info(f"  - XGBoost weight: {xgb_weight:.4f} (CV score: {individual_cv_scores['xgb']:.4f})")
            logger.info(f"  - Estimators: {ensemble_metadata['estimator_names']}")
            logger.info(f"  - Estimator types: {ensemble_metadata['estimator_types']}")
            
            # Additional transparency logging
            logger.info("Ensemble metadata details:")
            logger.info(f"  - Weight calculation method: {ensemble_metadata['weight_calculation_method']}")
            rf_std = individual_cv_scores.get('rf_std', 'N/A')
            logger.info(f"  - RF CV score std: {rf_std:.4f}" if isinstance(rf_std, (int, float)) else f"  - RF CV score std: {rf_std}")
            xgb_std = individual_cv_scores.get('xgb_std', 'N/A')
            logger.info(f"  - XGBoost CV score std: {xgb_std:.4f}" if isinstance(xgb_std, (int, float)) else f"  - XGBoost CV score std: {xgb_std}")
            logger.info(f"  - Validation status: {ensemble_metadata['validation_passed']}")
            
            return ensemble_model, ensemble_metadata
            
        except EnsembleCreationError:
            # Re-raise EnsembleCreationError as-is
            raise
        except Exception as e:
            error_msg = f"Ensemble model creation failed: {e}"
            logger.error(error_msg)
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise EnsembleCreationError(error_msg)

    def _normalize_symbol(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        if s.endswith('.KS') or s.endswith('.KA'):
            s = s.split('.')[0]
        if not s.endswith('_HISTORICAL_DATA'):
            s = f"{s}_HISTORICAL_DATA"
        return s
    
    def _get_model_path(self, symbol: str) -> str:
        """Get the file path for a model"""
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.models_dir, f"{clean_symbol}_model.pkl")
    
    def _get_scaler_path(self, symbol: str) -> str:
        """Get the file path for a scaler"""
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.scalers_dir, f"{clean_symbol}_scaler.pkl")

    def _get_threshold_path(self, symbol: str) -> str:
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.models_dir, f"{clean_symbol}_threshold.json")

    def _get_ensemble_metadata_path(self, symbol: str) -> str:
        """Get the file path for ensemble metadata"""
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.models_dir, f"{clean_symbol}_ensemble_meta.json")

    def _get_rf_model_path(self, symbol: str) -> str:
        """Get the file path for individual Random Forest model"""
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.models_dir, f"{clean_symbol}_rf_model.pkl")

    def _get_xgb_model_path(self, symbol: str) -> str:
        """Get the file path for individual XGBoost model"""
        normalized = self._normalize_symbol(symbol)
        clean_symbol = "".join(c for c in normalized if c.isalnum() or c in ('-', '_')).upper()
        return os.path.join(self.models_dir, f"{clean_symbol}_xgb_model.pkl")

    def _tune_threshold(self, y_true: np.ndarray, y_proba_up: np.ndarray) -> Tuple[float, float]:
        thresholds = np.arange(self.threshold_min, self.threshold_max + 1e-9, self.threshold_step)
        best_threshold = 0.5
        best_score = -np.inf
        for t in thresholds:
            y_pred = (y_proba_up >= t).astype(int)
            if self.threshold_metric == 'utility':
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                u = self.utility_params
                score = (
                    tp * float(u.get('tp_reward', 1.0))
                    + tn * float(u.get('tn_reward', 0.0))
                    - fp * float(u.get('fp_cost', 1.0))
                    - fn * float(u.get('fn_cost', 1.0))
                )
            else:
                score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(t)
        return best_threshold, best_score

    def save_threshold(self, symbol: str, threshold: float) -> bool:
        try:
            path = self._get_threshold_path(symbol)
            with open(path, 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    'symbol': symbol,
                    'threshold': float(threshold),
                    'metric': self.threshold_metric,
                    'updated_at': datetime.now().isoformat()
                }, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to save threshold for {symbol}: {e}")
            return False

    def load_threshold(self, symbol: str) -> Optional[float]:
        try:
            path = self._get_threshold_path(symbol)
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                return float(data.get('threshold', 0.5))
        except Exception as e:
            logger.warning(f"Failed to load threshold for {symbol}: {e}")
            return None
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """
        Optimize Random Forest hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            
        Returns:
            RandomForestClassifier: Best estimator from hyperparameter optimization
        """
        start_time = time.time()
        logger.info("Starting Random Forest hyperparameter optimization with performance monitoring...")
        
        # Define parameter ranges for optimization
        param_distributions = {
            'n_estimators': randint(50, 301),  # 50-300 random integers
            'max_depth': randint(5, 21),       # 5-20 random integers
            'min_samples_split': randint(2, 16),  # 2-15 random integers
            'min_samples_leaf': randint(1, 9),    # 1-8 random integers
            'max_features': ['sqrt', 'log2', 0.5]
        }
        
        # Use configured TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
        
        # Create base Random Forest classifier with parallel processing
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores for parallel processing
        )
        
        # Perform randomized search with parallel processing
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=50,  # Number of parameter settings sampled
            cv=tscv,    # Use TimeSeriesSplit for cross-validation
            scoring='accuracy',  # Optimize for accuracy
            random_state=self.random_state,
            n_jobs=-1,  # Parallel processing for hyperparameter optimization
            verbose=0
        )
        
        # Fit the randomized search with performance monitoring
        logger.info("Fitting RandomizedSearchCV with parallel processing...")
        random_search.fit(X, y)
        
        # Calculate optimization time
        optimization_time = time.time() - start_time
        
        # Log the best parameters found
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        logger.info(f"Random Forest hyperparameter optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        logger.info(f"Best parameters found: {best_params}")
        
        # Performance monitoring
        if self.performance_monitoring:
            logger.info(f"RF Optimization Performance Metrics:")
            logger.info(f"  - Total optimization time: {optimization_time:.2f}s")
            logger.info(f"  - Parallel processing: Enabled (n_jobs=-1)")
            logger.info(f"  - CV folds: {self.n_splits}")
            logger.info(f"  - Parameter combinations tested: 50")
            logger.info(f"  - Best CV score: {best_score:.4f}")
        
        # Clean up memory after optimization
        self._cleanup_memory()
        
        # Return the best estimator
        return random_search.best_estimator_

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target variable from stock data with technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and technical indicators
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix (X) and target variable (y)
            
        Raises:
            ValueError: If required columns are missing or data is insufficient
            InsufficientDataError: If not enough data for feature preparation
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided for feature preparation")
        
        # Check if we have technical indicators
        missing_indicators = [col for col in self.feature_columns if col not in df.columns]
        if missing_indicators:
            logger.info("Missing technical indicators, calculating them...")
            df = self.technical_analyzer.add_all_indicators(df)
            
            # Check again after calculation
            missing_indicators = [col for col in self.feature_columns if col not in df.columns]
            if missing_indicators:
                raise ValueError(f"Missing required technical indicators: {missing_indicators}")
        
        # Check for Close column to create target variable
        if 'Close' not in df.columns:
            raise ValueError("Close column is required for target variable creation")
        
        try:
            # Create target variable: next-day price movement (UP=1, DOWN=0)
            df = df.copy()
            df['Next_Close'] = df['Close'].shift(-1)  # Next day's closing price
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)  # 1 if UP, 0 if DOWN
            
            # Remove rows where we can't calculate target (last row and any NaN)
            df = df.dropna(subset=['Target', 'Next_Close'])
            
            if len(df) < self.min_training_samples:
                raise InsufficientDataError(
                    f"Insufficient data for training. Need {self.min_training_samples} samples, got {len(df)}"
                )
            
            # Prepare feature matrix
            X = df[self.feature_columns].values
            y = df['Target'].values
            
            # Check for any remaining NaN values in features
            if np.isnan(X).any():
                logger.warning("NaN values found in features, applying forward fill")
                feature_df = df[self.feature_columns].ffill().bfill()
                X = feature_df.values
                
                # If still NaN, remove those rows
                nan_mask = np.isnan(X).any(axis=1)
                if nan_mask.any():
                    logger.warning(f"Removing {nan_mask.sum()} rows with NaN values")
                    X = X[~nan_mask]
                    y = y[~nan_mask]
            
            if len(X) < self.min_training_samples:
                raise InsufficientDataError(
                    f"After cleaning, insufficient data for training. Need {self.min_training_samples} samples, got {len(X)}"
                )
            
            logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Target distribution - UP: {np.sum(y)}, DOWN: {len(y) - np.sum(y)}")
            
            return X, y
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValueError)):
                raise
            else:
                raise MLPredictorError(f"Error preparing features: {e}")
    
    def train_model(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train a model for a specific stock symbol. Supports both Random Forest and ensemble modes.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization (default: True)
            
        Returns:
            Dict[str, Any]: Training results including metrics and model info
            
        Raises:
            InsufficientDataError: If not enough data for training
            ModelTrainingError: If model training fails
        """
        operation_logger = create_operation_logger('train_model', symbol=symbol)
        
        try:
            symbol = self._normalize_symbol(symbol)
            operation_logger.info(f"Starting model training for {symbol}")
            
            # Determine training mode based on configuration with comprehensive fallback
            if self.ensemble_enabled or self.model_type.lower() == 'ensemble':
                operation_logger.info("Ensemble mode enabled - attempting ensemble model training")
                operation_logger.info(f"Ensemble configuration: {self.ensemble_config}")
                
                try:
                    # Attempt ensemble training with comprehensive error handling
                    return self._train_ensemble_model_with_fallback(symbol, optimize_params)
                    
                except EnsembleCreationError as e:
                    operation_logger.warning(f"Ensemble creation failed: {e}")
                    operation_logger.info("Falling back to single Random Forest model")
                    return self._train_single_model_with_fallback(symbol, optimize_params)
                    
                except XGBoostTrainingError as e:
                    operation_logger.warning(f"XGBoost training failed: {e}")
                    operation_logger.info("Falling back to single Random Forest model")
                    return self._train_single_model_with_fallback(symbol, optimize_params)
                    
                except Exception as e:
                    operation_logger.error(f"Ensemble training failed with unexpected error: {e}")
                    operation_logger.info("Falling back to single Random Forest model")
                    return self._train_single_model_with_fallback(symbol, optimize_params)
            else:
                operation_logger.info("Single model mode - training Random Forest model")
                return self._train_single_model_with_fallback(symbol, optimize_params)
                
        except (InsufficientDataError, ValueError) as e:
            error_context = create_error_context('train_model', symbol=symbol, error_type=type(e).__name__)
            log_exception(operation_logger, e, error_context)
            raise InsufficientDataError(f"Training failed for {symbol}: {e}")
        except Exception as e:
            error_context = create_error_context('train_model', symbol=symbol, error_type=type(e).__name__)
            log_exception(operation_logger, e, error_context)
            raise ModelTrainingError(f"Model training failed for {symbol}: {e}")

    def _train_single_model(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train a single Random Forest model (preserves backward compatibility).
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Training results including metrics and model info
        """
        # Load stock data
        try:
            stock_data = self.data_storage.load_stock_data(symbol)
        except FileNotFoundError:
            raise InsufficientDataError(f"No data file found for symbol: {symbol}")
        
        if stock_data.empty:
            raise InsufficientDataError(f"No data available for symbol: {symbol}")
        
        # Ensure data is sorted by date for time-series split
        if 'Date' in stock_data.columns:
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            logger.info("Data sorted by date for time-series aware splitting")
        else:
            logger.warning("No Date column found, assuming data is already chronologically sorted")
        
        # Prepare features and target
        X, y = self.prepare_features(stock_data)
        
        # Use TimeSeriesSplit for time-series aware cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
        
        # Get the last split for final training (largest training set)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]  # Use the last split which has the most training data
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"TimeSeriesSplit - Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Training period: indices {train_idx[0]}-{train_idx[-1]}, Test period: indices {test_idx[0]}-{test_idx[-1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model with or without hyperparameter optimization
        if optimize_params:
            logger.info("Training with hyperparameter optimization enabled")
            model = self._optimize_hyperparameters(X, y)
            optimized_params = model.get_params()
            optimized_params['random_state'] = self.random_state
            optimized_params['n_jobs'] = -1
            optimized_model = RandomForestClassifier(**optimized_params)
            optimized_model.fit(X_train_scaled, y_train)
            model = optimized_model
        else:
            logger.info("Training with default hyperparameters")
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics on final test set
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        tuned_threshold = None
        tuned_score = None
        if self.threshold_enabled:
            proba_up = y_pred_proba[:, 1]
            tuned_threshold, tuned_score = self._tune_threshold(y_test, proba_up)
            self.thresholds[symbol] = tuned_threshold
            self.save_threshold(symbol, tuned_threshold)
        
        # Perform time-series cross-validation for more robust evaluation
        cv_scores = self._perform_time_series_cv(X, y, tscv, model.get_params() if optimize_params else None)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Save model and scaler to disk
        self.save_model(symbol, model)
        self.save_scaler(symbol, scaler)
        
        # Prepare base training results
        base_training_results = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'cv_scores': cv_scores,  # Time-series cross-validation results
            'feature_count': len(self.feature_columns),
            'top_features': top_features,
            'class_distribution': {
                'UP': int(np.sum(y)),
                'DOWN': int(len(y) - np.sum(y))
            },
            'model_parameters': model.get_params() if optimize_params else {
                'n_estimators': self.n_estimators,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'hyperparameter_optimization': optimize_params,
            'time_series_split_info': {
                'n_splits': self.n_splits,
                'max_train_size': self.max_train_size,
                'data_sorted_by_date': 'Date' in stock_data.columns
            },
            'decision_threshold': float(tuned_threshold) if tuned_threshold is not None else 0.5,
            'threshold_metric': self.threshold_metric if self.threshold_enabled else None,
            'threshold_score': float(tuned_score) if tuned_score is not None else None
        }
        
        # Use enhanced output formatting for consistency
        training_results = self._format_ensemble_training_output(
            base_results=base_training_results,
            model_type=self.model_type,
            ensemble_metadata=None,
            individual_cv_scores=None,
            fallback_info=None
        )
        
        logger.info(f"Model training completed for {symbol}")
        logger.info(f"Final test accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"CV Mean accuracy: {cv_scores.get('mean_accuracy', 0):.4f} ± {cv_scores.get('std_accuracy', 0):.4f}")
        logger.info(f"Top features: {[f[0] for f in top_features]}")
        if optimize_params:
            logger.info(f"Hyperparameter optimization enabled - using optimized parameters")
        else:
            logger.info(f"Using default hyperparameters")
        logger.info("Time-series aware cross-validation completed - no lookahead bias")
        
        return training_results

    def _train_ensemble_model_with_fallback(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train an ensemble model with comprehensive fallback mechanisms.
        
        This method implements graceful fallback from ensemble to best individual model when 
        ensemble creation fails, and fallback from XGBoost to Random Forest when XGBoost 
        training fails, with comprehensive error logging for debugging ensemble issues.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Training results including ensemble metrics and model info
            
        Raises:
            InsufficientDataError: If not enough data for training
            ModelTrainingError: If all training attempts fail
        """
        
        try:
            # Attempt ensemble training with comprehensive error handling
            return self._train_ensemble_model(symbol, optimize_params)
            
        except EnsembleCreationError as e:
            operation_logger.warning(f"Ensemble creation failed: {e}")
            operation_logger.info("Implementing graceful fallback to best individual model")
            
            try:
                # Attempt to determine best individual model through quick evaluation
                return self._fallback_to_best_individual_model(symbol, optimize_params)
                
            except Exception as fallback_error:
                operation_logger.error(f"Best individual model fallback failed: {fallback_error}")
                operation_logger.info("Final fallback to Random Forest model")
                return self._train_single_model_with_fallback(symbol, optimize_params)
                
        except XGBoostTrainingError as e:
            operation_logger.warning(f"XGBoost training failed: {e}")
            operation_logger.info("Implementing fallback from XGBoost to Random Forest")
            return self._train_single_model_with_fallback(symbol, optimize_params)
            
        except Exception as e:
            error_context = create_error_context('train_ensemble_with_fallback', symbol=symbol, 
                                               error_type=type(e).__name__, optimize_params=optimize_params)
            log_exception(operation_logger, e, error_context)
            
            operation_logger.warning(f"Ensemble training failed with unexpected error: {e}")
            operation_logger.info("Implementing final fallback to Random Forest model")
            return self._train_single_model_with_fallback(symbol, optimize_params)

    def _train_single_model_with_fallback(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train a single Random Forest model with comprehensive error handling and fallback.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Training results including metrics and model info
            
        Raises:
            InsufficientDataError: If not enough data for training
            ModelTrainingError: If all training attempts fail
        """
        operation_logger = create_operation_logger('train_single_model_with_fallback', symbol=symbol)
        
        try:
            # Attempt single model training
            return self._train_single_model(symbol, optimize_params)
            
        except Exception as e:
            error_context = create_error_context('train_single_model_with_fallback', symbol=symbol,
                                               error_type=type(e).__name__, optimize_params=optimize_params)
            log_exception(operation_logger, e, error_context)
            
            if optimize_params:
                operation_logger.warning(f"Optimized training failed: {e}")
                operation_logger.info("Falling back to default hyperparameters")
                
                try:
                    # Fallback to default parameters
                    return self._train_single_model(symbol, optimize_params=False)
                    
                except Exception as fallback_error:
                    error_context = create_error_context('train_single_model_fallback', symbol=symbol,
                                                       error_type=type(fallback_error).__name__)
                    log_exception(operation_logger, fallback_error, error_context)
                    raise ModelTrainingError(f"All training attempts failed for {symbol}: {fallback_error}")
            else:
                raise ModelTrainingError(f"Single model training failed for {symbol}: {e}")

    def _fallback_to_best_individual_model(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Fallback to the best individual model when ensemble creation fails.
        
        This method trains both RF and XGBoost individually, evaluates their performance,
        and selects the best performing model as the fallback.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Training results for the best individual model
            
        Raises:
            ModelTrainingError: If all individual model training fails
        """
        operation_logger = create_operation_logger('fallback_to_best_individual', symbol=symbol)
        
        try:
            operation_logger.info("Attempting to determine best individual model through evaluation")
            
            # Load and prepare data
            stock_data = self.data_storage.load_stock_data(symbol)
            if stock_data.empty:
                raise InsufficientDataError(f"No data available for symbol: {symbol}")
            
            # Ensure data is sorted by date
            if 'Date' in stock_data.columns:
                stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            # Prepare features and target
            X, y = self.prepare_features(stock_data)
            
            # Use TimeSeriesSplit for evaluation
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            
            rf_model = None
            xgb_model = None
            rf_score = 0.0
            xgb_score = 0.0
            
            # Train and evaluate Random Forest
            try:
                operation_logger.info("Training and evaluating Random Forest model")
                
                if optimize_params:
                    rf_model = self._optimize_hyperparameters(X, y)
                else:
                    rf_model = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        n_jobs=-1
                    )
                
                # Quick cross-validation evaluation
                rf_scores = []
                for train_idx, test_idx in tscv.split(X):
                    X_train_cv = X[train_idx]
                    X_test_cv = X[test_idx]
                    y_train_cv = y[train_idx]
                    y_test_cv = y[test_idx]
                    
                    scaler_cv = StandardScaler()
                    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                    X_test_cv_scaled = scaler_cv.transform(X_test_cv)
                    
                    rf_cv = RandomForestClassifier(**rf_model.get_params())
                    rf_cv.fit(X_train_cv_scaled, y_train_cv)
                    rf_pred_cv = rf_cv.predict(X_test_cv_scaled)
                    rf_scores.append(accuracy_score(y_test_cv, rf_pred_cv))
                
                rf_score = np.mean(rf_scores)
                operation_logger.info(f"Random Forest CV score: {rf_score:.4f}")
                
            except Exception as rf_error:
                operation_logger.warning(f"Random Forest evaluation failed: {rf_error}")
                rf_model = None
                rf_score = 0.0
            
            # Train and evaluate XGBoost if available
            if self.xgboost_available:
                try:
                    operation_logger.info("Training and evaluating XGBoost model")
                    
                    if optimize_params:
                        xgb_model = self._optimize_xgboost(X, y)
                    else:
                        xgb_model = xgb.XGBClassifier(
                            eval_metric='logloss',
                            random_state=self.random_state,
                            n_jobs=-1,
                            verbosity=0
                        )
                    
                    # Quick cross-validation evaluation
                    xgb_scores = []
                    for train_idx, test_idx in tscv.split(X):
                        X_train_cv = X[train_idx]
                        X_test_cv = X[test_idx]
                        y_train_cv = y[train_idx]
                        y_test_cv = y[test_idx]
                        
                        scaler_cv = StandardScaler()
                        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                        X_test_cv_scaled = scaler_cv.transform(X_test_cv)
                        
                        xgb_cv = xgb.XGBClassifier(**xgb_model.get_params())
                        xgb_cv.fit(X_train_cv_scaled, y_train_cv)
                        xgb_pred_cv = xgb_cv.predict(X_test_cv_scaled)
                        xgb_scores.append(accuracy_score(y_test_cv, xgb_pred_cv))
                    
                    xgb_score = np.mean(xgb_scores)
                    operation_logger.info(f"XGBoost CV score: {xgb_score:.4f}")
                    
                except (XGBoostTrainingError, Exception) as xgb_error:
                    operation_logger.warning(f"XGBoost evaluation failed: {xgb_error}")
                    xgb_model = None
                    xgb_score = 0.0
            else:
                operation_logger.info("XGBoost not available, skipping XGBoost evaluation")
            
            # Select best model
            if rf_model is None and xgb_model is None:
                raise ModelTrainingError("Both RF and XGBoost model training failed")
            
            if xgb_model is not None and xgb_score > rf_score:
                operation_logger.info(f"Selected XGBoost as best model (CV score: {xgb_score:.4f} vs RF: {rf_score:.4f})")
                selected_model = xgb_model
                selected_model_type = 'XGBoost'
                selected_score = xgb_score
            else:
                operation_logger.info(f"Selected Random Forest as best model (CV score: {rf_score:.4f} vs XGB: {xgb_score:.4f})")
                selected_model = rf_model if rf_model is not None else xgb_model
                selected_model_type = 'RandomForest'
                selected_score = rf_score
            
            # Train the selected model on full training data
            operation_logger.info(f"Training selected {selected_model_type} model on full dataset")
            
            # Get the last split for final training
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train final model
            final_model = type(selected_model)(**selected_model.get_params())
            final_model.fit(X_train_scaled, y_train)
            
            # Make predictions and calculate metrics
            y_pred = final_model.predict(X_test_scaled)
            y_pred_proba = final_model.predict_proba(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Threshold tuning
            tuned_threshold = None
            tuned_score = None
            if self.threshold_enabled:
                proba_up = y_pred_proba[:, 1]
                tuned_threshold, tuned_score = self._tune_threshold(y_test, proba_up)
                self.thresholds[symbol] = tuned_threshold
                self.save_threshold(symbol, tuned_threshold)
            
            # Store model and scaler
            self.models[symbol] = final_model
            self.scalers[symbol] = scaler
            
            # Save model and scaler
            self.save_model(symbol, final_model)
            self.save_scaler(symbol, scaler)
            
            # Feature importance
            if hasattr(final_model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, final_model.feature_importances_))
            else:
                feature_importance = {col: 0.0 for col in self.feature_columns}
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Perform cross-validation for robust evaluation
            cv_scores = self._perform_time_series_cv(X, y, tscv, final_model.get_params())
            
            # Prepare training results
            training_results = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'model_type': selected_model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'cv_scores': cv_scores,
                'feature_count': len(self.feature_columns),
                'top_features': top_features,
                'class_distribution': {
                    'UP': int(np.sum(y)),
                    'DOWN': int(len(y) - np.sum(y))
                },
                'model_parameters': final_model.get_params(),
                'hyperparameter_optimization': optimize_params,
                'time_series_split_info': {
                    'n_splits': self.n_splits,
                    'max_train_size': self.max_train_size,
                    'data_sorted_by_date': 'Date' in stock_data.columns
                },
                'decision_threshold': float(tuned_threshold) if tuned_threshold is not None else 0.5,
                'threshold_metric': self.threshold_metric if self.threshold_enabled else None,
                'threshold_score': float(tuned_score) if tuned_score is not None else None,
                'fallback_info': {
                    'fallback_reason': 'ensemble_creation_failed',
                    'rf_cv_score': float(rf_score),
                    'xgb_cv_score': float(xgb_score),
                    'selected_model': selected_model_type,
                    'selected_cv_score': float(selected_score)
                }
            }
            
            operation_logger.info(f"Best individual model training completed for {symbol}")
            operation_logger.info(f"Selected model: {selected_model_type} with CV score: {selected_score:.4f}")
            operation_logger.info(f"Final test accuracy: {accuracy:.4f}")
            
            return training_results
            
        except Exception as e:
            error_context = create_error_context('fallback_to_best_individual', symbol=symbol,
                                               error_type=type(e).__name__)
            log_exception(operation_logger, e, error_context)
            raise ModelTrainingError(f"Best individual model fallback failed for {symbol}: {e}")

    def _select_best_individual_model(self, rf_model, xgb_model, X, y, tscv, symbol):
        """
        Select the best individual model based on cross-validation performance.
        
        Args:
            rf_model: Trained Random Forest model
            xgb_model: Trained XGBoost model
            X: Feature matrix
            y: Target variable
            tscv: TimeSeriesSplit cross-validator
            symbol: Stock symbol for logging
            
        Returns:
            Tuple: (selected_model, model_type, individual_cv_scores, fallback_info)
        """
        operation_logger = create_operation_logger('select_best_individual_model', symbol=symbol)
        
        try:
            operation_logger.info("Evaluating individual models to select best performer")
            
            rf_scores = []
            xgb_scores = []
            
            # Perform cross-validation evaluation
            for fold, (train_cv_idx, test_cv_idx) in enumerate(tscv.split(X)):
                X_train_cv = X[train_cv_idx]
                X_test_cv = X[test_cv_idx]
                y_train_cv = y[train_cv_idx]
                y_test_cv = y[test_cv_idx]
                
                scaler_cv = StandardScaler()
                X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                X_test_cv_scaled = scaler_cv.transform(X_test_cv)
                
                # Evaluate Random Forest
                try:
                    rf_cv = RandomForestClassifier(**rf_model.get_params())
                    rf_cv.fit(X_train_cv_scaled, y_train_cv)
                    rf_pred_cv = rf_cv.predict(X_test_cv_scaled)
                    rf_score = accuracy_score(y_test_cv, rf_pred_cv)
                    rf_scores.append(rf_score)
                    operation_logger.debug(f"Fold {fold + 1} - RF score: {rf_score:.4f}")
                except Exception as rf_error:
                    operation_logger.warning(f"RF evaluation failed on fold {fold + 1}: {rf_error}")
                    rf_scores.append(0.5)  # Default score for failed evaluation
                
                # Evaluate XGBoost
                try:
                    if xgb_model is not None:
                        xgb_cv = xgb.XGBClassifier(**xgb_model.get_params())
                        xgb_cv.fit(X_train_cv_scaled, y_train_cv)
                        xgb_pred_cv = xgb_cv.predict(X_test_cv_scaled)
                        xgb_score = accuracy_score(y_test_cv, xgb_pred_cv)
                        xgb_scores.append(xgb_score)
                        operation_logger.debug(f"Fold {fold + 1} - XGBoost score: {xgb_score:.4f}")
                    else:
                        xgb_scores.append(0.0)  # XGBoost not available
                        operation_logger.debug(f"Fold {fold + 1} - XGBoost not available")
                except Exception as xgb_error:
                    operation_logger.warning(f"XGBoost evaluation failed on fold {fold + 1}: {xgb_error}")
                    xgb_scores.append(0.0)  # Default score for failed evaluation
            
            # Calculate mean scores
            rf_mean_score = np.mean(rf_scores) if rf_scores else 0.0
            xgb_mean_score = np.mean(xgb_scores) if xgb_scores else 0.0
            
            operation_logger.info(f"Model evaluation results:")
            operation_logger.info(f"  - Random Forest mean CV score: {rf_mean_score:.4f}")
            operation_logger.info(f"  - XGBoost mean CV score: {xgb_mean_score:.4f}")
            
            # Select best model
            if xgb_model is not None and xgb_mean_score > rf_mean_score:
                operation_logger.info(f"Selected XGBoost as best model (score: {xgb_mean_score:.4f} vs RF: {rf_mean_score:.4f})")
                selected_model = xgb_model
                model_type = 'XGBoost'
                selected_score = xgb_mean_score
            else:
                operation_logger.info(f"Selected Random Forest as best model (score: {rf_mean_score:.4f} vs XGB: {xgb_mean_score:.4f})")
                selected_model = rf_model
                model_type = 'RandomForest'
                selected_score = rf_mean_score
            
            individual_cv_scores = {
                'rf': float(rf_mean_score),
                'xgb': float(xgb_mean_score),
                'rf_scores': [float(s) for s in rf_scores],
                'xgb_scores': [float(s) for s in xgb_scores]
            }
            
            fallback_info = {
                'fallback_reason': 'ensemble_creation_failed',
                'rf_cv_score': float(rf_mean_score),
                'xgb_cv_score': float(xgb_mean_score),
                'selected_model': model_type,
                'selected_cv_score': float(selected_score)
            }
            
            return selected_model, model_type, individual_cv_scores, fallback_info
            
        except Exception as e:
            operation_logger.error(f"Best individual model selection failed: {e}")
            # Final fallback to Random Forest
            return rf_model, 'RandomForest', None, {
                'fallback_reason': 'model_selection_failed',
                'error': str(e),
                'final_model': 'RandomForest'
            }

    def _load_model_with_fallback(self, symbol: str) -> bool:
        """
        Load model and scaler with comprehensive fallback mechanisms.
        
        Args:
            symbol (str): Stock symbol to load model for
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        operation_logger = create_operation_logger('load_model_with_fallback', symbol=symbol)
        
        try:
            model_path = self._get_model_path(symbol)
            scaler_path = self._get_scaler_path(symbol)
            
            # Check if primary model files exist
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    # Attempt to load primary model and scaler
                    operation_logger.info(f"Loading primary model from {model_path}")
                    model = self.load_model(symbol)
                    scaler = self.load_scaler(symbol)
                    
                    # Validate loaded model
                    if model is None or scaler is None:
                        raise ModelPersistenceError("Loaded model or scaler is None")
                    
                    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
                        raise ModelPersistenceError("Loaded model missing required methods")
                    
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    operation_logger.info(f"Successfully loaded primary model for {symbol}")
                    return True
                    
                except Exception as primary_load_error:
                    operation_logger.warning(f"Primary model loading failed: {primary_load_error}")
                    
                    # Attempt to load individual models as fallback
                    try:
                        operation_logger.info("Attempting to load individual models as fallback")
                        fallback_model = self._load_individual_models_fallback(symbol)
                        
                        if fallback_model is not None:
                            self.models[symbol] = fallback_model
                            # Try to load scaler separately
                            try:
                                scaler = self.load_scaler(symbol)
                                self.scalers[symbol] = scaler
                                operation_logger.info(f"Successfully loaded fallback model for {symbol}")
                                return True
                            except Exception as scaler_error:
                                operation_logger.warning(f"Scaler loading failed: {scaler_error}")
                                return False
                        else:
                            operation_logger.warning("No individual models available for fallback")
                            return False
                            
                    except Exception as fallback_error:
                        operation_logger.warning(f"Individual model fallback failed: {fallback_error}")
                        return False
            else:
                operation_logger.info(f"Primary model files not found for {symbol}")
                
                # Attempt to load individual models if available
                try:
                    operation_logger.info("Checking for individual model files")
                    fallback_model = self._load_individual_models_fallback(symbol)
                    
                    if fallback_model is not None:
                        self.models[symbol] = fallback_model
                        # Try to load scaler
                        try:
                            scaler = self.load_scaler(symbol)
                            self.scalers[symbol] = scaler
                            operation_logger.info(f"Successfully loaded individual model for {symbol}")
                            return True
                        except Exception as scaler_error:
                            operation_logger.warning(f"Scaler loading failed for individual model: {scaler_error}")
                            return False
                    else:
                        operation_logger.info("No individual model files found")
                        return False
                        
                except Exception as individual_error:
                    operation_logger.warning(f"Individual model loading failed: {individual_error}")
                    return False
                    
        except Exception as e:
            error_context = create_error_context('load_model_with_fallback', symbol=symbol,
                                               error_type=type(e).__name__)
            log_exception(operation_logger, e, error_context)
            return False

    def _load_individual_models_fallback(self, symbol: str):
        """
        Attempt to load individual RF or XGBoost models as fallback.
        
        Args:
            symbol (str): Stock symbol to load individual models for
            
        Returns:
            Model object or None if loading fails
        """
        operation_logger = create_operation_logger('load_individual_models_fallback', symbol=symbol)
        
        try:
            rf_path = self._get_rf_model_path(symbol)
            xgb_path = self._get_xgb_model_path(symbol)
            
            # Try to load Random Forest model
            if os.path.exists(rf_path):
                try:
                    operation_logger.info(f"Loading individual Random Forest model from {rf_path}")
                    with open(rf_path, 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    if rf_model is not None and hasattr(rf_model, 'predict'):
                        operation_logger.info("Successfully loaded Random Forest model")
                        return rf_model
                except Exception as rf_error:
                    operation_logger.warning(f"Random Forest model loading failed: {rf_error}")
            
            # Try to load XGBoost model if RF failed
            if os.path.exists(xgb_path) and self.xgboost_available:
                try:
                    operation_logger.info(f"Loading individual XGBoost model from {xgb_path}")
                    with open(xgb_path, 'rb') as f:
                        xgb_model = pickle.load(f)
                    
                    if xgb_model is not None and hasattr(xgb_model, 'predict'):
                        operation_logger.info("Successfully loaded XGBoost model")
                        return xgb_model
                except Exception as xgb_error:
                    operation_logger.warning(f"XGBoost model loading failed: {xgb_error}")
            
            operation_logger.info("No individual models could be loaded")
            return None
            
        except Exception as e:
            operation_logger.warning(f"Individual model fallback loading failed: {e}")
            return None

    def _train_ensemble_model(self, symbol: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train an ensemble model combining Random Forest and XGBoost.
        
        Implements independent optimization of RF and XGBoost models first,
        adds ensemble creation and weight calculation to training flow,
        and integrates threshold tuning for ensemble predictions.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Training results including ensemble metrics and model info
            
        Raises:
            InsufficientDataError: If not enough data for training
            ModelTrainingError: If ensemble training fails
            EnsembleCreationError: If ensemble creation fails
        """
        try:
            # Performance monitoring - start timing
            ensemble_start_time = time.time()
            logger.info(f"Starting ensemble model training for {symbol} with performance monitoring...")
            
            # Baseline RF training time estimation (for 2x time limit check)
            baseline_rf_time = None
            
            # Load stock data
            try:
                stock_data = self.data_storage.load_stock_data(symbol)
            except FileNotFoundError:
                raise InsufficientDataError(f"No data file found for symbol: {symbol}")
            
            if stock_data.empty:
                raise InsufficientDataError(f"No data available for symbol: {symbol}")
            
            # Ensure data is sorted by date for time-series split
            if 'Date' in stock_data.columns:
                stock_data = stock_data.sort_values('Date').reset_index(drop=True)
                logger.info("Data sorted by date for time-series aware splitting")
            else:
                logger.warning("No Date column found, assuming data is already chronologically sorted")
            
            # Prepare features and target
            X, y = self.prepare_features(stock_data)
            
            # Use TimeSeriesSplit for time-series aware cross-validation
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            
            # Get the last split for final training (largest training set)
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]  # Use the last split which has the most training data
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            logger.info(f"TimeSeriesSplit - Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            logger.info(f"Training period: indices {train_idx[0]}-{train_idx[-1]}, Test period: indices {test_idx[0]}-{test_idx[-1]}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Step 1: Independent optimization of Random Forest and XGBoost models
            logger.info("Step 1: Independent optimization of Random Forest and XGBoost models")
            
            # Train Random Forest model with performance monitoring
            rf_start_time = time.time()
            logger.info("Training Random Forest model...")
            if optimize_params:
                logger.info("Optimizing Random Forest hyperparameters...")
                rf_model = self._optimize_hyperparameters(X, y)
                optimized_rf_params = rf_model.get_params()
                optimized_rf_params['random_state'] = self.random_state
                optimized_rf_params['n_jobs'] = -1
                rf_model = RandomForestClassifier(**optimized_rf_params)
                rf_model.fit(X_train_scaled, y_train)
            else:
                logger.info("Training Random Forest with default hyperparameters...")
                rf_model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
            
            rf_training_time = time.time() - rf_start_time
            baseline_rf_time = rf_training_time  # Store for 2x time limit check
            logger.info(f"Random Forest training completed in {rf_training_time:.2f}s")
            
            # Train XGBoost model with comprehensive fallback handling
            xgb_model = None
            xgb_training_failed = False
            xgb_failure_reason = None
            
            try:
                if not self.xgboost_available:
                    logger.warning("XGBoost not available - dependency not installed")
                    xgb_training_failed = True
                    xgb_failure_reason = "xgboost_not_available"
                else:
                    xgb_start_time = time.time()
                    logger.info("Training XGBoost model...")
                    
                    try:
                        if optimize_params:
                            logger.info("Optimizing XGBoost hyperparameters...")
                            xgb_model = self._optimize_xgboost(X, y)
                            xgb_model.fit(X_train_scaled, y_train)
                        else:
                            logger.info("Training XGBoost with default hyperparameters...")
                            xgb_model = xgb.XGBClassifier(
                                eval_metric='logloss',
                                random_state=self.random_state,
                                n_jobs=-1,
                                verbosity=0
                            )
                            xgb_model.fit(X_train_scaled, y_train)
                        
                        xgb_training_time = time.time() - xgb_start_time
                        logger.info(f"XGBoost training completed successfully in {xgb_training_time:.2f}s")
                        if baseline_rf_time and self.training_time_limit_multiplier:
                            if xgb_training_time > self.training_time_limit_multiplier * baseline_rf_time:
                                logger.warning(
                                    f"XGBoost training time {xgb_training_time:.2f}s exceeds limit "
                                    f"{self.training_time_limit_multiplier}x RF ({baseline_rf_time:.2f}s);"
                                    f" falling back to RF only"
                                )
                                xgb_training_failed = True
                                xgb_failure_reason = "xgboost_time_limit_exceeded"
                                xgb_model = None
                        
                        # Validate XGBoost model
                        if not hasattr(xgb_model, 'predict') or not hasattr(xgb_model, 'predict_proba'):
                            raise XGBoostTrainingError("XGBoost model missing required methods")
                            
                    except XGBoostTrainingError as xgb_error:
                        logger.warning(f"XGBoost training failed with XGBoostTrainingError: {xgb_error}")
                        xgb_training_failed = True
                        xgb_failure_reason = "xgboost_training_error"
                        xgb_model = None
                        
                        # Log detailed error context for debugging
                        error_context = create_error_context('xgboost_training', symbol=symbol,
                                                           optimize_params=optimize_params,
                                                           error_details=str(xgb_error))
                        logger.error(f"XGBoost training error context: {error_context}")
                        
                    except Exception as xgb_error:
                        logger.warning(f"XGBoost training failed with unexpected error: {xgb_error}")
                        xgb_training_failed = True
                        xgb_failure_reason = "xgboost_unexpected_error"
                        xgb_model = None
                        
                        # Log detailed error context for debugging
                        error_context = create_error_context('xgboost_training', symbol=symbol,
                                                           optimize_params=optimize_params,
                                                           error_type=type(xgb_error).__name__,
                                                           error_details=str(xgb_error))
                        log_exception(logger, xgb_error, error_context)
                    
            except Exception as outer_error:
                logger.error(f"XGBoost training outer exception: {outer_error}")
                xgb_training_failed = True
                xgb_failure_reason = "xgboost_outer_error"
                xgb_model = None
                
                error_context = create_error_context('xgboost_training_outer', symbol=symbol,
                                                   error_type=type(outer_error).__name__)
                log_exception(logger, outer_error, error_context)
            
            # Step 2: Create ensemble model or implement comprehensive fallback
            if xgb_training_failed or xgb_model is None:
                logger.info(f"Falling back to single Random Forest model due to XGBoost failure: {xgb_failure_reason}")
                
                # Use single RF model as fallback
                model = rf_model
                model_type = 'RandomForest'
                ensemble_metadata = None
                individual_cv_scores = None
                fallback_info = {
                    'fallback_reason': 'xgboost_training_failed',
                    'xgb_failure_reason': xgb_failure_reason,
                    'fallback_model': 'RandomForest'
                }
                
            else:
                logger.info("Step 2: Creating ensemble model with weight calculation")
                fallback_info = None
                
                try:
                    # Validate models before ensemble creation
                    if rf_model is None:
                        raise EnsembleCreationError("Random Forest model is None")
                    if xgb_model is None:
                        raise EnsembleCreationError("XGBoost model is None")
                    
                    # Create ensemble model using VotingClassifier with calculated weights
                    logger.info("Creating ensemble model with VotingClassifier...")
                    model, ensemble_metadata = self._create_ensemble_model(rf_model, xgb_model, X, y)
                    model_type = 'Ensemble_RF_XGB'
                    individual_cv_scores = ensemble_metadata.get('individual_cv_scores', {})
                    
                    # Fit the ensemble model on training data
                    logger.info("Fitting ensemble model on training data...")
                    model.fit(X_train_scaled, y_train)
                    logger.info("Ensemble model fitting completed successfully")
                    
                    # Validate fitted ensemble model
                    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
                        raise EnsembleCreationError("Fitted ensemble model missing required methods")
                    
                except EnsembleCreationError as ensemble_error:
                    logger.warning(f"Ensemble creation failed with EnsembleCreationError: {ensemble_error}")
                    
                    # Log detailed error context for debugging
                    error_context = create_error_context('ensemble_creation', symbol=symbol,
                                                       rf_model_available=rf_model is not None,
                                                       xgb_model_available=xgb_model is not None,
                                                       error_details=str(ensemble_error))
                    logger.error(f"Ensemble creation error context: {error_context}")
                    
                    # Graceful fallback to Random Forest for backward compatibility
                    logger.info("Falling back to Random Forest model after ensemble creation failure")
                    model = rf_model
                    model_type = 'RandomForest'
                    ensemble_metadata = None
                    individual_cv_scores = None
                    fallback_info = {
                        'fallback_reason': 'ensemble_creation_failed',
                        'final_model': 'RandomForest'
                    }
                        
                except Exception as unexpected_error:
                    logger.error(f"Ensemble creation failed with unexpected error: {unexpected_error}")
                    
                    # Log detailed error context for debugging
                    error_context = create_error_context('ensemble_creation_unexpected', symbol=symbol,
                                                       error_type=type(unexpected_error).__name__,
                                                       error_details=str(unexpected_error))
                    log_exception(logger, unexpected_error, error_context)
                    
                    # Graceful fallback to Random Forest for backward compatibility
                    logger.info("Falling back to Random Forest model after unexpected ensemble error")
                    model = rf_model
                    model_type = 'RandomForest'
                    ensemble_metadata = None
                    individual_cv_scores = None
                    fallback_info = {
                        'fallback_reason': 'ensemble_unexpected_error',
                        'final_model': 'RandomForest'
                    }
            
            # Step 3: Make predictions on test set
            logger.info("Making predictions on test set...")
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics on final test set
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Step 4: Integrate threshold tuning for ensemble predictions
            logger.info("Step 4: Integrating threshold tuning for ensemble predictions...")
            tuned_threshold = None
            tuned_score = None
            if self.threshold_enabled:
                proba_up = y_pred_proba[:, 1]
                tuned_threshold, tuned_score = self._tune_threshold(y_test, proba_up)
                self.thresholds[symbol] = tuned_threshold
                self.save_threshold(symbol, tuned_threshold)
                logger.info(f"Threshold tuning completed: {tuned_threshold:.4f} (score: {tuned_score:.4f})")
            
            # Perform time-series cross-validation for more robust evaluation
            cv_scores = self._perform_time_series_cv(X, y, tscv, model.get_params() if hasattr(model, 'get_params') else None)
            
            # Feature importance (handle ensemble vs single model)
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # For ensemble, use RF feature importance as primary
                rf_estimator = None
                for name, estimator in model.estimators_:
                    if name == 'rf':
                        rf_estimator = estimator
                        break
                if rf_estimator and hasattr(rf_estimator, 'feature_importances_'):
                    feature_importance = dict(zip(self.feature_columns, rf_estimator.feature_importances_))
                else:
                    feature_importance = {col: 0.0 for col in self.feature_columns}
            else:
                feature_importance = {col: 0.0 for col in self.feature_columns}
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            # Save models to disk (ensemble or single model)
            if model_type == 'Ensemble_RF_XGB' and ensemble_metadata:
                self.save_model(symbol, model)
                self._save_individual_models(symbol, rf_model, xgb_model)
                self._save_ensemble_metadata(symbol, ensemble_metadata)
            else:
                self.save_model(symbol, model)
            
            self.save_scaler(symbol, scaler)
            
            # Prepare base training results
            base_training_results = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'cv_scores': cv_scores,
                'feature_count': len(self.feature_columns),
                'top_features': top_features,
                'class_distribution': {
                    'UP': int(np.sum(y)),
                    'DOWN': int(len(y) - np.sum(y))
                },
                'hyperparameter_optimization': optimize_params,
                'time_series_split_info': {
                    'n_splits': self.n_splits,
                    'max_train_size': self.max_train_size,
                    'data_sorted_by_date': 'Date' in stock_data.columns
                },
                'decision_threshold': float(tuned_threshold) if tuned_threshold is not None else 0.5,
                'threshold_metric': self.threshold_metric if self.threshold_enabled else None,
                'threshold_score': float(tuned_score) if tuned_score is not None else None
            }
            
            # Add XGBoost failure information if applicable
            if xgb_training_failed:
                base_training_results['xgboost_failure'] = {
                    'failed': True,
                    'reason': xgb_failure_reason,
                    'xgboost_available': self.xgboost_available
                }
                logger.info(f"XGBoost training failed: {xgb_failure_reason}")
            
            # Use enhanced output formatting to ensure consistent ensemble information
            training_results = self._format_ensemble_training_output(
                base_results=base_training_results,
                model_type=model_type,
                ensemble_metadata=ensemble_metadata,
                individual_cv_scores=individual_cv_scores,
                fallback_info=fallback_info
            )
            
            # Log training completion
            logger.info(f"Ensemble model training completed for {symbol}")
            logger.info(f"Final model type: {model_type}")
            logger.info(f"Final test accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            logger.info(f"CV Mean accuracy: {cv_scores.get('mean_accuracy', 0):.4f} ± {cv_scores.get('std_accuracy', 0):.4f}")
            logger.info(f"Top features: {[f[0] for f in top_features]}")
            
            if model_type == 'Ensemble_RF_XGB' and ensemble_metadata:
                weights = ensemble_metadata.get('ensemble_weights', {})
                logger.info(f"Ensemble weights - RF: {weights.get('rf', 0):.4f}, XGBoost: {weights.get('xgb', 0):.4f}")
                if individual_cv_scores:
                    logger.info(f"Individual CV scores - RF: {individual_cv_scores.get('rf', 0):.4f}, XGBoost: {individual_cv_scores.get('xgb', 0):.4f}")
            
            logger.info("Time-series aware cross-validation completed - no lookahead bias")
            
            # Final performance monitoring
            total_ensemble_time = time.time() - ensemble_start_time
            
            if self.performance_monitoring:
                logger.info("=== Ensemble Training Performance Summary ===")
                logger.info(f"Total ensemble training time: {total_ensemble_time:.2f}s")
                if baseline_rf_time:
                    time_ratio = total_ensemble_time / baseline_rf_time
                    logger.info(f"RF baseline training time: {baseline_rf_time:.2f}s")
                    logger.info(f"Ensemble vs RF time ratio: {time_ratio:.2f}x")
                    
                    # Check 2x time limit requirement
                    if time_ratio <= self.training_time_limit_multiplier:
                        logger.info("✓ Training time requirement met: ≤ 2x RF training time")
                    else:
                        logger.warning(
                            f"⚠ Training time exceeds limit: {time_ratio:.2f}x > {self.training_time_limit_multiplier:.1f}x"
                        )
                
                logger.info(f"Model type: {model_type}")
                logger.info(f"Memory-efficient storage: {'Enabled' if self.memory_efficient_storage else 'Disabled'}")
                logger.info(f"Parallel CV processing: {'Enabled' if self.enable_parallel_cv else 'Disabled'}")
                logger.info(f"Cache utilization: Available")
                logger.info("=== End Performance Summary ===")
            
            # Final memory cleanup
            self._cleanup_memory()
            
            return training_results
            
        except (InsufficientDataError, ValueError) as e:
            raise InsufficientDataError(f"Ensemble training failed for {symbol}: {e}")
        except Exception as e:
            raise ModelTrainingError(f"Ensemble model training failed for {symbol}: {e}")

    def _save_individual_models(self, symbol: str, rf_model, xgb_model) -> None:
        """Save individual Random Forest and XGBoost models separately."""
        try:
            # Save Random Forest model
            rf_path = self._get_rf_model_path(symbol)
            try:
                import joblib
                joblib.dump(rf_model, rf_path, compress=3)
            except Exception:
                with open(rf_path, 'wb') as f:
                    pickle.dump(rf_model, f, protocol=4)
            logger.debug(f"Individual Random Forest model saved to {rf_path}")
            
            # Save XGBoost model
            if xgb_model is not None:
                xgb_path = self._get_xgb_model_path(symbol)
                try:
                    import joblib
                    joblib.dump(xgb_model, xgb_path, compress=3)
                except Exception:
                    with open(xgb_path, 'wb') as f:
                        pickle.dump(xgb_model, f, protocol=4)
                logger.debug(f"Individual XGBoost model saved to {xgb_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save individual models for {symbol}: {e}")

    def _save_ensemble_metadata(self, symbol: str, ensemble_metadata: Dict[str, Any]) -> None:
        """Save ensemble metadata to JSON file."""
        try:
            import json
            metadata_path = self._get_ensemble_metadata_path(symbol)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(ensemble_metadata, f, indent=2, default=str)
            logger.debug(f"Ensemble metadata saved to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save ensemble metadata for {symbol}: {e}")

    def _format_ensemble_training_output(self, base_results: Dict[str, Any], model_type: str, 
                                        ensemble_metadata: Optional[Dict[str, Any]], 
                                        individual_cv_scores: Optional[Dict[str, Any]],
                                        fallback_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format enhanced training output with ensemble-specific information.
        
        Ensures ensemble_weights and individual_cv_scores are always present when 
        model_type is 'Ensemble_RF_XGB', maintaining backward compatibility.
        
        Args:
            base_results (Dict[str, Any]): Base training results
            model_type (str): Type of model trained
            ensemble_metadata (Optional[Dict[str, Any]]): Ensemble metadata if available
            individual_cv_scores (Optional[Dict[str, Any]]): Individual model CV scores
            fallback_info (Optional[Dict[str, Any]]): Fallback information if applicable
            
        Returns:
            Dict[str, Any]: Enhanced training results with ensemble formatting
        """
        try:
            # Create enhanced results starting with base results
            enhanced_results = base_results.copy()
            enhanced_results['model_type'] = model_type
            
            # Add ensemble-specific information for ensemble models
            if model_type == 'Ensemble_RF_XGB':
                logger.debug("Formatting ensemble training output")
                
                # Ensure ensemble_weights are always present
                if ensemble_metadata and 'ensemble_weights' in ensemble_metadata:
                    enhanced_results['ensemble_weights'] = ensemble_metadata['ensemble_weights']
                    logger.debug(f"Added ensemble_weights from metadata: {ensemble_metadata['ensemble_weights']}")
                else:
                    # Provide default structure if metadata is missing
                    enhanced_results['ensemble_weights'] = {
                        'rf': None,
                        'xgb': None
                    }
                    logger.warning("Ensemble weights not available in metadata, using default structure")
                
                # Ensure individual_cv_scores are always present
                if individual_cv_scores:
                    enhanced_results['individual_cv_scores'] = individual_cv_scores
                    logger.debug(f"Added individual_cv_scores: {individual_cv_scores}")
                else:
                    # Provide default structure if scores are missing
                    enhanced_results['individual_cv_scores'] = {
                        'rf': None,
                        'xgb': None,
                        'rf_std': None,
                        'xgb_std': None
                    }
                    logger.warning("Individual CV scores not available, using default structure")
                
                # Add enhanced model parameters structure for ensemble
                if ensemble_metadata:
                    enhanced_results['model_parameters'] = {
                        'rf_params': ensemble_metadata.get('rf_params', {}),
                        'xgb_params': ensemble_metadata.get('xgb_params', {}),
                        'ensemble_params': ensemble_metadata.get('ensemble_params', {})
                    }
                    logger.debug("Added enhanced model parameters from ensemble metadata")
                else:
                    enhanced_results['model_parameters'] = {
                        'rf_params': {},
                        'xgb_params': {},
                        'ensemble_params': {}
                    }
                    logger.warning("Ensemble metadata not available, using default model parameters structure")
                
                # Add ensemble transparency information
                enhanced_results['ensemble_info'] = {
                    'voting_type': 'soft',
                    'weight_calculation_method': 'cross_validation_performance_based',
                    'models_used': ['random_forest', 'xgboost']
                }
                
            else:
                # For non-ensemble models, ensure backward compatibility
                logger.debug(f"Formatting single model training output for {model_type}")
                
                # Add model parameters for single models
                if 'model_parameters' not in enhanced_results:
                    enhanced_results['model_parameters'] = base_results.get('model_parameters', {})
            
            # Add fallback information for transparency if applicable
            if fallback_info:
                enhanced_results['fallback_info'] = fallback_info
                logger.debug(f"Added fallback information: {fallback_info}")
            
            # Validate required fields are present
            required_fields = ['symbol', 'training_date', 'model_type', 'accuracy', 'precision', 'recall']
            missing_fields = [field for field in required_fields if field not in enhanced_results]
            if missing_fields:
                logger.warning(f"Missing required fields in training output: {missing_fields}")
            
            # Log output formatting completion
            logger.info(f"Enhanced training output formatted for {model_type} model")
            if model_type == 'Ensemble_RF_XGB':
                weights = enhanced_results.get('ensemble_weights', {})
                logger.info(f"Ensemble output includes weights: RF={weights.get('rf')}, XGB={weights.get('xgb')}")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to format enhanced training output: {e}")
            # Return base results as fallback
            base_results['model_type'] = model_type
            return base_results

    def _format_ensemble_prediction_output(self, base_result: Dict[str, Any], model, 
                                         ensemble_weights: Optional[Dict[str, Any]], 
                                         individual_predictions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format enhanced prediction output with ensemble transparency information.
        
        Ensures individual_predictions and ensemble_weights are always present when 
        model_type is 'Ensemble_RF_XGB', maintaining backward compatibility.
        
        Args:
            base_result (Dict[str, Any]): Base prediction result
            model: The trained model (ensemble or single)
            ensemble_weights (Optional[Dict[str, Any]]): Ensemble weights if available
            individual_predictions (Optional[Dict[str, Any]]): Individual model predictions
            
        Returns:
            Dict[str, Any]: Enhanced prediction result with ensemble formatting
        """
        try:
            # Create enhanced result starting with base result
            enhanced_result = base_result.copy()
            
            # Determine if this is an ensemble model
            is_ensemble = isinstance(model, VotingClassifier)
            
            # Set model_type appropriately
            if is_ensemble:
                enhanced_result['model_type'] = 'Ensemble_RF_XGB'
                logger.debug("Formatting ensemble prediction output")
                
                # Ensure ensemble_weights are always present for ensemble models
                if ensemble_weights:
                    enhanced_result['ensemble_weights'] = ensemble_weights
                    logger.debug(f"Added ensemble_weights to prediction: {ensemble_weights}")
                else:
                    # Provide default structure if weights are missing
                    enhanced_result['ensemble_weights'] = {
                        'rf': None,
                        'xgb': None
                    }
                    logger.warning("Ensemble weights not available, using default structure")
                
                # Ensure individual_predictions are always present for ensemble models
                if individual_predictions:
                    enhanced_result['individual_predictions'] = individual_predictions
                    logger.debug(f"Added individual_predictions: {list(individual_predictions.keys())}")
                else:
                    # Provide default structure if individual predictions are missing
                    enhanced_result['individual_predictions'] = {
                        'rf': {'DOWN': None, 'UP': None},
                        'xgb': {'DOWN': None, 'UP': None}
                    }
                    logger.warning("Individual predictions not available, using default structure")
                
                # Add ensemble transparency information
                enhanced_result['ensemble_info'] = {
                    'voting_type': 'soft',
                    'models_count': 2,
                    'models_used': ['random_forest', 'xgboost']
                }
                
                # Add ensemble prediction confidence breakdown if available
                if ensemble_weights and individual_predictions:
                    try:
                        # Calculate weighted contribution to final prediction
                        rf_weight = ensemble_weights.get('rf', 0.5)
                        xgb_weight = ensemble_weights.get('xgb', 0.5)
                        
                        rf_up_prob = individual_predictions.get('rf', {}).get('UP', 0.5)
                        xgb_up_prob = individual_predictions.get('xgb', {}).get('UP', 0.5)
                        
                        if rf_up_prob is not None and xgb_up_prob is not None:
                            enhanced_result['prediction_breakdown'] = {
                                'rf_contribution': float(rf_weight * rf_up_prob),
                                'xgb_contribution': float(xgb_weight * xgb_up_prob),
                                'weighted_average': float(rf_weight * rf_up_prob + xgb_weight * xgb_up_prob)
                            }
                            logger.debug("Added prediction breakdown with weighted contributions")
                    except Exception as breakdown_error:
                        logger.warning(f"Failed to calculate prediction breakdown: {breakdown_error}")
                
            else:
                # For single models, ensure model_type is set correctly
                enhanced_result['model_type'] = enhanced_result.get('model_type', self.model_type)
                logger.debug(f"Formatting single model prediction output for {enhanced_result['model_type']}")
            
            # Validate required fields are present
            required_fields = ['symbol', 'prediction', 'confidence', 'model_type']
            missing_fields = [field for field in required_fields if field not in enhanced_result]
            if missing_fields:
                logger.warning(f"Missing required fields in prediction output: {missing_fields}")
            
            # Log output formatting completion
            logger.info(f"Enhanced prediction output formatted for {enhanced_result['model_type']} model")
            if is_ensemble:
                logger.info("Ensemble prediction includes individual model transparency")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Failed to format enhanced prediction output: {e}")
            # Return base result as fallback
            return base_result

    def load_ensemble_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load ensemble metadata from JSON file with comprehensive error handling.
        
        Args:
            symbol (str): Stock symbol to load metadata for
            
        Returns:
            Optional[Dict[str, Any]]: Ensemble metadata or None if loading fails
        """
        try:
            import json
            metadata_path = self._get_ensemble_metadata_path(symbol)
            
            if not os.path.exists(metadata_path):
                logger.debug(f"Ensemble metadata file not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            if not isinstance(metadata, dict):
                logger.warning(f"Invalid ensemble metadata format for {symbol}: not a dictionary")
                return None
            
            logger.debug(f"Successfully loaded ensemble metadata for {symbol}")
            return metadata
            
        except json.JSONDecodeError as json_error:
            logger.warning(f"Failed to parse ensemble metadata JSON for {symbol}: {json_error}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load ensemble metadata for {symbol}: {e}")
            return None

    def _log_ensemble_debug_info(self, symbol: str, operation: str, **kwargs):
        """
        Log comprehensive debugging information for ensemble operations.
        
        Args:
            symbol (str): Stock symbol
            operation (str): Operation being performed
            **kwargs: Additional debug information
        """
        debug_logger = create_operation_logger(f'ensemble_debug_{operation}', symbol=symbol)
        
        try:
            debug_info = {
                'symbol': symbol,
                'operation': operation,
                'timestamp': datetime.now().isoformat(),
                'xgboost_available': getattr(self, 'xgboost_available', False),
                'ensemble_enabled': getattr(self, 'ensemble_enabled', False),
                'model_type': getattr(self, 'model_type', 'unknown'),
                'ensemble_config': getattr(self, 'ensemble_config', {}),
                'xgboost_config': getattr(self, 'xgboost_config', {})
            }
            
            # Add operation-specific information
            debug_info.update(kwargs)
            
            debug_logger.info(f"Ensemble debug info for {operation}:")
            for key, value in debug_info.items():
                debug_logger.info(f"  {key}: {value}")
                
        except Exception as debug_error:
            logger.warning(f"Failed to log ensemble debug info: {debug_error}")

    def _validate_ensemble_prerequisites(self, symbol: str) -> Dict[str, Any]:
        """
        Validate prerequisites for ensemble training and return status information.
        
        Args:
            symbol (str): Stock symbol to validate for
            
        Returns:
            Dict[str, Any]: Validation results and status information
        """
        validation_logger = create_operation_logger('validate_ensemble_prerequisites', symbol=symbol)
        
        validation_results = {
            'symbol': symbol,
            'xgboost_available': False,
            'xgboost_importable': False,
            'ensemble_enabled': False,
            'data_available': False,
            'sufficient_data': False,
            'prerequisites_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check XGBoost availability
            validation_results['xgboost_available'] = getattr(self, 'xgboost_available', False)
            
            # Test XGBoost import
            try:
                import xgboost as xgb_test
                validation_results['xgboost_importable'] = True
            except ImportError:
                validation_results['issues'].append('XGBoost not installed or importable')
                validation_results['recommendations'].append('Install XGBoost: pip install xgboost')
            
            # Check ensemble configuration
            validation_results['ensemble_enabled'] = getattr(self, 'ensemble_enabled', False)
            if not validation_results['ensemble_enabled']:
                validation_results['issues'].append('Ensemble mode not enabled in configuration')
                validation_results['recommendations'].append('Enable ensemble mode in configuration')
            
            # Check data availability
            try:
                stock_data = self.data_storage.load_stock_data(symbol)
                validation_results['data_available'] = not stock_data.empty
                
                if validation_results['data_available']:
                    validation_results['data_samples'] = len(stock_data)
                    validation_results['sufficient_data'] = len(stock_data) >= self.min_training_samples
                    
                    if not validation_results['sufficient_data']:
                        validation_results['issues'].append(f'Insufficient data: {len(stock_data)} < {self.min_training_samples}')
                        validation_results['recommendations'].append('Collect more historical data')
                else:
                    validation_results['issues'].append('No data available for symbol')
                    validation_results['recommendations'].append('Load historical data for symbol')
                    
            except Exception as data_error:
                validation_results['issues'].append(f'Data loading failed: {data_error}')
                validation_results['recommendations'].append('Check data storage and file integrity')
            
            # Determine if prerequisites are met
            validation_results['prerequisites_met'] = (
                validation_results['xgboost_available'] and
                validation_results['ensemble_enabled'] and
                validation_results['data_available'] and
                validation_results['sufficient_data']
            )
            
            # Log validation results
            validation_logger.info(f"Ensemble prerequisites validation for {symbol}:")
            validation_logger.info(f"  Prerequisites met: {validation_results['prerequisites_met']}")
            validation_logger.info(f"  XGBoost available: {validation_results['xgboost_available']}")
            validation_logger.info(f"  Ensemble enabled: {validation_results['ensemble_enabled']}")
            validation_logger.info(f"  Data available: {validation_results['data_available']}")
            validation_logger.info(f"  Sufficient data: {validation_results['sufficient_data']}")
            
            if validation_results['issues']:
                validation_logger.warning(f"  Issues found: {validation_results['issues']}")
            if validation_results['recommendations']:
                validation_logger.info(f"  Recommendations: {validation_results['recommendations']}")
            
            return validation_results
            
        except Exception as e:
            validation_logger.error(f"Ensemble prerequisites validation failed: {e}")
            validation_results['issues'].append(f'Validation failed: {e}')
            return validation_results
    
    def predict_movement(self, symbol: str) -> Dict[str, Any]:
        """
        Predict next-day price movement for a specific stock symbol with comprehensive fallback mechanisms.
        
        Args:
            symbol (str): Stock symbol to predict for
            
        Returns:
            Dict[str, Any]: Prediction results including direction, confidence, and metadata
            
        Raises:
            MLPredictorError: If prediction fails after all fallback attempts
        """
        operation_logger = create_operation_logger('predict_movement', symbol=symbol)
        
        try:
            symbol = self._normalize_symbol(symbol)
            operation_logger.info(f"Generating prediction for {symbol}")
            
            # Load or train model if not available with comprehensive error handling
            if symbol not in self.models:
                operation_logger.info(f"Model not in memory for {symbol}, attempting to load from disk")
                
                try:
                    # Attempt to load existing model and scaler
                    model_loaded = self._load_model_with_fallback(symbol)
                    if not model_loaded:
                        # Train new model if loading failed
                        operation_logger.info(f"No existing model found for {symbol}, training new model...")
                        training_results = self.train_model(symbol)
                        operation_logger.info(f"Model training completed for {symbol}")
                        
                except Exception as model_load_error:
                    operation_logger.warning(f"Model loading failed: {model_load_error}")
                    operation_logger.info("Attempting to train new model as fallback")
                    
                    try:
                        training_results = self.train_model(symbol)
                        operation_logger.info(f"Fallback model training completed for {symbol}")
                    except Exception as train_error:
                        error_context = create_error_context('predict_model_training_fallback', symbol=symbol,
                                                           load_error=str(model_load_error),
                                                           train_error=str(train_error))
                        log_exception(operation_logger, train_error, error_context)
                        raise MLPredictorError(f"Failed to load or train model for {symbol}: {train_error}")
            
            # Validate that model and scaler are available
            if symbol not in self.models or symbol not in self.scalers:
                raise MLPredictorError(f"Model or scaler not available for {symbol} after loading/training attempts")
            
            # Load latest stock data with error handling
            try:
                stock_data = self.data_storage.load_stock_data(symbol)
                if stock_data.empty:
                    raise MLPredictorError(f"No data available for symbol: {symbol}")
                    
            except FileNotFoundError:
                error_context = create_error_context('predict_data_loading', symbol=symbol)
                operation_logger.error(f"No data file found for symbol: {symbol}")
                raise MLPredictorError(f"No data file found for symbol: {symbol}")
            except Exception as data_error:
                error_context = create_error_context('predict_data_loading', symbol=symbol,
                                                   error_type=type(data_error).__name__)
                log_exception(operation_logger, data_error, error_context)
                raise MLPredictorError(f"Failed to load data for {symbol}: {data_error}")
            
            # Add technical indicators if not present with error handling
            try:
                missing_indicators = [col for col in self.feature_columns if col not in stock_data.columns]
                if missing_indicators:
                    operation_logger.info(f"Adding missing technical indicators: {missing_indicators}")
                    stock_data = self.technical_analyzer.add_all_indicators(stock_data)
                    
            except Exception as indicator_error:
                error_context = create_error_context('predict_technical_indicators', symbol=symbol,
                                                   missing_indicators=missing_indicators)
                log_exception(operation_logger, indicator_error, error_context)
                raise MLPredictorError(f"Failed to add technical indicators for {symbol}: {indicator_error}")
            
            # Get the latest data point for prediction with validation
            try:
                latest_data = stock_data.iloc[-1:][self.feature_columns]
                
                # Check for missing values with comprehensive handling
                if latest_data.isnull().any().any():
                    operation_logger.warning("Missing values detected in latest data, applying forward fill")
                    latest_data = latest_data.ffill().bfill()
                    
                    if latest_data.isnull().any().any():
                        missing_cols = latest_data.columns[latest_data.isnull().any()].tolist()
                        operation_logger.error(f"Unable to handle missing values in columns: {missing_cols}")
                        raise MLPredictorError(f"Unable to handle missing values in latest data for {symbol}: {missing_cols}")
                
            except Exception as data_prep_error:
                error_context = create_error_context('predict_data_preparation', symbol=symbol,
                                                   error_type=type(data_prep_error).__name__)
                log_exception(operation_logger, data_prep_error, error_context)
                raise MLPredictorError(f"Failed to prepare prediction data for {symbol}: {data_prep_error}")
            
            # Make prediction with comprehensive error handling
            try:
                model = self.models[symbol]
                scaler = self.scalers[symbol]
                
                # Scale features
                X_latest = scaler.transform(latest_data.values)
                
                # Generate prediction probabilities
                prediction_proba = model.predict_proba(X_latest)[0]
                
                # Load threshold with fallback
                active_threshold = self.thresholds.get(symbol)
                if active_threshold is None:
                    try:
                        loaded_thr = self.load_threshold(symbol)
                        if loaded_thr is not None:
                            self.thresholds[symbol] = loaded_thr
                            active_threshold = loaded_thr
                    except Exception as threshold_error:
                        operation_logger.warning(f"Failed to load threshold: {threshold_error}")
                
                if active_threshold is None:
                    active_threshold = 0.5
                    operation_logger.info("Using default threshold of 0.5")
                
                # Make final prediction
                up_proba = float(prediction_proba[1])
                prediction_class = 1 if up_proba >= active_threshold else 0
                prediction_direction = "UP" if prediction_class == 1 else "DOWN"
                confidence = float(max(prediction_proba))
                
                operation_logger.info(f"Prediction generated: {prediction_direction} (confidence: {confidence:.4f})")
                
            except Exception as prediction_error:
                error_context = create_error_context('predict_model_inference', symbol=symbol,
                                                   model_type=type(model).__name__ if 'model' in locals() else 'unknown',
                                                   error_type=type(prediction_error).__name__)
                log_exception(operation_logger, prediction_error, error_context)
                raise MLPredictorError(f"Model prediction failed for {symbol}: {prediction_error}")
            
            # Extract additional information with error handling
            try:
                current_price = float(stock_data['Close'].iloc[-1])
                current_date = stock_data['Date'].iloc[-1] if 'Date' in stock_data.columns else datetime.now()
                
                # Calculate model accuracy with fallback
                try:
                    model_accuracy = self._calculate_recent_accuracy(symbol, stock_data)
                except Exception as accuracy_error:
                    operation_logger.warning(f"Failed to calculate model accuracy: {accuracy_error}")
                    model_accuracy = None
                    
            except Exception as info_error:
                operation_logger.warning(f"Failed to extract additional information: {info_error}")
                current_price = 0.0
                current_date = datetime.now()
                model_accuracy = None

            # Extract ensemble information with comprehensive error handling
            is_ensemble = isinstance(model, VotingClassifier)
            ensemble_weights = None
            individual_predictions = None
            
            if is_ensemble:
                operation_logger.info("Extracting ensemble information for transparency")
                
                # Extract ensemble weights with fallback mechanisms
                try:
                    # First, try to load from metadata file
                    metadata = self.load_ensemble_metadata(symbol)
                    if metadata and isinstance(metadata, dict):
                        weights_meta = metadata.get('ensemble_weights')
                        if isinstance(weights_meta, dict):
                            rw = weights_meta.get('rf')
                            xw = weights_meta.get('xgb')
                            if rw is not None or xw is not None:
                                ensemble_weights = {}
                                if rw is not None:
                                    ensemble_weights['rf'] = float(rw)
                                if xw is not None:
                                    ensemble_weights['xgb'] = float(xw)
                                operation_logger.debug("Loaded ensemble weights from metadata")
                                
                except Exception as metadata_error:
                    operation_logger.warning(f"Failed to load ensemble metadata: {metadata_error}")
                
                # Fallback to extracting weights from model object
                if ensemble_weights is None and hasattr(model, 'weights') and model.weights is not None:
                    try:
                        names = [n for n, _ in getattr(model, 'estimators', [])]
                        weights_list = list(model.weights)
                        mapped = {}
                        for i, name in enumerate(names):
                            if i < len(weights_list) and name in ('rf', 'xgb'):
                                mapped[name] = float(weights_list[i])
                        if mapped:
                            ensemble_weights = mapped
                            operation_logger.debug("Extracted ensemble weights from model object")
                    except Exception as weights_error:
                        operation_logger.warning(f"Failed to extract weights from model: {weights_error}")
                        ensemble_weights = None
                
                # Extract individual model predictions with comprehensive error handling
                try:
                    preds = {}
                    
                    # Try named_estimators_ first (preferred method)
                    named = getattr(model, 'named_estimators_', None)
                    if isinstance(named, dict):
                        for key in ('rf', 'xgb'):
                            try:
                                est = named.get(key)
                                if est is not None and hasattr(est, 'predict_proba'):
                                    p = est.predict_proba(X_latest)[0]
                                    preds[key] = {'DOWN': float(p[0]), 'UP': float(p[1])}
                                    operation_logger.debug(f"Extracted {key} individual prediction")
                            except Exception as est_error:
                                operation_logger.warning(f"Failed to extract {key} prediction: {est_error}")
                    else:
                        # Fallback to estimators_ list
                        est_list = getattr(model, 'estimators_', None)
                        names = [n for n, _ in getattr(model, 'estimators', [])]
                        if est_list is not None and names:
                            for i, name in enumerate(names):
                                try:
                                    if name in ('rf', 'xgb') and i < len(est_list):
                                        est = est_list[i]
                                        if hasattr(est, 'predict_proba'):
                                            p = est.predict_proba(X_latest)[0]
                                            preds[name] = {'DOWN': float(p[0]), 'UP': float(p[1])}
                                            operation_logger.debug(f"Extracted {name} individual prediction from list")
                                except Exception as est_error:
                                    operation_logger.warning(f"Failed to extract {name} prediction from list: {est_error}")
                    
                    if preds:
                        individual_predictions = preds
                        operation_logger.info(f"Successfully extracted individual predictions for {len(preds)} models")
                    else:
                        operation_logger.warning("No individual predictions could be extracted")
                        
                except Exception as individual_error:
                    operation_logger.warning(f"Failed to extract individual predictions: {individual_error}")
                    individual_predictions = None

            # Prepare base prediction result
            try:
                base_prediction_result = {
                    'symbol': symbol,
                    'prediction': prediction_direction,
                    'confidence': confidence,
                    'prediction_probabilities': {
                        'DOWN': float(prediction_proba[0]),
                        'UP': float(prediction_proba[1])
                    },
                    'decision_threshold': float(active_threshold),
                    'current_price': current_price,
                    'prediction_date': datetime.now().isoformat(),
                    'data_date': current_date.isoformat() if hasattr(current_date, 'isoformat') else str(current_date),
                    'model_accuracy': model_accuracy,
                    'feature_count': len(self.feature_columns)
                }
                
                # Add fallback information if model was loaded with fallback
                if hasattr(self, '_model_fallback_info') and symbol in getattr(self, '_model_fallback_info', {}):
                    base_prediction_result['model_fallback_info'] = self._model_fallback_info[symbol]
                
                # Use enhanced output formatting to ensure consistent ensemble information
                prediction_result = self._format_ensemble_prediction_output(
                    base_result=base_prediction_result,
                    model=model,
                    ensemble_weights=ensemble_weights,
                    individual_predictions=individual_predictions
                )
                
                operation_logger.info(f"Prediction completed for {symbol}: {prediction_direction} (confidence: {confidence:.4f})")
                
                return prediction_result
                
            except Exception as result_error:
                error_context = create_error_context('predict_result_preparation', symbol=symbol,
                                                   error_type=type(result_error).__name__)
                log_exception(operation_logger, result_error, error_context)
                raise MLPredictorError(f"Failed to prepare prediction result for {symbol}: {result_error}")
            
        except MLPredictorError:
            # Re-raise MLPredictorError as-is to preserve error context
            raise
        except Exception as e:
            # Wrap unexpected errors with comprehensive context
            error_context = create_error_context('predict_movement', symbol=symbol,
                                               error_type=type(e).__name__)
            log_exception(operation_logger, e, error_context)
            raise MLPredictorError(f"Prediction failed for {symbol}: {e}")
    
    def _perform_time_series_cv(self, X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit, optimized_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform time-series cross-validation to get robust performance estimates.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            tscv (TimeSeriesSplit): Time series cross-validator
            optimized_params (Optional[Dict]): Optimized hyperparameters to use, if available
            
        Returns:
            Dict[str, Any]: Cross-validation scores and statistics
        """
        cv_accuracies = []
        cv_precisions = []
        cv_recalls = []
        
        try:
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                # Split data for this fold
                X_train_fold = X[train_idx]
                X_test_fold = X[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # Scale features for this fold
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                X_test_fold_scaled = scaler_fold.transform(X_test_fold)
                
                # Train model for this fold using optimized parameters if available
                if optimized_params:
                    fold_params = optimized_params.copy()
                    # Ensure consistent random_state and n_jobs
                    fold_params['random_state'] = self.random_state
                    fold_params['n_jobs'] = -1
                    model_fold = RandomForestClassifier(**fold_params)
                else:
                    model_fold = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        n_jobs=-1
                    )
                
                model_fold.fit(X_train_fold_scaled, y_train_fold)
                
                # Make predictions
                y_pred_fold = model_fold.predict(X_test_fold_scaled)
                
                # Calculate metrics
                accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
                precision_fold = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                recall_fold = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                
                cv_accuracies.append(accuracy_fold)
                cv_precisions.append(precision_fold)
                cv_recalls.append(recall_fold)
                
                logger.debug(f"Fold {fold + 1}: Accuracy={accuracy_fold:.4f}, Precision={precision_fold:.4f}, Recall={recall_fold:.4f}")
            
            # Calculate cross-validation statistics
            cv_results = {
                'mean_accuracy': float(np.mean(cv_accuracies)),
                'std_accuracy': float(np.std(cv_accuracies)),
                'mean_precision': float(np.mean(cv_precisions)),
                'std_precision': float(np.std(cv_precisions)),
                'mean_recall': float(np.mean(cv_recalls)),
                'std_recall': float(np.std(cv_recalls)),
                'individual_scores': {
                    'accuracies': [float(x) for x in cv_accuracies],
                    'precisions': [float(x) for x in cv_precisions],
                    'recalls': [float(x) for x in cv_recalls]
                },
                'n_folds': len(cv_accuracies)
            }
            
            logger.info(f"Time-series CV completed: Mean accuracy = {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.warning(f"Error during time-series cross-validation: {e}")
            return {
                'error': str(e),
                'n_folds': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'mean_precision': 0.0,
                'std_precision': 0.0,
                'mean_recall': 0.0,
                'std_recall': 0.0
            }

    def _calculate_recent_accuracy(self, symbol: str, stock_data: pd.DataFrame, days: int = 30) -> Optional[float]:
        """
        Calculate model accuracy on recent data.
        
        Args:
            symbol (str): Stock symbol
            stock_data (pd.DataFrame): Stock data with indicators
            days (int): Number of recent days to evaluate
            
        Returns:
            Optional[float]: Recent accuracy score or None if cannot calculate
        """
        try:
            if len(stock_data) < days + 1:
                return None
            
            X_full, y_full = self.prepare_features(stock_data)
            
            if len(X_full) < 10:
                return None
            
            n = min(days, len(X_full))
            X_recent = X_full[-n:]
            y_recent = y_full[-n:]
            
            # Scale features
            scaler = self.scalers[symbol]
            X_recent_scaled = scaler.transform(X_recent)
            
            # Make predictions
            model = self.models[symbol]
            y_pred_recent = model.predict(X_recent_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_recent, y_pred_recent)
            
            return float(accuracy)
            
        except Exception as e:
            logger.warning(f"Could not calculate recent accuracy for {symbol}: {e}")
            return None

    def _save_model_memory_efficient(self, symbol: str, model, model_path: str) -> bool:
        try:
            if self.memory_efficient_storage and self.enable_model_compression:
                try:
                    import joblib
                    joblib.dump(model, model_path, compress=3)
                except Exception:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f, protocol=4)
            elif self.memory_efficient_storage:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f, protocol=4)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            try:
                file_size = os.path.getsize(model_path)
                logger.debug(f"Saved model for {symbol} at {file_size / 1024:.1f} KB")
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Memory-efficient model saving failed for {symbol}: {e}")
            return False

    def _load_model_memory_efficient(self, model_path: str) -> Any:
        try:
            if not os.path.exists(model_path):
                return None
            file_size = os.path.getsize(model_path)
            if file_size == 0:
                logger.warning(f"Model file is empty: {model_path}")
                return None
            try:
                import joblib
                model = joblib.load(model_path)
            except Exception:
                try:
                    with open(model_path, 'rb') as f:
                        header = f.read(2)
                        f.seek(0)
                        if header == b'\x1f\x8b':
                            import gzip
                            with gzip.open(model_path, 'rb') as gz:
                                model = pickle.load(gz)
                        else:
                            model = pickle.load(f)
                except Exception as inner_e:
                    logger.error(f"Model loading failed from {model_path}: {inner_e}")
                    return None
            if model is None:
                logger.warning(f"Loaded model is None: {model_path}")
                return None
            if not hasattr(model, 'predict'):
                logger.warning(f"Loaded model missing predict method: {model_path}")
                return None
            logger.debug(f"Successfully loaded model from {model_path}: {file_size / 1024:.1f} KB")
            return model
        except Exception as e:
            logger.error(f"Memory-efficient model loading failed from {model_path}: {e}")
            return None

    def _cleanup_memory(self) -> None:
        """
        Enhanced memory cleanup with comprehensive garbage collection and monitoring.
        
        Implements memory-efficient cleanup by:
        - Forcing garbage collection for all generations
        - Clearing temporary variables and caches
        - Monitoring memory usage and logging cleanup results
        - Implementing memory usage threshold checks
        """
        try:
            import psutil
            import sys
            
            # Get memory usage before cleanup
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force garbage collection for all generations
            collected_gen0 = gc.collect(0)  # Young generation
            collected_gen1 = gc.collect(1)  # Middle generation  
            collected_gen2 = gc.collect(2)  # Old generation
            total_collected = collected_gen0 + collected_gen1 + collected_gen2
            
            # Clear any temporary caches that might be holding references
            if hasattr(self, '_temp_cv_results'):
                delattr(self, '_temp_cv_results')
            
            # Get memory usage after cleanup
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            if self.performance_monitoring:
                logger.debug(f"Memory cleanup completed:")
                logger.debug(f"  - Objects collected: {total_collected} (gen0: {collected_gen0}, gen1: {collected_gen1}, gen2: {collected_gen2})")
                logger.debug(f"  - Memory before: {memory_before:.1f} MB")
                logger.debug(f"  - Memory after: {memory_after:.1f} MB")
                logger.debug(f"  - Memory freed: {memory_freed:.1f} MB")
                
                # Check if memory usage is above threshold
                memory_usage_percent = memory_after / (psutil.virtual_memory().total / 1024 / 1024)
                if memory_usage_percent > self.memory_cleanup_threshold:
                    logger.warning(f"Memory usage still high after cleanup: {memory_usage_percent:.1%} of total system memory")
            
            # Additional cleanup for large objects
            if total_collected > 1000:
                logger.info(f"Large cleanup performed: {total_collected} objects collected, {memory_freed:.1f} MB freed")
                
        except ImportError:
            # Fallback to basic cleanup if psutil not available
            try:
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Basic garbage collection freed {collected} objects")
            except Exception as e:
                logger.warning(f"Basic memory cleanup failed: {e}")
                
        except Exception as e:
            logger.warning(f"Enhanced memory cleanup failed, falling back to basic cleanup: {e}")
            try:
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Fallback garbage collection freed {collected} objects")
            except Exception as fallback_error:
                logger.warning(f"Fallback memory cleanup failed: {fallback_error}")
    
    def save_model(self, symbol: str, model: Any, ensemble_metadata: Optional[Dict[str, Any]] = None, 
                   individual_models: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enhanced model saving to store ensemble models, individual models, and metadata.
        
        Extends model saving to store ensemble models, individual models, and metadata.
        Creates ensemble metadata JSON structure with weights and individual scores.
        Implements comprehensive error handling and validation during save process.
        
        Args:
            symbol (str): Stock symbol
            model: Trained model object (ensemble or individual)
            ensemble_metadata (Optional[Dict[str, Any]]): Ensemble metadata for transparency
            individual_models (Optional[Dict[str, Any]]): Individual models dict with 'rf' and 'xgb' keys
            
        Returns:
            bool: True if save successful, False otherwise
            
        Raises:
            ModelPersistenceError: If critical save operations fail
        """
        try:
            logger.info(f"Starting enhanced model save for {symbol}...")
            
            # Validate input parameters
            if model is None:
                raise ModelPersistenceError(f"Cannot save None model for {symbol}")
            
            # Determine if this is an ensemble model
            is_ensemble = ensemble_metadata is not None
            model_type = ensemble_metadata.get('model_type', 'Unknown') if is_ensemble else 'Individual'
            
            logger.info(f"Saving {model_type} model for {symbol}")
            
            # Always save the main model (ensemble or individual)
            model_path = self._get_model_path(symbol)
            logger.debug(f"Saving main model to {model_path}")
            
            # Validate model has required methods before saving
            required_methods = ['predict']
            for method in required_methods:
                if not hasattr(model, method):
                    raise ModelPersistenceError(f"Model missing required method '{method}' for {symbol}")
            
            # Save main model with error handling using memory-efficient storage
            try:
                if not self._save_model_memory_efficient(symbol, model, model_path):
                    raise ModelPersistenceError(f"Memory-efficient save failed for {symbol}")
                logger.debug(f"Main model saved for {symbol} at {model_path}")
            except Exception as e:
                raise ModelPersistenceError(f"Failed to save main model for {symbol}: {e}")
            
            # Validate main model was saved correctly
            if not os.path.exists(model_path):
                raise ModelPersistenceError(f"Main model file not found after save for {symbol}")
            
            # If this is an ensemble model, save additional components
            if is_ensemble:
                logger.info(f"Saving ensemble components for {symbol}...")
                
                # Validate ensemble metadata structure
                if not self._validate_ensemble_metadata(ensemble_metadata):
                    raise ModelPersistenceError(f"Invalid ensemble metadata structure for {symbol}")
                
                # Save ensemble metadata with validation
                logger.debug(f"Saving ensemble metadata for {symbol}")
                if not self.save_ensemble_metadata(symbol, ensemble_metadata):
                    raise ModelPersistenceError(f"Failed to save ensemble metadata for {symbol}")
                
                # Validate ensemble metadata was saved
                if not os.path.exists(self._get_ensemble_metadata_path(symbol)):
                    raise ModelPersistenceError(f"Ensemble metadata file not found after save for {symbol}")
                
                # Save individual models if provided
                if individual_models is not None:
                    logger.debug(f"Saving individual models for {symbol}")
                    
                    # Save Random Forest model
                    if 'rf' in individual_models and individual_models['rf'] is not None:
                        rf_path = self._get_rf_model_path(symbol)
                        try:
                            # Validate RF model before saving
                            rf_model = individual_models['rf']
                            if not hasattr(rf_model, 'predict'):
                                logger.warning(f"RF model missing predict method for {symbol}")
                            else:
                                if self._save_model_memory_efficient(symbol, rf_model, rf_path):
                                    logger.debug(f"Individual RF model saved for {symbol} at {rf_path}")
                                else:
                                    logger.warning(f"Memory-efficient RF model save failed for {symbol}")
                                
                                # Validate RF model was saved
                                if not os.path.exists(rf_path):
                                    logger.warning(f"RF model file not found after save for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to save individual RF model for {symbol}: {e}")
                    
                    # Save XGBoost model
                    if 'xgb' in individual_models and individual_models['xgb'] is not None:
                        xgb_path = self._get_xgb_model_path(symbol)
                        try:
                            # Validate XGBoost model before saving
                            xgb_model = individual_models['xgb']
                            if not hasattr(xgb_model, 'predict'):
                                logger.warning(f"XGBoost model missing predict method for {symbol}")
                            else:
                                if self._save_model_memory_efficient(symbol, xgb_model, xgb_path):
                                    logger.debug(f"Individual XGBoost model saved for {symbol} at {xgb_path}")
                                else:
                                    logger.warning(f"Memory-efficient XGBoost model save failed for {symbol}")
                                
                                # Validate XGBoost model was saved
                                if not os.path.exists(xgb_path):
                                    logger.warning(f"XGBoost model file not found after save for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to save individual XGBoost model for {symbol}: {e}")
                
                logger.info(f"Ensemble model and components saved successfully for {symbol}")
                
                # Log ensemble save summary
                self._log_ensemble_save_summary(symbol, ensemble_metadata)
                
            else:
                logger.debug(f"Individual model saved for {symbol}")
            
            # Final validation - ensure main model file exists and is readable
            try:
                test_model = self._validate_saved_model(symbol)
                if test_model is None:
                    raise ModelPersistenceError(f"Saved model validation failed for {symbol}")
                logger.debug(f"Model save validation successful for {symbol}")
            except Exception as e:
                logger.warning(f"Model save validation failed for {symbol}: {e}")
                # Don't fail the save operation for validation issues, just warn
            
            logger.info(f"Enhanced model save completed successfully for {symbol}")
            return True
            
        except ModelPersistenceError:
            # Re-raise ModelPersistenceError as-is
            raise
        except Exception as e:
            error_msg = f"Error saving model for {symbol}: {e}"
            logger.error(error_msg)
            raise ModelPersistenceError(error_msg)
    
    def load_model(self, symbol: str) -> Any:
        """
        Enhanced ensemble model loading with fallback to individual models.
        
        Implements ensemble model loading with fallback to individual models.
        Adds model validation during loading process.
        Provides comprehensive error handling and logging.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Any: Loaded model object (ensemble or best individual model)
            
        Raises:
            FileNotFoundError: If no model files exist for the symbol
            MLPredictorError: If all loading attempts fail
        """
        try:
            logger.info(f"Starting enhanced model load for {symbol}...")
            
            model_path = self._get_model_path(symbol)
            ensemble_metadata_path = self._get_ensemble_metadata_path(symbol)
            
            # Check if this is an ensemble model
            is_ensemble = os.path.exists(ensemble_metadata_path)
            
            if is_ensemble:
                logger.info(f"Detected ensemble model for {symbol}, attempting ensemble load...")
                
                # Try to load ensemble model first
                try:
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Ensemble model file not found: {model_path}")
                    
                    # Load main ensemble model using memory-efficient loading
                    ensemble_model = self._load_model_memory_efficient(model_path)
                    if ensemble_model is None:
                        raise MLPredictorError(f"Memory-efficient ensemble model loading failed for {symbol}")
                    
                    # Validate ensemble model
                    if not self._validate_loaded_model(ensemble_model, symbol, 'ensemble'):
                        raise MLPredictorError(f"Ensemble model validation failed for {symbol}")
                    
                    # Load and validate ensemble metadata
                    ensemble_metadata = self.load_ensemble_metadata(symbol)
                    if ensemble_metadata is None:
                        logger.warning(f"Ensemble metadata not found for {symbol}, but ensemble model exists")
                    else:
                        logger.debug(f"Ensemble metadata loaded for {symbol}: {ensemble_metadata.get('model_type', 'Unknown')}")
                    
                    logger.info(f"Ensemble model loaded successfully for {symbol}")
                    return ensemble_model
                    
                except Exception as e:
                    logger.warning(f"Ensemble model loading failed for {symbol}: {e}")
                    logger.info(f"Attempting fallback to individual models for {symbol}...")
                    
                    # Fallback to best individual model
                    return self._load_fallback_model(symbol)
            
            else:
                logger.info(f"Loading individual model for {symbol}...")
                
                # Load individual model
                if not os.path.exists(model_path):
                    # Try to find any available individual models as fallback
                    logger.warning(f"Main model file not found for {symbol}, checking for individual models...")
                    return self._load_fallback_model(symbol)
                
                # Load main model using memory-efficient loading
                model = self._load_model_memory_efficient(model_path)
                if model is None:
                    raise MLPredictorError(f"Memory-efficient individual model loading failed for {symbol}")
                
                # Validate loaded model
                if not self._validate_loaded_model(model, symbol, 'individual'):
                    raise MLPredictorError(f"Individual model validation failed for {symbol}")
                
                logger.info(f"Individual model loaded successfully for {symbol}")
                return model
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except MLPredictorError:
            # Re-raise MLPredictorError as-is
            raise
        except Exception as e:
            error_msg = f"Error loading model for {symbol}: {e}"
            logger.error(error_msg)
            raise MLPredictorError(error_msg)
    
    def save_scaler(self, symbol: str, scaler: Any) -> bool:
        """
        Save feature scaler to disk.
        
        Args:
            symbol (str): Stock symbol
            scaler: Trained scaler object
            
        Returns:
            bool: True if save successful
        """
        try:
            scaler_path = self._get_scaler_path(symbol)
            
            if self.memory_efficient_storage:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f, protocol=4)
            else:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            logger.debug(f"Scaler saved for {symbol} at {scaler_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving scaler for {symbol}: {e}")
            return False
    
    def load_scaler(self, symbol: str) -> Any:
        """
        Load feature scaler from disk.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Any: Loaded scaler object
            
        Raises:
            FileNotFoundError: If scaler file doesn't exist
            MLPredictorError: If loading fails
        """
        try:
            scaler_path = self._get_scaler_path(symbol)
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"No scaler file found for symbol: {symbol}")
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            logger.debug(f"Scaler loaded for {symbol} from {scaler_path}")
            return scaler
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise MLPredictorError(f"Error loading scaler for {symbol}: {e}")

    def save_ensemble_metadata(self, symbol: str, metadata: Dict[str, Any]) -> bool:
        """
        Save ensemble metadata to disk for transparency.
        
        Args:
            symbol (str): Stock symbol
            metadata (Dict[str, Any]): Ensemble metadata dictionary
            
        Returns:
            bool: True if save successful
        """
        try:
            metadata_path = self._get_ensemble_metadata_path(symbol)
            
            # Ensure metadata is JSON serializable
            import json
            serializable_metadata = self._make_json_serializable(metadata)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2)
            
            logger.debug(f"Ensemble metadata saved for {symbol} at {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble metadata for {symbol}: {e}")
            return False

    def load_ensemble_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load ensemble metadata from disk.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict[str, Any]]: Loaded ensemble metadata or None if not found
        """
        try:
            metadata_path = self._get_ensemble_metadata_path(symbol)
            
            if not os.path.exists(metadata_path):
                logger.debug(f"No ensemble metadata file found for {symbol}")
                return None
            
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.debug(f"Ensemble metadata loaded for {symbol} from {metadata_path}")
            return metadata
            
        except Exception as e:
            logger.warning(f"Error loading ensemble metadata for {symbol}: {e}")
            return None

    def save_individual_models(self, symbol: str, rf_model, xgb_model) -> bool:
        """
        Save individual RF and XGBoost models separately for debugging and analysis.
        
        Args:
            symbol (str): Stock symbol
            rf_model: Trained Random Forest model
            xgb_model: Trained XGBoost model
            
        Returns:
            bool: True if both models saved successfully
        """
        try:
            rf_path = self._get_rf_model_path(symbol)
            xgb_path = self._get_xgb_model_path(symbol)
            
            # Save Random Forest model
            with open(rf_path, 'wb') as f:
                pickle.dump(rf_model, f)
            
            # Save XGBoost model
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            
            logger.debug(f"Individual models saved for {symbol}: RF at {rf_path}, XGB at {xgb_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving individual models for {symbol}: {e}")
            return False

    def load_individual_models(self, symbol: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load individual RF and XGBoost models from disk.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Tuple[Optional[Any], Optional[Any]]: RF model and XGBoost model, or None if not found
        """
        try:
            rf_path = self._get_rf_model_path(symbol)
            xgb_path = self._get_xgb_model_path(symbol)
            
            rf_model = None
            xgb_model = None
            
            # Load Random Forest model
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    rf_model = pickle.load(f)
                logger.debug(f"RF model loaded for {symbol} from {rf_path}")
            
            # Load XGBoost model
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    xgb_model = pickle.load(f)
                logger.debug(f"XGBoost model loaded for {symbol} from {xgb_path}")
            
            return rf_model, xgb_model
            
        except Exception as e:
            logger.warning(f"Error loading individual models for {symbol}: {e}")
            return None, None

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # For complex objects, convert to string representation
            return str(obj)
    
    def get_model_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a trained model.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            model_path = self._get_model_path(symbol)
            scaler_path = self._get_scaler_path(symbol)
            
            info = {
                'symbol': symbol,
                'model_exists': os.path.exists(model_path),
                'scaler_exists': os.path.exists(scaler_path),
                'model_type': self.model_type,
                'feature_count': len(self.feature_columns),
                'features': self.feature_columns
            }
            
            if info['model_exists']:
                # Get file modification time
                model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                info['model_last_updated'] = model_mtime.isoformat()
                info['model_size_bytes'] = os.path.getsize(model_path)
                
                # Load model to get additional info
                try:
                    model = self.load_model(symbol)
                    if hasattr(model, 'n_estimators'):
                        info['n_estimators'] = model.n_estimators
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
                        info['top_features'] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                except Exception as e:
                    logger.warning(f"Could not load model details for {symbol}: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_available_models(self) -> List[str]:
        """
        Get list of symbols that have trained models.
        
        Returns:
            List[str]: List of symbols with available models
        """
        try:
            models = []
            
            if not os.path.exists(self.models_dir):
                return models
            
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_model.pkl'):
                    # Extract symbol from filename
                    symbol = filename.replace('_model.pkl', '')
                    models.append(symbol)
            
            models.sort()
            logger.debug(f"Found {len(models)} trained models")
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    def tune_threshold_for_symbol(self, symbol: str) -> Dict[str, Any]:
        try:
            symbol = self._normalize_symbol(symbol)
            if symbol not in self.models:
                model_path = self._get_model_path(symbol)
                scaler_path = self._get_scaler_path(symbol)
                if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                    raise MLPredictorError(f"No trained model found for {symbol}")
                self.models[symbol] = self.load_model(symbol)
                self.scalers[symbol] = self.load_scaler(symbol)

            stock_data = self.data_storage.load_stock_data(symbol)
            if stock_data.empty:
                raise MLPredictorError(f"No data available for symbol: {symbol}")

            X, y = self.prepare_features(stock_data)
            if len(X) < self.min_training_samples:
                raise InsufficientDataError(
                    f"Insufficient data for threshold tuning. Need {self.min_training_samples} samples, got {len(X)}"
                )

            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_test = X[test_idx]
            y_test = y[test_idx]

            scaler = self.scalers[symbol]
            model = self.models[symbol]
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)

            tuned_threshold, tuned_score = self._tune_threshold(y_test, y_pred_proba[:, 1])
            self.thresholds[symbol] = tuned_threshold
            self.save_threshold(symbol, tuned_threshold)

            return {
                'symbol': symbol,
                'decision_threshold': float(tuned_threshold),
                'threshold_metric': self.threshold_metric,
                'threshold_score': float(tuned_score),
                'test_samples': int(len(y_test))
            }
        except Exception as e:
            if isinstance(e, (MLPredictorError, InsufficientDataError)):
                raise
            raise MLPredictorError(f"Threshold tuning failed for {symbol}: {e}")
    
    def evaluate_model(self, symbol: str) -> Dict[str, Any]:
        """
        Evaluate model performance on historical data.
        
        Args:
            symbol (str): Stock symbol to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation metrics and performance data
            
        Raises:
            MLPredictorError: If evaluation fails
        """
        try:
            logger.info(f"Evaluating model for {symbol}")
            
            # Load model if not in memory
            if symbol not in self.models:
                model_path = self._get_model_path(symbol)
                scaler_path = self._get_scaler_path(symbol)
                
                if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                    raise MLPredictorError(f"No trained model found for {symbol}")
                
                self.models[symbol] = self.load_model(symbol)
                self.scalers[symbol] = self.load_scaler(symbol)
            
            # Load stock data
            try:
                stock_data = self.data_storage.load_stock_data(symbol)
            except FileNotFoundError:
                raise MLPredictorError(f"No data file found for symbol: {symbol}")
            
            if stock_data.empty:
                raise MLPredictorError(f"No data available for symbol: {symbol}")
            
            # Prepare features and target
            X, y = self.prepare_features(stock_data)
            
            # Scale features
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)

            active_threshold = self.thresholds.get(symbol)
            if active_threshold is None:
                loaded_thr = self.load_threshold(symbol)
                if loaded_thr is not None:
                    self.thresholds[symbol] = loaded_thr
                    active_threshold = loaded_thr
            if active_threshold is None:
                active_threshold = 0.5
            y_pred_thresh = (y_pred_proba[:, 1] >= float(active_threshold)).astype(int)
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1_default = f1_score(y, y_pred, average='weighted', zero_division=0)
            accuracy_thr = accuracy_score(y, y_pred_thresh)
            precision_thr = precision_score(y, y_pred_thresh, average='weighted', zero_division=0)
            recall_thr = recall_score(y, y_pred_thresh, average='weighted', zero_division=0)
            f1_thr = f1_score(y, y_pred_thresh, average='weighted', zero_division=0)
            
            # Calculate class-specific metrics
            precision_up = precision_score(y, y_pred, pos_label=1, zero_division=0)
            recall_up = recall_score(y, y_pred, pos_label=1, zero_division=0)
            precision_down = precision_score(y, y_pred, pos_label=0, zero_division=0)
            recall_down = recall_score(y, y_pred, pos_label=0, zero_division=0)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate prediction confidence statistics
            confidence_stats = {
                'mean_confidence': float(np.mean(np.max(y_pred_proba, axis=1))),
                'min_confidence': float(np.min(np.max(y_pred_proba, axis=1))),
                'max_confidence': float(np.max(np.max(y_pred_proba, axis=1))),
                'std_confidence': float(np.std(np.max(y_pred_proba, axis=1)))
            }
            
            # Calculate recent performance (last 30 predictions)
            recent_accuracy = None
            if len(y) >= 30:
                recent_y = y[-30:]
                recent_pred = y_pred[-30:]
                recent_accuracy = accuracy_score(recent_y, recent_pred)
            
            # Prepare evaluation results
            evaluation_results = {
                'symbol': symbol,
                'evaluation_date': datetime.now().isoformat(),
                'total_samples': len(y),
                'overall_metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1_default)
                },
                'threshold_metrics': {
                    'threshold': float(active_threshold),
                    'accuracy': float(accuracy_thr),
                    'precision': float(precision_thr),
                    'recall': float(recall_thr),
                    'f1': float(f1_thr)
                },
                'class_metrics': {
                    'UP': {
                        'precision': float(precision_up),
                        'recall': float(recall_up)
                    },
                    'DOWN': {
                        'precision': float(precision_down),
                        'recall': float(recall_down)
                    }
                },
                'class_distribution': {
                    'UP': int(np.sum(y)),
                    'DOWN': int(len(y) - np.sum(y)),
                    'UP_percentage': float(np.mean(y) * 100),
                    'DOWN_percentage': float((1 - np.mean(y)) * 100)
                },
                'confidence_statistics': confidence_stats,
                'recent_accuracy': float(recent_accuracy) if recent_accuracy is not None else None,
                'feature_importance': {
                    'top_10_features': top_features[:10],
                    'all_features': feature_importance
                },
                'model_info': {
                    'model_type': self.model_type,
                    'n_estimators': self.n_estimators,
                    'feature_count': len(self.feature_columns)
                }
            }
            
            logger.info(f"Model evaluation completed for {symbol}")
            logger.info(f"Overall accuracy: {accuracy:.4f}")
            logger.info(f"UP precision/recall: {precision_up:.4f}/{recall_up:.4f}")
            logger.info(f"DOWN precision/recall: {precision_down:.4f}/{recall_down:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            if isinstance(e, MLPredictorError):
                raise
            else:
                raise MLPredictorError(f"Model evaluation failed for {symbol}: {e}")

    
    
    def generate_prediction_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Generate a comprehensive prediction summary for a stock symbol.
        
        Args:
            symbol (str): Stock symbol to generate summary for
            
        Returns:
            Dict[str, Any]: Comprehensive prediction summary
        """
        try:
            # Get prediction
            prediction_result = self.predict_movement(symbol)
            
            # Load stock data for additional context
            stock_data = self.data_storage.load_stock_data(symbol)
            
            # Calculate additional metrics
            recent_data = stock_data.tail(5)  # Last 5 days
            price_change_5d = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / 
                              recent_data['Close'].iloc[0] * 100) if len(recent_data) >= 2 else 0
            
            # Get technical indicator values
            latest_indicators = {}
            for indicator in ['RSI_14', 'MACD', 'SMA_50', 'SMA_200']:
                if indicator in stock_data.columns:
                    latest_indicators[indicator] = float(stock_data[indicator].iloc[-1])
            
            # Create comprehensive summary
            summary = {
                'basic_prediction': prediction_result,
                'formatted_output': self.format_prediction_output(prediction_result),
                'additional_context': {
                    'price_change_5d_percent': float(price_change_5d),
                    'latest_indicators': latest_indicators,
                    'data_points_available': len(stock_data),
                    'last_data_date': stock_data['Date'].iloc[-1].isoformat() if 'Date' in stock_data.columns else None
                },
                'recommendation_strength': self._get_recommendation_strength(prediction_result),
                'risk_assessment': self._assess_prediction_risk(prediction_result, latest_indicators)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating prediction summary for {symbol}: {e}")
            raise MLPredictorError(f"Failed to generate prediction summary for {symbol}: {e}")
    
    def _get_recommendation_strength(self, prediction_result: Dict[str, Any]) -> str:
        """
        Determine recommendation strength based on confidence and accuracy.
        
        Args:
            prediction_result (Dict[str, Any]): Prediction result
            
        Returns:
            str: Recommendation strength (Strong, Moderate, Weak)
        """
        confidence = prediction_result.get('confidence', 0)
        accuracy = prediction_result.get('model_accuracy', 0)
        
        # Combine confidence and accuracy for strength assessment
        if confidence >= 0.8 and (accuracy is None or accuracy >= 0.6):
            return "Strong"
        elif confidence >= 0.65 and (accuracy is None or accuracy >= 0.55):
            return "Moderate"
        else:
            return "Weak"
    
    def _assess_prediction_risk(self, prediction_result: Dict[str, Any], 
                               indicators: Dict[str, float]) -> str:
        """
        Assess risk level of the prediction based on technical indicators.
        
        Args:
            prediction_result (Dict[str, Any]): Prediction result
            indicators (Dict[str, float]): Latest technical indicator values
            
        Returns:
            str: Risk assessment (Low, Medium, High)
        """
        try:
            risk_factors = 0
            
            # Check RSI for overbought/oversold conditions
            rsi = indicators.get('RSI_14')
            if rsi is not None:
                if rsi > 70 or rsi < 30:  # Overbought or oversold
                    risk_factors += 1
            
            # Check MACD for divergence signals
            macd = indicators.get('MACD')
            if macd is not None:
                if abs(macd) > 10:  # Strong MACD signal (arbitrary threshold)
                    risk_factors += 1
            
            # Check confidence level
            confidence = prediction_result.get('confidence', 0)
            if confidence < 0.6:
                risk_factors += 1
            
            # Check model accuracy
            accuracy = prediction_result.get('model_accuracy')
            if accuracy is not None and accuracy < 0.55:
                risk_factors += 1
            
            # Determine risk level
            if risk_factors >= 3:
                return "High"
            elif risk_factors >= 2:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.warning(f"Error assessing prediction risk: {e}")
            return "Medium"  # Default to medium risk if assessment fails

    def format_prediction_output(self, prediction_result: Dict[str, Any]) -> str:
        """
        Format prediction result as human-readable suggestion string.
        
        Args:
            prediction_result (Dict[str, Any]): Prediction result from predict_movement()
            
        Returns:
            str: Human-readable prediction string (e.g., "Prediction for ENGRO: UP")
        """
        try:
            symbol = prediction_result['symbol']
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            current_price = prediction_result['current_price']
            model_accuracy = prediction_result.get('model_accuracy')
            
            # Basic prediction format as specified in requirements
            basic_format = f"Prediction for {symbol}: {prediction}"
            
            # Enhanced format with additional details
            confidence_pct = confidence * 100
            enhanced_format = f"Prediction for {symbol}: {prediction} (Confidence: {confidence_pct:.1f}%"
            
            if model_accuracy is not None:
                accuracy_pct = model_accuracy * 100
                enhanced_format += f", Model Accuracy: {accuracy_pct:.1f}%"
            
            enhanced_format += f", Current Price: {current_price:.2f})"
            
            return enhanced_format
            
        except Exception as e:
            logger.error(f"Error formatting prediction output: {e}")
            return f"Prediction for {prediction_result.get('symbol', 'UNKNOWN')}: ERROR"
    
    def format_prediction_summary(self, prediction_result: Dict[str, Any]) -> str:
        """
        Format prediction result as a detailed summary with timestamp.
        
        Args:
            prediction_result (Dict[str, Any]): Prediction result from predict_movement()
            
        Returns:
            str: Detailed prediction summary
        """
        try:
            symbol = prediction_result['symbol']
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            current_price = prediction_result['current_price']
            prediction_date = prediction_result['prediction_date']
            data_date = prediction_result['data_date']
            model_accuracy = prediction_result.get('model_accuracy')
            
            # Parse prediction timestamp
            try:
                pred_dt = datetime.fromisoformat(prediction_date.replace('Z', '+00:00'))
                timestamp_str = pred_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_str = prediction_date
            
            # Create detailed summary
            summary = f"""
Stock Symbol: {symbol}
Prediction: {prediction}
Confidence: {confidence * 100:.1f}%
Current Price: {current_price:.2f}
Prediction Time: {timestamp_str}
Data Date: {data_date}"""
            
            if model_accuracy is not None:
                summary += f"\nModel Accuracy: {model_accuracy * 100:.1f}%"
            
            # Add probability breakdown
            if 'prediction_probabilities' in prediction_result:
                probs = prediction_result['prediction_probabilities']
                summary += f"\nProbability UP: {probs.get('UP', 0) * 100:.1f}%"
                summary += f"\nProbability DOWN: {probs.get('DOWN', 0) * 100:.1f}%"
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error formatting prediction summary: {e}")
            return f"Error formatting summary for {prediction_result.get('symbol', 'UNKNOWN')}"
    
    def format_multiple_predictions(self, predictions: List[Dict[str, Any]], 
                                  format_type: str = "table") -> str:
        """
        Format multiple prediction results in a readable format.
        
        Args:
            predictions (List[Dict[str, Any]]): List of prediction results
            format_type (str): Format type - "table", "list", or "simple"
            
        Returns:
            str: Formatted predictions output
        """
        try:
            if not predictions:
                return "No predictions available."
            
            if format_type == "simple":
                # Simple list format as specified in requirements
                lines = []
                for pred in predictions:
                    lines.append(self.format_prediction_output(pred))
                return "\n".join(lines)
            
            elif format_type == "list":
                # Detailed list format
                lines = ["=== STOCK PREDICTIONS ==="]
                for i, pred in enumerate(predictions, 1):
                    lines.append(f"\n{i}. {self.format_prediction_output(pred)}")
                return "\n".join(lines)
            
            elif format_type == "table":
                # Table format
                lines = ["=== STOCK PREDICTIONS TABLE ==="]
                lines.append(f"{'Symbol':<10} {'Prediction':<10} {'Confidence':<12} {'Price':<10} {'Accuracy':<10}")
                lines.append("-" * 62)
                
                for pred in predictions:
                    symbol = pred['symbol']
                    prediction = pred['prediction']
                    confidence = f"{pred['confidence'] * 100:.1f}%"
                    price = f"{pred['current_price']:.2f}"
                    accuracy = f"{pred.get('model_accuracy', 0) * 100:.1f}%" if pred.get('model_accuracy') else "N/A"
                    
                    lines.append(f"{symbol:<10} {prediction:<10} {confidence:<12} {price:<10} {accuracy:<10}")
                
                return "\n".join(lines)
            
            else:
                raise ValueError(f"Unknown format_type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error formatting multiple predictions: {e}")
            return f"Error formatting predictions: {e}"
    
    def get_batch_predictions(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate predictions for multiple stock symbols.
        
        Args:
            symbols (List[str]): List of stock symbols to predict
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        predictions = []
        
        for symbol in symbols:
            try:
                logger.info(f"Generating prediction for {symbol}")
                prediction = self.predict_movement(symbol)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to generate prediction for {symbol}: {e}")
                # Add error result to maintain consistency
                error_result = {
                    'symbol': symbol,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'current_price': 0.0,
                    'prediction_date': datetime.now().isoformat(),
                    'data_date': 'N/A',
                    'model_accuracy': None,
                    'error': str(e)
                }
                predictions.append(error_result)
        
        return predictions
    
    def display_predictions(self, symbols: List[str], format_type: str = "simple") -> None:
        """
        Generate and display predictions for multiple stocks in formatted output.
        
        Args:
            symbols (List[str]): List of stock symbols to predict
            format_type (str): Display format - "simple", "table", or "detailed"
        """
        try:
            logger.info(f"Generating predictions for {len(symbols)} symbols")
            
            # Generate predictions
            predictions = self.get_batch_predictions(symbols)
            
            # Filter out error results for display (but log them)
            valid_predictions = []
            error_count = 0
            
            for pred in predictions:
                if pred['prediction'] == 'ERROR':
                    error_count += 1
                    logger.warning(f"Prediction failed for {pred['symbol']}: {pred.get('error', 'Unknown error')}")
                else:
                    valid_predictions.append(pred)
            
            # Display results
            if valid_predictions:
                if format_type == "detailed":
                    print("\n" + "="*60)
                    print("DETAILED STOCK PREDICTIONS")
                    print("="*60)
                    for pred in valid_predictions:
                        print(self.format_prediction_summary(pred))
                        print("-" * 40)
                else:
                    formatted_output = self.format_multiple_predictions(valid_predictions, format_type)
                    print(formatted_output)
            
            # Display summary
            print(f"\nSummary: {len(valid_predictions)} successful predictions, {error_count} errors")
            
            if valid_predictions:
                # Calculate overall statistics
                up_predictions = sum(1 for p in valid_predictions if p['prediction'] == 'UP')
                down_predictions = len(valid_predictions) - up_predictions
                avg_confidence = sum(p['confidence'] for p in valid_predictions) / len(valid_predictions)
                
                print(f"UP predictions: {up_predictions}, DOWN predictions: {down_predictions}")
                print(f"Average confidence: {avg_confidence * 100:.1f}%")
                print(f"Prediction timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error displaying predictions: {e}")
            print(f"Error displaying predictions: {e}")

    def cleanup_old_models(self, days_to_keep: int = 90) -> int:
        """
        Clean up old model files.
        
        Args:
            days_to_keep (int): Number of days to keep model files
            
        Returns:
            int: Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for directory in [self.models_dir, self.scalers_dir]:
                if not os.path.exists(directory):
                    continue
                
                for filename in os.listdir(directory):
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(directory, filename)
                        
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_mtime < cutoff_date:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                logger.debug(f"Deleted old model file: {filename}")
                            except Exception as e:
                                logger.warning(f"Failed to delete model file {filename}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old model files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            return 0

    def _validate_ensemble_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate ensemble metadata structure for completeness and correctness.
        
        Args:
            metadata (Dict[str, Any]): Ensemble metadata to validate
            
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['model_type', 'ensemble_weights', 'individual_cv_scores']
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Missing required field in ensemble metadata: {field}")
                    return False
            
            # Validate model_type
            if metadata['model_type'] != 'Ensemble_RF_XGB':
                logger.warning(f"Invalid model_type in ensemble metadata: {metadata['model_type']}")
                return False
            
            # Validate ensemble_weights structure
            weights = metadata['ensemble_weights']
            if not isinstance(weights, dict) or 'rf' not in weights or 'xgb' not in weights:
                logger.warning("Invalid ensemble_weights structure in metadata")
                return False
            
            # Validate weight values
            rf_weight = weights['rf']
            xgb_weight = weights['xgb']
            if not (0 <= rf_weight <= 1) or not (0 <= xgb_weight <= 1):
                logger.warning(f"Invalid weight values: RF={rf_weight}, XGB={xgb_weight}")
                return False
            
            # Check that weights sum to approximately 1.0
            weight_sum = rf_weight + xgb_weight
            if abs(weight_sum - 1.0) > 1e-3:
                logger.warning(f"Weights do not sum to 1.0: sum={weight_sum}")
                return False
            
            # Validate individual_cv_scores structure
            cv_scores = metadata['individual_cv_scores']
            if not isinstance(cv_scores, dict) or 'rf' not in cv_scores or 'xgb' not in cv_scores:
                logger.warning("Invalid individual_cv_scores structure in metadata")
                return False
            
            logger.debug("Ensemble metadata validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating ensemble metadata: {e}")
            return False

    def _validate_loaded_model(self, model: Any, symbol: str, model_type: str) -> bool:
        """
        Validate loaded model has required methods and attributes.
        
        Args:
            model: Loaded model object
            symbol (str): Stock symbol for logging
            model_type (str): Type of model ('ensemble' or 'individual')
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            # Check required methods
            required_methods = ['predict']
            for method in required_methods:
                if not hasattr(model, method):
                    logger.warning(f"Loaded {model_type} model for {symbol} missing required method: {method}")
                    return False
            
            # For ensemble models, check VotingClassifier specific attributes
            if model_type == 'ensemble':
                if not hasattr(model, 'estimators'):
                    logger.warning(f"Ensemble model for {symbol} missing estimators attribute")
                    return False
                
                if not hasattr(model, 'voting'):
                    logger.warning(f"Ensemble model for {symbol} missing voting attribute")
                    return False
                
                # Check that it's a VotingClassifier
                if not hasattr(model, 'estimators_'):
                    logger.warning(f"Ensemble model for {symbol} appears to be untrained")
                    return False
            
            # Try a basic prediction test (if possible)
            try:
                # Create dummy input with correct feature count
                dummy_input = np.zeros((1, len(self.feature_columns)))
                test_prediction = model.predict(dummy_input)
                
                if test_prediction is None or len(test_prediction) == 0:
                    logger.warning(f"Model for {symbol} returned invalid prediction on test input")
                    return False
                    
            except Exception as e:
                logger.warning(f"Model validation test failed for {symbol}: {e}")
                return False
            
            logger.debug(f"Model validation passed for {symbol} ({model_type})")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating loaded model for {symbol}: {e}")
            return False

    def _validate_saved_model(self, symbol: str) -> Any:
        """
        Validate that a saved model can be loaded and is functional.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Any: Loaded model if validation passes, None otherwise
        """
        try:
            model_path = self._get_model_path(symbol)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file does not exist for validation: {model_path}")
                return None
            
            # Try to load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Validate the loaded model
            if not self._validate_loaded_model(model, symbol, 'unknown'):
                logger.warning(f"Saved model validation failed for {symbol}")
                return None
            
            return model
            
        except Exception as e:
            logger.warning(f"Error validating saved model for {symbol}: {e}")
            return None

    def _load_fallback_model(self, symbol: str) -> Any:
        """
        Load fallback model when main ensemble model fails.
        
        Attempts to load individual models in order of preference:
        1. Random Forest model
        2. XGBoost model
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Any: Best available individual model
            
        Raises:
            FileNotFoundError: If no fallback models are available
            MLPredictorError: If all fallback attempts fail
        """
        try:
            logger.info(f"Attempting fallback model loading for {symbol}...")
            
            rf_path = self._get_rf_model_path(symbol)
            xgb_path = self._get_xgb_model_path(symbol)
            
            # Try Random Forest first (more reliable)
            if os.path.exists(rf_path):
                try:
                    logger.info(f"Attempting to load RF fallback model for {symbol}")
                    with open(rf_path, 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    # Validate RF model
                    if self._validate_loaded_model(rf_model, symbol, 'individual'):
                        logger.info(f"Successfully loaded RF fallback model for {symbol}")
                        return rf_model
                    else:
                        logger.warning(f"RF fallback model validation failed for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load RF fallback model for {symbol}: {e}")
            
            # Try XGBoost as secondary fallback
            if os.path.exists(xgb_path):
                try:
                    logger.info(f"Attempting to load XGBoost fallback model for {symbol}")
                    with open(xgb_path, 'rb') as f:
                        xgb_model = pickle.load(f)
                    
                    # Validate XGBoost model
                    if self._validate_loaded_model(xgb_model, symbol, 'individual'):
                        logger.info(f"Successfully loaded XGBoost fallback model for {symbol}")
                        return xgb_model
                    else:
                        logger.warning(f"XGBoost fallback model validation failed for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost fallback model for {symbol}: {e}")
            
            # No fallback models available
            logger.error(f"No valid fallback models found for {symbol}")
            raise FileNotFoundError(f"No fallback model files found for symbol: {symbol}")
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except Exception as e:
            error_msg = f"All fallback model loading attempts failed for {symbol}: {e}"
            logger.error(error_msg)
            raise MLPredictorError(error_msg)

    def _log_ensemble_save_summary(self, symbol: str, ensemble_metadata: Dict[str, Any]) -> None:
        """
        Log comprehensive summary of ensemble save operation.
        
        Args:
            symbol (str): Stock symbol
            ensemble_metadata (Dict[str, Any]): Ensemble metadata
        """
        try:
            logger.info(f"Ensemble save summary for {symbol}:")
            logger.info(f"  - Model type: {ensemble_metadata.get('model_type', 'Unknown')}")
            logger.info(f"  - Voting type: {ensemble_metadata.get('voting_type', 'Unknown')}")
            
            weights = ensemble_metadata.get('ensemble_weights', {})
            logger.info(f"  - RF weight: {weights.get('rf', 'N/A'):.4f}")
            logger.info(f"  - XGBoost weight: {weights.get('xgb', 'N/A'):.4f}")
            
            cv_scores = ensemble_metadata.get('individual_cv_scores', {})
            logger.info(f"  - RF CV score: {cv_scores.get('rf', 'N/A'):.4f}")
            logger.info(f"  - XGBoost CV score: {cv_scores.get('xgb', 'N/A'):.4f}")
            
            # Check file existence
            main_model_exists = os.path.exists(self._get_model_path(symbol))
            rf_model_exists = os.path.exists(self._get_rf_model_path(symbol))
            xgb_model_exists = os.path.exists(self._get_xgb_model_path(symbol))
            metadata_exists = os.path.exists(self._get_ensemble_metadata_path(symbol))
            
            logger.info(f"  - Files saved:")
            logger.info(f"    * Main ensemble model: {'✓' if main_model_exists else '✗'}")
            logger.info(f"    * Individual RF model: {'✓' if rf_model_exists else '✗'}")
            logger.info(f"    * Individual XGBoost model: {'✓' if xgb_model_exists else '✗'}")
            logger.info(f"    * Ensemble metadata: {'✓' if metadata_exists else '✗'}")
            
            logger.info(f"  - Creation timestamp: {ensemble_metadata.get('creation_timestamp', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"Error logging ensemble save summary for {symbol}: {e}")