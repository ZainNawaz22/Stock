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

from .config_loader import get_section, get_value
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

        self._validate_optional_dependencies()
    
    def _ensure_model_directories(self) -> None:
        """Create model and scaler directories if they don't exist"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.scalers_dir, exist_ok=True)
            logger.debug(f"Ensured model directories exist: {self.models_dir}, {self.scalers_dir}")
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
        
        logger.info("Starting XGBoost hyperparameter optimization...")
        
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
            
            # Perform randomized search
            logger.info("Performing RandomizedSearchCV for XGBoost with TimeSeriesSplit...")
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
            
            # Log the best parameters found
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            logger.info(f"XGBoost hyperparameter optimization completed")
            logger.info(f"Best XGBoost cross-validation score: {best_score:.4f}")
            logger.info(f"Best XGBoost parameters found: {best_params}")
            
            # Log parameter comparison for transparency
            logger.info("XGBoost optimization details:")
            logger.info(f"  - n_estimators: {best_params.get('n_estimators', 'N/A')}")
            logger.info(f"  - max_depth: {best_params.get('max_depth', 'N/A')}")
            logger.info(f"  - learning_rate: {best_params.get('learning_rate', 'N/A'):.4f}")
            logger.info(f"  - subsample: {best_params.get('subsample', 'N/A'):.4f}")
            logger.info(f"  - colsample_bytree: {best_params.get('colsample_bytree', 'N/A'):.4f}")
            logger.info(f"  - min_child_weight: {best_params.get('min_child_weight', 'N/A')}")
            logger.info(f"  - gamma: {best_params.get('gamma', 'N/A'):.4f}")
            
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
            logger.info("Starting ensemble weight calculation using cross-validation performance...")
            
            # Use configured TimeSeriesSplit for consistent evaluation
            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
            
            # Initialize score collectors
            rf_cv_scores = []
            xgb_cv_scores = []
            
            # Perform cross-validation for both models
            logger.info(f"Evaluating models using {self.n_splits}-fold TimeSeriesSplit cross-validation...")
            
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
                
                # Evaluate Random Forest model
                try:
                    # Create RF model with same parameters as the trained model
                    rf_params = rf_model.get_params()
                    rf_fold = RandomForestClassifier(**rf_params)
                    rf_fold.fit(X_train_fold_scaled, y_train_fold)
                    
                    # Make predictions and calculate accuracy
                    rf_pred_fold = rf_fold.predict(X_test_fold_scaled)
                    rf_accuracy_fold = accuracy_score(y_test_fold, rf_pred_fold)
                    rf_cv_scores.append(rf_accuracy_fold)
                    
                    logger.debug(f"Fold {fold + 1} - RF accuracy: {rf_accuracy_fold:.4f}")
                    
                except Exception as e:
                    logger.warning(f"RF evaluation failed on fold {fold + 1}: {e}")
                    # Use a default low score for failed folds
                    rf_cv_scores.append(0.5)
                
                # Evaluate XGBoost model
                try:
                    if self.xgboost_available and xgb_model is not None:
                        # Create XGBoost model with same parameters as the trained model
                        xgb_params = xgb_model.get_params()
                        xgb_fold = xgb.XGBClassifier(**xgb_params)
                        xgb_fold.fit(X_train_fold_scaled, y_train_fold)
                        
                        # Make predictions and calculate accuracy
                        xgb_pred_fold = xgb_fold.predict(X_test_fold_scaled)
                        xgb_accuracy_fold = accuracy_score(y_test_fold, xgb_pred_fold)
                        xgb_cv_scores.append(xgb_accuracy_fold)
                        
                        logger.debug(f"Fold {fold + 1} - XGBoost accuracy: {xgb_accuracy_fold:.4f}")
                    else:
                        # XGBoost not available, use default low score
                        xgb_cv_scores.append(0.5)
                        logger.debug(f"Fold {fold + 1} - XGBoost not available, using default score")
                        
                except Exception as e:
                    logger.warning(f"XGBoost evaluation failed on fold {fold + 1}: {e}")
                    # Use a default low score for failed folds
                    xgb_cv_scores.append(0.5)
            
            # Calculate mean CV scores for each model
            rf_mean_cv_score = float(np.mean(rf_cv_scores))
            xgb_mean_cv_score = float(np.mean(xgb_cv_scores))
            
            logger.info(f"Cross-validation results:")
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
                'xgb_scores': [float(score) for score in xgb_cv_scores]
            }
            
            # Log weight calculation results with transparency
            logger.info("Ensemble weight calculation completed:")
            logger.info(f"  - Random Forest weight: {rf_weight:.4f} (based on CV score: {rf_mean_cv_score:.4f})")
            logger.info(f"  - XGBoost weight: {xgb_weight:.4f} (based on CV score: {xgb_mean_cv_score:.4f})")
            logger.info(f"  - Weight sum verification: {rf_weight + xgb_weight:.6f}")
            
            # Additional transparency logging
            logger.info("Weight calculation transparency:")
            logger.info(f"  - RF individual fold scores: {[f'{score:.4f}' for score in rf_cv_scores]}")
            logger.info(f"  - XGBoost individual fold scores: {[f'{score:.4f}' for score in xgb_cv_scores]}")
            logger.info(f"  - Weight calculation method: Performance-based normalization")
            
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
            logger.info(f"  - RF CV score std: {individual_cv_scores.get('rf_std', 'N/A'):.4f}")
            logger.info(f"  - XGBoost CV score std: {individual_cv_scores.get('xgb_std', 'N/A'):.4f}")
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
        logger.info("Starting hyperparameter optimization...")
        
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
        
        # Create base Random Forest classifier
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=50,  # Number of parameter settings sampled
            cv=tscv,    # Use TimeSeriesSplit for cross-validation
            scoring='accuracy',  # Optimize for accuracy
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the randomized search
        random_search.fit(X, y)
        
        # Log the best parameters found
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        logger.info(f"Best parameters found: {best_params}")
        
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
        Train a Random Forest model for a specific stock symbol.
        
        Args:
            symbol (str): Stock symbol to train model for
            optimize_params (bool): Whether to use hyperparameter optimization (default: True)
            
        Returns:
            Dict[str, Any]: Training results including metrics and model info
            
        Raises:
            InsufficientDataError: If not enough data for training
            ModelTrainingError: If model training fails
        """
        try:
            symbol = self._normalize_symbol(symbol)
            logger.info(f"Starting model training for {symbol}")
            
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
            
            # Prepare training results
            training_results = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'model_type': self.model_type,
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
            
        except (InsufficientDataError, ValueError) as e:
            raise InsufficientDataError(f"Training failed for {symbol}: {e}")
        except Exception as e:
            raise ModelTrainingError(f"Model training failed for {symbol}: {e}")
    
    def predict_movement(self, symbol: str) -> Dict[str, Any]:
        """
        Predict next-day price movement for a specific stock symbol.
        
        Args:
            symbol (str): Stock symbol to predict for
            
        Returns:
            Dict[str, Any]: Prediction results including direction, confidence, and metadata
            
        Raises:
            MLPredictorError: If prediction fails
        """
        try:
            symbol = self._normalize_symbol(symbol)
            logger.info(f"Generating prediction for {symbol}")
            
            # Load or train model if not available
            if symbol not in self.models:
                model_path = self._get_model_path(symbol)
                scaler_path = self._get_scaler_path(symbol)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    # Load existing model and scaler
                    self.models[symbol] = self.load_model(symbol)
                    self.scalers[symbol] = self.load_scaler(symbol)
                    logger.info(f"Loaded existing model for {symbol}")
                else:
                    # Train new model
                    logger.info(f"No existing model found for {symbol}, training new model...")
                    training_results = self.train_model(symbol)
                    logger.info(f"Model training completed for {symbol}")
            
            # Load latest stock data
            try:
                stock_data = self.data_storage.load_stock_data(symbol)
            except FileNotFoundError:
                raise MLPredictorError(f"No data file found for symbol: {symbol}")
            
            if stock_data.empty:
                raise MLPredictorError(f"No data available for symbol: {symbol}")
            
            # Add technical indicators if not present
            missing_indicators = [col for col in self.feature_columns if col not in stock_data.columns]
            if missing_indicators:
                stock_data = self.technical_analyzer.add_all_indicators(stock_data)
            
            # Get the latest data point for prediction
            latest_data = stock_data.iloc[-1:][self.feature_columns]
            
            # Check for missing values
            if latest_data.isnull().any().any():
                logger.warning("Missing values in latest data, applying forward fill")
                latest_data = latest_data.ffill().bfill()
                
                if latest_data.isnull().any().any():
                    raise MLPredictorError(f"Unable to handle missing values in latest data for {symbol}")
            
            # Scale features
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            X_latest = scaler.transform(latest_data.values)
            
            # Make prediction
            prediction_proba = model.predict_proba(X_latest)[0]
            active_threshold = self.thresholds.get(symbol)
            if active_threshold is None:
                loaded_thr = self.load_threshold(symbol)
                if loaded_thr is not None:
                    self.thresholds[symbol] = loaded_thr
                    active_threshold = loaded_thr
            if active_threshold is None:
                active_threshold = 0.5
            up_proba = float(prediction_proba[1])
            prediction_class = 1 if up_proba >= active_threshold else 0
            
            prediction_direction = "UP" if prediction_class == 1 else "DOWN"
            confidence = float(max(prediction_proba))
            
            # Get current price and other metadata
            current_price = float(stock_data['Close'].iloc[-1])
            current_date = stock_data['Date'].iloc[-1] if 'Date' in stock_data.columns else datetime.now()
            
            # Calculate model accuracy from recent predictions (if available)
            model_accuracy = self._calculate_recent_accuracy(symbol, stock_data)
            
            prediction_result = {
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
                'model_type': self.model_type,
                'feature_count': len(self.feature_columns)
            }
            
            logger.info(f"Prediction for {symbol}: {prediction_direction} (confidence: {confidence:.4f})")
            
            return prediction_result
            
        except Exception as e:
            if isinstance(e, MLPredictorError):
                raise
            else:
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
    
    def save_model(self, symbol: str, model: Any) -> bool:
        """
        Save trained model to disk.
        
        Args:
            symbol (str): Stock symbol
            model: Trained model object
            
        Returns:
            bool: True if save successful
        """
        try:
            model_path = self._get_model_path(symbol)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.debug(f"Model saved for {symbol} at {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            return False
    
    def load_model(self, symbol: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Any: Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            MLPredictorError: If loading fails
        """
        try:
            model_path = self._get_model_path(symbol)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model file found for symbol: {symbol}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.debug(f"Model loaded for {symbol} from {model_path}")
            return model
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise MLPredictorError(f"Error loading model for {symbol}: {e}")
    
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