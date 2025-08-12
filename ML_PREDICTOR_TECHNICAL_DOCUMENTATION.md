# ML Predictor Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Feature Engineering](#feature-engineering)
6. [Model Training & Optimization](#model-training--optimization)
7. [Prediction System](#prediction-system)
8. [Performance Evaluation](#performance-evaluation)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## System Overview

The ML Predictor module is a sophisticated machine learning system designed for stock price movement prediction in the PSX (Pakistan Stock Exchange) AI Advisor. It employs Random Forest classification to predict next-day price movements (UP/DOWN) based on technical indicators and historical price data.

### Key Features
- **Time-Series Aware**: Implements proper time-series cross-validation to prevent lookahead bias
- **Hyperparameter Optimization**: Automated hyperparameter tuning using RandomizedSearchCV
- **Decision Threshold Tuning**: Dynamic threshold optimization for improved prediction accuracy
- **Feature Engineering**: Comprehensive technical indicator calculation
- **Model Persistence**: Automatic model and scaler serialization/deserialization
- **Batch Processing**: Support for multiple stock symbol predictions
- **Performance Monitoring**: Comprehensive evaluation metrics and model assessment

### Technology Stack
- **ML Framework**: scikit-learn (Random Forest Classifier)
- **Data Processing**: pandas, numpy
- **Technical Analysis**: Custom implementation with TA-Lib principles
- **Configuration**: YAML-based configuration management
- **Logging**: Structured logging with error tracking

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Storage  │    │ Technical       │    │   ML Predictor  │
│                 │    │ Analysis        │    │                 │
│ - Stock Data    │───▶│ - Indicators    │───▶│ - Feature Prep  │
│ - Model Cache   │    │ - Calculations  │    │ - Model Training│
│ - Scalers       │    │ - Validation    │    │ - Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Configuration   │
                    │ Management      │
                    │ - ML Params     │
                    │ - Thresholds    │
                    │ - Paths         │
                    └─────────────────┘
```

### Component Dependencies
- **MLPredictor**: Main orchestrator class
- **DataStorage**: Handles data loading and persistence
- **TechnicalAnalyzer**: Calculates technical indicators
- **ConfigLoader**: Manages configuration settings
- **Exception Handling**: Custom exception hierarchy

## Core Components

### MLPredictor Class
The central class that orchestrates the entire ML pipeline.

**Key Responsibilities:**
- Model training and optimization
- Feature preparation and scaling
- Prediction generation
- Model persistence and loading
- Performance evaluation

**Initialization Parameters:**
```python
def __init__(self, model_type: str = "RandomForest"):
    # Loads configuration from config.yaml
    # Sets up directories for models and scalers
    # Initializes technical analyzer and data storage
```

### Feature Engineering System
The system uses 16 technical indicators as features:

**Core Technical Indicators:**
- **RSI_14**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **MACD_Signal**: MACD signal line
- **MACD_Histogram**: MACD histogram
- **ROC_12**: Rate of Change (12-period)
- **SMA_20**: Simple Moving Average (20-period)
- **SMA_50**: Simple Moving Average (50-period)
- **BB_Upper**: Bollinger Bands upper band
- **BB_Middle**: Bollinger Bands middle band
- **BB_Lower**: Bollinger Bands lower band
- **Volume_MA_20**: Volume Moving Average (20-period)
- **OBV**: On-Balance Volume
- **Return_1d**: 1-day return
- **Return_2d**: 2-day return
- **Return_5d**: 5-day return
- **Volatility_20d**: 20-day rolling volatility

## Machine Learning Pipeline

### 1. Data Preparation Pipeline
```
Raw Stock Data → Technical Indicators → Feature Matrix → Target Variable → Scaling
```

**Steps:**
1. **Data Loading**: Load historical OHLCV data
2. **Technical Analysis**: Calculate all required indicators
3. **Feature Matrix Creation**: Extract feature columns
4. **Target Variable**: Create binary target (UP=1, DOWN=0)
5. **Data Cleaning**: Handle missing values and outliers
6. **Scaling**: Standardize features using StandardScaler

### 2. Model Training Pipeline
```
Feature Matrix → TimeSeriesSplit → Hyperparameter Optimization → Model Training → Evaluation
```

**Time-Series Cross-Validation:**
- Uses `TimeSeriesSplit` to prevent lookahead bias
- Configurable number of splits (default: 5)
- Maintains temporal order of data

**Hyperparameter Optimization:**
```python
param_distributions = {
    'n_estimators': randint(50, 301),
    'max_depth': randint(5, 21),
    'min_samples_split': randint(2, 16),
    'min_samples_leaf': randint(1, 9),
    'max_features': ['sqrt', 'log2', 0.5]
}
```

### 3. Prediction Pipeline
```
Latest Data → Feature Extraction → Scaling → Model Prediction → Threshold Application → Result
```

## Feature Engineering

### Technical Indicator Calculations

**RSI (Relative Strength Index):**
```python
def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
    # Calculate price changes
    delta = data.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**MACD (Moving Average Convergence Divergence):**
```python
def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })
```

**Bollinger Bands:**
```python
def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return pd.DataFrame({
        'BB_Upper': upper_band,
        'BB_Middle': sma,
        'BB_Lower': lower_band
    })
```

### Feature Selection Strategy
- **Momentum Indicators**: RSI, ROC, Returns
- **Trend Indicators**: SMA, MACD
- **Volatility Indicators**: Bollinger Bands, Volatility
- **Volume Indicators**: OBV, Volume MA
- **Price Action**: Multiple return periods

## Model Training & Optimization

### Training Process
1. **Data Validation**: Ensure sufficient data (minimum 200 samples)
2. **Time-Series Split**: Use TimeSeriesSplit for temporal validation
3. **Feature Scaling**: Standardize features using StandardScaler
4. **Hyperparameter Optimization**: RandomizedSearchCV with TimeSeriesSplit
5. **Model Training**: Train Random Forest with optimized parameters
6. **Threshold Tuning**: Optimize decision threshold for better performance
7. **Cross-Validation**: Perform time-series cross-validation
8. **Model Persistence**: Save model and scaler to disk

### Hyperparameter Optimization
```python
def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    param_distributions = {
        'n_estimators': randint(50, 301),
        'max_depth': randint(5, 21),
        'min_samples_split': randint(2, 16),
        'min_samples_leaf': randint(1, 9),
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    tscv = TimeSeriesSplit(n_splits=self.n_splits)
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=self.random_state),
        param_distributions=param_distributions,
        n_iter=50,
        cv=tscv,
        scoring='accuracy',
        random_state=self.random_state,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    return random_search.best_estimator_
```

### Decision Threshold Tuning
The system implements dynamic threshold tuning to optimize prediction accuracy:

```python
def _tune_threshold(self, y_true: np.ndarray, y_proba_up: np.ndarray) -> Tuple[float, float]:
    thresholds = np.arange(self.threshold_min, self.threshold_max + 1e-9, self.threshold_step)
    best_threshold = 0.5
    best_score = -np.inf
    
    for t in thresholds:
        y_pred = (y_proba_up >= t).astype(int)
        if self.threshold_metric == 'utility':
            # Calculate utility-based score
            score = self._calculate_utility_score(y_true, y_pred)
        else:
            # Calculate F1 score
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score
```

## Prediction System

### Prediction Workflow
1. **Model Loading**: Load trained model and scaler
2. **Data Preparation**: Get latest stock data and calculate indicators
3. **Feature Extraction**: Extract feature vector from latest data
4. **Scaling**: Apply StandardScaler transformation
5. **Prediction**: Generate probability predictions
6. **Threshold Application**: Apply tuned decision threshold
7. **Result Formatting**: Format prediction results

### Prediction Output Format
```python
{
    'symbol': 'ENGRO_HISTORICAL_DATA',
    'prediction': 'UP',
    'confidence': 0.75,
    'prediction_probabilities': {
        'DOWN': 0.25,
        'UP': 0.75
    },
    'decision_threshold': 0.6,
    'current_price': 150.50,
    'prediction_date': '2024-01-15T10:30:00',
    'data_date': '2024-01-14T00:00:00',
    'model_accuracy': 0.68,
    'model_type': 'RandomForest',
    'feature_count': 16
}
```

### Batch Prediction
The system supports batch prediction for multiple symbols:

```python
def get_batch_predictions(self, symbols: List[str]) -> List[Dict[str, Any]]:
    predictions = []
    for symbol in symbols:
        try:
            prediction = self.predict_movement(symbol)
            predictions.append(prediction)
        except Exception as e:
            # Handle errors gracefully
            error_result = {
                'symbol': symbol,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
            predictions.append(error_result)
    return predictions
```

## Performance Evaluation

### Evaluation Metrics
The system provides comprehensive evaluation metrics:

**Classification Metrics:**
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for UP and DOWN predictions
- **Recall**: Recall for UP and DOWN predictions
- **F1 Score**: Harmonic mean of precision and recall

**Time-Series Specific Metrics:**
- **Cross-Validation Scores**: Time-series CV performance
- **Recent Accuracy**: Performance on recent data
- **Confidence Statistics**: Prediction confidence analysis

### Evaluation Process
```python
def evaluate_model(self, symbol: str) -> Dict[str, Any]:
    # Load model and data
    # Prepare features and targets
    # Make predictions on full dataset
    # Calculate comprehensive metrics
    # Generate evaluation report
```

### Performance Monitoring
- **Feature Importance**: Track most important features
- **Model Drift**: Monitor performance over time
- **Confidence Analysis**: Analyze prediction confidence distribution
- **Class Distribution**: Monitor UP/DOWN class balance

## Configuration

### Configuration Structure
The system uses YAML-based configuration:

```yaml
machine_learning:
  min_training_samples: 200
  random_state: 42
  n_estimators: 100
  n_splits: 5
  max_train_size: null
  
  threshold_tuning:
    enabled: true
    metric: "f1"
    min_threshold: 0.3
    max_threshold: 0.7
    step: 0.05
    utility:
      tp_reward: 1.0
      tn_reward: 0.0
      fp_cost: 1.0
      fn_cost: 1.0

storage:
  data_directory: "data"
  models_directory: "data/models"
  scalers_directory: "data/scalers"
```

### Environment-Specific Configuration
- **Development**: Default settings with debug logging
- **Production**: Optimized settings with error logging
- **Testing**: Minimal settings for unit tests

## API Reference

### Core Methods

#### `train_model(symbol: str, optimize_params: bool = True) -> Dict[str, Any]`
Train a Random Forest model for a specific stock symbol.

**Parameters:**
- `symbol`: Stock symbol to train model for
- `optimize_params`: Whether to use hyperparameter optimization

**Returns:**
- Training results including metrics and model info

**Example:**
```python
predictor = MLPredictor()
results = predictor.train_model("ENGRO", optimize_params=True)
print(f"Training accuracy: {results['accuracy']:.4f}")
```

#### `predict_movement(symbol: str) -> Dict[str, Any]`
Predict next-day price movement for a specific stock symbol.

**Parameters:**
- `symbol`: Stock symbol to predict for

**Returns:**
- Prediction results including direction, confidence, and metadata

**Example:**
```python
prediction = predictor.predict_movement("ENGRO")
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

#### `evaluate_model(symbol: str) -> Dict[str, Any]`
Evaluate model performance on historical data.

**Parameters:**
- `symbol`: Stock symbol to evaluate

**Returns:**
- Comprehensive evaluation metrics and performance data

**Example:**
```python
evaluation = predictor.evaluate_model("ENGRO")
print(f"Overall accuracy: {evaluation['overall_metrics']['accuracy']:.4f}")
```

#### `get_batch_predictions(symbols: List[str]) -> List[Dict[str, Any]]`
Generate predictions for multiple stock symbols.

**Parameters:**
- `symbols`: List of stock symbols to predict

**Returns:**
- List of prediction results

**Example:**
```python
symbols = ["ENGRO", "HBL", "MCB"]
predictions = predictor.get_batch_predictions(symbols)
for pred in predictions:
    print(f"{pred['symbol']}: {pred['prediction']}")
```

### Utility Methods

#### `get_model_info(symbol: str) -> Dict[str, Any]`
Get information about a trained model.

#### `get_available_models() -> List[str]`
Get list of symbols that have trained models.

#### `cleanup_old_models(days_to_keep: int = 90) -> int`
Clean up old model files.

#### `tune_threshold_for_symbol(symbol: str) -> Dict[str, Any]`
Tune decision threshold for a specific symbol.

## Best Practices

### Model Training
1. **Data Quality**: Ensure clean, sufficient historical data
2. **Feature Engineering**: Use comprehensive technical indicators
3. **Time-Series Validation**: Always use time-series cross-validation
4. **Hyperparameter Tuning**: Enable optimization for better performance
5. **Threshold Tuning**: Use threshold tuning for improved accuracy

### Prediction Usage
1. **Model Loading**: Load models once and reuse
2. **Error Handling**: Implement proper error handling for failed predictions
3. **Batch Processing**: Use batch predictions for multiple symbols
4. **Performance Monitoring**: Regularly evaluate model performance

### Configuration Management
1. **Environment Separation**: Use different configs for dev/prod
2. **Parameter Tuning**: Adjust parameters based on data characteristics
3. **Logging**: Configure appropriate logging levels
4. **Storage Management**: Regular cleanup of old models

### Performance Optimization
1. **Parallel Processing**: Use n_jobs=-1 for parallel training
2. **Memory Management**: Clean up old models regularly
3. **Caching**: Cache frequently used models in memory
4. **Batch Operations**: Use batch predictions for efficiency

## Troubleshooting

### Common Issues

#### Insufficient Data Error
**Problem:** "Insufficient data for training"
**Solution:** Ensure at least 200 data points are available

#### Missing Technical Indicators
**Problem:** "Missing required technical indicators"
**Solution:** Check if technical analysis module is working correctly

#### Model Loading Errors
**Problem:** "No model file found"
**Solution:** Train model first or check model file paths

#### Prediction Errors
**Problem:** "Prediction failed"
**Solution:** Check data availability and model state

### Debugging Tips
1. **Enable Debug Logging**: Set logging level to DEBUG
2. **Check Data Quality**: Validate input data format and completeness
3. **Monitor Performance**: Use evaluation methods to assess model health
4. **Verify Configuration**: Ensure all configuration parameters are correct

### Performance Issues
1. **Slow Training**: Reduce hyperparameter search iterations
2. **Memory Issues**: Clean up old models and reduce batch sizes
3. **Prediction Delays**: Cache models in memory for faster access

### Error Recovery
1. **Model Corruption**: Retrain model from scratch
2. **Data Issues**: Validate and clean input data
3. **Configuration Errors**: Reset to default configuration

---

## Version Information
- **Module Version**: 1.0.0
- **Last Updated**: 2024-01-15
- **Compatibility**: Python 3.8+, scikit-learn 1.0+

## Contributing
When contributing to the ML Predictor module:
1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include proper logging and documentation
4. Test with time-series cross-validation
5. Update this documentation for any new features
