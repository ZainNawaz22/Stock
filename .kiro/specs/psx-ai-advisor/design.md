# Design Document

## Overview

The PSX AI Advisor is a comprehensive stock analysis system that downloads and extracts daily stock data from the Pakistan Stock Exchange (PSX) Closing Rate Summary PDF, calculates technical indicators, and uses machine learning to predict next-day price movements. The system features both a Python-based backend for data processing and a modern web interface that provides interactive dashboards, real-time charts, and intuitive visualization of stock analysis and predictions.

## Architecture

The system follows a modern full-stack architecture with clear separation of concerns:

```
PSX AI Advisor
├── Backend Services
│   ├── Data Acquisition Layer (PDF Download + Extraction)
│   ├── Data Processing Layer (Pandas + Technical Analysis)
│   ├── Storage Layer (CSV Files + Database)
│   ├── Machine Learning Layer (Scikit-learn)
│   └── API Layer (FastAPI REST Endpoints)
└── Frontend Application
    ├── Web Interface (React/Vue.js)
    ├── Chart Visualization (Chart.js/D3.js)
    ├── Real-time Updates (WebSocket/SSE)
    └── Responsive Design (CSS Framework)
```

### Key Architectural Principles

- **Modularity**: Each component has a single responsibility
- **Extensibility**: Easy to add new indicators or data sources
- **Reliability**: Robust error handling and retry mechanisms
- **Performance**: Efficient data processing and storage
- **Maintainability**: Clean, well-documented code structure

## Components and Interfaces

### 1. Data Acquisition Module (`data_acquisition.py`)

**Purpose**: Download and extract stock data from PSX Closing Rate Summary PDF

**Key Functions**:
- `download_daily_pdf(date)`: Download the Closing Rate Summary PDF for a specific date
- `extract_stock_data(pdf_path)`: Extract OHLCV data from the PDF file
- `parse_pdf_content(pdf_content)`: Parse PDF content to structured data
- `retry_with_backoff(func, max_retries=3)`: Retry mechanism for network failures

**Dependencies**: 
- Requests for HTTP operations
- PyPDF2 or pdfplumber for PDF parsing
- Pandas for data structuring

**Interface**:
```python
class PSXDataAcquisition:
    def __init__(self, base_url="https://dps.psx.com.pk"):
        self.base_url = base_url
        self.downloads_url = f"{base_url}/downloads"
    
    def download_daily_pdf(self, date: str = None) -> str
    def extract_stock_data(self, pdf_path: str) -> pd.DataFrame
    def get_all_stock_data(self, date: str = None) -> pd.DataFrame
```

### 2. Technical Analysis Module (`technical_analysis.py`)

**Purpose**: Calculate technical indicators from raw OHLCV data

**Key Functions**:
- `calculate_sma(data, period)`: Simple Moving Average
- `calculate_rsi(data, period)`: Relative Strength Index
- `calculate_macd(data, fast, slow, signal)`: MACD indicator
- `add_all_indicators(df)`: Add all required indicators to DataFrame

**Dependencies**:
- Pandas for data manipulation
- Pandas-TA or TA-Lib for technical indicators

**Interface**:
```python
class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame
```

### 3. Data Storage Module (`data_storage.py`)

**Purpose**: Manage persistent storage of stock data in CSV format

**Key Functions**:
- `save_stock_data(symbol, data)`: Save/append data to CSV file
- `load_stock_data(symbol)`: Load historical data from CSV
- `ensure_data_directory()`: Create data directory structure
- `validate_data_integrity(symbol)`: Check data consistency

**Interface**:
```python
class DataStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool
    def load_stock_data(self, symbol: str) -> pd.DataFrame
    def get_available_symbols(self) -> List[str]
    def validate_data_integrity(self, symbol: str) -> bool
```

### 4. Machine Learning Module (`ml_predictor.py`)

**Purpose**: Train models and generate predictions for stock price movements

**Key Functions**:
- `prepare_features(df)`: Prepare feature matrix from technical indicators
- `train_model(symbol)`: Train Random Forest model for specific stock
- `predict_movement(symbol)`: Predict next-day price movement
- `evaluate_model(symbol)`: Calculate model performance metrics

**Dependencies**:
- Scikit-learn for machine learning
- NumPy for numerical operations

**Interface**:
```python
class MLPredictor:
    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type
        self.models = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
    def train_model(self, symbol: str) -> Dict[str, Any]
    def predict_movement(self, symbol: str) -> Dict[str, Any]
    def save_model(self, symbol: str, model: Any) -> bool
    def load_model(self, symbol: str) -> Any
```

### 5. API Backend Module (`api_server.py`)

**Purpose**: Provide REST API endpoints for web interface and orchestrate backend workflows

**Key Functions**:
- `get_stock_data(symbol, period)`: Return stock data with technical indicators
- `get_predictions()`: Return current ML predictions for all stocks
- `get_stock_list()`: Return available stock symbols
- `run_daily_analysis()`: Execute complete daily workflow
- `get_system_status()`: Return system health and last update info

**Dependencies**:
- FastAPI for REST API framework
- Uvicorn for ASGI server
- Pydantic for data validation

**Interface**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PSX AI Advisor API")

@app.get("/api/stocks")
async def get_stocks() -> List[StockInfo]

@app.get("/api/stocks/{symbol}/data")
async def get_stock_data(symbol: str, days: int = 30) -> StockDataResponse

@app.get("/api/predictions")
async def get_predictions() -> List[PredictionResult]

@app.post("/api/analysis/run")
async def trigger_analysis() -> AnalysisStatus
```

### 6. Web Frontend Application

**Purpose**: Provide interactive web interface for data visualization and analysis

**Technology Stack**:
- **Frontend Framework**: React.js with TypeScript
- **Charting Library**: Chart.js or Recharts for interactive charts
- **UI Framework**: Material-UI or Tailwind CSS for responsive design
- **State Management**: React Context API or Redux Toolkit
- **HTTP Client**: Axios for API communication

**Key Components**:
- `Dashboard`: Main overview with key metrics and predictions
- `StockChart`: Interactive OHLCV charts with technical indicators
- `PredictionPanel`: Display ML predictions with confidence scores
- `StockList`: Filterable and sortable list of all stocks
- `AnalysisControls`: Trigger manual analysis and system controls

**Component Structure**:
```
src/
├── components/
│   ├── Dashboard/
│   ├── StockChart/
│   ├── PredictionPanel/
│   ├── StockList/
│   └── common/
├── services/
│   ├── api.ts
│   └── websocket.ts
├── hooks/
│   ├── useStockData.ts
│   └── usePredictions.ts
├── types/
│   └── api.ts
└── utils/
    └── formatters.ts
```

### 7. Main Application Module (`main.py`)

**Purpose**: Orchestrate the entire workflow and provide CLI interface

**Key Functions**:
- `run_daily_analysis()`: Execute complete daily workflow
- `display_predictions()`: Show formatted predictions
- `handle_errors()`: Centralized error handling
- `setup_logging()`: Configure logging system

## Data Models

### Stock Data Schema

```python
StockData = {
    'Date': datetime,
    'Symbol': str,
    'Open': float,
    'High': float,
    'Low': float,
    'Close': float,
    'Volume': int,
    'SMA_50': float,
    'SMA_200': float,
    'RSI_14': float,
    'MACD': float,
    'MACD_Signal': float,
    'MACD_Histogram': float,
    'Target': int  # 1 for UP, 0 for DOWN (for ML training)
}
```

### Prediction Output Schema

```python
PredictionResult = {
    'symbol': str,
    'current_price': float,
    'prediction': str,  # 'UP' or 'DOWN'
    'confidence': float,  # 0.0 to 1.0
    'model_accuracy': float,
    'last_updated': datetime
}
```

### API Response Schemas

```python
# Stock data response for API
StockDataResponse = {
    'symbol': str,
    'data': List[StockData],
    'technical_indicators': Dict[str, List[float]],
    'last_updated': datetime,
    'total_records': int
}

# Stock info for listing
StockInfo = {
    'symbol': str,
    'name': str,
    'sector': str,
    'current_price': float,
    'change_percent': float,
    'volume': int,
    'last_updated': datetime
}

# System status response
SystemStatus = {
    'status': str,  # 'healthy', 'updating', 'error'
    'last_analysis': datetime,
    'total_stocks': int,
    'predictions_available': int,
    'system_uptime': str
}
```

## Error Handling

### Error Categories and Strategies

1. **Network Errors**
   - Implement exponential backoff retry mechanism
   - Fallback to cached data if available
   - Log failures for manual intervention

2. **Data Quality Issues**
   - Validate data ranges and formats
   - Handle missing values with forward fill
   - Skip corrupted data points with warnings

3. **Model Training Failures**
   - Require minimum data points (e.g., 200 days)
   - Handle insufficient data gracefully
   - Provide fallback predictions based on technical indicators

4. **File System Errors**
   - Create directories automatically
   - Handle permission issues
   - Implement data backup mechanisms

### Error Handling Implementation

```python
class PSXAdvisorError(Exception):
    """Base exception for PSX Advisor"""
    pass

class DataScrapingError(PSXAdvisorError):
    """Raised when data scraping fails"""
    pass

class InsufficientDataError(PSXAdvisorError):
    """Raised when not enough data for analysis"""
    pass

class ModelTrainingError(PSXAdvisorError):
    """Raised when model training fails"""
    pass
```

## Testing Strategy

### Unit Testing

- **Data Acquisition Tests**: Mock PDF downloads, test PDF parsing logic
- **Technical Analysis Tests**: Verify indicator calculations against known values
- **Storage Tests**: Test CSV read/write operations, data integrity
- **ML Tests**: Test feature preparation, model training with synthetic data

### Integration Testing

- **End-to-End Workflow**: Test complete pipeline with sample data
- **Error Scenarios**: Test system behavior under various failure conditions
- **Performance Tests**: Ensure system meets timing requirements

### Test Data Strategy

- Use historical PSX data samples for realistic testing
- Create synthetic data for edge cases
- Mock external dependencies (PDF downloads) for reliable testing

## Configuration Management

### Configuration File Structure (`config.yaml`)

```yaml
data_sources:
  psx_base_url: "https://dps.psx.com.pk"
  downloads_endpoint: "/downloads"
  pdf_filename_pattern: "closing_rates/{date}.pdf"

technical_indicators:
  sma_periods: [50, 200]
  rsi_period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9

machine_learning:
  model_type: "RandomForest"
  min_training_samples: 200
  test_size: 0.2
  random_state: 42
  n_estimators: 100

storage:
  data_directory: "data"
  backup_directory: "backups"
  max_file_age_days: 365

api_server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["http://localhost:3000", "http://localhost:5173"]
  max_request_size: "10MB"
  rate_limit: "100/minute"

web_interface:
  title: "PSX AI Advisor"
  default_chart_period: 30
  auto_refresh_interval: 300  # seconds
  max_stocks_per_page: 50
  theme: "light"

logging:
  level: "INFO"
  file: "psx_advisor.log"
  max_size_mb: 10
  backup_count: 5

performance:
  max_concurrent_requests: 5
  request_timeout: 30
  retry_attempts: 3
  retry_delay: 2
```

## Security Considerations

### Data Protection
- No sensitive financial data stored permanently
- Local CSV files with appropriate file permissions
- No API keys or credentials in code

### Data Download Ethics
- Use official download endpoints provided by PSX
- Implement reasonable delays between requests
- Respect server resources and avoid excessive requests

### Legal Compliance
- Acknowledge PSX data usage restrictions
- Include appropriate disclaimers about investment advice
- Ensure personal use only as specified in requirements

## Web Interface Architecture

### Frontend Design Principles

- **Responsive Design**: Mobile-first approach with breakpoints for tablet and desktop
- **Performance**: Lazy loading, code splitting, and optimized bundle sizes
- **Accessibility**: WCAG 2.1 compliance with proper ARIA labels and keyboard navigation
- **User Experience**: Intuitive navigation, consistent design patterns, and fast interactions

### Key UI Components

#### Dashboard Layout
```
┌─────────────────────────────────────────────────────────┐
│ Header (Logo, Navigation, System Status)                │
├─────────────────────────────────────────────────────────┤
│ Summary Cards (Total Stocks, Predictions, Performance) │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────────────────────┐ │
│ │ Top Predictions │ │ Market Overview Chart           │ │
│ │ (UP/DOWN)       │ │ (Index Performance)             │ │
│ └─────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Stock List with Search/Filter                           │
└─────────────────────────────────────────────────────────┘
```

#### Stock Detail View
```
┌─────────────────────────────────────────────────────────┐
│ Stock Header (Symbol, Name, Current Price, Change)     │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Interactive OHLCV Chart with Technical Indicators   │ │
│ │ (Candlestick + SMA + RSI + MACD)                   │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│ │ Prediction  │ │ Technical   │ │ Historical          │ │
│ │ Panel       │ │ Indicators  │ │ Performance         │ │
│ └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Real-time Data Updates

- **WebSocket Connection**: For live price updates and new predictions
- **Polling Fallback**: HTTP polling every 5 minutes as backup
- **Optimistic Updates**: Immediate UI updates with server confirmation
- **Error Handling**: Graceful degradation when real-time connection fails

### Chart Visualization Features

- **Interactive Charts**: Zoom, pan, crosshair, and tooltip interactions
- **Technical Indicators**: Toggle visibility of SMA, RSI, MACD overlays
- **Time Periods**: 1D, 7D, 1M, 3M, 6M, 1Y view options
- **Chart Types**: Candlestick, line, and area chart options
- **Export**: Save charts as PNG/SVG for reports

## Performance Optimization

### Data Acquisition Performance
- Efficient PDF download and caching
- Optimized PDF parsing with appropriate libraries
- Cache daily PDF files to avoid re-downloading

### Data Processing Performance
- Vectorized operations with Pandas/NumPy
- Incremental processing for new data only
- Memory-efficient data structures

### Storage Optimization
- Compress CSV files for long-term storage
- Implement data archiving for old records
- Use efficient data types (float32 vs float64)

### Web Interface Performance
- **Frontend Optimization**: Code splitting, lazy loading, and tree shaking
- **API Caching**: Redis cache for frequently requested data
- **CDN Integration**: Static asset delivery through CDN
- **Database Indexing**: Optimize queries for stock data retrieval
- **Compression**: Gzip compression for API responses

## Deployment and Operations

### System Requirements
- Python 3.8+
- Node.js 16+ (for frontend development)
- 4GB RAM minimum (increased for web server)
- 2GB disk space for data storage
- Stable internet connection

### Installation Process

#### Backend Setup
1. Clone repository
2. Install Python dependencies: `pip install -r requirements.txt` (includes fastapi, uvicorn, requests, pandas, pdfplumber, pandas-ta, scikit-learn)
3. Configure settings in `config.yaml`
4. Run initial setup: `python setup.py`
5. Start API server: `uvicorn api_server:app --host 0.0.0.0 --port 8000`

#### Frontend Setup
1. Navigate to frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Configure API endpoint in `.env` file
4. Start development server: `npm run dev`
5. Build for production: `npm run build`

#### Production Deployment
1. Use Docker containers for both backend and frontend
2. Set up reverse proxy (Nginx) for serving static files
3. Configure SSL certificates for HTTPS
4. Set up process manager (PM2) for Node.js applications

### Monitoring and Maintenance
- Daily log review for errors
- Weekly data integrity checks
- Monthly model performance evaluation
- Quarterly dependency updates

## Future Enhancements

### Phase 2 Potential Features
- Email/SMS alerts for predictions
- Portfolio tracking and recommendations
- Integration with additional data sources
- Advanced ML models (LSTM, Transformer)
- Mobile application (React Native)
- Advanced charting features (candlestick patterns, drawing tools)
- User authentication and personalized dashboards

### Scalability Considerations
- Database migration from CSV files
- Distributed processing for multiple markets
- Real-time data streaming capabilities
- Cloud deployment options