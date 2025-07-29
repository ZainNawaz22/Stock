# PSX AI Advisor Installation Guide

This guide provides step-by-step instructions for installing and setting up the PSX AI Advisor system.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 2GB RAM
- **Storage**: At least 1GB free disk space
- **Internet**: Stable connection for data downloads

### Python Dependencies
The system requires the following Python packages (automatically installed):
- `requests` - HTTP requests for data download
- `pandas` - Data manipulation and analysis
- `pandas-ta` - Technical analysis indicators
- `scikit-learn` - Machine learning algorithms
- `pyyaml` - YAML configuration file parsing
- `fastapi` - Web API framework (for web interface)
- `uvicorn` - ASGI server (for web interface)

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd psx-ai-advisor
   ```

2. **Run Automated Setup**
   ```bash
   python setup.py
   ```
   
   This will:
   - Install all required dependencies
   - Create necessary directories
   - Generate default configuration
   - Validate system setup

3. **Verify Installation**
   ```bash
   python setup.py --check-only
   ```

### Method 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Directories**
   ```bash
   mkdir data backups logs
   mkdir data/models data/scalers
   ```

3. **Create Configuration**
   ```bash
   python setup.py --no-deps
   ```

## Environment Configuration

### Development Environment (Default)
```bash
python setup.py --environment development
```
- Detailed logging enabled
- Debug information included
- Lower retry attempts for faster testing

### Production Environment
```bash
python setup.py --environment production
```
- Optimized for performance
- Minimal logging
- Higher concurrency settings

### Testing Environment
```bash
python setup.py --environment testing
```
- Separate test data directories
- Error-only logging
- Reduced data requirements

## Configuration Options

### Main Configuration File (`config.yaml`)
The system uses a YAML configuration file with the following sections:

#### Data Sources
```yaml
data_sources:
  psx_base_url: "https://dps.psx.com.pk"
  downloads_endpoint: "/download/closing_rates"
  pdf_filename_pattern: "{date}.pdf"
```

#### Technical Indicators
```yaml
technical_indicators:
  sma_periods: [50, 200]
  rsi_period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9
```

#### Machine Learning
```yaml
machine_learning:
  model_type: "RandomForest"
  min_training_samples: 200
  test_size: 0.2
  random_state: 42
  n_estimators: 100
  n_splits: 5
```

#### Storage
```yaml
storage:
  data_directory: "data"
  backup_directory: "backups"
  max_file_age_days: 365
```

#### Logging
```yaml
logging:
  level: "INFO"
  file: "psx_advisor.log"
  max_size_mb: 10
  backup_count: 5
  console_output: true
  log_directory: "logs"
```

### Environment-Specific Configuration
Create environment-specific config files:
- `config.development.yaml` - Development overrides
- `config.production.yaml` - Production overrides  
- `config.testing.yaml` - Testing overrides

### Environment Variables
Override configuration using environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `PSX_ENVIRONMENT` | Environment name | `production` |
| `PSX_BASE_URL` | PSX base URL | `https://dps.psx.com.pk` |
| `PSX_DATA_DIR` | Data directory | `./data` |
| `PSX_BACKUP_DIR` | Backup directory | `./backups` |
| `PSX_LOG_LEVEL` | Logging level | `INFO` |
| `PSX_LOG_DIR` | Log directory | `./logs` |
| `PSX_MODEL_TYPE` | ML model type | `RandomForest` |
| `PSX_MIN_SAMPLES` | Min training samples | `200` |
| `PSX_RETRY_ATTEMPTS` | Network retry attempts | `3` |
| `PSX_REQUEST_TIMEOUT` | Request timeout (seconds) | `30` |

## Verification and Testing

### System Check
Run comprehensive system validation:
```bash
python setup.py --check-only
```

This checks:
- Python version compatibility
- Required files and directories
- Configuration validity
- Dependency installation
- File permissions

### Test Run
Verify the system works:
```bash
python main.py --help
python main.py
```

## Troubleshooting

### Common Issues

#### 1. Python Version Error
**Error**: `Python 3.x.x (requires 3.8+)`
**Solution**: Upgrade Python to version 3.8 or higher

#### 2. Missing Dependencies
**Error**: `Missing packages: pandas, requests, ...`
**Solution**: 
```bash
pip install -r requirements.txt
```

#### 3. Permission Errors
**Error**: `No write permission for data`
**Solution**: 
```bash
chmod 755 data backups logs
```

#### 4. Configuration Errors
**Error**: `Configuration validation failed`
**Solution**: 
```bash
python setup.py --force-config
```

#### 5. Network Issues
**Error**: Connection timeouts during data download
**Solution**: 
- Check internet connection
- Increase timeout in config:
  ```yaml
  performance:
    request_timeout: 60
    retry_attempts: 5
  ```

### Getting Help

1. **Check Logs**: Review `logs/psx_advisor.log` for detailed error information
2. **Validate Config**: Run `python setup.py --check-only`
3. **Reset Setup**: Run `python setup.py --force-config`
4. **Environment Issues**: Try different environment settings

## Directory Structure

After successful installation:
```
psx-ai-advisor/
├── config.yaml                 # Main configuration
├── main.py                     # Main application
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── INSTALLATION.md             # This file
├── README.md                   # Project documentation
├── psx_ai_advisor/            # Main package
│   ├── __init__.py
│   ├── config_loader.py       # Configuration management
│   ├── data_acquisition.py    # Data download/parsing
│   ├── technical_analysis.py  # Technical indicators
│   ├── data_storage.py        # Data persistence
│   ├── ml_predictor.py        # Machine learning
│   ├── logging_config.py      # Logging setup
│   ├── exceptions.py          # Custom exceptions
│   └── fallback_mechanisms.py # Error handling
├── data/                      # Stock data storage
│   ├── models/               # ML model files
│   └── scalers/              # Data scalers
├── backups/                  # Data backups
├── logs/                     # Application logs
└── __pycache__/              # Python cache
```

## Next Steps

After successful installation:

1. **Review Configuration**: Check `config.yaml` and adjust settings as needed
2. **Run First Analysis**: Execute `python main.py` to download and analyze data
3. **Schedule Regular Runs**: Set up daily execution via cron (Linux/Mac) or Task Scheduler (Windows)
4. **Monitor Logs**: Check `logs/psx_advisor.log` for system status and errors
5. **Backup Data**: Regularly backup the `data/` directory

## Advanced Configuration

### Custom Data Sources
Modify `data_sources` section to use alternative data endpoints:
```yaml
data_sources:
  psx_base_url: "https://custom-endpoint.com"
  downloads_endpoint: "/api/data"
```

### Performance Tuning
Adjust performance settings for your system:
```yaml
performance:
  max_concurrent_requests: 10    # Higher for faster systems
  request_timeout: 60           # Increase for slow connections
  retry_attempts: 5             # More retries for unreliable networks
```

### Custom Indicators
Add custom technical indicators by modifying the `technical_indicators` section and extending the `TechnicalAnalyzer` class.

### Model Customization
Experiment with different ML models by changing:
```yaml
machine_learning:
  model_type: "RandomForest"     # or "GradientBoosting", "SVM"
  n_estimators: 200             # More trees for better accuracy
  min_training_samples: 500     # More data for better models
```