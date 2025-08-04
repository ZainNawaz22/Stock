# PSX Data Loader Usage Guide

This guide explains how to use the new data loading functionality in the PSX AI Advisor system to download and update stock data from Yahoo Finance with fail-safe mechanisms.

## Overview

The PSX Data Loader provides:
- **Automated data loading** from Yahoo Finance for Pakistani stocks
- **Fail-safe mechanisms** to ensure data integrity
- **KSE-100 focus** with comprehensive stock list
- **Parallel processing** for efficient updates
- **Backup and recovery** capabilities
- **REST API endpoints** for remote control

## Quick Start

### 1. Install Dependencies

First, ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

This will install `yfinance>=0.2.18` and other required packages.

### 2. Basic Usage - Python API

```python
from psx_ai_advisor.data_loader import PSXDataLoader

# Initialize the loader
loader = PSXDataLoader()

# Update a single stock
success = loader.update_single_stock("HBL", period="5y")

# Update multiple stocks
symbols = ["HBL", "UBL", "MCB", "NBP"]
results = loader.update_multiple_stocks(symbols, period="2y")

# Update all KSE-100 stocks
results = loader.update_kse100_stocks(period="5y")

# Update all existing stocks
results = loader.update_existing_stocks(period="1y")
```

### 3. Basic Usage - REST API

Start the API server:

```bash
python api_server.py
```

Then use the endpoints:

```bash
# Update all KSE-100 stocks (background task)
curl -X POST "http://localhost:8000/data/update/kse100?period=5y"

# Update existing stocks
curl -X POST "http://localhost:8000/data/update/existing?period=2y"

# Update specific stocks
curl -X POST "http://localhost:8000/data/update/stocks?symbols=HBL&symbols=UBL&period=1y"

# Check task status
curl "http://localhost:8000/data/update/status/kse100_20250130_143022"

# List all tasks
curl "http://localhost:8000/data/update/tasks"
```

## Detailed Usage

### KSE-100 Stock Symbols

The loader includes a comprehensive list of KSE-100 stocks:

```python
from psx_ai_advisor.data_loader import PSXDataLoader

# View all KSE-100 symbols
print(f"Total KSE-100 stocks: {len(PSXDataLoader.KSE_100_SYMBOLS)}")
print("Symbols:", PSXDataLoader.KSE_100_SYMBOLS[:10])  # First 10

# The loader automatically converts PSX symbols to Yahoo Finance format
# Example: "HBL" becomes "HBL.KA" for Yahoo Finance
```

### Supported Time Periods

- `1y` - 1 year of data
- `2y` - 2 years of data
- `5y` - 5 years of data (default)
- `10y` - 10 years of data
- `max` - Maximum available data

### Data Format

Downloaded data follows the existing CSV format:

| Column | Description |
|--------|-------------|
| Date | Trading date (timezone-aware) |
| Open | Opening price |
| High | Highest price |
| Low | Lowest price |
| Close | Closing price |
| Volume | Trading volume |
| Dividends | Dividend amount |
| Stock Splits | Stock split ratio |
| Capital Gains | Capital gains (always 0.0) |

### Fail-Safe Mechanisms

The loader implements multiple fail-safe mechanisms:

1. **Temporary File Creation**: Data is downloaded to temporary files first
2. **Data Validation**: Downloaded data is validated before saving
3. **Automatic Backup**: Existing files are backed up before replacement
4. **Error Recovery**: Failed downloads don't affect existing data
5. **Atomic Operations**: File replacement is atomic (all-or-nothing)

### Error Handling

```python
loader = PSXDataLoader()

# Individual stock update with error handling
try:
    success = loader.update_single_stock("INVALID_SYMBOL")
    if not success:
        print("Update failed - check logs for details")
except Exception as e:
    print(f"Update error: {e}")

# Batch update with detailed results
results = loader.update_multiple_stocks(["HBL", "INVALID", "UBL"])
for symbol, success in results.items():
    if success:
        print(f"✅ {symbol}: Updated successfully")
    else:
        print(f"❌ {symbol}: Update failed")
```

## REST API Endpoints

### Data Update Endpoints

#### 1. Update KSE-100 Stocks

```http
POST /data/update/kse100?period=5y
```

Updates all KSE-100 stocks in the background.

**Response:**
```json
{
  "task_id": "kse100_20250130_143022",
  "status": "queued",
  "message": "KSE-100 data update started in background",
  "period": "5y",
  "stocks_count": 98,
  "tracking_url": "/data/update/status/kse100_20250130_143022"
}
```

#### 2. Update Existing Stocks

```http
POST /data/update/existing?period=2y
```

Updates all stocks that already have data files.

#### 3. Update Custom Stocks

```http
POST /data/update/stocks?symbols=HBL&symbols=UBL&symbols=MCB&period=1y
```

Updates specific stocks provided in the symbols parameter.

#### 4. Check Task Status

```http
GET /data/update/status/{task_id}
```

**Response:**
```json
{
  "task_id": "kse100_20250130_143022",
  "type": "kse100",
  "status": "completed",
  "created_time": "2025-01-30T14:30:22",
  "start_time": "2025-01-30T14:30:23",
  "end_time": "2025-01-30T14:35:45",
  "period": "5y",
  "runtime_seconds": 322.5,
  "summary": {
    "total_stocks": 98,
    "successful": 95,
    "failed": 3,
    "success_rate": 0.969
  }
}
```

#### 5. List All Tasks

```http
GET /data/update/tasks
```

#### 6. Delete Completed Task

```http
DELETE /data/update/tasks/{task_id}
```

#### 7. Get Loader Information

```http
GET /data/loader/info
```

## Command Line Usage

### Direct Script Usage

```bash
# Update all KSE-100 stocks
python psx_ai_advisor/data_loader.py kse100

# Update all existing stocks
python psx_ai_advisor/data_loader.py existing

# Update specific stock
python psx_ai_advisor/data_loader.py HBL
```

### Test Script

```bash
# Run comprehensive tests
python test_data_loader.py
```

## Advanced Configuration

### Custom Data Directory

```python
loader = PSXDataLoader(data_dir="custom_data", backup_dir="custom_backups")
```

### Parallel Processing Control

```python
# Update with custom worker count
results = loader.update_multiple_stocks(
    symbols=["HBL", "UBL", "MCB"], 
    period="2y", 
    max_workers=2  # Limit concurrent downloads
)
```

### Backup Management

```python
loader = PSXDataLoader()

# Get update summary
summary = loader.get_update_summary()
print("Backup directory:", summary["session_backup_dir"])

# Restore from backup if needed
success = loader.restore_from_backup("HBL")
```

## Monitoring and Logging

The data loader provides comprehensive logging:

```python
from psx_ai_advisor.logging_config import get_logger

logger = get_logger(__name__)
# Check logs/psx_advisor.log for detailed information
```

Log levels include:
- **INFO**: Successful operations, progress updates
- **WARNING**: Data quality issues, old data warnings
- **ERROR**: Download failures, validation errors

## Best Practices

### 1. Start Small
```python
# Test with a few stocks first
test_symbols = ["HBL", "UBL", "MCB"]
results = loader.update_multiple_stocks(test_symbols, period="1y")
```

### 2. Use Appropriate Periods
```python
# For daily updates, use shorter periods
loader.update_existing_stocks(period="1y")

# For initial setup, use longer periods
loader.update_kse100_stocks(period="5y")
```

### 3. Monitor Task Progress
```python
# For long-running operations, use background tasks via API
# Check status periodically instead of blocking operations
```

### 4. Handle Network Issues
```python
# Implement retry logic for failed stocks
failed_symbols = [symbol for symbol, success in results.items() if not success]
if failed_symbols:
    print(f"Retrying {len(failed_symbols)} failed stocks...")
    retry_results = loader.update_multiple_stocks(failed_symbols, period="1y")
```

## Troubleshooting

### Common Issues

1. **Network Timeouts**: Reduce `max_workers` parameter
2. **Invalid Symbols**: Check symbol exists in Yahoo Finance with .KA suffix
3. **No Data Available**: Some stocks may not have data for requested period
4. **File Permission Errors**: Ensure write access to data directory

### Debugging

```python
# Enable detailed logging
import logging
logging.getLogger("psx_ai_advisor").setLevel(logging.DEBUG)

# Test individual components
loader = PSXDataLoader()
data = loader.download_single_stock("HBL", period="1y")
if data is not None:
    print("Download successful")
    is_valid = loader.validate_data(data, "HBL")
    print(f"Validation: {is_valid}")
```

## Integration Examples

### Scheduled Updates

```python
import schedule
import time

def daily_update():
    """Run daily data updates."""
    from psx_ai_advisor.data_loader import update_existing_data
    results = update_existing_data(period="1y")
    print(f"Daily update: {sum(results.values())} successful")

# Schedule daily updates at 6 PM
schedule.every().day.at("18:00").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Custom Data Pipeline

```python
def custom_pipeline():
    """Custom data update pipeline."""
    loader = PSXDataLoader()
    
    # Step 1: Update high-priority stocks
    priority_stocks = ["HBL", "UBL", "MCB", "NBP", "BAFL"]
    priority_results = loader.update_multiple_stocks(priority_stocks, period="2y")
    
    # Step 2: Update remaining KSE-100 stocks if priority update successful
    if all(priority_results.values()):
        remaining_stocks = [s for s in loader.KSE_100_SYMBOLS if s not in priority_stocks]
        remaining_results = loader.update_multiple_stocks(remaining_stocks, period="1y")
        
        return {**priority_results, **remaining_results}
    
    return priority_results
```

This completes the comprehensive data loading implementation for your PSX AI Advisor system with fail-safe mechanisms, KSE-100 focus, and both Python API and REST API interfaces.
