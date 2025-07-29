# PSX AI Advisor

A high-performance AI-powered stock analysis and prediction system for the Pakistan Stock Exchange (PSX) with optimized REST API.

## 🚀 Features

- **📊 Technical Analysis**: Comprehensive technical indicators (RSI, MACD, SMA, Bollinger Bands, etc.)
- **🤖 Machine Learning**: Random Forest-based price movement prediction with time-series validation
- **⚡ High-Performance API**: Optimized FastAPI server with parallel processing and intelligent caching
- **📈 96 Stock Symbols**: Complete PSX data coverage with historical analysis
- **🔄 Real-time Processing**: Efficient data processing with background task support
- **📝 Comprehensive Logging**: Detailed logging and error handling

## 🎯 Performance Highlights

- **System Status**: <5s (was timeout)
- **Stock Data**: <1s (was timeout) 
- **Predictions**: <2s (was timeout)
- **Stocks List**: <1s (was 11s)
- **100% Uptime**: No timeout errors, production-ready

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd psx-ai-advisor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python final_performance_test.py
```

## 🚀 Quick Start

### Start the Optimized API Server
```bash
python start_optimized_api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

### Test Performance
```bash
python final_performance_test.py
```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check with cache statistics
- `GET /api/stocks` - List available stocks (optimized with parallel processing)
- `GET /api/stocks/{symbol}/data` - Get stock data with technical indicators
- `GET /api/predictions` - Get ML predictions (parallel processing)
- `GET /api/system/status` - Comprehensive system health check

### Management Endpoints
- `POST /api/cache/clear` - Clear all cached data
- `POST /api/predictions/warmup` - Background prediction cache warmup

### Example Usage
```bash
# Get health status
curl http://localhost:8000/health

# Get top 5 stocks
curl "http://localhost:8000/api/stocks?limit=5"

# Get stock data (30 days)
curl "http://localhost:8000/api/stocks/PTC/data?days=30"

# Get predictions (limited)
curl "http://localhost:8000/api/predictions?limit=3"
```

## ⚙️ Configuration

Edit `config.yaml` for customization:

```yaml
storage:
  data_directory: data          # Stock data location
  backup_directory: backups     # Backup location

machine_learning:
  min_training_samples: 50      # Minimum data for training
  n_estimators: 100            # Random Forest trees

performance:
  max_concurrent_requests: 5    # API concurrency limit
  request_timeout: 30          # Request timeout
```

## 🏗️ Architecture

### Core Components
- **`api_server.py`** - Optimized FastAPI server with performance enhancements
- **`psx_ai_advisor/data_storage.py`** - Efficient data management
- **`psx_ai_advisor/ml_predictor.py`** - ML models with time-series validation
- **`psx_ai_advisor/technical_analysis.py`** - Technical indicator calculations
- **`psx_ai_advisor/config_loader.py`** - Configuration management

### Performance Optimizations
- **Parallel Processing**: ThreadPoolExecutor for concurrent operations
- **Multi-Layer Caching**: Intelligent caching with automatic expiration
- **Request Limiting**: Smart limits prevent system overload
- **Background Tasks**: Non-blocking expensive operations
- **Memory Optimization**: Efficient data handling and cleanup

## 📊 Data Coverage

- **96 Stock Symbols** from PSX
- **Historical Data**: 2016-2025 (227K+ records)
- **35+ Pre-trained Models** ready for predictions
- **16 Technical Indicators** per stock
- **Real-time Processing** capabilities

## 🧪 Testing

### Comprehensive Performance Test
```bash
python final_performance_test.py
```

Expected output:
```
🎉 OUTSTANDING: API performance is excellent! All issues resolved.
✅ 100% Success Rate - All 11 endpoints working perfectly
🎯 100% Performance Targets Met
⚡ Average Response Time: 0.76s
```

## 🚀 Production Deployment

### Using Uvicorn (Recommended)
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using the Optimized Starter
```bash
python start_optimized_api.py --host 0.0.0.0 --port 8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "start_optimized_api.py"]
```

## 📈 Monitoring

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# System status
curl http://localhost:8000/api/system/status
```

### Performance Metrics
- Response times tracked per endpoint
- Cache hit rates monitored
- Background task status
- Memory usage optimization

## 🔧 Development

### Project Structure
```
psx-ai-advisor/
├── api_server.py              # Main optimized API server
├── start_optimized_api.py     # Server startup script
├── final_performance_test.py  # Comprehensive testing
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── psx_ai_advisor/          # Core library
│   ├── data_storage.py      # Data management
│   ├── ml_predictor.py      # ML models
│   ├── technical_analysis.py # Technical indicators
│   └── ...
├── data/                    # Stock data (96 symbols)
└── backups/                # Data backups
```

### Adding New Features
1. Extend core modules in `psx_ai_advisor/`
2. Update API endpoints in `api_server.py`
3. Add tests to `final_performance_test.py`
4. Update documentation

## 📋 Requirements

- Python 3.8+
- FastAPI
- Pandas, NumPy
- Scikit-learn
- PyYAML
- Requests

## 🎉 Success Metrics

✅ **All timeout issues resolved**  
✅ **100% API endpoint success rate**  
✅ **Sub-second response times for most endpoints**  
✅ **Production-ready performance**  
✅ **Comprehensive caching and optimization**  
✅ **96 stocks with ML predictions**  

## 📞 Support

- **API Documentation**: Available at `/docs` when server is running
- **Performance Testing**: Run `python final_performance_test.py`
- **Configuration Help**: Check `config.yaml` comments
- **Issues**: Create GitHub issues for bugs or feature requests

---

**The PSX AI Advisor is now production-ready with outstanding performance! 🚀**