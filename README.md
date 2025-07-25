# PSX AI Advisor

An intelligent stock market analysis tool for the Pakistan Stock Exchange (PSX) that provides AI-powered investment recommendations based on technical indicators and machine learning models.

## Features

- **Automated Data Acquisition**: Downloads daily closing rate summaries from PSX
- **PDF Processing**: Extracts stock data from PSX PDF reports
- **Technical Analysis**: Calculates SMA, RSI, MACD, and other indicators
- **Machine Learning**: Uses Random Forest models for price predictions
- **Investment Recommendations**: Generates buy/sell/hold recommendations
- **Risk Assessment**: Evaluates investment risks and portfolio optimization

## Project Structure

```
psx-ai-advisor/
├── psx_ai_advisor/           # Main package
│   ├── __init__.py
│   ├── config_loader.py      # Configuration management
│   ├── data_acquisition.py   # PDF download and data extraction
│   ├── pdf_processor.py      # PDF parsing functionality
│   ├── technical_analysis.py # Technical indicators
│   ├── ml_models.py          # Machine learning models
│   └── recommendation_engine.py # Investment recommendations
├── .kiro/specs/              # Feature specifications
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── main.py                   # Main application entry point
└── tests/                    # Test files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/psx-ai-advisor.git
cd psx-ai-advisor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
   - Review and modify `config.yaml` as needed
   - Ensure data directories exist

## Usage

### Basic Usage

```python
from psx_ai_advisor.data_acquisition import PSXDataAcquisition

# Download daily PDF
psx_data = PSXDataAcquisition()
pdf_path = psx_data.download_daily_pdf()
print(f"Downloaded: {pdf_path}")
```

### Configuration

The application uses `config.yaml` for configuration. Key sections include:

- `data_sources`: PSX URL and endpoints
- `technical_indicators`: SMA periods, RSI settings, MACD parameters
- `machine_learning`: Model configuration
- `storage`: Data directories and retention
- `performance`: Request timeouts and retry settings

## Development

This project follows a spec-driven development approach using Kiro IDE. Feature specifications are located in `.kiro/specs/psx-ai-advisor/`:

- `requirements.md`: Feature requirements in EARS format
- `design.md`: Technical design and architecture
- `tasks.md`: Implementation task list

### Running Tests

```bash
python test_data_acquisition.py
python test_config.py
```

## Requirements

- Python 3.8+
- requests >= 2.31.0
- pdfplumber >= 0.9.0
- pandas >= 2.0.0
- pandas-ta >= 0.3.14b
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- numpy >= 1.24.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the spec-driven development process
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions.

## Status

🚧 **In Development** - Currently implementing core data acquisition and processing features.

### Completed Features
- ✅ Configuration management system
- ✅ PDF download functionality with retry mechanism
- ✅ Basic error handling and logging

### In Progress
- 🔄 PDF data extraction and parsing
- 🔄 Technical analysis indicators
- 🔄 Machine learning model implementation

### Planned Features
- 📋 Investment recommendation engine
- 📋 Risk assessment tools
- 📋 Portfolio optimization
- 📋 Web interface