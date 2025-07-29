#!/usr/bin/env python3
"""
Main application workflow and CLI interface for PSX AI Advisor

This module provides the main orchestration function and command-line interface
for the complete daily analysis workflow including data acquisition, processing,
and prediction generation.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.config_loader import get_config, get_section, get_value
from psx_ai_advisor.data_acquisition import PSXDataAcquisition, PSXDataAcquisitionError
from psx_ai_advisor.data_storage import DataStorage, DataStorageError
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
from psx_ai_advisor.ml_predictor import MLPredictor, MLPredictorError, InsufficientDataError


class PSXAdvisorError(Exception):
    """Base exception for PSX Advisor application"""
    pass


class PSXAdvisor:
    """
    Main PSX AI Advisor application class that orchestrates the complete workflow
    """
    
    def __init__(self):
        """Initialize the PSX Advisor application"""
        # Setup logging first
        self._setup_logging()
        
        # Initialize components
        self.data_acquisition = PSXDataAcquisition()
        self.data_storage = DataStorage()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor()
        
        # Application state
        self.predictions = {}
        self.errors = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("PSX AI Advisor initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        try:
            # Load logging configuration
            logging_config = get_section('logging')
            log_level = logging_config.get('level', 'INFO')
            log_file = logging_config.get('file', 'psx_advisor.log')
            max_size_mb = logging_config.get('max_size_mb', 10)
            backup_count = logging_config.get('backup_count', 5)
            
            # Configure logging
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            # Setup file handler with rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Setup console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('psx_advisor.log')
                ]
            )
            logging.getLogger(__name__).warning(f"Failed to setup advanced logging: {e}")
    
    def run_daily_analysis(self, date: Optional[str] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the complete daily analysis workflow
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            symbols (List[str], optional): Specific symbols to analyze. If None, analyzes all available.
            
        Returns:
            Dict[str, Any]: Analysis results including predictions and summary
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting PSX AI Advisor Daily Analysis")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        analysis_results = {
            'start_time': start_time.isoformat(),
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'symbols_analyzed': [],
            'predictions': {},
            'errors': [],
            'summary': {}
        }
        
        try:
            # Step 1: Data Acquisition
            self.logger.info("Step 1: Downloading and parsing stock data...")
            stock_data = self._acquire_stock_data(date)
            
            if stock_data.empty:
                raise PSXAdvisorError("No stock data acquired")
            
            # Step 2: Process individual stocks
            available_symbols = self._get_symbols_to_process(stock_data, symbols)
            self.logger.info(f"Processing {len(available_symbols)} symbols: {', '.join(available_symbols[:10])}{'...' if len(available_symbols) > 10 else ''}")
            
            # Step 3: Process each symbol
            for i, symbol in enumerate(available_symbols, 1):
                try:
                    self.logger.info(f"Processing {symbol} ({i}/{len(available_symbols)})...")
                    
                    # Get symbol-specific data
                    symbol_data = stock_data[stock_data['Symbol'] == symbol].copy()
                    
                    if symbol_data.empty:
                        self.logger.warning(f"No data found for symbol: {symbol}")
                        continue
                    
                    # Process this symbol
                    prediction_result = self._process_symbol(symbol, symbol_data)
                    
                    if prediction_result:
                        analysis_results['predictions'][symbol] = prediction_result
                        analysis_results['symbols_analyzed'].append(symbol)
                        self.logger.info(f"✓ {symbol}: {prediction_result['prediction']} (confidence: {prediction_result['confidence']:.3f})")
                    
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    self.logger.error(error_msg)
                    analysis_results['errors'].append({
                        'symbol': symbol,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
            
            # Step 4: Generate summary
            analysis_results['summary'] = self._generate_summary(analysis_results)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            analysis_results['end_time'] = end_time.isoformat()
            analysis_results['execution_time_seconds'] = execution_time
            
            self.logger.info("=" * 60)
            self.logger.info("Daily Analysis Completed Successfully")
            self.logger.info(f"Processed: {len(analysis_results['symbols_analyzed'])} symbols")
            self.logger.info(f"Errors: {len(analysis_results['errors'])}")
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")
            self.logger.info("=" * 60)
            
            return analysis_results
            
        except Exception as e:
            error_msg = f"Daily analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            analysis_results['errors'].append({
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            })
            
            return analysis_results
    
    def _acquire_stock_data(self, date: Optional[str] = None) -> Any:
        """
        Acquire stock data from PSX CSV
        
        Args:
            date (str, optional): Date to acquire data for
            
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            self.logger.info("Downloading CSV from PSX...")
            
            # Try to get data for the specified date or today
            stock_data = self.data_acquisition.get_all_stock_data(date)
            
            if stock_data.empty:
                # If no data for today, try previous business days
                self.logger.warning("No data for specified date, trying recent dates...")
                available_dates = self.data_acquisition.get_available_dates(7)
                
                for alt_date in available_dates:
                    try:
                        self.logger.info(f"Trying date: {alt_date}")
                        stock_data = self.data_acquisition.get_all_stock_data(alt_date)
                        if not stock_data.empty:
                            self.logger.info(f"Successfully acquired data for {alt_date}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to get data for {alt_date}: {e}")
                        continue
            
            if stock_data.empty:
                raise PSXAdvisorError("Could not acquire stock data for any recent date")
            
            self.logger.info(f"Successfully acquired data for {len(stock_data)} records")
            return stock_data
            
        except Exception as e:
            raise PSXAdvisorError(f"Data acquisition failed: {e}")
    
    def _get_symbols_to_process(self, stock_data: Any, symbols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of symbols to process
        
        Args:
            stock_data (pd.DataFrame): Available stock data
            symbols (List[str], optional): Specific symbols to process
            
        Returns:
            List[str]: List of symbols to process
        """
        if symbols:
            # Use provided symbols, but filter to only those with data
            available_symbols = set(stock_data['Symbol'].unique())
            requested_symbols = set(symbols)
            valid_symbols = list(requested_symbols.intersection(available_symbols))
            
            missing_symbols = requested_symbols - available_symbols
            if missing_symbols:
                self.logger.warning(f"Requested symbols not found in data: {', '.join(missing_symbols)}")
            
            return valid_symbols
        else:
            # Use all available symbols
            return list(stock_data['Symbol'].unique())
    
    def _process_symbol(self, symbol: str, symbol_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process a single symbol through the complete workflow
        
        Args:
            symbol (str): Stock symbol
            symbol_data (pd.DataFrame): Symbol-specific data
            
        Returns:
            Optional[Dict[str, Any]]: Prediction result or None if failed
        """
        try:
            # Step 1: Load existing historical data
            try:
                historical_data = self.data_storage.load_stock_data(symbol)
                self.logger.debug(f"Loaded {len(historical_data)} historical records for {symbol}")
            except FileNotFoundError:
                historical_data = None
                self.logger.debug(f"No historical data found for {symbol}")
            
            # Step 2: Save/append new data
            self.logger.debug(f"Saving new data for {symbol}...")
            success = self.data_storage.save_stock_data(symbol, symbol_data)
            
            if not success:
                self.logger.error(f"Failed to save data for {symbol}")
                return None
            
            # Step 3: Load complete data for analysis
            complete_data = self.data_storage.load_stock_data(symbol)
            
            # Step 4: Add technical indicators
            self.logger.debug(f"Calculating technical indicators for {symbol}...")
            complete_data = self.technical_analyzer.add_all_indicators(complete_data)
            
            # Step 5: Generate prediction
            self.logger.debug(f"Generating prediction for {symbol}...")
            prediction_result = self.ml_predictor.predict_movement(symbol)
            
            return prediction_result
            
        except InsufficientDataError as e:
            self.logger.warning(f"Insufficient data for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for the analysis
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        predictions = analysis_results['predictions']
        
        if not predictions:
            return {
                'total_symbols': 0,
                'successful_predictions': 0,
                'up_predictions': 0,
                'down_predictions': 0,
                'average_confidence': 0.0,
                'high_confidence_predictions': 0
            }
        
        # Calculate summary statistics
        up_count = sum(1 for p in predictions.values() if p['prediction'] == 'UP')
        down_count = sum(1 for p in predictions.values() if p['prediction'] == 'DOWN')
        
        confidences = [p['confidence'] for p in predictions.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        high_confidence_count = sum(1 for c in confidences if c > 0.7)
        
        return {
            'total_symbols': len(analysis_results['symbols_analyzed']),
            'successful_predictions': len(predictions),
            'up_predictions': up_count,
            'down_predictions': down_count,
            'up_percentage': (up_count / len(predictions) * 100) if predictions else 0.0,
            'down_percentage': (down_count / len(predictions) * 100) if predictions else 0.0,
            'average_confidence': avg_confidence,
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_percentage': (high_confidence_count / len(predictions) * 100) if predictions else 0.0,
            'errors_count': len(analysis_results['errors'])
        }
    
    def display_predictions(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display formatted predictions to terminal
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results from run_daily_analysis
        """
        print("\n" + "=" * 80)
        print("PSX AI ADVISOR - DAILY PREDICTIONS")
        print("=" * 80)
        
        # Display summary
        summary = analysis_results.get('summary', {})
        print(f"\nSUMMARY:")
        print(f"  Date: {analysis_results.get('date', 'N/A')}")
        print(f"  Symbols Analyzed: {summary.get('total_symbols', 0)}")
        print(f"  Successful Predictions: {summary.get('successful_predictions', 0)}")
        print(f"  Execution Time: {analysis_results.get('execution_time_seconds', 0):.2f} seconds")
        
        if summary.get('successful_predictions', 0) > 0:
            print(f"  UP Predictions: {summary.get('up_predictions', 0)} ({summary.get('up_percentage', 0):.1f}%)")
            print(f"  DOWN Predictions: {summary.get('down_predictions', 0)} ({summary.get('down_percentage', 0):.1f}%)")
            print(f"  Average Confidence: {summary.get('average_confidence', 0):.3f}")
            print(f"  High Confidence (>70%): {summary.get('high_confidence_predictions', 0)} ({summary.get('high_confidence_percentage', 0):.1f}%)")
        
        # Display predictions
        predictions = analysis_results.get('predictions', {})
        if predictions:
            print(f"\nPREDICTIONS:")
            print("-" * 80)
            print(f"{'SYMBOL':<10} {'PREDICTION':<10} {'CONFIDENCE':<12} {'CURRENT PRICE':<15} {'MODEL ACC':<10}")
            print("-" * 80)
            
            # Sort by confidence (highest first)
            sorted_predictions = sorted(
                predictions.items(),
                key=lambda x: x[1]['confidence'],
                reverse=True
            )
            
            for symbol, pred in sorted_predictions:
                confidence_str = f"{pred['confidence']:.3f}"
                price_str = f"Rs {pred['current_price']:.2f}"
                accuracy_str = f"{pred.get('model_accuracy', 0):.3f}" if pred.get('model_accuracy') else "N/A"
                
                # Color coding for terminal (if supported)
                prediction_display = pred['prediction']
                if pred['prediction'] == 'UP':
                    prediction_display = f"🟢 {pred['prediction']}"
                else:
                    prediction_display = f"🔴 {pred['prediction']}"
                
                print(f"{symbol:<10} {prediction_display:<10} {confidence_str:<12} {price_str:<15} {accuracy_str:<10}")
        
        # Display errors if any
        errors = analysis_results.get('errors', [])
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            print("-" * 80)
            for error in errors[:5]:  # Show only first 5 errors
                symbol = error.get('symbol', 'GENERAL')
                error_msg = error.get('error', 'Unknown error')
                print(f"  {symbol}: {error_msg}")
            
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors (check log file for details)")
        
        print("\n" + "=" * 80)
        print("Analysis completed. Check psx_advisor.log for detailed logs.")
        print("=" * 80)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PSX AI Advisor - Automated stock analysis and prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run analysis for today
  python main.py --date 2024-01-15       # Run analysis for specific date
  python main.py --symbols ENGRO HBL     # Analyze specific symbols only
  python main.py --quiet                 # Run with minimal output
        """
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date to analyze in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific stock symbols to analyze (default: all available)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, show only final results'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying predictions (useful for automated runs)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PSX AI Advisor v1.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize the advisor
        advisor = PSXAdvisor()
        
        # Adjust logging level if quiet mode
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Run the analysis
        results = advisor.run_daily_analysis(
            date=args.date,
            symbols=args.symbols
        )
        
        # Display results unless suppressed
        if not args.no_display:
            advisor.display_predictions(results)
        
        # Exit with appropriate code
        if results.get('errors'):
            sys.exit(1)  # Exit with error if there were any errors
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        logging.getLogger(__name__).error(f"Fatal error: {e}")
        logging.getLogger(__name__).error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()