"""
PSX Data Loader Module

This module provides data loading functionality for Pakistan Stock Exchange (PSX) data.
"""

import os
import shutil
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class PSXDataLoader:
    """
    PSX Data Loader for downloading and managing Pakistan Stock Exchange data.
    """
    
    # KSE-100 stock symbols (major Pakistani stocks with .KS suffix for Yahoo Finance)
    KSE_100_SYMBOLS = [
        'EFOODS.KS', 'NESTLE.KS', 'SEARL.KS', 'UNILEVER.KS', 'COLG.KS',
        'NATF.KS', 'ISIL.KS', 'PSMC.KS', 'KAPCO.KS', 'HUBC.KS',
        'KTML.KS', 'KEL.KS', 'LUCK.KS', 'MLCF.KS', 'DGKC.KS',
        'PIOC.KS', 'CHCC.KS', 'FCCL.KS', 'KOHC.KS', 'PAEL.KS',
        'EFERT.KS', 'FFBL.KS', 'FFC.KS', 'FATIMA.KS', 'ENGRO.KS',
        'EPCL.KS', 'OGDC.KS', 'PPL.KS', 'POL.KS', 'MARI.KS',
        'OGDC.KS', 'SSGC.KS', 'SNGP.KS', 'SHEL.KS', 'HASCOL.KS',
        'HBL.KS', 'UBL.KS', 'MCB.KS', 'ABL.KS', 'BAFL.KS',
        'FABL.KS', 'NBP.KS', 'AKBL.KS', 'BOP.KS', 'MEBL.KS',
        'SILK.KS', 'GTYR.KS', 'CNERGY.KS', 'UNITY.KS', 'DAWH.KS',
        'THALL.KS', 'TRG.KS', 'SYSTEMS.KS', 'TPL.KS', 'AIRLINK.KS',
        'AVN.KS', 'LOADS.KS', 'JLICL.KS', 'JDWS.KS', 'ILP.KS',
        'HGFA.KS', 'HINOON.KS', 'HCAR.KS', 'GLAXO.KS', 'GHGL.KS',
        'GATM.KS', 'FHAM.KS', 'FML.KS', 'EFUG.KS', 'DCR.KS',
        'BNWM.KS', 'BAHL.KS', 'ATRL.KS', 'ATLH.KS', 'ARPL.KS',
        'APL.KS', 'ANL.KS', 'AICL.KS', 'AGIL.KS', 'ABOT.KS',
        'HMB.KS', 'IBFL.KS', 'IDYM.KS', 'INDU.KS', 'INIL.KS',
        'ISL.KS', 'LOTCHEM.KS', 'MTL.KS', 'MUREB.KS', 'PTC.KS',
        'TOMCL.KS', 'WTL.KS', 'BYCO.KS', 'ADMM.KS', 'SCBPL.KS',
        'PAKOXY.KS', 'PAKT.KS', 'SITC.KS', 'FECTC.KS', 'GCIL.KS'
    ]
    
    def __init__(self, data_dir: str = 'data', temp_dir: str = 'temp_data'):
        """
        Initialize PSX Data Loader.
        
        Args:
            data_dir (str): Directory to store data files
            temp_dir (str): Directory for temporary files
        """
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"PSXDataLoader initialized with data_dir: {self.data_dir}")
    
    def download_stock_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Download stock data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'PTC.KS')
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            logger.info(f"Downloading data for {symbol} (period: {period})")
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Add symbol and company name columns
            base = symbol.replace('.KS', '').replace('.KA', '').upper()
            hist_symbol = f"{base}_HISTORICAL_DATA"
            data['Symbol'] = hist_symbol
            data['Company_Name'] = base
            
            # Calculate additional fields
            data['Previous_Close'] = data['Close'].shift(1)
            data['Change'] = data['Close'] - data['Previous_Close']
            
            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            return None
    
    def get_yahoo_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if '.' in s:
            return s
        return f"{s}.KA"

    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        if data is None or data.empty:
            return False
        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(set(data.columns)):
            return False
        return True

    def _final_file_path(self, symbol: str) -> Path:
        s = symbol.strip().upper()
        if s.endswith('.KS') or s.endswith('.KA'):
            s = s.split('.')[0]
        if not s.endswith('_HISTORICAL_DATA'):
            s = f"{s}_HISTORICAL_DATA"
        return self.data_dir / f"{s}.csv"

    def download_single_stock(self, symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
        yahoo_symbol = self.get_yahoo_symbol(symbol)
        return self.download_stock_data(yahoo_symbol, period)

    def update_single_stock(self, symbol: str, period: str = '2y', merge_with_existing: bool = False) -> bool:
        try:
            base_symbol = symbol.split('.')[0].upper()
            data = self.download_single_stock(base_symbol, period)
            if not self.validate_data(data, symbol):
                return False
            file_symbol = f"{base_symbol}_HISTORICAL_DATA"
            temp_file = self.temp_dir / f"{file_symbol}.tmp.csv"
            data.to_csv(temp_file, index=False)
            final_file = self._final_file_path(file_symbol)
            if final_file.exists():
                try:
                    os.replace(str(temp_file), str(final_file))
                finally:
                    if temp_file.exists():
                        try:
                            os.remove(str(temp_file))
                        except Exception:
                            pass
            else:
                os.replace(str(temp_file), str(final_file))
            return True
        except Exception as e:
            logger.error(f"Failed to update {symbol}: {e}")
            try:
                bs = symbol.split('.')[0].upper()
                fs = f"{bs}_HISTORICAL_DATA"
                temp_path = self.temp_dir / f"{fs}.tmp.csv"
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            return False

    def update_multiple_stocks(self, symbols: List[str], period: str = '2y', max_workers: int = 4, merge_with_existing: bool = False) -> Dict[str, bool]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: Dict[str, bool] = {}
        cleaned_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        if not cleaned_symbols:
            return results
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.update_single_stock, s, period, merge_with_existing): s for s in cleaned_symbols}
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                ok = False
                try:
                    ok = bool(future.result())
                except Exception:
                    ok = False
                results[s] = ok
        return results

    def update_kse100_stocks(self, period: str = '5y', max_workers: int = 4, merge_with_existing: bool = False) -> Dict[str, bool]:
        return self.update_multiple_stocks(self.get_kse100_symbols(), period, max_workers, merge_with_existing)

    def update_existing_stocks(self, period: str = '2y', max_workers: int = 4, merge_with_existing: bool = False) -> Dict[str, bool]:
        existing = self.get_available_symbols()
        return self.update_multiple_stocks(existing, period, max_workers, merge_with_existing)

    def get_update_summary(self) -> Dict[str, Any]:
        return {
            "session_backup_dir": str(self.temp_dir),
            "data_directory": str(self.data_dir),
            "kse100_symbols_count": len(self.KSE_100_SYMBOLS)
        }

    def download_kse100_data(self, period: str = '2y', max_workers: int = 4) -> Dict[str, Any]:
        """
        Download data for all KSE-100 symbols.
        
        Args:
            period (str): Data period
            max_workers (int): Maximum number of worker threads
            
        Returns:
            Dict[str, Any]: Summary of download operation
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Starting KSE-100 data download (period: {period}, workers: {max_workers})")
        
        successful_downloads = []
        failed_downloads = []
        
        def download_single_stock(symbol):
            """Download data for a single stock."""
            try:
                base_symbol = symbol.split('.')[0].upper()
                yahoo_symbol = self.get_yahoo_symbol(base_symbol)
                data = self.download_stock_data(yahoo_symbol, period)
                if data is not None:
                    temp_file = self.temp_dir / f"{base_symbol}_HISTORICAL_DATA.tmp.csv"
                    data.to_csv(temp_file, index=False)
                    return {'symbol': f"{base_symbol}_HISTORICAL_DATA", 'status': 'success', 'records': len(data)}
                else:
                    return {'symbol': f"{base_symbol}_HISTORICAL_DATA", 'status': 'no_data', 'error': 'No data returned'}
            except Exception as e:
                return {'symbol': f"{symbol.split('.')[0].upper()}_HISTORICAL_DATA", 'status': 'error', 'error': str(e)}
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(download_single_stock, symbol): symbol 
                for symbol in self.KSE_100_SYMBOLS
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result['status'] == 'success':
                    successful_downloads.append(result)
                else:
                    failed_downloads.append(result)
        
        for result in successful_downloads:
            symbol = result['symbol'].upper()
            temp_file = self.temp_dir / f"{symbol}.tmp.csv"
            final_file = self._final_file_path(symbol)
            if temp_file.exists():
                try:
                    os.replace(str(temp_file), str(final_file))
                finally:
                    if temp_file.exists():
                        try:
                            os.remove(str(temp_file))
                        except Exception:
                            pass
        
        summary = {
            'total_symbols': len(self.KSE_100_SYMBOLS),
            'successful_downloads': len(successful_downloads),
            'failed_downloads': len(failed_downloads),
            'success_rate': len(successful_downloads) / len(self.KSE_100_SYMBOLS) * 100,
            'successful_symbols': [r['symbol'] for r in successful_downloads],
            'failed_symbols': [r['symbol'] for r in failed_downloads],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"KSE-100 download completed: {summary['successful_downloads']}/{summary['total_symbols']} successful")
        return summary
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols that have data files available.
        
        Returns:
            List[str]: List of available stock symbols
        """
        import glob
        
        csv_files = glob.glob(str(self.data_dir / "*.csv"))
        symbols = []
        for file_path in csv_files:
            filename = Path(file_path).name
            if filename.lower().endswith('.csv'):
                symbol = filename[:-4].upper()
                if symbol.endswith('_HISTORICAL_DATA'):
                    symbols.append(symbol)
        return sorted(list(set(symbols)))
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from CSV file.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Stock data or None if not found
        """
        file_path = self._final_file_path(symbol)
        
        if not file_path.exists():
            logger.warning(f"Data file not found for symbol: {symbol}")
            return None
        
        try:
            data = pd.read_csv(file_path)
            logger.debug(f"Loaded {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    @classmethod
    def get_kse100_symbols(cls) -> List[str]:
        """
        Get the list of KSE-100 symbols.
        
        Returns:
            List[str]: List of KSE-100 stock symbols
        """
        return cls.KSE_100_SYMBOLS.copy()
