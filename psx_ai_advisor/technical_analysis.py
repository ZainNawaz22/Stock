"""
Technical Analysis Module for PSX AI Advisor

This module provides technical indicator calculations for stock data analysis.
It implements various technical indicators including Simple Moving Averages (SMA),
Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    A class for calculating technical indicators from OHLCV stock data.
    
    This class provides methods to calculate various technical indicators
    and append them to stock data DataFrames for further analysis.
    """
    
    def __init__(self):
        """Initialize the TechnicalAnalyzer."""
        logger.info("TechnicalAnalyzer initialized")
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA) for given data and period.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            period (int): Number of periods for the moving average
            
        Returns:
            pd.Series: Simple Moving Average values
            
        Raises:
            ValueError: If period is invalid or data is insufficient
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        
        if len(data) < period:
            logger.warning(f"Insufficient data for SMA calculation. Need {period} points, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
        
        try:
            # Calculate Simple Moving Average using pandas rolling window
            sma = data.rolling(window=period, min_periods=period).mean()
            logger.debug(f"Calculated SMA_{period} for {len(data)} data points")
            return sma
        except Exception as e:
            logger.error(f"Error calculating SMA_{period}: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for given data.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            period (int): Number of periods for RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI values (0-100 scale)
            
        Raises:
            ValueError: If period is invalid or data is insufficient
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        
        if len(data) < period + 1:  # RSI needs at least period+1 points
            logger.warning(f"Insufficient data for RSI calculation. Need {period + 1} points, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
        
        try:
            # Calculate price changes
            delta = data.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses using exponential moving average
            avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
            avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            logger.debug(f"Calculated RSI_{period} for {len(data)} data points")
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI_{period}: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            fast (int): Fast EMA period (default: 12)
            slow (int): Slow EMA period (default: 26)
            signal (int): Signal line EMA period (default: 9)
            
        Returns:
            pd.DataFrame: DataFrame with MACD, MACD_Signal, and MACD_Histogram columns
            
        Raises:
            ValueError: If parameters are invalid or data is insufficient
        """
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("All periods must be positive integers")
        
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        
        min_required = slow + signal
        if len(data) < min_required:
            logger.warning(f"Insufficient data for MACD calculation. Need {min_required} points, got {len(data)}")
            return pd.DataFrame(index=data.index, columns=['MACD', 'MACD_Signal', 'MACD_Histogram'])
        
        try:
            # Calculate exponential moving averages
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            
            # Calculate MACD line (difference between fast and slow EMA)
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line (EMA of MACD line)
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram (difference between MACD and signal line)
            histogram = macd_line - signal_line
            
            # Create result DataFrame
            result_df = pd.DataFrame(index=data.index)
            result_df['MACD'] = macd_line
            result_df['MACD_Signal'] = signal_line
            result_df['MACD_Histogram'] = histogram
            
            logger.debug(f"Calculated MACD({fast},{slow},{signal}) for {len(data)} data points")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating MACD({fast},{slow},{signal}): {str(e)}")
            return pd.DataFrame(index=data.index, columns=['MACD', 'MACD_Signal', 'MACD_Histogram'])
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required technical indicators to the DataFrame.
        
        This method calculates and appends the following indicators:
        - SMA_50: 50-day Simple Moving Average
        - SMA_200: 200-day Simple Moving Average  
        - RSI_14: 14-day Relative Strength Index
        - MACD: Moving Average Convergence Divergence
        - MACD_Signal: MACD Signal Line
        - MACD_Histogram: MACD Histogram
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data (must have 'Close' column)
            
        Returns:
            pd.DataFrame: DataFrame with all technical indicators added as new columns
            
        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to add_all_indicators")
            return df
        
        # Validate required columns
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        try:
            # Calculate Simple Moving Averages
            logger.info("Calculating Simple Moving Averages...")
            result_df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            result_df['SMA_200'] = self.calculate_sma(df['Close'], 200)
            
            # Calculate RSI
            logger.info("Calculating Relative Strength Index...")
            result_df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
            
            # Calculate MACD
            logger.info("Calculating MACD...")
            macd_df = self.calculate_macd(df['Close'])
            result_df['MACD'] = macd_df['MACD']
            result_df['MACD_Signal'] = macd_df['MACD_Signal']
            result_df['MACD_Histogram'] = macd_df['MACD_Histogram']
            
            # Log summary of indicators added
            indicators_added = ['SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram']
            logger.info(f"Successfully added {len(indicators_added)} technical indicators: {', '.join(indicators_added)}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of technical indicators for the latest data point.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            
        Returns:
            Dict[str, Any]: Summary of latest indicator values
        """
        if df.empty:
            return {}
        
        latest_row = df.iloc[-1]
        summary = {}
        
        # Technical indicator columns to summarize
        indicator_columns = ['SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram']
        
        for col in indicator_columns:
            if col in df.columns:
                value = latest_row[col]
                summary[col] = value if pd.notna(value) else None
        
        # Add some derived insights
        if 'Close' in df.columns and 'SMA_50' in summary and 'SMA_200' in summary:
            current_price = latest_row['Close']
            sma_50 = summary['SMA_50']
            sma_200 = summary['SMA_200']
            
            if pd.notna(sma_50) and pd.notna(sma_200):
                summary['price_above_sma_50'] = current_price > sma_50
                summary['price_above_sma_200'] = current_price > sma_200
                summary['golden_cross'] = sma_50 > sma_200  # Bullish signal
        
        if 'RSI_14' in summary and pd.notna(summary['RSI_14']):
            rsi_value = summary['RSI_14']
            summary['rsi_overbought'] = rsi_value > 70
            summary['rsi_oversold'] = rsi_value < 30
        
        return summary