"""
Technical Analysis Module for PSX AI Advisor

This module provides technical indicator calculations for stock data analysis.
It implements various technical indicators including Simple Moving Averages (SMA),
Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .exceptions import TechnicalAnalysisError, InsufficientDataError, ValidationError, create_error_context
from .logging_config import get_logger, log_exception, create_operation_logger

# Set up logging
logger = get_logger(__name__)


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
            ValidationError: If period is invalid
            InsufficientDataError: If data is insufficient
            TechnicalAnalysisError: If calculation fails
        """
        operation_name = f"calculate_sma_{period}"
        context = create_error_context(operation_name, period=period, data_length=len(data))
        
        if period <= 0:
            exc = ValidationError("Period must be a positive integer", 'INVALID_PERIOD', context)
            log_exception(logger, exc, context, operation_name)
            raise exc
        
        if len(data) < period:
            logger.warning(f"Insufficient data for SMA calculation. Need {period} points, got {len(data)}")
            # Return empty series instead of raising exception for graceful degradation
            return pd.Series(index=data.index, dtype=float)
        
        try:
            # Calculate Simple Moving Average using pandas rolling window
            sma = data.rolling(window=period, min_periods=period).mean()
            logger.debug(f"Calculated SMA_{period} for {len(data)} data points")
            return sma
        except Exception as e:
            error_context = {**context, 'calculation_error': str(e)}
            exc = TechnicalAnalysisError(f"Error calculating SMA_{period}: {str(e)}", 'SMA_CALCULATION_ERROR', error_context)
            log_exception(logger, exc, error_context, operation_name)
            # Return empty series for graceful degradation
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
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for given data.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            period (int): Period for moving average and standard deviation (default: 20)
            std_dev (float): Number of standard deviations for bands (default: 2)
            
        Returns:
            pd.DataFrame: DataFrame with BB_Upper, BB_Middle, BB_Lower columns
        """
        if len(data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands. Need {period} points, got {len(data)}")
            return pd.DataFrame(index=data.index, columns=['BB_Upper', 'BB_Middle', 'BB_Lower'])
        
        try:
            # Calculate middle band (SMA)
            middle_band = data.rolling(window=period, min_periods=period).mean()
            
            # Calculate standard deviation
            rolling_std = data.rolling(window=period, min_periods=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            result_df = pd.DataFrame(index=data.index)
            result_df['BB_Upper'] = upper_band
            result_df['BB_Middle'] = middle_band
            result_df['BB_Lower'] = lower_band
            
            logger.debug(f"Calculated Bollinger Bands({period},{std_dev}) for {len(data)} data points")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.DataFrame(index=data.index, columns=['BB_Upper', 'BB_Middle', 'BB_Lower'])
    
    def calculate_price_rate_of_change(self, data: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Price Rate of Change (ROC) indicator.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            period (int): Number of periods for ROC calculation (default: 12)
            
        Returns:
            pd.Series: Rate of Change values as percentages
        """
        if len(data) < period + 1:
            logger.warning(f"Insufficient data for ROC calculation. Need {period + 1} points, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
        
        try:
            # Calculate Rate of Change: ((Current Price - Price n periods ago) / Price n periods ago) * 100
            roc = ((data - data.shift(period)) / data.shift(period)) * 100
            
            logger.debug(f"Calculated ROC_{period} for {len(data)} data points")
            return roc
            
        except Exception as e:
            logger.error(f"Error calculating ROC_{period}: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_obv(self, close_data: pd.Series, volume_data: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV) indicator.
        
        Args:
            close_data (pd.Series): Closing price data
            volume_data (pd.Series): Volume data
            
        Returns:
            pd.Series: On-Balance Volume values
        """
        if len(close_data) != len(volume_data):
            raise ValueError("Close and volume data must have the same length")
        
        if len(close_data) < 2:
            logger.warning("Insufficient data for OBV calculation. Need at least 2 points")
            return pd.Series(index=close_data.index, dtype=float)
        
        try:
            # Calculate price changes
            price_change = close_data.diff()
            
            # Initialize OBV
            obv = pd.Series(index=close_data.index, dtype=float)
            obv.iloc[0] = volume_data.iloc[0]
            
            # Calculate OBV
            for i in range(1, len(close_data)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume_data.iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume_data.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            logger.debug(f"Calculated OBV for {len(close_data)} data points")
            return obv
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series(index=close_data.index, dtype=float)
    
    def calculate_returns(self, data: pd.Series, periods: list = [1, 2, 5]) -> pd.DataFrame:
        """
        Calculate returns for multiple periods.
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            periods (list): List of periods to calculate returns for
            
        Returns:
            pd.DataFrame: DataFrame with return columns for each period
        """
        result_df = pd.DataFrame(index=data.index)
        
        try:
            for period in periods:
                if len(data) > period:
                    # Calculate percentage returns
                    returns = ((data - data.shift(period)) / data.shift(period)) * 100
                    result_df[f'Return_{period}d'] = returns
                    logger.debug(f"Calculated {period}-day returns")
                else:
                    result_df[f'Return_{period}d'] = pd.Series(index=data.index, dtype=float)
                    logger.warning(f"Insufficient data for {period}-day returns")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def calculate_rolling_volatility(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            data (pd.Series): Price data (typically closing prices)
            period (int): Rolling window period (default: 20)
            
        Returns:
            pd.Series: Rolling volatility values
        """
        if len(data) < period + 1:
            logger.warning(f"Insufficient data for volatility calculation. Need {period + 1} points, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
        
        try:
            # Calculate daily returns
            returns = data.pct_change()
            
            # Calculate rolling standard deviation of returns
            volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)  # Annualized
            
            logger.debug(f"Calculated {period}-day rolling volatility for {len(data)} data points")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {str(e)}")
            return pd.Series(index=data.index, dtype=float)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required technical indicators to the DataFrame.
        
        This method calculates and appends the following indicators:
        
        Core Technical Indicators:
        - RSI_14: 14-period Relative Strength Index
        - MACD, MACD_Signal, MACD_Histogram: Moving Average Convergence Divergence
        - ROC_12: 12-period Price Rate of Change
        
        Moving Averages & Bollinger Bands:
        - SMA_20: 20-day Simple Moving Average
        - SMA_50: 50-day Simple Moving Average
        - BB_Upper, BB_Middle, BB_Lower: Bollinger Bands (20-period, 2 std dev)
        
        Volume Indicators:
        - Volume_MA_20: 20-day Volume Moving Average
        - OBV: On-Balance Volume
        
        Derived Features for ML:
        - Return_1d, Return_2d, Return_5d: Previous 1, 2, and 5-day returns
        - Volatility_20d: 20-day rolling volatility
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all technical indicators added as new columns
            
        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to add_all_indicators")
            return df
        
        # Validate required columns
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        try:
            logger.info("Calculating comprehensive technical indicators...")
            
            # Core Technical Indicators
            logger.info("Calculating RSI...")
            result_df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
            
            logger.info("Calculating MACD...")
            macd_df = self.calculate_macd(df['Close'])
            result_df['MACD'] = macd_df['MACD']
            result_df['MACD_Signal'] = macd_df['MACD_Signal']
            result_df['MACD_Histogram'] = macd_df['MACD_Histogram']
            
            logger.info("Calculating Price Rate of Change...")
            result_df['ROC_12'] = self.calculate_price_rate_of_change(df['Close'], 12)
            
            # Moving Averages & Bollinger Bands
            logger.info("Calculating Moving Averages...")
            result_df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            result_df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            
            logger.info("Calculating Bollinger Bands...")
            bb_df = self.calculate_bollinger_bands(df['Close'], 20, 2)
            result_df['BB_Upper'] = bb_df['BB_Upper']
            result_df['BB_Middle'] = bb_df['BB_Middle']
            result_df['BB_Lower'] = bb_df['BB_Lower']
            
            # Volume Indicators
            logger.info("Calculating Volume indicators...")
            result_df['Volume_MA_20'] = self.calculate_sma(df['Volume'], 20)
            result_df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
            
            # Derived Features for ML
            logger.info("Calculating return features...")
            returns_df = self.calculate_returns(df['Close'], [1, 2, 5])
            for col in returns_df.columns:
                result_df[col] = returns_df[col]
            
            logger.info("Calculating rolling volatility...")
            result_df['Volatility_20d'] = self.calculate_rolling_volatility(df['Close'], 20)
            
            # Log summary of indicators added
            indicators_added = [
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_12',
                'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                'Volume_MA_20', 'OBV', 'Return_1d', 'Return_2d', 'Return_5d', 'Volatility_20d'
            ]
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
        
        # All technical indicator columns to summarize
        indicator_columns = [
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_12',
            'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_MA_20', 'OBV', 'Return_1d', 'Return_2d', 'Return_5d', 'Volatility_20d'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                value = latest_row[col]
                summary[col] = value if pd.notna(value) else None
        
        # Add derived insights
        if 'Close' in df.columns:
            current_price = latest_row['Close']
            
            # Moving Average insights
            if 'SMA_20' in summary and 'SMA_50' in summary:
                sma_20 = summary['SMA_20']
                sma_50 = summary['SMA_50']
                
                if pd.notna(sma_20) and pd.notna(sma_50):
                    summary['price_above_sma_20'] = current_price > sma_20
                    summary['price_above_sma_50'] = current_price > sma_50
                    summary['sma_20_above_sma_50'] = sma_20 > sma_50  # Short-term bullish
            
            # Bollinger Bands insights
            if all(col in summary for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                bb_upper = summary['BB_Upper']
                bb_lower = summary['BB_Lower']
                bb_middle = summary['BB_Middle']
                
                if all(pd.notna(val) for val in [bb_upper, bb_lower, bb_middle]):
                    summary['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)  # 0-1 scale
                    summary['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle < 0.1  # Low volatility
        
        # RSI insights
        if 'RSI_14' in summary and pd.notna(summary['RSI_14']):
            rsi_value = summary['RSI_14']
            summary['rsi_overbought'] = rsi_value > 70
            summary['rsi_oversold'] = rsi_value < 30
            summary['rsi_bullish'] = rsi_value > 50
        
        # MACD insights
        if all(col in summary for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            macd = summary['MACD']
            macd_signal = summary['MACD_Signal']
            macd_hist = summary['MACD_Histogram']
            
            if all(pd.notna(val) for val in [macd, macd_signal, macd_hist]):
                summary['macd_bullish'] = macd > macd_signal
                summary['macd_strengthening'] = macd_hist > 0
        
        # Volume insights
        if 'Volume' in df.columns and 'Volume_MA_20' in summary:
            current_volume = latest_row['Volume']
            volume_ma = summary['Volume_MA_20']
            
            if pd.notna(volume_ma) and volume_ma > 0:
                summary['volume_above_average'] = current_volume > volume_ma
                summary['volume_ratio'] = current_volume / volume_ma
        
        return summary