"""
Demonstration of TechnicalAnalyzer with sufficient data

This script shows how the technical indicators work when there's enough historical data.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the psx_ai_advisor module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'psx_ai_advisor'))

from psx_ai_advisor.technical_analysis import TechnicalAnalyzer


def create_realistic_stock_data(symbol="DEMO", days=250, start_price=100):
    """Create realistic stock data with trends and volatility."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)
    
    # Create a trending price series with volatility
    trend = np.linspace(0, 20, days)  # Upward trend
    noise = np.random.normal(0, 2, days)  # Daily volatility
    momentum = np.cumsum(np.random.normal(0, 0.5, days))  # Momentum component
    
    close_prices = start_price + trend + noise + momentum
    close_prices = np.maximum(close_prices, 1)  # Ensure positive prices
    
    # Generate OHLV data based on close prices
    data = []
    for i, close in enumerate(close_prices):
        # Generate realistic OHLV based on close price
        daily_range = abs(np.random.normal(0, close * 0.02))  # 2% average daily range
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)
        volume = int(np.random.uniform(50000, 200000))
        
        data.append({
            'Date': dates[i],
            'Symbol': symbol,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


def demonstrate_technical_analysis():
    """Demonstrate technical analysis with sufficient data."""
    print("=== PSX AI Advisor - Technical Analysis Demonstration ===\n")
    
    # Create realistic stock data
    stock_data = create_realistic_stock_data("DEMO", 250, 100)
    print(f"Created {len(stock_data)} days of stock data for demonstration")
    print(f"Price range: {stock_data['Close'].min():.2f} to {stock_data['Close'].max():.2f}")
    print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}\n")
    
    # Initialize technical analyzer
    analyzer = TechnicalAnalyzer()
    
    # Add all technical indicators
    print("Adding technical indicators...")
    enriched_data = analyzer.add_all_indicators(stock_data)
    
    # Display the latest values
    print("\n=== Latest Technical Indicator Values ===")
    latest = enriched_data.iloc[-1]
    
    print(f"Stock: {latest['Symbol']}")
    print(f"Date: {enriched_data.index[-1].date()}")
    print(f"Close Price: ${latest['Close']:.2f}")
    print()
    
    # Moving Averages
    print("Moving Averages:")
    if pd.notna(latest['SMA_50']):
        print(f"  SMA 50:  ${latest['SMA_50']:.2f}")
        trend_50 = "↑" if latest['Close'] > latest['SMA_50'] else "↓"
        print(f"  Price vs SMA 50: {trend_50} ({'Above' if latest['Close'] > latest['SMA_50'] else 'Below'})")
    
    if pd.notna(latest['SMA_200']):
        print(f"  SMA 200: ${latest['SMA_200']:.2f}")
        trend_200 = "↑" if latest['Close'] > latest['SMA_200'] else "↓"
        print(f"  Price vs SMA 200: {trend_200} ({'Above' if latest['Close'] > latest['SMA_200'] else 'Below'})")
    
    if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        golden_cross = latest['SMA_50'] > latest['SMA_200']
        print(f"  Golden Cross: {'Yes ✓' if golden_cross else 'No ✗'} (Bullish signal)")
    
    print()
    
    # RSI
    print("Relative Strength Index (RSI):")
    if pd.notna(latest['RSI_14']):
        rsi = latest['RSI_14']
        print(f"  RSI 14: {rsi:.2f}")
        if rsi > 70:
            print("  Signal: Overbought ⚠️ (Consider selling)")
        elif rsi < 30:
            print("  Signal: Oversold 📈 (Consider buying)")
        else:
            print("  Signal: Neutral ➡️")
    
    print()
    
    # MACD
    print("MACD (Moving Average Convergence Divergence):")
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
        macd = latest['MACD']
        signal = latest['MACD_Signal']
        histogram = latest['MACD_Histogram']
        
        print(f"  MACD Line: {macd:.4f}")
        print(f"  Signal Line: {signal:.4f}")
        print(f"  Histogram: {histogram:.4f}")
        
        if macd > signal:
            print("  Signal: Bullish 📈 (MACD above signal)")
        else:
            print("  Signal: Bearish 📉 (MACD below signal)")
    
    print()
    
    # Get comprehensive summary
    summary = analyzer.get_indicator_summary(enriched_data)
    
    print("=== Trading Signals Summary ===")
    signals = []
    
    if summary.get('price_above_sma_50'):
        signals.append("✓ Price above 50-day MA (Short-term bullish)")
    else:
        signals.append("✗ Price below 50-day MA (Short-term bearish)")
    
    if summary.get('price_above_sma_200'):
        signals.append("✓ Price above 200-day MA (Long-term bullish)")
    else:
        signals.append("✗ Price below 200-day MA (Long-term bearish)")
    
    if summary.get('golden_cross'):
        signals.append("✓ Golden Cross (50-MA > 200-MA)")
    else:
        signals.append("✗ Death Cross (50-MA < 200-MA)")
    
    if not summary.get('rsi_overbought') and not summary.get('rsi_oversold'):
        signals.append("✓ RSI in normal range")
    elif summary.get('rsi_overbought'):
        signals.append("⚠️ RSI indicates overbought")
    elif summary.get('rsi_oversold'):
        signals.append("📈 RSI indicates oversold")
    
    for signal in signals:
        print(f"  {signal}")
    
    # Show recent trend
    print(f"\n=== Recent Price Trend (Last 5 Days) ===")
    recent_data = enriched_data[['Close', 'SMA_50', 'RSI_14']].tail(5)
    print(recent_data.to_string(float_format='%.2f'))
    
    print(f"\n=== Technical Analysis Complete ===")
    print(f"All {len(['SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram'])} indicators calculated successfully!")
    
    return enriched_data


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for demo
    
    try:
        result = demonstrate_technical_analysis()
        print(f"\nDemo completed successfully! Generated {result.shape[0]} rows with {result.shape[1]} columns.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()