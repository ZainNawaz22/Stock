#!/usr/bin/env python3
"""
Detailed technical analysis for PTC stock with proper historical context
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer

def detailed_ptc_analysis():
    """Perform detailed technical analysis for PTC stock"""
    print("🔍 Detailed Technical Analysis for PTC Stock")
    print("=" * 60)
    
    try:
        # Load PTC data
        data_acq = PSXDataAcquisition()
        csv_path = "data/PTC_historical_data.csv"
        
        print("📊 Loading PTC historical data...")
        stock_data = data_acq.get_all_stock_data(csv_path=csv_path)
        
        # Sort by date ascending for proper indicator calculation
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Loaded {len(stock_data)} records for PTC")
        print(f"📅 Date range: {stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Add technical indicators
        print("\n🔧 Calculating technical indicators...")
        analyzer = TechnicalAnalyzer()
        enriched_data = analyzer.add_all_indicators(stock_data)
        
        # Get data with valid indicators (skip initial NaN values)
        valid_data = enriched_data.dropna(subset=['SMA_50', 'SMA_200', 'RSI_14']).copy()
        
        if len(valid_data) == 0:
            print("❌ Not enough data for complete technical analysis")
            return False
        
        print(f"✅ Technical indicators calculated for {len(valid_data)} records with complete data")
        
        # Get the most recent data
        latest = valid_data.iloc[-1]  # Most recent with all indicators
        recent_data = valid_data.tail(30)  # Last 30 records with indicators
        
        print(f"\n📈 Current PTC Stock Information:")
        print(f"Date: {latest['Date'].strftime('%Y-%m-%d')}")
        print(f"Close Price: ${latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,}")
        print(f"Daily Change: ${latest['Change']:.2f} ({(latest['Change']/latest['Previous_Close']*100):.1f}%)")
        
        # Display technical indicators
        print(f"\n📊 Technical Indicators (Latest Values):")
        print(f"SMA 50:  ${latest['SMA_50']:.2f}")
        print(f"SMA 200: ${latest['SMA_200']:.2f}")
        print(f"RSI 14:  {latest['RSI_14']:.1f}")
        print(f"MACD:    {latest['MACD']:.4f}")
        print(f"MACD Signal: {latest['MACD_Signal']:.4f}")
        print(f"MACD Histogram: {latest['MACD_Histogram']:.4f}")
        
        # Price vs Moving Averages
        print(f"\n📏 Price Position Analysis:")
        close = latest['Close']
        sma50 = latest['SMA_50']
        sma200 = latest['SMA_200']
        
        print(f"Price vs SMA50: {((close - sma50) / sma50 * 100):+.1f}%")
        print(f"Price vs SMA200: {((close - sma200) / sma200 * 100):+.1f}%")
        print(f"SMA50 vs SMA200: {((sma50 - sma200) / sma200 * 100):+.1f}%")
        
        # Technical analysis insights
        print(f"\n🎯 Technical Analysis Insights:")
        
        # RSI Analysis
        rsi = latest['RSI_14']
        if rsi > 70:
            print(f"• RSI ({rsi:.1f}): 🔴 OVERBOUGHT - Consider selling")
        elif rsi < 30:
            print(f"• RSI ({rsi:.1f}): 🟢 OVERSOLD - Consider buying")
        elif rsi > 50:
            print(f"• RSI ({rsi:.1f}): 🟡 BULLISH MOMENTUM")
        else:
            print(f"• RSI ({rsi:.1f}): 🟡 BEARISH MOMENTUM")
        
        # MACD Analysis
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        macd_hist = latest['MACD_Histogram']
        
        if macd > macd_signal:
            trend = "🟢 BULLISH"
        else:
            trend = "🔴 BEARISH"
        
        if macd_hist > 0:
            momentum = "🟢 STRENGTHENING"
        else:
            momentum = "🔴 WEAKENING"
        
        print(f"• MACD: {trend} - MACD {'above' if macd > macd_signal else 'below'} signal line")
        print(f"• MACD Momentum: {momentum} - Histogram: {macd_hist:.4f}")
        
        # Moving Average Analysis
        if sma50 > sma200:
            ma_trend = "🟢 BULLISH (Golden Cross)"
        else:
            ma_trend = "🔴 BEARISH (Death Cross)"
        
        print(f"• Moving Average Trend: {ma_trend}")
        
        if close > sma50 > sma200:
            position = "🟢 STRONG BULLISH - Price above both MAs"
        elif close > sma50 and sma50 < sma200:
            position = "🟡 MIXED - Price above SMA50 but below SMA200"
        elif close < sma50 and sma50 > sma200:
            position = "🟡 MIXED - Price below SMA50 but above SMA200"
        else:
            position = "🔴 STRONG BEARISH - Price below both MAs"
        
        print(f"• Price Position: {position}")
        
        # Volume Analysis
        avg_volume = recent_data['Volume'].mean()
        current_volume = latest['Volume']
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 1.5:
            volume_signal = "🟢 HIGH VOLUME - Strong interest"
        elif volume_ratio < 0.5:
            volume_signal = "🔴 LOW VOLUME - Weak interest"
        else:
            volume_signal = "🟡 NORMAL VOLUME"
        
        print(f"• Volume: {volume_signal} (Current: {current_volume:,}, Avg: {avg_volume:,.0f})")
        
        # Price trend analysis
        print(f"\n📈 Price Trend Analysis:")
        
        # Calculate price changes over different periods
        periods = [5, 10, 20, 30]
        for period in periods:
            if len(recent_data) >= period:
                old_price = recent_data.iloc[-period]['Close']
                price_change = ((close - old_price) / old_price) * 100
                trend_emoji = "🟢" if price_change > 0 else "🔴"
                print(f"• {period}-day change: {trend_emoji} {price_change:+.1f}%")
        
        # Support and Resistance levels
        print(f"\n🎯 Support & Resistance Levels (Last 30 days):")
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        
        print(f"• Resistance (30-day high): ${recent_high:.2f}")
        print(f"• Support (30-day low): ${recent_low:.2f}")
        print(f"• Current position: {((close - recent_low) / (recent_high - recent_low) * 100):.1f}% of range")
        
        # Display recent data table
        print(f"\n📋 Recent Trading Data (Last 15 Days):")
        display_data = recent_data.tail(15)[['Date', 'Close', 'Volume', 'RSI_14', 'MACD', 'SMA_50', 'SMA_200']].copy()
        display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
        display_data['Close'] = display_data['Close'].round(2)
        display_data['Volume'] = display_data['Volume'].astype(int)
        display_data['RSI_14'] = display_data['RSI_14'].round(1)
        display_data['MACD'] = display_data['MACD'].round(4)
        display_data['SMA_50'] = display_data['SMA_50'].round(2)
        display_data['SMA_200'] = display_data['SMA_200'].round(2)
        
        print(display_data.to_string(index=False))
        
        # Overall signal
        print(f"\n🎯 Overall Technical Signal:")
        
        signals = []
        if rsi > 70:
            signals.append("SELL")
        elif rsi < 30:
            signals.append("BUY")
        
        if macd > macd_signal and macd_hist > 0:
            signals.append("BUY")
        elif macd < macd_signal and macd_hist < 0:
            signals.append("SELL")
        
        if close > sma50 > sma200:
            signals.append("BUY")
        elif close < sma50 < sma200:
            signals.append("SELL")
        
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        
        if buy_signals > sell_signals:
            overall_signal = f"🟢 BULLISH ({buy_signals} buy vs {sell_signals} sell signals)"
        elif sell_signals > buy_signals:
            overall_signal = f"🔴 BEARISH ({sell_signals} sell vs {buy_signals} buy signals)"
        else:
            overall_signal = f"🟡 NEUTRAL ({buy_signals} buy vs {sell_signals} sell signals)"
        
        print(overall_signal)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = detailed_ptc_analysis()
    if success:
        print(f"\n🎉 PTC detailed technical analysis completed!")
    else:
        print(f"\n❌ Failed to complete PTC analysis")
    
    sys.exit(0 if success else 1)