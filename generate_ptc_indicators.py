#!/usr/bin/env python3
"""
Generate technical indicators for PTC stock
"""

import sys
import os
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
from psx_ai_advisor.data_storage import DataStorage

def generate_ptc_indicators():
    """Generate and display technical indicators for PTC stock"""
    print("Generating Technical Indicators for PTC Stock")
    print("=" * 50)
    
    try:
        # Load PTC data
        data_acq = PSXDataAcquisition()
        csv_path = "data/PTC_historical_data.csv"
        
        if not os.path.exists(csv_path):
            print(f"❌ PTC CSV file not found: {csv_path}")
            return False
        
        print("📊 Loading PTC historical data...")
        stock_data = data_acq.get_all_stock_data(csv_path=csv_path)
        print(f"✅ Loaded {len(stock_data)} records for {stock_data['Symbol'].iloc[0]}")
        print(f"📅 Date range: {stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Add technical indicators
        print("\n🔧 Calculating technical indicators...")
        analyzer = TechnicalAnalyzer()
        enriched_data = analyzer.add_all_indicators(stock_data)
        
        # Get the most recent data (last 30 days)
        recent_data = enriched_data.head(30).copy()
        
        print("✅ Technical indicators calculated successfully!")
        
        # Display current stock info
        latest = recent_data.iloc[0]
        print(f"\n📈 Current PTC Stock Information:")
        print(f"Date: {latest['Date'].strftime('%Y-%m-%d')}")
        print(f"Close Price: ${latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,}")
        print(f"Daily Change: ${latest['Change']:.2f}")
        
        # Display technical indicators
        print(f"\n📊 Technical Indicators (Latest Values):")
        print(f"SMA 50:  ${latest['SMA_50']:.2f}" if pd.notna(latest['SMA_50']) else "SMA 50:  Not available (need 50+ days)")
        print(f"SMA 200: ${latest['SMA_200']:.2f}" if pd.notna(latest['SMA_200']) else "SMA 200: Not available (need 200+ days)")
        print(f"RSI 14:  {latest['RSI_14']:.2f}" if pd.notna(latest['RSI_14']) else "RSI 14:  Not available")
        print(f"MACD:    {latest['MACD']:.4f}")
        print(f"MACD Signal: {latest['MACD_Signal']:.4f}")
        print(f"MACD Histogram: {latest['MACD_Histogram']:.4f}")
        
        # Technical analysis insights
        print(f"\n🎯 Technical Analysis Insights:")
        
        # RSI Analysis
        if pd.notna(latest['RSI_14']):
            rsi = latest['RSI_14']
            if rsi > 70:
                print(f"• RSI ({rsi:.1f}): OVERBOUGHT - Potential sell signal")
            elif rsi < 30:
                print(f"• RSI ({rsi:.1f}): OVERSOLD - Potential buy signal")
            else:
                print(f"• RSI ({rsi:.1f}): NEUTRAL - No strong signal")
        
        # MACD Analysis
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        if macd > macd_signal:
            print(f"• MACD: BULLISH - MACD above signal line")
        else:
            print(f"• MACD: BEARISH - MACD below signal line")
        
        # Moving Average Analysis
        if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
            sma50 = latest['SMA_50']
            sma200 = latest['SMA_200']
            close = latest['Close']
            
            if sma50 > sma200:
                print(f"• Moving Averages: BULLISH - SMA50 above SMA200 (Golden Cross)")
            else:
                print(f"• Moving Averages: BEARISH - SMA50 below SMA200 (Death Cross)")
            
            if close > sma50:
                print(f"• Price vs SMA50: BULLISH - Price above 50-day average")
            else:
                print(f"• Price vs SMA50: BEARISH - Price below 50-day average")
        
        # Display recent price action table
        print(f"\n📋 Recent Price Action (Last 10 Days):")
        display_columns = ['Date', 'Close', 'Volume', 'Change', 'RSI_14', 'MACD']
        recent_display = recent_data[display_columns].head(10).copy()
        recent_display['Date'] = recent_display['Date'].dt.strftime('%Y-%m-%d')
        recent_display['Close'] = recent_display['Close'].round(2)
        recent_display['Change'] = recent_display['Change'].round(2)
        recent_display['RSI_14'] = recent_display['RSI_14'].round(1)
        recent_display['MACD'] = recent_display['MACD'].round(4)
        
        print(recent_display.to_string(index=False))
        
        # Save enriched data
        print(f"\n💾 Saving enriched data...")
        storage = DataStorage()
        result = storage.save_stock_data("PTC", enriched_data)
        
        if result:
            print("✅ PTC data with technical indicators saved successfully!")
            
            # Show storage stats
            stats = storage.get_storage_stats()
            print(f"📁 Storage: {stats['total_records']} total records, {stats['total_size_mb']:.2f} MB")
        else:
            print("❌ Failed to save data")
        
        # Export to CSV for analysis
        output_file = "PTC_with_indicators.csv"
        enriched_data.to_csv(output_file, index=False)
        print(f"📄 Data exported to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_ptc_indicators()
    if success:
        print(f"\n🎉 PTC technical indicators generated successfully!")
        print(f"📊 Ready for machine learning and further analysis!")
    else:
        print(f"\n❌ Failed to generate PTC technical indicators")
    
    sys.exit(0 if success else 1)