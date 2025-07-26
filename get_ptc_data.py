#!/usr/bin/env python3
"""
Get PTC stock data and technical indicators
"""

import sys
import os
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_acquisition import PSXDataAcquisition
from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
from psx_ai_advisor.data_storage import DataStorage

def get_ptc_data():
    """Get PTC data and technical indicators"""
    print("🔍 Fetching PTC stock data and technical indicators...")
    
    try:
        # Initialize components
        data_acq = PSXDataAcquisition()
        tech_analyzer = TechnicalAnalyzer()
        data_storage = DataStorage()
        
        # Get the latest stock data (use 2025-07-25 since today's data isn't available)
        print("\n📊 Loading stock data for 2025-07-25...")
        stock_data = data_acq.get_all_stock_data("2025-07-25")
        
        # Filter for PTC
        ptc_data = stock_data[stock_data['Symbol'] == 'PTC']
        
        if ptc_data.empty:
            print("❌ PTC not found in today's data. Let me check what symbols are available...")
            print(f"\nAvailable symbols (showing first 20):")
            symbols = sorted(stock_data['Symbol'].unique())
            for i, symbol in enumerate(symbols[:20]):
                print(f"  {symbol}")
            if len(symbols) > 20:
                print(f"  ... and {len(symbols) - 20} more symbols")
            
            # Check if there's a similar symbol
            similar_symbols = [s for s in symbols if 'PTC' in s.upper()]
            if similar_symbols:
                print(f"\n🔍 Found similar symbols: {', '.join(similar_symbols)}")
            
            return False
        
        print(f"✅ Found PTC data!")
        
        # Display current PTC data
        ptc_row = ptc_data.iloc[0]
        print(f"\n📈 PTC Stock Data for {ptc_row['Date'].strftime('%Y-%m-%d')}:")
        print(f"  Company: {ptc_row['Company_Name']}")
        print(f"  Open: {ptc_row['Open']:.2f}")
        print(f"  High: {ptc_row['High']:.2f}")
        print(f"  Low: {ptc_row['Low']:.2f}")
        print(f"  Close: {ptc_row['Close']:.2f}")
        print(f"  Volume: {ptc_row['Volume']:,}")
        print(f"  Previous Close: {ptc_row['Previous_Close']:.2f}")
        print(f"  Change: {ptc_row['Change']:+.2f}")
        print(f"  Change %: {(ptc_row['Change'] / ptc_row['Previous_Close'] * 100):+.2f}%")
        
        # Show market performance context
        print(f"\n📊 Market Context:")
        print(f"  Total stocks in market: {len(stock_data)}")
        
        # Top performers by volume
        top_volume = stock_data.nlargest(5, 'Volume')[['Symbol', 'Company_Name', 'Volume', 'Close', 'Change']]
        print(f"\n  Top 5 by volume:")
        for idx, row in top_volume.iterrows():
            print(f"    {row['Symbol']}: {row['Volume']:,} shares")
        
        # PTC's volume ranking
        ptc_volume_rank = (stock_data['Volume'] > ptc_row['Volume']).sum() + 1
        print(f"\n  PTC volume ranking: #{ptc_volume_rank} out of {len(stock_data)} stocks")
        
        # Get historical data for technical indicators
        print(f"\n📊 Loading historical data for technical indicators...")
        
        # Try to load existing historical data
        try:
            historical_data = data_storage.load_stock_data('PTC')
            if not historical_data.empty:
                print(f"✅ Loaded {len(historical_data)} historical records for PTC")
                
                # Add today's data if it's not already there
                latest_date = historical_data['Date'].max()
                current_date = ptc_row['Date']
                
                if current_date > latest_date:
                    # Append today's data
                    historical_data = pd.concat([historical_data, ptc_data], ignore_index=True)
                    historical_data = historical_data.sort_values('Date').reset_index(drop=True)
                    print(f"✅ Added today's data to historical records")
                
                # Calculate technical indicators
                print(f"\n🔧 Calculating technical indicators...")
                historical_with_indicators = tech_analyzer.add_all_indicators(historical_data)
                
                # Get the latest indicators
                latest_indicators = historical_with_indicators.iloc[-1]
                
                print(f"\n📊 Technical Indicators for PTC:")
                print(f"  SMA 50: {latest_indicators['SMA_50']:.2f}" if pd.notna(latest_indicators['SMA_50']) else "  SMA 50: Not enough data (need 50+ days)")
                print(f"  SMA 200: {latest_indicators['SMA_200']:.2f}" if pd.notna(latest_indicators['SMA_200']) else "  SMA 200: Not enough data (need 200+ days)")
                print(f"  RSI 14: {latest_indicators['RSI_14']:.2f}" if pd.notna(latest_indicators['RSI_14']) else "  RSI 14: Not enough data (need 15+ days)")
                print(f"  MACD: {latest_indicators['MACD']:.4f}" if pd.notna(latest_indicators['MACD']) else "  MACD: Not enough data (need 26+ days)")
                print(f"  MACD Signal: {latest_indicators['MACD_Signal']:.4f}" if pd.notna(latest_indicators['MACD_Signal']) else "  MACD Signal: Not enough data (need 35+ days)")
                print(f"  MACD Histogram: {latest_indicators['MACD_Histogram']:.4f}" if pd.notna(latest_indicators['MACD_Histogram']) else "  MACD Histogram: Not enough data (need 35+ days)")
                
                # Get indicator summary
                summary = tech_analyzer.get_indicator_summary(historical_with_indicators)
                
                print(f"\n📈 Technical Analysis Summary:")
                if 'price_above_sma_50' in summary:
                    print(f"  Price above SMA 50: {'Yes' if summary['price_above_sma_50'] else 'No'}")
                if 'price_above_sma_200' in summary:
                    print(f"  Price above SMA 200: {'Yes' if summary['price_above_sma_200'] else 'No'}")
                if 'golden_cross' in summary:
                    print(f"  Golden Cross (SMA 50 > SMA 200): {'Yes' if summary['golden_cross'] else 'No'}")
                if 'rsi_overbought' in summary:
                    print(f"  RSI Overbought (>70): {'Yes' if summary['rsi_overbought'] else 'No'}")
                if 'rsi_oversold' in summary:
                    print(f"  RSI Oversold (<30): {'Yes' if summary['rsi_oversold'] else 'No'}")
                
                # Save updated data
                data_storage.save_stock_data('PTC', historical_with_indicators)
                print(f"\n💾 Updated PTC data saved successfully")
                
            else:
                print("⚠️ No historical data found for PTC. Only current day data available.")
                print("Technical indicators require historical data for accurate calculations.")
                print("\n📋 Data Requirements for Technical Indicators:")
                print("  • SMA 50: Need 50+ trading days")
                print("  • SMA 200: Need 200+ trading days") 
                print("  • RSI 14: Need 15+ trading days")
                print("  • MACD: Need 26+ trading days")
                print("  • MACD Signal: Need 35+ trading days")
                
                # Save current data as starting point
                data_storage.save_stock_data('PTC', ptc_data)
                print("\n💾 Saved current PTC data as starting point for future analysis")
                
        except Exception as e:
            print(f"⚠️ Could not load historical data: {e}")
            print("Using only current day data...")
            print("\n📋 Data Requirements for Technical Indicators:")
            print("  • SMA 50: Need 50+ trading days")
            print("  • SMA 200: Need 200+ trading days") 
            print("  • RSI 14: Need 15+ trading days")
            print("  • MACD: Need 26+ trading days")
            print("  • MACD Signal: Need 35+ trading days")
            
            # Save current data
            data_storage.save_stock_data('PTC', ptc_data)
            print("\n💾 Saved current PTC data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting PTC data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = get_ptc_data()
    if success:
        print("\n🎉 PTC data retrieval completed successfully!")
        print("\n💡 To get technical indicators, you'll need to collect more historical data.")
        print("   Run this script daily to build up the historical dataset.")
    else:
        print("\n❌ PTC data retrieval failed")
    
    sys.exit(0 if success else 1)