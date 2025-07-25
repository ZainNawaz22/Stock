#!/usr/bin/env python3
"""
Demonstration script for DataStorage functionality
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psx_ai_advisor.data_storage import DataStorage


def demo_data_storage():
    """Demonstrate DataStorage functionality"""
    print("=== PSX AI Advisor - DataStorage Demo ===\n")
    
    # Initialize DataStorage
    storage = DataStorage()
    
    print(f"Data directory: {storage.data_dir}")
    print(f"Backup directory: {storage.backup_dir}")
    print()
    
    # Create sample data for demonstration
    print("Creating sample stock data...")
    
    # Sample data for ENGRO
    engro_data = []
    base_date = datetime.now() - timedelta(days=10)
    base_price = 250.0
    
    for i in range(5):
        date = base_date + timedelta(days=i)
        price = base_price + i * 2.5
        
        engro_data.append({
            'Date': date,
            'Symbol': 'ENGRO',
            'Company_Name': 'Engro Corporation Limited',
            'Open': price - 1.0,
            'High': price + 3.0,
            'Low': price - 2.0,
            'Close': price,
            'Volume': 100000 + i * 5000,
            'Previous_Close': price - 2.5 if i > 0 else price,
            'Change': 2.5 if i > 0 else 0.0
        })
    
    engro_df = pd.DataFrame(engro_data)
    
    # Sample data for HBL
    hbl_data = []
    for i in range(3):
        date = base_date + timedelta(days=i)
        price = 150.0 + i * 1.5
        
        hbl_data.append({
            'Date': date,
            'Symbol': 'HBL',
            'Company_Name': 'Habib Bank Limited',
            'Open': price - 0.5,
            'High': price + 2.0,
            'Low': price - 1.5,
            'Close': price,
            'Volume': 75000 + i * 3000,
            'Previous_Close': price - 1.5 if i > 0 else price,
            'Change': 1.5 if i > 0 else 0.0
        })
    
    hbl_df = pd.DataFrame(hbl_data)
    
    # Demonstrate saving data
    print("Saving stock data...")
    
    result1 = storage.save_stock_data("ENGRO", engro_df)
    result2 = storage.save_stock_data("HBL", hbl_df)
    
    print(f"ENGRO save result: {result1}")
    print(f"HBL save result: {result2}")
    print()
    
    # Demonstrate loading data
    print("Loading stock data...")
    
    loaded_engro = storage.load_stock_data("ENGRO")
    loaded_hbl = storage.load_stock_data("HBL")
    
    print(f"ENGRO records loaded: {len(loaded_engro)}")
    print(f"HBL records loaded: {len(loaded_hbl)}")
    print()
    
    # Show sample data
    print("Sample ENGRO data:")
    print(loaded_engro[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].head(3))
    print()
    
    # Demonstrate appending new data
    print("Appending new data to ENGRO...")
    
    new_engro_data = pd.DataFrame([{
        'Date': base_date + timedelta(days=5),
        'Symbol': 'ENGRO',
        'Company_Name': 'Engro Corporation Limited',
        'Open': 262.0,
        'High': 265.0,
        'Low': 260.0,
        'Close': 263.5,
        'Volume': 125000,
        'Previous_Close': 260.0,
        'Change': 3.5
    }])
    
    storage.save_stock_data("ENGRO", new_engro_data)
    
    updated_engro = storage.load_stock_data("ENGRO")
    print(f"ENGRO records after append: {len(updated_engro)}")
    print()
    
    # Demonstrate available symbols
    print("Available symbols:")
    symbols = storage.get_available_symbols()
    print(symbols)
    print()
    
    # Demonstrate data summary
    print("Data summary for ENGRO:")
    summary = storage.get_data_summary("ENGRO")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # Demonstrate data integrity validation
    print("Data integrity validation:")
    for symbol in symbols:
        is_valid = storage.validate_data_integrity(symbol)
        print(f"  {symbol}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    print()
    
    # Demonstrate storage statistics
    print("Storage statistics:")
    stats = storage.get_storage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=== Demo completed successfully! ===")


if __name__ == "__main__":
    try:
        demo_data_storage()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)