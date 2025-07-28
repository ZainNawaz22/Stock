#!/usr/bin/env python3
"""
Simple test for ML predictor imports
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test individual imports
    print("Testing pandas import...")
    import pandas as pd
    print("✓ pandas imported successfully")
    
    print("Testing numpy import...")
    import numpy as np
    print("✓ numpy imported successfully")
    
    print("Testing sklearn import...")
    from sklearn.ensemble import RandomForestClassifier
    print("✓ sklearn imported successfully")
    
    print("Testing config loader import...")
    from psx_ai_advisor.config_loader import get_section
    print("✓ config_loader imported successfully")
    
    print("Testing data storage import...")
    from psx_ai_advisor.data_storage import DataStorage
    print("✓ data_storage imported successfully")
    
    print("Testing technical analysis import...")
    from psx_ai_advisor.technical_analysis import TechnicalAnalyzer
    print("✓ technical_analysis imported successfully")
    
    print("Testing ml_predictor module import...")
    import psx_ai_advisor.ml_predictor
    print("✓ ml_predictor module imported successfully")
    
    print("Module contents:", [x for x in dir(psx_ai_advisor.ml_predictor) if not x.startswith('_')])
    
    # Try to execute the file content directly
    print("Testing direct execution...")
    with open('psx_ai_advisor/ml_predictor.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}")
    print(f"Last 200 characters: {content[-200:]}")
    
    # Try to execute in a namespace
    namespace = {}
    exec(content, namespace)
    print("Available in namespace:", [x for x in namespace.keys() if not x.startswith('_')])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()