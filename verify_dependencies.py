"""
Verify that all required dependencies are correctly specified
"""

import sys

def check_dependency(module_name, import_statement=None):
    """Check if a dependency can be imported"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        print(f"✓ {module_name} - Available")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - Not available: {e}")
        return False

def main():
    """Check all dependencies from requirements.txt"""
    print("Checking PSX AI Advisor Dependencies")
    print("=" * 40)
    
    dependencies = [
        ("requests", None),
        ("pandas", None),
        ("pandas_ta", "import pandas_ta"),
        ("sklearn", "import sklearn"),
        ("yaml", "import yaml"),
        ("numpy", None),
        ("fastapi", None),
        ("uvicorn", None)
    ]
    
    all_available = True
    for dep_name, import_stmt in dependencies:
        if not check_dependency(dep_name, import_stmt):
            all_available = False
    
    print("\n" + "=" * 40)
    if all_available:
        print("All dependencies are available!")
    else:
        print("Some dependencies are missing. Run: pip install -r requirements.txt")
    
    return 0 if all_available else 1

if __name__ == "__main__":
    exit(main())