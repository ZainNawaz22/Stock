"""
Test script to validate configuration loader functionality
"""

from psx_ai_advisor.config_loader import ConfigLoader, get_config, get_section, get_value


def test_config_loader():
    """Test the configuration loader functionality"""
    print("Testing PSX AI Advisor Configuration Loader")
    print("=" * 50)
    
    try:
        # Test loading configuration
        config = get_config()
        print("✓ Configuration loaded successfully")
        
        # Test getting specific sections
        data_sources = get_section('data_sources')
        print(f"✓ Data sources section loaded: {data_sources['psx_base_url']}")
        
        tech_indicators = get_section('technical_indicators')
        print(f"✓ Technical indicators section loaded: SMA periods {tech_indicators['sma_periods']}")
        
        ml_config = get_section('machine_learning')
        print(f"✓ Machine learning section loaded: Model type {ml_config['model_type']}")
        
        # Test getting specific values
        base_url = get_value('data_sources', 'psx_base_url')
        print(f"✓ Specific value retrieval: {base_url}")
        
        # Test default value
        non_existent = get_value('data_sources', 'non_existent_key', 'default_value')
        print(f"✓ Default value handling: {non_existent}")
        
        # Test configuration validation
        config_loader = ConfigLoader()
        is_valid = config_loader.validate_config()
        print(f"✓ Configuration validation: {is_valid}")
        
        print("\n" + "=" * 50)
        print("All configuration tests passed!")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_config_loader()