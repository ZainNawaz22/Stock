"""
Configuration loader utility for PSX AI Advisor
"""

import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Utility class to load and manage configuration settings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            return self._config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific section from the configuration
        
        Args:
            section_name (str): Name of the configuration section
            
        Returns:
            Dict[str, Any]: Configuration section
            
        Raises:
            KeyError: If section doesn't exist
        """
        config = self.get_config()
        if section_name not in config:
            raise KeyError(f"Configuration section '{section_name}' not found")
        return config[section_name]
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific value from the configuration
        
        Args:
            section (str): Configuration section name
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        try:
            section_config = self.get_section(section)
            return section_config.get(key, default)
        except KeyError:
            return default
    
    def validate_config(self) -> bool:
        """
        Validate that all required configuration sections exist
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If required sections are missing
        """
        required_sections = [
            'data_sources',
            'technical_indicators',
            'machine_learning',
            'storage',
            'logging',
            'performance'
        ]
        
        config = self.get_config()
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate data_sources section
        data_sources = config['data_sources']
        required_data_keys = ['psx_base_url', 'downloads_endpoint']
        missing_data_keys = [key for key in required_data_keys if key not in data_sources]
        if missing_data_keys:
            raise ValueError(f"Missing required data_sources keys: {missing_data_keys}")
        
        # Validate technical_indicators section
        tech_indicators = config['technical_indicators']
        required_tech_keys = ['sma_periods', 'rsi_period', 'macd']
        missing_tech_keys = [key for key in required_tech_keys if key not in tech_indicators]
        if missing_tech_keys:
            raise ValueError(f"Missing required technical_indicators keys: {missing_tech_keys}")
        
        # Validate machine_learning section
        ml_config = config['machine_learning']
        required_ml_keys = ['model_type', 'min_training_samples', 'test_size', 'random_state']
        missing_ml_keys = [key for key in required_ml_keys if key not in ml_config]
        if missing_ml_keys:
            raise ValueError(f"Missing required machine_learning keys: {missing_ml_keys}")
        
        return True


# Global configuration instance
config_loader = ConfigLoader()


def get_config() -> Dict[str, Any]:
    """
    Convenience function to get the global configuration
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return config_loader.get_config()


def get_section(section_name: str) -> Dict[str, Any]:
    """
    Convenience function to get a configuration section
    
    Args:
        section_name (str): Name of the configuration section
        
    Returns:
        Dict[str, Any]: Configuration section
    """
    return config_loader.get_section(section_name)


def get_value(section: str, key: str, default: Any = None) -> Any:
    """
    Convenience function to get a configuration value
    
    Args:
        section (str): Configuration section name
        key (str): Configuration key
        default (Any): Default value if key not found
        
    Returns:
        Any: Configuration value or default
    """
    return config_loader.get_value(section, key, default)