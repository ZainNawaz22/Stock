"""
Configuration loader utility for PSX AI Advisor
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Utility class to load and manage configuration settings"""
    
    def __init__(self, config_path: str = "config.yaml", environment: Optional[str] = None):
        """
        Initialize the configuration loader
        
        Args:
            config_path (str): Path to the configuration file
            environment (str): Environment name for environment-specific configs
        """
        self.config_path = config_path
        self.environment = environment or os.getenv('PSX_ENVIRONMENT', 'development')
        self._config = None
        self._default_config = self._get_default_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment-specific overrides
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        # Start with default configuration
        config = self._default_config.copy()
        
        # Load main config file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    file_config = yaml.safe_load(file) or {}
                    config = self._merge_configs(config, file_config)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing configuration file: {e}")
        
        # Load environment-specific config if it exists
        env_config_path = self._get_env_config_path()
        if env_config_path and os.path.exists(env_config_path):
            try:
                with open(env_config_path, 'r', encoding='utf-8') as file:
                    env_config = yaml.safe_load(file) or {}
                    config = self._merge_configs(config, env_config)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing environment config file: {e}")
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        self._config = config
        return self._config
    
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
        required_ml_keys = ['model_type', 'min_training_samples', 'random_state']
        missing_ml_keys = [key for key in required_ml_keys if key not in ml_config]
        if missing_ml_keys:
            raise ValueError(f"Missing required machine_learning keys: {missing_ml_keys}")
        
        # Validate model_type value
        valid_model_types = ['RandomForest', 'random_forest', 'ensemble', 'xgboost']
        model_type = ml_config.get('model_type', '').lower()
        if model_type not in [t.lower() for t in valid_model_types]:
            raise ValueError(f"Invalid model_type '{ml_config.get('model_type')}'. Must be one of: {valid_model_types}")
        
        # Validate ensemble configuration if model_type is ensemble
        if model_type == 'ensemble':
            if 'ensemble' not in ml_config:
                raise ValueError("Ensemble configuration section is required when model_type is 'ensemble'")
            self._validate_ensemble_config(ml_config)
        
        return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "data_sources": {
                "psx_base_url": "https://dps.psx.com.pk",
                "downloads_endpoint": "/download/closing_rates",
                "pdf_filename_pattern": "{date}.pdf"
            },
            "technical_indicators": {
                "sma_periods": [50, 200],
                "rsi_period": 14,
                "macd": {
                    "fast": 12,
                    "slow": 26,
                    "signal": 9
                }
            },
            "machine_learning": {
                "model_type": "RandomForest",
                "min_training_samples": 200,
                "random_state": 42,
                "n_estimators": 100,
                "n_splits": 5,
                "max_train_size": None,
                "threshold_tuning": {
                    "enabled": True,
                    "metric": "f1",
                    "min_threshold": 0.3,
                    "max_threshold": 0.7,
                    "step": 0.05,
                    "utility": {
                        "tp_reward": 1.0,
                        "tn_reward": 0.0,
                        "fp_cost": 1.0,
                        "fn_cost": 1.0
                    }
                },
                "ensemble": {
                    "models": ["random_forest", "xgboost"],
                    "voting": "soft",
                    "optimize_weights": True,
                    "fallback_strategy": "best_individual"
                },
                "xgboost": {
                    "eval_metric": "logloss",
                    "early_stopping_rounds": 10,
                    "verbose": False
                }
            },
            "storage": {
                "data_directory": "data",
                "backup_directory": "backups",
                "max_file_age_days": 365
            },
            "logging": {
                "level": "INFO",
                "file": "psx_advisor.log",
                "max_size_mb": 10,
                "backup_count": 5,
                "console_output": True,
                "console_level": "INFO",
                "detailed_errors": True,
                "log_directory": "logs"
            },
            "performance": {
                "max_concurrent_requests": 5,
                "request_timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 2
            }
        }
    
    def _get_env_config_path(self) -> Optional[str]:
        """
        Get environment-specific config file path
        
        Returns:
            Optional[str]: Path to environment config file
        """
        if self.environment:
            base_path = Path(self.config_path).parent
            env_config_name = f"config.{self.environment}.yaml"
            return str(base_path / env_config_name)
        return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries
        
        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_ensemble_config(self, ml_config: Dict[str, Any]) -> None:
        """
        Validate ensemble-specific configuration
        
        Args:
            ml_config (Dict[str, Any]): Machine learning configuration section
            
        Raises:
            ValueError: If ensemble configuration is invalid
        """
        ensemble_config = ml_config['ensemble']
        
        # Validate required ensemble keys
        required_ensemble_keys = ['models', 'voting', 'optimize_weights']
        missing_ensemble_keys = [key for key in required_ensemble_keys if key not in ensemble_config]
        if missing_ensemble_keys:
            raise ValueError(f"Missing required ensemble configuration keys: {missing_ensemble_keys}")
        
        # Validate models list
        models = ensemble_config.get('models', [])
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError("Ensemble models must be a list with at least 2 models")
        
        valid_models = ['random_forest', 'xgboost']
        invalid_models = [model for model in models if model not in valid_models]
        if invalid_models:
            raise ValueError(f"Invalid ensemble models: {invalid_models}. Valid models are: {valid_models}")
        
        # Validate voting type
        voting = ensemble_config.get('voting', '')
        valid_voting_types = ['soft', 'hard']
        if voting not in valid_voting_types:
            raise ValueError(f"Invalid voting type '{voting}'. Must be one of: {valid_voting_types}")
        
        # Validate optimize_weights
        optimize_weights = ensemble_config.get('optimize_weights')
        if not isinstance(optimize_weights, bool):
            raise ValueError("optimize_weights must be a boolean value")
        
        # Validate optional fallback_strategy
        fallback_strategy = ensemble_config.get('fallback_strategy')
        if fallback_strategy is not None:
            valid_fallback_strategies = ['best_individual', 'random_forest', 'xgboost']
            if fallback_strategy not in valid_fallback_strategies:
                raise ValueError(f"Invalid fallback_strategy '{fallback_strategy}'. Must be one of: {valid_fallback_strategies}")
        
        # Validate XGBoost configuration if xgboost is in models
        if 'xgboost' in models and 'xgboost' in ml_config:
            self._validate_xgboost_config(ml_config['xgboost'])
    
    def _validate_xgboost_config(self, xgb_config: Dict[str, Any]) -> None:
        """
        Validate XGBoost-specific configuration
        
        Args:
            xgb_config (Dict[str, Any]): XGBoost configuration section
            
        Raises:
            ValueError: If XGBoost configuration is invalid
        """
        # Validate eval_metric
        eval_metric = xgb_config.get('eval_metric')
        if eval_metric is not None:
            valid_eval_metrics = ['logloss', 'error', 'auc', 'aucpr']
            if eval_metric not in valid_eval_metrics:
                raise ValueError(f"Invalid XGBoost eval_metric '{eval_metric}'. Must be one of: {valid_eval_metrics}")
        
        # Validate early_stopping_rounds
        early_stopping = xgb_config.get('early_stopping_rounds')
        if early_stopping is not None:
            if not isinstance(early_stopping, int) or early_stopping < 1:
                raise ValueError("XGBoost early_stopping_rounds must be a positive integer")
        
        # Validate verbose
        verbose = xgb_config.get('verbose')
        if verbose is not None and not isinstance(verbose, bool):
            raise ValueError("XGBoost verbose must be a boolean value")

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration
        
        Args:
            config (Dict[str, Any]): Base configuration
            
        Returns:
            Dict[str, Any]: Configuration with environment overrides
        """
        # Define environment variable mappings
        env_mappings = {
            'PSX_BASE_URL': ('data_sources', 'psx_base_url'),
            'PSX_DATA_DIR': ('storage', 'data_directory'),
            'PSX_BACKUP_DIR': ('storage', 'backup_directory'),
            'PSX_LOG_LEVEL': ('logging', 'level'),
            'PSX_LOG_DIR': ('logging', 'log_directory'),
            'PSX_MODEL_TYPE': ('machine_learning', 'model_type'),
            'PSX_MIN_SAMPLES': ('machine_learning', 'min_training_samples'),
            'PSX_RETRY_ATTEMPTS': ('performance', 'retry_attempts'),
            'PSX_REQUEST_TIMEOUT': ('performance', 'request_timeout')
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if key in ['min_training_samples', 'retry_attempts', 'request_timeout']:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        continue
                
                elif key in ['console_output', 'detailed_errors']:
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                
                if section in config:
                    config[section][key] = env_value
        
        return config
    
    def get_environment(self) -> str:
        """
        Get current environment name
        
        Returns:
            str: Environment name
        """
        return self.environment
    
    def set_environment(self, environment: str) -> None:
        """
        Set environment and reload configuration
        
        Args:
            environment (str): Environment name
        """
        self.environment = environment
        self._config = None  # Force reload on next access
    
    def is_ensemble_enabled(self) -> bool:
        """
        Check if ensemble model is enabled
        
        Returns:
            bool: True if model_type is 'ensemble'
        """
        try:
            model_type = self.get_value('machine_learning', 'model_type', 'RandomForest')
            return model_type.lower() == 'ensemble'
        except Exception:
            return False
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """
        Get ensemble configuration with defaults
        
        Returns:
            Dict[str, Any]: Ensemble configuration
        """
        try:
            ml_config = self.get_section('machine_learning')
            ensemble_config = ml_config.get('ensemble', {})
            
            # Apply defaults for missing keys
            defaults = {
                'models': ['random_forest', 'xgboost'],
                'voting': 'soft',
                'optimize_weights': True,
                'fallback_strategy': 'best_individual'
            }
            
            for key, default_value in defaults.items():
                if key not in ensemble_config:
                    ensemble_config[key] = default_value
            
            return ensemble_config
        except Exception:
            # Return default ensemble config if section is missing
            return {
                'models': ['random_forest', 'xgboost'],
                'voting': 'soft',
                'optimize_weights': True,
                'fallback_strategy': 'best_individual'
            }
    
    def get_xgboost_config(self) -> Dict[str, Any]:
        """
        Get XGBoost configuration with defaults
        
        Returns:
            Dict[str, Any]: XGBoost configuration
        """
        try:
            ml_config = self.get_section('machine_learning')
            xgb_config = ml_config.get('xgboost', {})
            
            # Apply defaults for missing keys
            defaults = {
                'eval_metric': 'logloss',
                'early_stopping_rounds': 10,
                'verbose': False
            }
            
            for key, default_value in defaults.items():
                if key not in xgb_config:
                    xgb_config[key] = default_value
            
            return xgb_config
        except Exception:
            # Return default XGBoost config if section is missing
            return {
                'eval_metric': 'logloss',
                'early_stopping_rounds': 10,
                'verbose': False
            }


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


def is_ensemble_enabled() -> bool:
    """
    Convenience function to check if ensemble model is enabled
    
    Returns:
        bool: True if model_type is 'ensemble'
    """
    return config_loader.is_ensemble_enabled()


def get_ensemble_config() -> Dict[str, Any]:
    """
    Convenience function to get ensemble configuration
    
    Returns:
        Dict[str, Any]: Ensemble configuration with defaults
    """
    return config_loader.get_ensemble_config()


def get_xgboost_config() -> Dict[str, Any]:
    """
    Convenience function to get XGBoost configuration
    
    Returns:
        Dict[str, Any]: XGBoost configuration with defaults
    """
    return config_loader.get_xgboost_config()