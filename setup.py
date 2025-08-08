#!/usr/bin/env python3
"""
PSX AI Advisor Setup Script

This script handles initial system configuration, environment setup,
and validation of the PSX AI Advisor system.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import shutil


class PSXAdvisorSetup:
    """Setup and configuration manager for PSX AI Advisor"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / "config.yaml"
        self.requirements_file = self.project_root / "requirements.txt"
        
    def create_default_config(self, environment: str = "development") -> Dict[str, Any]:
        """
        Create default configuration based on environment
        
        Args:
            environment (str): Target environment (development, production, testing)
            
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        base_config = {
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
                "max_train_size": None
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
        
        # Environment-specific overrides
        if environment == "development":
            base_config["logging"]["level"] = "DEBUG"
            base_config["logging"]["console_level"] = "DEBUG"
            base_config["logging"]["detailed_errors"] = True
            base_config["performance"]["retry_attempts"] = 2
            
        elif environment == "production":
            base_config["logging"]["level"] = "WARNING"
            base_config["logging"]["console_level"] = "INFO"
            base_config["logging"]["detailed_errors"] = False
            base_config["performance"]["max_concurrent_requests"] = 10
            
        elif environment == "testing":
            base_config["logging"]["level"] = "ERROR"
            base_config["logging"]["console_output"] = False
            base_config["storage"]["data_directory"] = "test_data"
            base_config["storage"]["backup_directory"] = "test_backups"
            base_config["machine_learning"]["min_training_samples"] = 50
            
        return base_config
    
    def create_directories(self) -> None:
        """Create necessary directories for the application"""
        directories = [
            "data",
            "data/models",
            "data/scalers", 
            "backups",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
    
    def validate_dependencies(self) -> bool:
        """
        Validate that all required Python packages are installed
        
        Returns:
            bool: True if all dependencies are satisfied
        """
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
            
        try:
            with open(self.requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            missing_packages = []
            for requirement in requirements:
                package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0]
                try:
                    __import__(package_name.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package_name)
            
            if missing_packages:
                print(f"❌ Missing packages: {', '.join(missing_packages)}")
                print("Run: pip install -r requirements.txt")
                return False
            else:
                print("✓ All dependencies satisfied")
                return True
                
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            return False
    
    def validate_config(self, config_path: Path = None) -> bool:
        """
        Validate configuration file
        
        Args:
            config_path (Path): Path to config file (defaults to self.config_file)
            
        Returns:
            bool: True if configuration is valid
        """
        if config_path is None:
            config_path = self.config_file
            
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        try:
            from psx_ai_advisor.config_loader import ConfigLoader
            config_loader = ConfigLoader(str(config_path))
            config_loader.validate_config()
            print("✓ Configuration file is valid")
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """
        Install Python dependencies from requirements.txt
        
        Returns:
            bool: True if installation successful
        """
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            print("Installing dependencies...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def create_config_file(self, environment: str = "development", force: bool = False) -> bool:
        """
        Create configuration file
        
        Args:
            environment (str): Target environment
            force (bool): Overwrite existing config file
            
        Returns:
            bool: True if config file created successfully
        """
        if self.config_file.exists() and not force:
            print(f"⚠️  Configuration file already exists: {self.config_file}")
            response = input("Overwrite? (y/N): ").lower().strip()
            if response != 'y':
                print("Configuration file creation skipped")
                return True
        
        try:
            config = self.create_default_config(environment)
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"✓ Created configuration file: {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to create configuration file: {e}")
            return False
    
    def run_system_check(self) -> bool:
        """
        Run comprehensive system check
        
        Returns:
            bool: True if all checks pass
        """
        print("Running system check...")
        print("=" * 50)
        
        checks = [
            ("Python version", self._check_python_version),
            ("Project structure", self._check_project_structure),
            ("Configuration file", lambda: self.validate_config()),
            ("Dependencies", self.validate_dependencies),
            ("Directories", self._check_directories),
            ("Permissions", self._check_permissions)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            print(f"\nChecking {check_name}...")
            try:
                if check_func():
                    print(f"✓ {check_name}: PASS")
                else:
                    print(f"❌ {check_name}: FAIL")
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name}: ERROR - {e}")
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("✓ All system checks passed!")
        else:
            print("❌ Some system checks failed. Please review and fix issues.")
        
        return all_passed
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"  Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"  Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
            return False
    
    def _check_project_structure(self) -> bool:
        """Check if required project files exist"""
        required_files = [
            "main.py",
            "requirements.txt",
            "psx_ai_advisor/__init__.py",
            "psx_ai_advisor/config_loader.py",
            "psx_ai_advisor/data_acquisition.py",
            "psx_ai_advisor/technical_analysis.py",
            "psx_ai_advisor/data_storage.py",
            "psx_ai_advisor/ml_predictor.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"  Missing files: {', '.join(missing_files)}")
            return False
        else:
            print("  All required files present")
            return True
    
    def _check_directories(self) -> bool:
        """Check if required directories exist"""
        required_dirs = ["data", "backups", "logs"]
        missing_dirs = []
        
        for directory in required_dirs:
            if not (self.project_root / directory).exists():
                missing_dirs.append(directory)
        
        if missing_dirs:
            print(f"  Missing directories: {', '.join(missing_dirs)}")
            return False
        else:
            print("  All required directories present")
            return True
    
    def _check_permissions(self) -> bool:
        """Check if we have necessary file permissions"""
        test_dirs = ["data", "backups", "logs"]
        
        for directory in test_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists():
                # Test write permission
                test_file = dir_path / ".permission_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception:
                    print(f"  No write permission for {directory}")
                    return False
        
        print("  File permissions OK")
        return True
    
    def setup_environment(self, environment: str = "development", 
                         install_deps: bool = True, 
                         force_config: bool = False) -> bool:
        """
        Complete environment setup
        
        Args:
            environment (str): Target environment
            install_deps (bool): Whether to install dependencies
            force_config (bool): Force config file creation
            
        Returns:
            bool: True if setup successful
        """
        print(f"Setting up PSX AI Advisor for {environment} environment...")
        print("=" * 60)
        
        steps = [
            ("Creating directories", self.create_directories),
            ("Creating configuration", lambda: self.create_config_file(environment, force_config))
        ]
        
        if install_deps:
            steps.append(("Installing dependencies", self.install_dependencies))
        
        steps.append(("Running system check", self.run_system_check))
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            try:
                if callable(step_func):
                    result = step_func()
                    if result is False:
                        print(f"❌ {step_name} failed")
                        return False
                    elif result is None:
                        # Functions like create_directories return None on success
                        pass
                else:
                    step_func
                print(f"✓ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("✓ PSX AI Advisor setup completed successfully!")
        print("\nNext steps:")
        print("1. Review configuration in config.yaml")
        print("2. Run: python main.py --help")
        print("3. Start with: python main.py")
        
        return True


def main():
    """Main setup script entry point"""
    parser = argparse.ArgumentParser(description="PSX AI Advisor Setup Script")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "production", "testing"],
        default="development",
        help="Target environment (default: development)"
    )
    parser.add_argument(
        "--no-deps", 
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--force-config",
        action="store_true", 
        help="Force configuration file creation"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run system check without setup"
    )
    
    args = parser.parse_args()
    
    setup = PSXAdvisorSetup()
    
    if args.check_only:
        success = setup.run_system_check()
    else:
        success = setup.setup_environment(
            environment=args.environment,
            install_deps=not args.no_deps,
            force_config=args.force_config
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()