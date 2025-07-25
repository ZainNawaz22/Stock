"""
PSX AI Advisor - Main application entry point
"""

from psx_ai_advisor.config_loader import get_config, get_section


def main():
    """Main application entry point"""
    try:
        # Load and validate configuration
        config = get_config()
        print("Configuration loaded successfully!")
        
        # Display configuration sections
        print("\nConfiguration sections:")
        for section in config.keys():
            print(f"  - {section}")
        
        print("\nPSX AI Advisor is ready to run!")
        
    except Exception as e:
        print(f"Error initializing PSX AI Advisor: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())