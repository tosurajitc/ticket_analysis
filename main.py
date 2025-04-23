"""
Main entry point for the Incident Management Analytics application.
This file initializes the Streamlit application and routing logic.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "app.log"))
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to initialize and run the application.
    """
    try:
        logger.info("Starting Incident Management Analytics application")
        
        # Import here to avoid circular imports
        from config.config import Config
        from ui.app import main as run_app
        
        # Load configuration using the Config class
        config = Config()
        logger.info(f"Configuration loaded successfully: {config['app_name']}")
        
        # Run the Streamlit app
        run_app()
        
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()