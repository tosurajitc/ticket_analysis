import pandas as pd
import logging
import traceback
import sys
import os

def diagnose_csv_processing(file_path):
    """
    Comprehensive diagnostic function to investigate CSV processing issues
    
    Args:
        file_path (str): Path to the CSV file
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Attempting to load CSV file: {file_path}")
        
        # Method 1: Standard CSV loading
        logger.info("Method 1: Standard CSV Loading")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Standard Loading Success - Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Detailed column analysis
            logger.info("\nDetailed Column Analysis:")
            for col in df.columns:
                logger.info(f"\nColumn: {col}")
                logger.info(f"Data Type: {df[col].dtype}")
                
                # Check for potential datetime columns
                if 'date' in col.lower():
                    try:
                        converted_dates = pd.to_datetime(df[col], errors='coerce')
                        valid_dates = converted_dates.notna()
                        
                        logger.info(f"Datetime Conversion:")
                        logger.info(f"Total entries: {len(df[col])}")
                        logger.info(f"Valid date entries: {valid_dates.sum()}")
                        logger.info(f"Sample valid dates: {converted_dates[valid_dates].head()}")
                    except Exception as date_err:
                        logger.error(f"Datetime conversion error: {date_err}")
        
        except Exception as e:
            logger.error(f"CSV Loading Failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Method 2: With different parsing options
        logger.info("\nMethod 2: Advanced CSV Loading")
        try:
            df_advanced = pd.read_csv(
                file_path, 
                parse_dates=['created_date', 'resolved_date'],  # Specify known date columns
                infer_datetime_format=True,
                keep_default_na=False
            )
            logger.info(f"Advanced Loading Success - Shape: {df_advanced.shape}")
            logger.info(f"Columns: {list(df_advanced.columns)}")
            
            # Verify specific columns
            date_columns = ['created_date', 'resolved_date']
            for col in date_columns:
                if col in df_advanced.columns:
                    logger.info(f"\nColumn Analysis for {col}:")
                    logger.info(f"Data Type: {df_advanced[col].dtype}")
                    logger.info(f"Sample values: {df_advanced[col].head()}")
        
        except Exception as e:
            logger.error(f"Advanced CSV Loading Failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Additional pandas operations diagnostic
        logger.info("\nPandas Operations Diagnostic")
        try:
            # Test basic DataFrame operations
            logger.info("Testing DataFrame attribute access")
            logger.info(f"DataFrame type: {type(df)}")
            logger.info(f"Columns type: {type(df.columns)}")
            
            # Check .dtypes (plural) instead of .dtype
            logger.info("Checking column dtypes")
            logger.info(f"Columns dtypes: {df.dtypes}")
        
        except Exception as ops_err:
            logger.error(f"DataFrame operations diagnostic failed: {ops_err}")
            logger.error(traceback.format_exc())
        
    except Exception as main_error:
        logger.error(f"Diagnostic process failed: {main_error}")
        logger.error(traceback.format_exc())

# Allow running as script with file path argument
if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose_csv_processing(sys.argv[1])
    else:
        print("Please provide CSV file path as argument")