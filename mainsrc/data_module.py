# mainsrc/data_module.py
# Wrapper for data_extraction and data_processing

import sys
import os
import streamlit as st
import pandas as pd

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from data_extraction import DataExtractor
from data_processing import DataProcessor

class DataModule:
    """
    Wrapper for data extraction and processing functionality.
    Provides a unified interface for data operations.
    """
    
    def __init__(self):
        """Initialize the data module with extractors and processors."""
        self.extractor = DataExtractor()
        self.processor = DataProcessor()
    
    def extract_data(self, uploaded_file):
        """
        Extract data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Extracted data or None if error
        """
        try:
            return self.extractor.extract(uploaded_file)
        except Exception as e:
            st.error(f"Error extracting data: {str(e)}")
            return None
    
    def process_data(self, df):
        """
        Process and transform the raw data.
        
        Args:
            df: Pandas DataFrame to process
            
        Returns:
            pandas.DataFrame: Processed data
        """
        try:
            return self.processor.process(df)
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return df
    
    def validate_data(self, df):
        """
        Validate the data has expected columns and format.
        
        Args:
            df: Pandas DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if df is None:
            return False
        
        if len(df) == 0:
            st.error("The uploaded file contains no data.")
            return False
        
        # Check for essential columns
        essential_columns = ['number'] 
        missing_columns = [col for col in essential_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing essential columns: {', '.join(missing_columns)}")
            # Continue anyway, but warn the user
        
        return True