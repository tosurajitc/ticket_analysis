
# Module for extracting ticket data from uploaded files

import pandas as pd
import io
import streamlit as st

class DataExtractor:
    """
    Class for extracting data from uploaded CSV or Excel files.
    Handles chunking for large datasets.
    """
    
    def __init__(self, chunk_size=1000):
        """
        Initialize the DataExtractor.
        
        Args:
            chunk_size (int): Size of chunks for processing large datasets
        """
        self.chunk_size = chunk_size
    
    def extract(self, uploaded_file):
        """
        Extract data from the uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Extracted data
        """
        try:
            # Get file extension
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()
            
            # Read file based on extension
            if file_extension == 'csv':
                return self._extract_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                return self._extract_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            st.error(f"Error extracting data: {str(e)}")
            raise
    
    def _extract_csv(self, uploaded_file):
        """
        Extract data from CSV file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Extracted data
        """
        # Check if file is large (>10MB as an example threshold)
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            return self._process_large_csv(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    
    def _extract_excel(self, uploaded_file):
        """
        Extract data from Excel file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Extracted data
        """
        # Check if file is large (>10MB as an example threshold)
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            return self._process_large_excel(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    
    def _process_large_csv(self, uploaded_file):
        """
        Process large CSV files in chunks.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Combined data from all chunks
        """
        # Create a text IO buffer from the uploaded file's content
        text_io = io.StringIO(uploaded_file.getvalue().decode('utf-8'))
        
        # Process file in chunks
        chunks = []
        with st.spinner('Processing large CSV file in chunks...'):
            # Read and process chunks
            for chunk in pd.read_csv(text_io, chunksize=self.chunk_size):
                chunks.append(chunk)
                
            # Combine all chunks
            df = pd.concat(chunks, ignore_index=True)
            
        return df
    
    def _process_large_excel(self, uploaded_file):
        """
        Process large Excel files.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Combined data from all chunks
        """
        # Since Excel files can't be read in chunks directly with pandas,
        # we'll use a different approach
        with st.spinner('Processing large Excel file...'):
            # Create a bytes IO buffer
            bytes_io = io.BytesIO(uploaded_file.getvalue())
            
            # Use pandas ExcelFile to get sheet names
            excel_file = pd.ExcelFile(bytes_io)
            
            # Read the first sheet (or allow user to select a sheet in a more complex implementation)
            sheet_name = excel_file.sheet_names[0]
            
            # Read the Excel file
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
        return df
    
    def clean_column_names(self, df):
        """
        Clean column names by stripping whitespace and converting to lowercase.
        
        Args:
            df (pandas.DataFrame): DataFrame with columns to clean
            
        Returns:
            pandas.DataFrame: DataFrame with cleaned column names
        """
        df.columns = df.columns.str.strip().str.lower()
        return df