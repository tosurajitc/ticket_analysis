"""
Data loading module for the Incident Management Analytics application.
This module handles loading and initial preprocessing of incident data from various file formats.
"""

import io
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from data.data_validator import DataValidator
from config.constants import MANDATORY_INCIDENT_COLUMNS, OPTIONAL_INCIDENT_COLUMNS

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Simplified class for loading incident data from converted and formatted files.
    Focuses on loading data from file paths, removing upload functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader with application configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.supported_file_types = config["data"]["supported_file_types"]
        self.chunk_size = config["data"]["chunk_size"]
        self.mandatory_columns = config["data"]["mandatory_columns"]
        self.optional_columns = config["data"]["optional_columns"]
        self.date_format = config["data"]["date_format"]
        self.max_sample_size = config["data"]["max_sample_size"]
    
    def load_data(self, file_content: bytes, file_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load incident data from file content with enhanced logging and error handling.
        
        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            
        Returns:
            Tuple containing:
                - DataFrame with incident data
                - Dictionary with metadata about the loaded data
        
        Raises:
            ValueError: If the file format is not supported or data is invalid
        """
        logger.info(f"Starting data loading process for file: {file_name}")
        
        # Check if this is a formatted file
        is_formatted = False
        if file_name.endswith('_formatted.csv') or '_formatted_' in file_name:
            is_formatted = True
            logger.info(f"Detected pre-formatted file: {file_name}")
        
        # Extract file extension
        file_ext = Path(file_name).suffix.lower()
        # Normalize the extension by removing the dot for comparison
        normalized_ext = file_ext.lstrip('.')
        
        # Normalize supported file types for comparison
        normalized_supported_types = [ext.lstrip('.') for ext in self.supported_file_types]
        
        # Check if file type is supported
        if normalized_ext not in normalized_supported_types:
            supported_types_display = ', '.join(self.supported_file_types)
            error_msg = f"Unsupported file type: {normalized_ext}. Supported types are: {supported_types_display}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extended error handling with multiple encoding attempts
        load_errors = []
        
        try:
            # For pre-formatted files, use direct loading with fewer attempts
            if is_formatted and normalized_ext == 'csv':
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_content), 
                        encoding='utf-8',
                        low_memory=False
                    )
                    logger.info("Successfully loaded pre-formatted CSV file")
                except Exception as e:
                    # Try alternate encoding if UTF-8 fails
                    try:
                        df = pd.read_csv(
                            io.BytesIO(file_content), 
                            encoding='latin-1',
                            low_memory=False
                        )
                        logger.info("Successfully loaded pre-formatted CSV with latin-1 encoding")
                    except Exception as e2:
                        load_errors.append(f"Pre-formatted CSV load attempts failed: {str(e)}, {str(e2)}")
                        raise ValueError(f"Could not load pre-formatted CSV file: {load_errors}")
            else:
                # Regular loading logic for non-formatted files - CSV only
                if normalized_ext == 'csv':
                    # Try multiple encodings and parsing strategies
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'utf-16']
                    delimiters_to_try = [',', ';', '\t', '|']
                    
                    for encoding in encodings_to_try:
                        for delimiter in delimiters_to_try:
                            try:
                                df = pd.read_csv(
                                    io.BytesIO(file_content), 
                                    encoding=encoding, 
                                    delimiter=delimiter,
                                    low_memory=False,
                                    parse_dates=True
                                )
                                
                                # Additional validation
                                if not df.empty and len(df.columns) > 1:
                                    logger.info(f"Successfully loaded CSV with encoding {encoding} and delimiter {delimiter}")
                                    break
                            except Exception as e:
                                load_errors.append(f"CSV load attempt failed: {str(e)}")
                                continue
                        else:
                            continue
                        break
                    else:
                        raise ValueError(f"Could not load CSV file. Attempts failed: {load_errors}")
                else:
                    raise ValueError(f"Only CSV files are supported for converted data")
            
            # Log basic information about loaded data
            logger.info(f"Loaded data shape: {df.shape}")
            logger.info(f"Loaded data columns: {list(df.columns)}")
            
            # Validate data structure and content
            validator = DataValidator()
            validation_results = validator.validate(df, self.mandatory_columns)
            
            # If validation failed
            if not validation_results.get("is_valid", False):
                logger.error(f"Data validation failed: {validation_results['errors']}")
                raise ValueError(f"Invalid data format: {validation_results['errors']}")
            
            # Check if there's enough data for meaningful analysis
            if len(df) < 5:  # Reduced threshold for converted data
                logger.warning("Limited data for analysis")
                logger.info(f"Proceeding with limited data: {len(df)} records")
            
            # Process and clean the data - skip for formatted files to avoid duplicating efforts
            if is_formatted:
                # For pre-formatted files, do minimal processing
                metadata = {
                    "formatted": True,
                    "file_name": file_name,
                    "file_type": normalized_ext,
                    "original_columns": list(df.columns),
                    "row_count": len(df)
                }
            else:
                # Full preprocessing for non-formatted files
                df, metadata = self._preprocess_data(df)
            
            logger.info(f"Successfully loaded data with {len(df)} incidents")
            return df, metadata
        
        except Exception as e:
            logger.error(f"Comprehensive error loading data: {str(e)}", exc_info=True)
            
            # Collect and log all potential error details
            error_details = {
                "file_name": file_name,
                "file_type": normalized_ext,
                "load_errors": load_errors,
                "exception": str(e)
            }
            
            raise ValueError(f"Error loading data: {error_details}")
    
    def load_from_path(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file path with enhanced error recovery.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with incident data or None if loading fails
        """
        try:
            # Validate file existence
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Get filename
            file_name = os.path.basename(file_path)
            
            # Use the comprehensive load_data method
            try:
                df, _ = self.load_data(file_content, file_name)
                return df
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                
                # Check if error is related to formatting
                if "format" in str(e).lower() or "parse" in str(e).lower() or "conversion" in str(e).lower():
                    logger.info(f"Attempting to format file {file_path} before loading")
                    
                    try:
                        # Format the file
                        from data.data_formatter import DataFormatter
                        formatter = DataFormatter()
                        
                        # Format in place
                        formatted_path = f"{file_path}.formatted"
                        format_metadata = formatter.format_file(file_path, formatted_path)
                        
                        if format_metadata.get('success', False):
                            # Try to load the formatted file
                            with open(formatted_path, 'rb') as f:
                                formatted_content = f.read()
                            
                            try:
                                df, _ = self.load_data(formatted_content, f"{file_name}_formatted")
                                return df
                            except Exception as load_err:
                                logger.error(f"Error loading formatted file: {str(load_err)}")
                                # Last resort - try with pandas directly
                                try:
                                    return pd.read_csv(formatted_path)
                                except:
                                    pass
                    except Exception as format_err:
                        logger.error(f"Error during format recovery: {str(format_err)}", exc_info=True)
    
        except ValueError as ve:
            # Log specific validation errors
            logger.error(f"Validation error loading file: {str(ve)}")
            return None
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error loading file: {str(e)}", exc_info=True)
            return None
            
        return None  # Explicit return None if all attempts fail
        


    def load_processed_data(self) -> pd.DataFrame:
        """
        Returns the processed data from session state if available.
        This method is used by various components that expect to get processed data from the loader.
        
        Returns:
            DataFrame with processed incident data or None if not available
        """
        import streamlit as st
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            return st.session_state.processed_data
        elif 'raw_data' in st.session_state and st.session_state.raw_data is not None:
            # Return raw data as fallback
            return st.session_state.raw_data
        else:
            logger.warning("No processed data available in session state")
            return None
    
 
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and preprocess incident data.
        
        Args:
            df: DataFrame with raw incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata about preprocessing
        """
        original_row_count = len(df)
        original_col_count = len(df.columns)
        
        # Track preprocessing steps for metadata
        preprocessing_steps = []
        
        # 1. Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        preprocessing_steps.append("Standardized column names")
        
        # 2. Map columns to standard names based on content
        df, column_mapping = self._map_columns_to_standard(df)
        preprocessing_steps.append("Mapped columns to standard names")
        
        # 3. Remove duplicate incidents
        original_dupes = 0
        if 'incident_id' in df.columns:
            original_dupes = df.duplicated(subset=['incident_id']).sum()
            if original_dupes > 0:
                df = df.drop_duplicates(subset=['incident_id'])
                preprocessing_steps.append(f"Removed {original_dupes} duplicate incidents")
        
        # 4. Convert date columns to datetime
        date_columns = []
        for col in df.columns:
            col_str = str(col).lower()
            if 'date' in col_str or 'time' in col_str:
                date_columns.append(col)
                    
        for col in date_columns:
            try:
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    preprocessing_steps.append(f"Converted {col} to datetime")
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        # 5. Ensure priority is standardized if exists
        if 'priority' in df.columns:
            df['priority'] = df['priority'].astype(str).str.lower()
            
            # Map various priority terms to standard values
            priority_mapping = {
                'p1': 'critical',
                'p2': 'high',
                'p3': 'medium',
                'p4': 'low',
                'p5': 'low',
                '1': 'critical',
                '2': 'high',
                '3': 'medium',
                '4': 'low',
                '5': 'low',
                'critical': 'critical',
                'high': 'high',
                'medium': 'medium',
                'normal': 'medium',
                'low': 'low'
            }
            
            # Fixed: Use explicit string operations
            def map_priority(x):
                x_str = str(x).lower()
                for k, v in priority_mapping.items():
                    if k in x_str:
                        return v
                return x
            
            df['priority'] = df['priority'].apply(map_priority)
            preprocessing_steps.append("Standardized priority values")
        
        # 6. Calculate derived metrics if possible
        derived_columns = []
        
        # Resolution time if both dates exist
        if 'created_date' in df.columns and 'resolved_date' in df.columns:
            try:
                df['resolution_time_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
                
                # Filter valid values
                def filter_valid_hours(x):
                    if pd.isnull(x):
                        return None
                    if x < 0 or x > 8760:  # Negative or > 1 year
                        return None
                    return x
                
                df['resolution_time_hours'] = df['resolution_time_hours'].apply(filter_valid_hours)
                derived_columns.append('resolution_time_hours')
                preprocessing_steps.append("Calculated resolution time in hours")
            except Exception as e:
                logger.warning(f"Could not calculate resolution time: {str(e)}")
        
        # 7. Remove rows with null values in critical columns
        null_counts_before = df.isnull().sum()
        critical_columns = []
        for col in self.mandatory_columns:
            if col in df.columns:
                critical_columns.append(col)
                
        if critical_columns:
            original_null_count = df[critical_columns].isnull().any(axis=1).sum()
            if original_null_count > 0:
                df = df.dropna(subset=critical_columns)
                preprocessing_steps.append(f"Removed {original_null_count} rows with missing values in critical columns")
        
        # 8. Limit to maximum sample size if needed
        if len(df) > self.max_sample_size:
            df = df.sample(n=self.max_sample_size, random_state=42)
            preprocessing_steps.append(f"Sampled {self.max_sample_size} incidents from {len(df)} total")
        
        # Compile metadata
        start_date = None
        end_date = None
        
        # FIX: Properly check if created_date column exists and isn't all null
        date_range_info = {"start": None, "end": None}
        if 'created_date' in df.columns:
            try:
                # Check for non-null values
                valid_dates = df['created_date'].dropna()
                if len(valid_dates) > 0:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    if pd.notnull(min_date):
                        date_range_info["start"] = min_date.strftime('%Y-%m-%d')
                    if pd.notnull(max_date):
                        date_range_info["end"] = max_date.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Error extracting date range: {str(e)}")
        
        # Fix: Use the corrected _determine_available_analyses method
        available_analyses = {}
        try:
            available_analyses = self._determine_available_analyses(df)
        except Exception as e:
            logger.warning(f"Error determining available analyses: {str(e)}")
            # Provide fallback default analyses
            available_analyses = {
                "time_series": 'created_date' in df.columns,
                "priority_distribution": 'priority' in df.columns,
                "status_distribution": 'status' in df.columns,
                "category_analysis": 'category' in df.columns,
                "resolution_time_analysis": 'resolution_time_hours' in df.columns or ('created_date' in df.columns and 'resolved_date' in df.columns),
                "assignee_analysis": 'assignee' in df.columns,
                "root_cause_analysis": 'description' in df.columns,
                "predictive_analysis": 'created_date' in df.columns and ('priority' in df.columns or 'category' in df.columns or 'status' in df.columns)
            }
            
        metadata = {
            "original_row_count": original_row_count,
            "processed_row_count": len(df),
            "original_column_count": original_col_count,
            "processed_column_count": len(df.columns),
            "column_mapping": column_mapping,
            "preprocessing_steps": preprocessing_steps,
            "derived_columns": derived_columns,
            "null_value_counts": {col: int(val) for col, val in df.isnull().sum().items()},  # Convert to int for serialization
            "data_summary": {
                "incident_count": len(df),
                "date_range": date_range_info,
                "available_analyses": available_analyses
            }
        }
        
        return df, metadata


    
    def _clean_column_name(self, column_name: Any) -> str:
        """
        Clean and standardize a column name.
        
        Args:
            column_name: Original column name
            
        Returns:
            Cleaned column name
        """
        if not isinstance(column_name, str):
            column_name = str(column_name)
        
        # Convert to lowercase and replace spaces and special chars with underscores
        cleaned = column_name.lower().strip()
        cleaned = ''.join(c if c.isalnum() else '_' for c in cleaned)
        
        # Replace multiple underscores with a single one
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def _map_columns_to_standard(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map columns to standard names based on content and naming patterns.
        
        Args:
            df: DataFrame with raw column names
            
        Returns:
            Tuple containing:
                - DataFrame with standardized column names
                - Dictionary mapping original to standardized names
        """
        # Define common patterns for each standard column
        column_patterns = {
            'incident_id': ['incident_id', 'ticket_id', 'id', 'number', 'reference', 'case'],
            'created_date': ['created', 'open', 'opened', 'raised', 'reported', 'start', 'logged'],
            'resolved_date': ['resolved', 'closed', 'completed', 'fixed', 'end', 'resolution'],
            'priority': ['priority', 'severity', 'urgency', 'importance'],
            'status': ['status', 'state', 'condition'],
            'category': ['category', 'type', 'class', 'group'],
            'subcategory': ['subcategory', 'subtype', 'subclass'],
            'source': ['source', 'origin', 'reported_by', 'channel'],
            'assignee': ['assignee', 'assigned_to', 'owner', 'resolver', 'handler'],
            'assignment_group': ['assignment_group', 'team', 'squad', 'group', 'department'],
            'affected_system': ['affected_system', 'system', 'application', 'service', 'component'],
            'impact': ['impact', 'effect', 'consequence'],
            'description': ['description', 'details', 'summary', 'problem'],
            'resolution_notes': ['resolution', 'solution', 'fix', 'workaround'],
        }
        
        # Track the mapping
        column_mapping = {}
        standardized_columns = set()
        
        # First pass: direct matches
        for col in df.columns:
            cleaned_col = self._clean_column_name(col)
            for std_col, patterns in column_patterns.items():
                # Fixed: direct string equality check
                if cleaned_col == std_col:
                    if std_col not in standardized_columns:
                        column_mapping[col] = std_col
                        standardized_columns.add(std_col)
                        break
                        
                # Check against patterns
                match_found = False
                for pattern in patterns:
                    if cleaned_col == pattern:
                        match_found = True
                        break
                        
                if match_found and std_col not in standardized_columns:
                    column_mapping[col] = std_col
                    standardized_columns.add(std_col)
                    break
        
        # Second pass: pattern matches within column name
        for col in df.columns:
            if col in column_mapping:
                continue  # Skip already mapped columns
                
            cleaned_col = self._clean_column_name(col)
            for std_col, patterns in column_patterns.items():
                if std_col in standardized_columns:
                    continue  # Skip already mapped standard columns
                
                # Fixed: Individual string containment checks
                match_found = False
                for pattern in patterns:
                    if pattern in cleaned_col:
                        match_found = True
                        break
                        
                if match_found:
                    column_mapping[col] = std_col
                    standardized_columns.add(std_col)
                    break
        
        # Apply the mapping to the DataFrame
        df_new = df.copy()
        for orig_col, new_col in column_mapping.items():
            df_new = df_new.rename(columns={orig_col: new_col})
        
        # Keep unmapped columns with their cleaned names
        for col in df.columns:
            if col not in column_mapping:
                cleaned_col = self._clean_column_name(col)
                df_new = df_new.rename(columns={col: cleaned_col})
                column_mapping[col] = cleaned_col
        
        return df_new, column_mapping
        
    def _determine_available_analyses(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Determine which analyses are possible with the given data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of analysis types and whether they're available
        """
        analyses = {
            "time_series": False,
            "priority_distribution": False,
            "status_distribution": False,
            "category_analysis": False,
            "resolution_time_analysis": False,
            "assignee_analysis": False,
            "root_cause_analysis": False,
            "predictive_analysis": False
        }
        
        # Time series analysis needs created_date
        if 'created_date' in df.columns and df['created_date'].notna().any():
            analyses["time_series"] = True
            
        # Priority distribution
        if 'priority' in df.columns and df['priority'].notna().any():
            analyses["priority_distribution"] = True
            
        # Status distribution
        if 'status' in df.columns and df['status'].notna().any():
            analyses["status_distribution"] = True
            
        # Category analysis
        if 'category' in df.columns and df['category'].notna().any():
            analyses["category_analysis"] = True
            
        # Resolution time analysis
        if ('resolution_time_hours' in df.columns and df['resolution_time_hours'].notna().any()) or \
        ('created_date' in df.columns and 'resolved_date' in df.columns and 
            df['created_date'].notna().any() and df['resolved_date'].notna().any()):
            analyses["resolution_time_analysis"] = True
            
        # Assignee analysis
        if 'assignee' in df.columns and df['assignee'].notna().any():
            analyses["assignee_analysis"] = True
            
        # Root cause analysis
        if 'description' in df.columns and df['description'].notna().any():
            analyses["root_cause_analysis"] = True
            
        # Predictive analysis (needs at least time series + one other dimension)
        predictive_requirements = [
            analyses["priority_distribution"], 
            analyses["category_analysis"],
            analyses["status_distribution"]
        ]
        if analyses["time_series"] and any(predictive_requirements):
            analyses["predictive_analysis"] = True
            
        return analyses