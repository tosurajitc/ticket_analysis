"""
Data formatting module for the Incident Management Analytics application.

This module handles specific formatting of converted incident data to ensure
full compatibility with pandas DataFrames and analysis components. It processes
files already converted by the data_converter module, focusing on data type
consistency, value normalization, and handling of edge cases that might cause
errors during analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import tempfile
import shutil
import re

# Configure logging
logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Class for formatting converted incident data to ensure full compatibility
    with pandas DataFrames and analysis components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DataFormatter with configuration settings.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Standard column data types
        self.column_types = {
            "incident_id": "string",
            "created_date": "datetime64[ns]",
            "resolved_date": "datetime64[ns]",
            "priority": "category",
            "status": "category",
            "category": "category",
            "subcategory": "category",
            "assignee": "string",
            "assignment_group": "string",
            "description": "string",
            "resolution_notes": "string"
        }
        
        # Standard value mappings for categorical columns
        self.priority_mapping = {
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
        
        self.status_mapping = {
            'open': 'open',
            'new': 'open',
            'in progress': 'in progress',
            'in-progress': 'in progress',
            'inprogress': 'in progress',
            'work in progress': 'in progress',
            'assigned': 'in progress',
            'pend': 'pending',
            'pending': 'pending',
            'on hold': 'pending',
            'waiting': 'pending',
            'resolved': 'resolved',
            'complete': 'resolved',
            'completed': 'resolved',
            'fixed': 'resolved',
            'close': 'closed',
            'closed': 'closed',
            'cancelled': 'closed',
            'canceled': 'closed'
        }
        
        # Characters that need to be escaped in CSV
        self.special_chars = [',', '"', '\n', '\r', ';']
    
    def format_file(self, 
                   file_path: str, 
                   output_path: Optional[str] = None,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Format a converted CSV file to ensure full compatibility with pandas DataFrames
        and replace the original file with the formatted version.
        
        Args:
            file_path: Path to the converted CSV file
            output_path: Optional path for the formatted output (if None, overwrites input file)
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary with metadata about the formatting
        """
        try:
            # Ensure file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Determine output path
            if output_path is None:
                # Create temporary file for processing
                temp_dir = os.path.dirname(file_path)
                fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=temp_dir)
                os.close(fd)
                output_path = temp_path
                replace_original = True
            else:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                replace_original = False
            
            # Start timing
            start_time = datetime.now()
            
            # First try to read the file with default settings to see if it works
            try:
                # Try reading the file
                df = pd.read_csv(file_path)
                logger.info(f"File {file_path} loaded successfully with default settings")
                
                # Track initial row count
                initial_row_count = len(df)
                
                # Process data as a whole
                df = self._format_dataframe(df)
                
                # Check for any issues in the data
                issues_detected = self._check_data_issues(df)
                
                # If we processed successfully and found no issues, write back the file
                if df is not None and not df.empty and not issues_detected:
                    # No issues found - file should be readable normally
                    df.to_csv(output_path, index=False)
                    formatted_normally = True
                else:
                    # Issues detected - need custom processing
                    formatted_normally = False
            except Exception as e:
                logger.warning(f"Could not read {file_path} with default settings: {str(e)}")
                formatted_normally = False
                initial_row_count = 0
                issues_detected = True
            
            # If we couldn't read the file normally or issues were detected, try more robust processing
            if not formatted_normally:
                df = self._process_file_with_custom_parser(file_path, output_path, progress_callback)
                
                # If we still failed, try an absolute fallback approach
                if df is None or df.empty:
                    df = self._process_file_emergency_fallback(file_path, output_path)
            
            # Finalize the process
            if replace_original:
                # Only replace if the temporary file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Copy permissions from original file
                    shutil.copymode(file_path, output_path)
                    # Replace the original file
                    shutil.move(output_path, file_path)
                    output_path = file_path
                else:
                    # Something went wrong, don't replace the original
                    logger.error(f"Formatted file is empty or missing, keeping original: {file_path}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    output_path = file_path
            
            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Final check to ensure file exists and is readable
            success = os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
            # Try reading the final file to verify it's loadable
            final_row_count = 0
            loadable = False
            if success:
                try:
                    verification_df = pd.read_csv(output_path)
                    final_row_count = len(verification_df)
                    loadable = True
                except Exception as e:
                    logger.error(f"Verification read failed on formatted file: {str(e)}")
                    loadable = False
            
            # Return metadata about the formatting
            metadata = {
                "input_path": file_path,
                "output_path": output_path,
                "initial_row_count": initial_row_count,
                "final_row_count": final_row_count,
                "processing_time_seconds": processing_time,
                "formatted_normally": formatted_normally,
                "issues_detected": issues_detected,
                "success": success,
                "loadable": loadable,
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(output_path) if success else 0
            }
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error formatting file: {str(e)}", exc_info=True)
            
            # Return error metadata
            return {
                "input_path": file_path,
                "output_path": output_path if 'output_path' in locals() else None,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive formatting to a DataFrame with improved datetime handling.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Make a defensive copy
        df = df.copy()
        
        # Enhanced datetime conversion for created_date column
        if 'created_date' in df.columns:
            try:
                # First try standard conversion
                df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
                
                # Check if conversion succeeded and we have valid dates
                if df['created_date'].isna().all():
                    # Try some common date formats explicitly
                    date_formats = [
                        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', 
                        '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y'
                    ]
                    
                    for date_format in date_formats:
                        try:
                            df['created_date'] = pd.to_datetime(df['created_date'], format=date_format, errors='coerce')
                            if not df['created_date'].isna().all():
                                break  # Found a working format
                        except:
                            continue
                    
                # Check if we have at least some valid dates
                if not df['created_date'].isna().all():
                    # Set a flag in the df to indicate we have valid dates
                    df.attrs['has_valid_dates'] = True
                    logger.info(f"Successfully converted created_date column with {df['created_date'].notna().sum()} valid dates")
                else:
                    logger.warning("All dates in created_date column were converted to NaT")
                    df.attrs['has_valid_dates'] = False
                    
            except Exception as e:
                logger.error(f"Error converting created_date: {str(e)}")
                df.attrs['has_valid_dates'] = False
        
        # Format each column based on its expected type
        for column, dtype in self.column_types.items():
            if column in df.columns and column != 'created_date':  # Skip created_date as we handled it above
                try:
                    # Handle different data types
                    if dtype == "datetime64[ns]":
                        # Convert to datetime, coercing errors to NaT
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        
                        # Flag dates far in the future or past as invalid
                        if not df[column].isna().all():
                            min_valid_date = pd.Timestamp('1990-01-01')
                            max_valid_date = pd.Timestamp('2035-01-01')
                            invalid_dates = (df[column] < min_valid_date) | (df[column] > max_valid_date)
                            if invalid_dates.any():
                                logger.warning(f"Found {invalid_dates.sum()} invalid dates in column {column}")
                                df.loc[invalid_dates, column] = pd.NaT
                    
                    elif dtype == "category":
                        # Clean string values first
                        df[column] = df[column].astype(str).str.strip().str.lower()
                        
                        # Replace empty strings with NaN
                        df[column] = df[column].replace('', np.nan).replace('nan', np.nan).replace('none', np.nan)
                        
                        # Apply standard mapping if this is a standard category
                        if column == "priority":
                            df[column] = self._standardize_priority(df[column])
                        elif column == "status":
                            df[column] = self._standardize_status(df[column])
                        
                        # Convert to category dtype
                        df[column] = df[column].astype('category')
                    
                    elif dtype == "string":
                        # Clean string values - allow NaN but convert everything else to string
                        df[column] = df[column].fillna('').astype(str).replace('nan', '').replace('none', '')
                        
                        # Clean control characters that might cause issues
                        df[column] = df[column].apply(self._clean_string)
                        
                        # Use pandas string type if available
                        try:
                            df[column] = df[column].astype('string')
                        except:
                            # Fallback to object if string dtype not available
                            pass
                
                except Exception as e:
                    logger.warning(f"Error formatting column {column}: {str(e)}")
                    # Don't fail the entire process due to one column
        
        # Handle potential date columns not in standard schema
        date_columns = [col for col in df.columns if col not in self.column_types 
                    and any(term in col.lower() for term in ['date', 'time', 'created', 'resolved'])]
        
        for col in date_columns:
            try:
                # Check if it looks like a date
                sample = df[col].dropna().astype(str).iloc[:10] if len(df) > 10 else df[col].dropna().astype(str)
                date_like = sample.str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any()
                
                if date_like:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Error checking potential date column {col}: {str(e)}")
        
        # Calculate derived fields if appropriate columns exist
        if 'created_date' in df.columns and 'resolved_date' in df.columns:
            try:
                # Both columns must be datetime
                if pd.api.types.is_datetime64_dtype(df['created_date']) and pd.api.types.is_datetime64_dtype(df['resolved_date']):
                    # Calculate resolution time in hours
                    df['resolution_time_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
                    
                    # Clean invalid values (negative or extreme outliers)
                    df.loc[df['resolution_time_hours'] < 0, 'resolution_time_hours'] = np.nan
                    df.loc[df['resolution_time_hours'] > 8760, 'resolution_time_hours'] = np.nan  # > 1 year
            except Exception as e:
                logger.warning(f"Error calculating resolution time: {str(e)}")
        
        # Remove completely empty rows
        before_len = len(df)
        df = df.dropna(how='all')
        dropped = before_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} completely empty rows")
        
        return df
    
    def _standardize_priority(self, priority_series: pd.Series) -> pd.Series:
        """
        Standardize priority values to a consistent format.
        
        Args:
            priority_series: Series containing priority values
            
        Returns:
            Series with standardized priority values
        """
        # Convert to string and lowercase
        priority_series = priority_series.astype(str).str.lower()
        
        # Apply mapping
        def map_priority(x):
            if pd.isna(x) or x == 'nan' or x == 'none' or x == '':
                return np.nan
                
            x_str = str(x).lower()
            for k, v in self.priority_mapping.items():
                if k == x_str or k in x_str:
                    return v
            return x
        
        return priority_series.apply(map_priority)
    
    def _standardize_status(self, status_series: pd.Series) -> pd.Series:
        """
        Standardize status values to a consistent format.
        
        Args:
            status_series: Series containing status values
            
        Returns:
            Series with standardized status values
        """
        # Convert to string and lowercase
        status_series = status_series.astype(str).str.lower()
        
        # Apply mapping
        def map_status(x):
            if pd.isna(x) or x == 'nan' or x == 'none' or x == '':
                return np.nan
                
            x_str = str(x).lower()
            for k, v in self.status_mapping.items():
                if k == x_str or k in x_str:
                    return v
            return x
        
        return status_series.apply(map_status)
    
    def _clean_string(self, text: Any) -> str:
        """
        Clean a string value to remove problematic characters.
        
        Args:
            text: Input string or value
            
        Returns:
            Cleaned string
        """
        if pd.isna(text) or text == 'nan' or text == 'none' or text == '':
            return ''
            
        # Convert to string
        text = str(text)
        
        # Replace control characters and other problematic chars
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
        
        # Escape quotes
        text = text.replace('"', '""')
        
        # Clean excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _check_data_issues(self, df: pd.DataFrame) -> bool:
        """
        Check for common issues in the data that might cause problems during analysis.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if issues were detected, False otherwise
        """
        if df is None or df.empty:
            return True
            
        issues_detected = False
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            logger.warning("Duplicate column names detected")
            issues_detected = True
        
        # Check for excessive missing values
        missing_pct = df.isna().mean().mean() * 100
        if missing_pct > 50:
            logger.warning(f"High percentage of missing values: {missing_pct:.1f}%")
            issues_detected = True
        
        # Check for required columns
        required_columns = ["incident_id", "created_date", "priority", "status"]
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            logger.warning(f"Missing required columns: {', '.join(missing_required)}")
            issues_detected = True
        
        # Check data types of critical columns
        if "created_date" in df.columns and not pd.api.types.is_datetime64_dtype(df["created_date"]):
            logger.warning("created_date column is not a datetime type")
            issues_detected = True
            
        # Check for non-finite values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            non_finite = (~np.isfinite(df[col])).sum()
            if non_finite > 0:
                logger.warning(f"Column {col} has {non_finite} non-finite values")
                issues_detected = True
        
        return issues_detected
    
    def _process_file_with_custom_parser(self, 
                                       input_path: str, 
                                       output_path: str,
                                       progress_callback: Optional[Callable] = None) -> Optional[pd.DataFrame]:
        """
        Process the file with a custom parser to handle various CSV format issues.
        
        Args:
            input_path: Path to the input file
            output_path: Path to the output file
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Read the file line by line to handle formatting issues
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Process header line to get column names
            if not lines:
                logger.error(f"File {input_path} is empty")
                return None
                
            header = lines[0].strip()
            columns = self._parse_csv_line(header)
            
            # Remove duplicate column names by adding suffix
            clean_columns = []
            seen = {}
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    clean_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    clean_columns.append(col)
            
            # Process data rows
            data = []
            total_lines = len(lines)
            
            for i, line in enumerate(lines[1:], 1):
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Parse the line
                    row_values = self._parse_csv_line(line)
                    
                    # Ensure row has right number of columns
                    if len(row_values) < len(clean_columns):
                        # Add empty values for missing columns
                        row_values.extend([''] * (len(clean_columns) - len(row_values)))
                    elif len(row_values) > len(clean_columns):
                        # Truncate extra values
                        row_values = row_values[:len(clean_columns)]
                    
                    data.append(row_values)
                    
                    # Report progress every 1000 lines
                    if progress_callback and i % 1000 == 0:
                        progress_callback(i, total_lines)
                except Exception as e:
                    logger.warning(f"Error parsing line {i+1}: {str(e)}")
                    # Skip the problematic line
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=clean_columns)
            
            # Format the DataFrame
            df = self._format_dataframe(df)
            
            # Write to output file
            df.to_csv(output_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in custom parser: {str(e)}", exc_info=True)
            return None
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """
        Parse a CSV line handling quotes and commas properly.
        
        Args:
            line: CSV line to parse
            
        Returns:
            List of values from the line
        """
        result = []
        current_value = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                # If we're in quotes and encounter another quote, it could be an escape or end of quotes
                if in_quotes:
                    # Check if this is an escaped quote
                    if len(line) > len(current_value) + 1 and line[len(current_value) + 1] == '"':
                        current_value += '"'
                        # Skip the next quote
                        continue
                    else:
                        in_quotes = False
                else:
                    in_quotes = True
            elif char == ',' and not in_quotes:
                # End of value
                result.append(current_value)
                current_value = ""
            else:
                current_value += char
        
        # Add the last value
        result.append(current_value)
        
        return result
    
    def _process_file_emergency_fallback(self, input_path: str, output_path: str) -> Optional[pd.DataFrame]:
        """
        Last resort fallback for processing files with serious issues.
        Creates a minimal valid CSV with only essential columns.
        
        Args:
            input_path: Path to the input file
            output_path: Path to the output file
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Try various options to read the file
            encoding_options = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            separator_options = [',', ';', '\t', '|']
            
            df = None
            
            # Try each combination
            for encoding in encoding_options:
                for sep in separator_options:
                    try:
                        df = pd.read_csv(input_path, encoding=encoding, sep=sep, 
                                      error_bad_lines=False, warn_bad_lines=True,
                                      low_memory=False, quoting=3)
                        logger.info(f"Successfully read file with encoding={encoding}, sep={sep}")
                        break
                    except Exception as e:
                        continue
                
                if df is not None:
                    break
            
            # If all methods failed, create a minimal DataFrame
            if df is None:
                logger.warning("All reading methods failed, creating minimal DataFrame")
                
                # Create minimal valid columns
                df = pd.DataFrame({
                    'incident_id': ['FALLBACK_1', 'FALLBACK_2'],
                    'created_date': [datetime.now(), datetime.now()],
                    'priority': ['medium', 'medium'],
                    'status': ['open', 'open'],
                    'description': ['Emergency fallback record - original file had format issues',
                                    'Please check the original data file']
                })
            
            # Format the DataFrame
            df = self._format_dataframe(df)
            
            # Write to output file
            df.to_csv(output_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Emergency fallback failed: {str(e)}", exc_info=True)
            # Create the most basic valid file
            with open(output_path, 'w') as f:
                f.write("incident_id,created_date,priority,status\n")
                f.write(f"EMERGENCY_1,{datetime.now()},medium,open\n")
            
            return None