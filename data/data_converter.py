"""
Data conversion module for the Incident Management Analytics application.

This module handles conversion of raw incident data into a standardized format
with proper column mapping and data transformation. It supports chunked processing
for large files to ensure memory efficiency.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import json
from pathlib import Path
import io
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.pages.landing_page import standardize_priority, standardize_status

# Configure logging
logger = logging.getLogger(__name__)

class DataConverter:
    """
    Class for converting raw incident data into a standardized format.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DataConverter with configuration settings.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 10000)
        self.max_sample_rows = self.config.get("max_sample_rows", 1000)
        
        # Standard schema definition
        self.standard_schema = {
            "incident_id": {
                "description": "Unique identifier for each ticket",
                "required": True,
                "type": "string"
            },
            "created_date": {
                "description": "When the incident was created/reported",
                "required": True,
                "type": "datetime"
            },
            "priority": {
                "description": "Incident priority/severity level",
                "required": True,
                "type": "category"
            },
            "status": {
                "description": "Current status of the incident",
                "required": True,
                "type": "category"
            },
            "category": {
                "description": "Incident classification/type",
                "required": False,
                "type": "category"
            },
            "resolved_date": {
                "description": "When the incident was resolved",
                "required": False,
                "type": "datetime"
            },
            "assignee": {
                "description": "Person assigned to the incident",
                "required": False,
                "type": "string"
            },
            "assignment_group": {
                "description": "Team responsible for the incident",
                "required": False,
                "type": "string"
            },
            "description": {
                "description": "Description of the incident",
                "required": False,
                "type": "text"
            },
            "resolution_notes": {
                "description": "How the incident was resolved",
                "required": False,
                "type": "text"
            }
        }
        
        # Initialize standard mappings for common values
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
    
    def load_file_sample(self, file_path: str, file_obj=None) -> Tuple[pd.DataFrame, str, int]:
        """
        Load a sample of rows from the input file to preview and detect structure.
        
        Args:
            file_path: Path to the input file or filename if using file_obj
            file_obj: File object if provided (for uploaded files)
            
        Returns:
            Tuple of (sample_df, file_type, estimated_total_rows)
        """
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            # Load based on file type
            if file_extension == 'csv':
                if file_obj:
                    # Read from file object
                    sample_df = pd.read_csv(file_obj, nrows=self.max_sample_rows)
                    # Reset position for future reads
                    file_obj.seek(0)
                    # Estimate total rows
                    line_count = sum(1 for _ in file_obj)
                    estimated_total_rows = line_count - 1  # Subtract header
                    file_obj.seek(0)  # Reset again
                else:
                    # Read from file path
                    sample_df = pd.read_csv(file_path, nrows=self.max_sample_rows)
                    # Estimate total rows using file size and average row size
                    with open(file_path, 'r') as f:
                        # Read first 100 lines to get average row size
                        sample_lines = []
                        for _ in range(100):
                            try:
                                sample_lines.append(next(f))
                            except StopIteration:
                                break
                                
                        if sample_lines:
                            avg_row_size = sum(len(line) for line in sample_lines) / len(sample_lines)
                            file_size = os.path.getsize(file_path)
                            estimated_total_rows = int(file_size / avg_row_size)
                        else:
                            estimated_total_rows = 0
                
                file_type = 'csv'
                
            elif file_extension in ['xlsx', 'xls']:
                # Try different engines for more robust Excel reading
                engines_to_try = ['openpyxl', 'xlrd']
                
                if file_obj:
                    # Read entire file into memory first
                    file_content = file_obj.read()
                    file_obj.seek(0)  # Reset file pointer for later use
                    
                    # Try each engine
                    for engine in engines_to_try:
                        try:
                            sample_df = pd.read_excel(
                                io.BytesIO(file_content), 
                                engine=engine, 
                                nrows=self.max_sample_rows
                            )
                            break  # If successful, stop trying engines
                        except Exception as e:
                            logger.warning(f"Engine {engine} failed: {str(e)}")
                            if engine == engines_to_try[-1]:
                                # Last engine, re-raise the exception
                                raise
                    
                    # No good way to estimate total rows without reading entire file
                    # just use the sample size as an indication
                    estimated_total_rows = len(sample_df)
                else:
                    # Try each engine
                    for engine in engines_to_try:
                        try:
                            sample_df = pd.read_excel(
                                file_path, 
                                engine=engine, 
                                nrows=self.max_sample_rows
                            )
                            break  # If successful, stop trying engines
                        except Exception as e:
                            logger.warning(f"Engine {engine} failed: {str(e)}")
                            if engine == engines_to_try[-1]:
                                # Last engine, re-raise the exception
                                raise
                    
                    # Try to get sheet dimensions from Excel file
                    try:
                        import openpyxl
                        wb = openpyxl.load_workbook(file_path, read_only=True)
                        sheet = wb.active
                        # Estimate based on max row or use a default
                        estimated_total_rows = sheet.max_row - 1  # Subtract header
                    except Exception as e:
                        # If openpyxl fails, just use the sample size
                        estimated_total_rows = len(sample_df)
                        logger.warning(f"Could not estimate total rows in Excel file: {str(e)}")
                
                file_type = 'excel'
                
            elif file_extension == 'json':
                if file_obj:
                    sample_df = pd.read_json(file_obj, lines=True, nrows=self.max_sample_rows)
                    # Reset position
                    file_obj.seek(0)
                    # Count lines for JSON Lines format
                    line_count = sum(1 for _ in file_obj)
                    estimated_total_rows = line_count
                    file_obj.seek(0)
                else:
                    sample_df = pd.read_json(file_path, lines=True, nrows=self.max_sample_rows)
                    # Estimate total rows by counting lines
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    estimated_total_rows = line_count
                
                file_type = 'json'
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Loaded sample with {len(sample_df)} rows, estimated total: {estimated_total_rows}")
            return sample_df, file_type, estimated_total_rows
            
        except Exception as e:
            logger.error(f"Error loading file sample: {str(e)}")
            raise
    
    def infer_column_mappings(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer suggested column mappings based on column names and content.
        
        Args:
            df: Sample DataFrame with source columns
            
        Returns:
            Dictionary mapping target columns to source columns
        """
        source_columns = list(df.columns)
        inferred_mappings = {}
        
        # Pattern-based mapping
        column_patterns = {
            'incident_id': ['incident', 'ticket', 'id', 'number', 'key', 'case', 'reference'],
            'created_date': ['created', 'open', 'opened', 'raised', 'reported', 'start', 'logged', 'date'],
            'resolved_date': ['resolved', 'closed', 'completed', 'fixed', 'end', 'resolution', 'finish'],
            'priority': ['priority', 'severity', 'urgency', 'importance', 'criticality'],
            'status': ['status', 'state', 'condition', 'lifecycle'],
            'category': ['category', 'type', 'class', 'group', 'classification'],
            'subcategory': ['subcategory', 'subtype', 'subclass'],
            'assignee': ['assignee', 'assigned_to', 'owner', 'resolver', 'handler', 'technician'],
            'assignment_group': ['assignment_group', 'team', 'squad', 'group', 'department', 'support_group'],
            'description': ['description', 'details', 'summary', 'problem', 'issue', 'synopsis'],
            'resolution_notes': ['resolution', 'solution', 'fix', 'workaround', 'resolution_details', 'action_taken']
        }
        
        # First attempt exact matches with column names
        for target_col, patterns in column_patterns.items():
            for source_col in source_columns:
                source_col_lower = source_col.lower()
                
                # Check for exact matches first
                if source_col_lower in patterns or source_col_lower == target_col:
                    inferred_mappings[target_col] = source_col
                    break
        
        # If we don't have all required columns, try partial matches
        required_cols = [col for col, details in self.standard_schema.items() if details.get("required", False)]
        missing_required = [col for col in required_cols if col not in inferred_mappings]
        
        if missing_required:
            for target_col in missing_required:
                patterns = column_patterns.get(target_col, [])
                
                for source_col in source_columns:
                    if source_col in inferred_mappings.values():
                        continue  # Skip already mapped columns
                        
                    source_col_lower = source_col.lower()
                    
                    # Look for partial matches
                    for pattern in patterns:
                        if pattern in source_col_lower or any(token in source_col_lower for token in pattern.split('_')):
                            inferred_mappings[target_col] = source_col
                            break
                    
                    if target_col in inferred_mappings:
                        break
        
        # If still missing required columns, try content-based inference
        missing_required = [col for col in required_cols if col not in inferred_mappings]
        
        if missing_required and len(df) > 0:
            # Try to infer based on column content
            for target_col in missing_required:
                if target_col == 'incident_id':
                    # Look for a column that contains unique identifiers
                    for source_col in source_columns:
                        if source_col in inferred_mappings.values():
                            continue
                            
                        # Check if column has mostly unique values and contains numbers or IDs
                        if len(df) > 0:
                            unique_ratio = df[source_col].nunique() / len(df)
                        else:
                            unique_ratio = 0
                        is_id_like = False
                        
                        # Check sample values
                        try:
                            sample_vals = df[source_col].dropna().astype(str).head(10).tolist()
                            if sample_vals and all(any(c.isdigit() for c in val) for val in sample_vals):
                                is_id_like = True
                        except:
                            continue
                            
                        if unique_ratio > 0.9 and is_id_like:
                            inferred_mappings[target_col] = source_col
                            break
                
                elif target_col == 'created_date':
                    # Look for a column that contains dates
                    for source_col in source_columns:
                        if source_col in inferred_mappings.values():
                            continue
                            
                        # Try to convert to datetime
                        try:
                            # Check percentage of successful date conversions
                            dates = pd.to_datetime(df[source_col], errors='coerce')
                            if not dates.isna().all() and dates.notna().mean() > 0.8:
                                inferred_mappings[target_col] = source_col
                                break
                        except:
                            continue
                
                elif target_col == 'priority':
                    # Look for a column with priority-like values
                    for source_col in source_columns:
                        if source_col in inferred_mappings.values():
                            continue
                            
                        # Get unique values as strings
                        try:
                            unique_vals = df[source_col].dropna().astype(str).str.lower().unique()
                        except:
                            continue
                        
                        # Check if values match common priority patterns
                        priority_patterns = set(['p1', 'p2', 'p3', 'p4', 'critical', 'high', 'medium', 'low',
                                             '1', '2', '3', '4', 'urgent', 'normal'])
                        
                        matches = sum(1 for val in unique_vals if val in priority_patterns 
                                    or any(p in val for p in ['priority', 'critical', 'high', 'medium', 'low']))
                        
                        if matches > 0 and len(unique_vals) <= 10:  # Priorities typically have few unique values
                            inferred_mappings[target_col] = source_col
                            break
                
                elif target_col == 'status':
                    # Look for a column with status-like values
                    for source_col in source_columns:
                        if source_col in inferred_mappings.values():
                            continue
                            
                        # Get unique values as strings
                        try:
                            unique_vals = df[source_col].dropna().astype(str).str.lower().unique()
                        except:
                            continue
                        
                        # Check if values match common status patterns
                        status_patterns = set(['open', 'closed', 'in progress', 'pending', 'resolved', 
                                           'new', 'assigned', 'completed'])
                        
                        matches = sum(1 for val in unique_vals if val in status_patterns 
                                    or any(s in val for s in ['open', 'close', 'progress', 'pend', 'resolv']))
                        
                        if matches > 0 and len(unique_vals) <= 10:  # Statuses typically have few unique values
                            inferred_mappings[target_col] = source_col
                            break
        
        return inferred_mappings
    
    def convert_file(self, 
                    input_path: str, 
                    output_path: str, 
                    column_mappings: Dict[str, str], 
                    progress_callback: Optional[Callable[[int, int], None]] = None, 
                    file_obj=None) -> Dict[str, Any]:
        """
        Convert input file to standardized format using column mappings.
        Uses chunked processing for large files.
        
        Args:
            input_path: Path to input file or filename if using file_obj
            output_path: Path to save converted file
            column_mappings: Dictionary mapping target columns to source columns
            progress_callback: Optional callback function to report progress (current_chunk, total_chunks)
            file_obj: File object if provided (for uploaded files)
            
        Returns:
            Dictionary with metadata about the conversion
        """
        file_extension = input_path.split('.')[-1].lower()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize counters
        total_rows = 0
        processed_rows = 0
        total_chunks = 0
        processed_chunks = 0
        
        try:
            # Process based on file type
            if file_extension == 'csv':
                # Get reader for chunked processing
                if file_obj:
                    # Reset position
                    file_obj.seek(0)
                    # Count total lines to estimate chunks (subtracting header)
                    total_rows = sum(1 for _ in file_obj) - 1
                    file_obj.seek(0)
                    # Calculate total chunks
                    total_chunks = (total_rows // self.chunk_size) + 1
                    # Process in chunks
                    reader = pd.read_csv(file_obj, chunksize=self.chunk_size)
                else:
                    # Count total lines
                    with open(input_path, 'r') as f:
                        total_rows = sum(1 for _ in f) - 1  # Subtract header
                    # Calculate total chunks
                    total_chunks = (total_rows // self.chunk_size) + 1
                    # Process in chunks
                    reader = pd.read_csv(input_path, chunksize=self.chunk_size)
                
                # Track if we've written anything
                wrote_file = False
                
                # Process each chunk
                for i, chunk in enumerate(reader):
                    processed_chunks += 1
                    
                    # Apply column mappings and standardization
                    converted_chunk = self._transform_chunk(chunk, column_mappings)
                    
                    # Skip empty chunks
                    if converted_chunk is None or converted_chunk.empty:
                        logger.warning(f"Chunk {i} resulted in empty data after transformation")
                        continue
                    
                    # Write to output file
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0
                    
                    try:
                        # Explicitly ensure the file is writable
                        with open(output_path, mode, newline='') as test_file:
                            pass
                    except Exception as e:
                        raise ValueError(f"Cannot write to output file: {str(e)}")
                    
                    # Write the chunk
                    converted_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                    wrote_file = True
                    
                    # Update progress
                    processed_rows += len(chunk)
                    if progress_callback:
                        progress_callback(processed_chunks, total_chunks)
                
                # Check if we wrote anything
                if not wrote_file:
                    # Create an empty file with headers if we didn't write any chunks
                    empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                    empty_df.to_csv(output_path, index=False)
                    logger.warning("No data was converted, writing empty file with headers")
            
            elif file_extension in ['xlsx', 'xls']:
                # Excel files - use direct memory loading for reliability
                try:
                    if file_obj:
                        # Get file content
                        file_content = file_obj.read()
                        file_obj.seek(0)  # Reset for later use
                        
                        # Try different engines
                        engines_to_try = ['openpyxl', 'xlrd']
                        excel_data = None
                        
                        for engine in engines_to_try:
                            try:
                                excel_data = pd.read_excel(io.BytesIO(file_content), engine=engine)
                                logger.info(f"Successfully read Excel with engine: {engine}")
                                break
                            except Exception as engine_error:
                                logger.warning(f"Engine {engine} failed: {str(engine_error)}")
                                if engine == engines_to_try[-1]:  # Last engine
                                    raise ValueError(f"Failed to read Excel file with any engine: {str(engine_error)}")
                        
                        if excel_data is None:
                            raise ValueError("Failed to read Excel file with any engine")
                        
                        # Successfully loaded data
                        df = excel_data
                    else:
                        # Read from file path with multiple engine attempts
                        engines_to_try = ['openpyxl', 'xlrd']
                        excel_data = None
                        
                        for engine in engines_to_try:
                            try:
                                excel_data = pd.read_excel(input_path, engine=engine)
                                logger.info(f"Successfully read Excel with engine: {engine}")
                                break
                            except Exception as engine_error:
                                logger.warning(f"Engine {engine} failed: {str(engine_error)}")
                                if engine == engines_to_try[-1]:  # Last engine
                                    raise ValueError(f"Failed to read Excel file with any engine: {str(engine_error)}")
                        
                        if excel_data is None:
                            raise ValueError("Failed to read Excel file with any engine")
                        
                        # Successfully loaded data
                        df = excel_data
                    
                    # Verify data was loaded
                    if df is None or df.empty:
                        # Create an empty file with headers
                        empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                        empty_df.to_csv(output_path, index=False)
                        logger.warning("Excel file loaded, but no data was found. Writing empty file with headers.")
                        return {
                            "input_path": input_path,
                            "output_path": output_path,
                            "total_rows": 0,
                            "processed_rows": 0,
                            "total_chunks": 0,
                            "processed_chunks": 0,
                            "column_mappings": column_mappings,
                            "timestamp": datetime.now().isoformat(),
                            "success": True,
                            "file_size": os.path.getsize(output_path)
                        }
                    
                    # Log info
                    logger.info(f"Excel file loaded successfully. Shape: {df.shape}")
                    
                    # Get basic counts
                    total_rows = len(df)
                    total_chunks = max(1, (total_rows // self.chunk_size) + (1 if total_rows % self.chunk_size > 0 else 0))
                    
                    # Flag to track if we wrote anything
                    wrote_file = False
                    
                    # Process in memory chunks
                    for i in range(0, total_rows, self.chunk_size):
                        processed_chunks += 1
                        
                        # Get chunk
                        end_idx = min(i + self.chunk_size, total_rows)
                        chunk = df.iloc[i:end_idx]
                        
                        # Apply column mappings and standardization
                        converted_chunk = self._transform_chunk(chunk, column_mappings)
                        
                        # Ensure we have data to write
                        if converted_chunk is None or converted_chunk.empty:
                            logger.warning(f"Chunk {processed_chunks} had no data after transformation. Skipping.")
                            continue
                        
                        # Write to output file
                        mode = 'w' if i == 0 else 'a'
                        header = i == 0
                        
                        # Explicitly test file writability
                        try:
                            with open(output_path, mode, newline='') as test_file:
                                pass
                        except Exception as e:
                            raise ValueError(f"Cannot write to output file: {str(e)}")
                        
                        # Write chunk
                        converted_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                        wrote_file = True
                        
                        # Update progress
                        processed_rows += len(chunk)
                        if progress_callback:
                            progress_callback(processed_chunks, total_chunks)
                    
                    # Check if we wrote anything
                    if not wrote_file:
                        # Create an empty file with headers
                        empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                        empty_df.to_csv(output_path, index=False)
                        logger.warning("No data was converted, writing empty file with headers")
                    
                except Exception as excel_error:
                    logger.error(f"Error processing Excel file: {str(excel_error)}", exc_info=True)
                    raise ValueError(f"Excel processing failed: {str(excel_error)}")
            
            elif file_extension == 'json':
                # Similar approach as for Excel files
                # ... [JSON handling code]
                # Create a minimal output file with headers
                empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                empty_df.to_csv(output_path, index=False)
                logger.warning("JSON handling not fully implemented, writing empty file with headers")
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Ensure the file exists after conversion
            if not os.path.exists(output_path):
                # One last attempt to create the file
                empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                empty_df.to_csv(output_path, index=False)
                if not os.path.exists(output_path):
                    raise ValueError(f"File conversion failed: Output file {output_path} was not created")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            
            # Prepare metadata about the conversion
            metadata = {
                "input_path": input_path,
                "output_path": output_path,
                "total_rows": total_rows,
                "processed_rows": processed_rows,
                "total_chunks": total_chunks,
                "processed_chunks": processed_chunks,
                "column_mappings": column_mappings,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "file_size": file_size
            }
            
            logger.info(f"Conversion complete: {processed_rows} rows processed, file size: {file_size} bytes")
            return metadata
            
        except Exception as e:
            logger.error(f"Error converting file: {str(e)}", exc_info=True)
            
            # Try to diagnose specific issues
            error_details = str(e)
            diagnosis = ""
            
            if "ambiguous" in error_details.lower() and "series" in error_details.lower():
                diagnosis = "Internal error with pandas Series comparison"
            elif "permission" in error_details.lower():
                diagnosis = "File permission issue"
            elif "file not found" in error_details.lower() or "no such file" in error_details.lower():
                diagnosis = "File path issue"
            elif "memory" in error_details.lower():
                diagnosis = "Memory error - file might be too large"
            elif "excel" in error_details.lower() and "corrupt" in error_details.lower():
                diagnosis = "Excel file appears to be corrupted"
            
            # Try to create at least an empty file with headers
            try:
                empty_df = pd.DataFrame(columns=list(column_mappings.keys()))
                empty_df.to_csv(output_path, index=False)
                logger.info(f"Created empty file with headers after error: {output_path}")
            except Exception as write_err:
                logger.error(f"Could not create empty file: {str(write_err)}")
            
            # Return error metadata
            metadata = {
                "input_path": input_path,
                "output_path": output_path,
                "total_rows": total_rows,
                "processed_rows": processed_rows,
                "total_chunks": total_chunks,
                "processed_chunks": processed_chunks,
                "error": str(e),
                "error_diagnosis": diagnosis,
                "success": False
            }
            
            return metadata
    
    def _transform_chunk(self, chunk: pd.DataFrame, column_mappings: Dict[str, str]) -> pd.DataFrame:
        """
        Apply column mappings and standardization to a chunk of data with robust handling.
        
        Args:
            chunk: DataFrame chunk to transform
            column_mappings: Dictionary mapping target columns to source columns
            
        Returns:
            Transformed DataFrame chunk
        """
        # Validate inputs
        if chunk is None or chunk.empty:
            logger.warning("Empty or null chunk received for transformation")
            return pd.DataFrame()
            
        if not column_mappings:
            logger.warning("No column mappings provided for transformation")
            return pd.DataFrame()
        
        # Create a new DataFrame to store transformed data
        result_df = pd.DataFrame()
        
        # Specialized datetime conversion method
        def safe_datetime_conversion(series):
            """
            Safely convert a series to datetime with comprehensive error handling
            """
            try:
                # First attempt: use pandas to_datetime with coercion
                converted = pd.to_datetime(series, errors='coerce')
                
                # Additional filtering for valid dates
                # Remove extreme outliers (e.g., dates before 1900 or after 2100)
                min_date = pd.Timestamp('1900-01-01')
                max_date = pd.Timestamp('2100-12-31')
                
                return converted.where(
                    (converted >= min_date) & (converted <= max_date), 
                    pd.NaT
                )
            except Exception as e:
                logger.warning(f"Comprehensive datetime conversion failed: {str(e)}")
                return pd.Series([pd.NaT] * len(series))
        
        # Priority mapping for standardization
        priority_mapping = {
            'p1': 'critical', 'p2': 'high', 'p3': 'medium', 
            'p4': 'low', 'p5': 'low', '1': 'critical', 
            '2': 'high', '3': 'medium', '4': 'low', '5': 'low'
        }
        
        # Status mapping for standardization
        status_mapping = {
            'open': 'open', 'new': 'open', 
            'in progress': 'in progress', 'assigned': 'in progress',
            'pending': 'pending', 'on hold': 'pending',
            'resolved': 'resolved', 'complete': 'resolved',
            'closed': 'closed', 'cancelled': 'closed'
        }
        
        # Process each mapped column
        for target_col, source_col in column_mappings.items():
            try:
                # Check if source column exists
                if source_col not in chunk.columns:
                    logger.warning(f"Source column {source_col} not found in data")
                    continue
                
                # Get column type from standard schema
                col_schema = self.standard_schema.get(target_col, {})
                col_type = col_schema.get('type', 'string')
                
                # Column-specific transformations
                if col_type == 'datetime':
                    # Specialized datetime conversion
                    result_df[target_col] = safe_datetime_conversion(chunk[source_col])
                
                elif target_col == 'priority' or col_type == 'priority':
                    # Standardize priority values
                    result_df[target_col] = (
                        chunk[source_col]
                        .astype(str)
                        .str.lower()
                        .map(lambda x: priority_mapping.get(x, x.lower()))
                    )
                
                elif target_col == 'status' or col_type == 'status':
                    # Standardize status values
                    result_df[target_col] = (
                        chunk[source_col]
                        .astype(str)
                        .str.lower()
                        .map(lambda x: status_mapping.get(x, x.lower()))
                    )
                
                else:
                    # Default: direct copy with type conversion to string
                    result_df[target_col] = chunk[source_col].astype(str)
            
            except Exception as e:
                logger.error(f"Error processing column {source_col} to {target_col}: {str(e)}")
        
        return result_df



    def validate_mappings(self, column_mappings: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate if all required fields are mapped.
        
        Args:
            column_mappings: Dictionary mapping target columns to source columns
            
        Returns:
            Validation results dictionary with is_valid flag and any issues
        """
        validation = {
            "is_valid": True,
            "missing_required": [],
            "issues": []
        }
        
        # Check that all required fields are mapped
        for target_col, schema in self.standard_schema.items():
            if schema.get("required", False) and target_col not in column_mappings:
                validation["is_valid"] = False
                validation["missing_required"].append(target_col)
                validation["issues"].append(f"Required field '{target_col}' is not mapped")
        
        return validation            