"""
Landing page for the incident management analytics dashboard with enhanced data conversion feature.

This module renders the main dashboard view with a data preparation section that allows users
to map columns from their input data to standardized fields and convert the data for analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visualization.interactive_elements import (
    create_date_filter,
    create_multiselect_filter,
    create_interactive_time_series,
    create_interactive_distribution,
    interactive_kpi_cards
)

from ui.components import (
    page_header,
    data_insufficiency_message,
    data_summary_section,
    incident_trend_section,
    incident_distribution_section,
    dynamic_insight_card,
    resolution_analysis_section
)

from ui.utils.ui_helpers import (
    detect_key_columns,
    generate_data_profile,
    display_data_status,
    apply_theme_to_figure,
    format_duration,
    format_percentage,
    sanitize_column_name
)

from ui.utils.session_state import (
    initialize_session_state,
    update_date_filter,
    get_filtered_data,
    cache_analysis_result,
    get_cached_analysis_result
)

# Make sure these imports are correctly pointing to the right modules and functions
from models.anomaly_detection import detect_anomalies
from models.forecasting import generate_forecast
from analysis.insights_generator import generate_insights


def render_data_conversion_section():
    """
    Render the data conversion interface for mapping columns and converting data.
    This replaces the existing file upload functionality with a more structured approach.
    """
    st.subheader("Data Preparation and Column Mapping")
    
    st.info("""
    This tool helps you standardize your incident data for analysis. 
    You can process up to approximately 100,000 records at a time (50,000 for Excel files).
    Larger files will be processed in chunks.
    """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your incident data file (CSV, Excel, or JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload your raw incident data file. We'll help you map the columns to our standard format."
    )
    
    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Try to read the uploaded file
        try:
            # Detect file type from extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Display loading message
            with st.spinner("Reading file..."):
                # Read based on file type
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file, nrows=1000)  # Read first 1000 rows for preview
                    st.session_state.file_type = 'csv'
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(uploaded_file, nrows=1000)
                    st.session_state.file_type = 'excel'
                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file, lines=True, nrows=1000)
                    st.session_state.file_type = 'json'
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    return
                
                # Store the preview dataframe in session state
                st.session_state.preview_data = df
                st.session_state.file_read_success = True
                
                # Estimate total rows in the file
                # This is an approximation for CSV files
                if file_extension == 'csv':
                    # Reset file pointer
                    uploaded_file.seek(0)
                    # Count lines in the file
                    line_count = sum(1 for _ in uploaded_file)
                    # Adjust for header
                    total_rows = line_count - 1
                    st.session_state.estimated_total_rows = total_rows
                else:
                    # For Excel/JSON just show the preview count
                    st.session_state.estimated_total_rows = len(df)
                    
                # Reset file pointer again
                uploaded_file.seek(0)
        
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.session_state.file_read_success = False
            return
    
    # If file was successfully loaded, show the column mapping interface
    if 'file_read_success' in st.session_state and st.session_state.file_read_success:
        render_column_mapping_interface()


def render_column_mapping_interface():
    """
    Render the interface for mapping source columns to target columns.
    """
    if 'preview_data' not in st.session_state:
        return
    
    df = st.session_state.preview_data
    
    # Display data preview with limited rows
    st.subheader("Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Display estimated total rows
    if 'estimated_total_rows' in st.session_state:
        estimate = st.session_state.estimated_total_rows
        if estimate > 100000:
            st.warning(f"Estimated {estimate:,} rows in file. Large files will be processed in chunks.")
        else:
            st.success(f"Estimated {estimate:,} rows in file. This is within the recommended processing limit.")
    
    # Define required and recommended columns
    required_columns = {
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
        }
    }
    
    recommended_columns = {
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
    
    # Create column mapping section
    st.subheader("Column Mapping")
    st.markdown("Map your source columns to our standard fields. **Required fields are marked with** *")
    
    # Initialize column_mappings in session state if not already done
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}
    
    # Get source columns
    source_columns = list(df.columns)
    
    # Create two columns for the mapping interface - one for required, one for recommended
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Required Fields")
        # Create dropdown for each required field
        for target_col, details in required_columns.items():
            # Try to auto-select a matching column
            default_index = 0  # Default to "None"
            for i, source_col in enumerate(source_columns):
                # Check if source column name contains the target column name
                if target_col.lower() in source_col.lower():
                    default_index = i + 1  # +1 because we add "None" at the beginning
                    break
            
            # Create the dropdown
            options = ["None"] + source_columns
            selected = st.selectbox(
                f"{target_col}* ({details['description']})",
                options=options,
                index=default_index,
                key=f"required_{target_col}"
            )
            
            if selected != "None":
                st.session_state.column_mappings[target_col] = selected
            elif target_col in st.session_state.column_mappings:
                del st.session_state.column_mappings[target_col]
    
    with col2:
        st.markdown("### Recommended Fields")
        # Create dropdown for each recommended field
        for target_col, details in recommended_columns.items():
            # Try to auto-select a matching column
            default_index = 0  # Default to "None"
            for i, source_col in enumerate(source_columns):
                # Check if source column name contains the target column name
                if target_col.lower() in source_col.lower():
                    default_index = i + 1  # +1 because we add "None" at the beginning
                    break
            
            # Create the dropdown
            options = ["None"] + source_columns
            selected = st.selectbox(
                f"{target_col} ({details['description']})",
                options=options,
                index=default_index,
                key=f"recommended_{target_col}"
            )
            
            if selected != "None":
                st.session_state.column_mappings[target_col] = selected
            elif target_col in st.session_state.column_mappings:
                del st.session_state.column_mappings[target_col]
    
    # Output file configuration
    st.subheader("Output File Configuration")
    
    # Output directory
    default_output_dir = os.path.join(os.getcwd(), "converted_data")
    # Ensure the directory exists
    os.makedirs(default_output_dir, exist_ok=True)
    
    output_dir = st.text_input(
        "Output Directory",
        value=default_output_dir,
        help="Directory where the converted file will be saved"
    )
    
    # Output file name (default based on input file with timestamp)
    if 'uploaded_file' in st.session_state:
        input_filename = st.session_state.uploaded_file.name
        base_name = os.path.splitext(input_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output_name = f"{base_name}_converted_{timestamp}.csv"
    else:
        default_output_name = f"converted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    output_filename = st.text_input(
        "Output Filename",
        value=default_output_name,
        help="Name of the converted file"
    )
    
    # Full output path
    output_path = os.path.join(output_dir, output_filename)
    st.session_state.output_path = output_path
    
    # Checkbox to confirm overwrite if file exists
    if os.path.exists(output_path):
        st.warning(f"File already exists: {output_path}")
        allow_overwrite = st.checkbox("Allow overwrite", value=False)
        if not allow_overwrite:
            st.error("Please change the output filename or allow overwrite to continue")
            return
    
    # Validate required mappings
    missing_required = []
    for target_col, details in required_columns.items():
        if details["required"] and target_col not in st.session_state.column_mappings:
            missing_required.append(target_col)
    
    if missing_required:
        st.error(f"Missing required column mappings: {', '.join(missing_required)}")
        st.button("Convert Data", disabled=True)
    else:
        # Show convert button
        if st.button("Convert Data"):
            convert_and_save_data()


def convert_and_save_data():
    """
    Convert the uploaded data based on column mappings and save to the output path.
    Uses chunked processing for large files.
    """
    if 'uploaded_file' not in st.session_state or 'column_mappings' not in st.session_state:
        st.error("Missing uploaded file or column mappings")
        return
    
    # Create conversion progress container
    progress_container = st.empty()
    progress_container.info("Preparing to convert data...")
    
    try:
        # First, attempt to safely read the file directly using pandas to check for Excel corruption
        file_content = st.session_state.uploaded_file.read()
        st.session_state.uploaded_file.seek(0)  # Reset file pointer
        
        # Get file extension
        file_name = st.session_state.uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Perform a preliminary check for Excel files
        if file_extension in ['xlsx', 'xls']:
            try:
                # Try a direct read with various engines to see if the file is valid
                import pandas as pd
                import io
                
                # First try openpyxl (for .xlsx)
                try:
                    pd.read_excel(io.BytesIO(file_content), engine='openpyxl', nrows=5)
                except Exception as e1:
                    # Try xlrd (for .xls)
                    try:
                        pd.read_excel(io.BytesIO(file_content), engine='xlrd', nrows=5)
                    except Exception as e2:
                        # If both fail, show detailed error
                        error_msg = f"Excel file appears to be corrupt or in an unsupported format.\nErrors:\n{str(e1)}\n{str(e2)}"
                        progress_container.error("Excel file validation failed")
                        st.error(error_msg)
                        return
                
                # Reset file pointer again
                st.session_state.uploaded_file.seek(0)
                
            except Exception as e:
                progress_container.error("Failed to validate Excel file")
                st.error(f"Error validating Excel file: {str(e)}")
                return
        
        # Import data converter
        from data.data_converter import DataConverter
        
        # Create a new DataConverter instance
        data_converter = DataConverter()
        
        # Validate column mappings
        mapping_validation = data_converter.validate_mappings(st.session_state.column_mappings)
        if not mapping_validation['is_valid']:
            st.error("Invalid column mappings:")
            for issue in mapping_validation['issues']:
                st.error(issue)
            return
        
        # Ensure output path is set
        if 'output_path' not in st.session_state or not st.session_state.output_path:
            # Generate a default output path if not set
            import os
            from datetime import datetime
            default_output_dir = os.path.join(os.getcwd(), "converted_data")
            os.makedirs(default_output_dir, exist_ok=True)
            default_filename = f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.output_path = os.path.join(default_output_dir, default_filename)
        
        # Ensure output directory exists
        import os
        output_dir = os.path.dirname(os.path.abspath(st.session_state.output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress update callback
        def update_progress(current_chunk, total_chunks):
            percent = min(current_chunk / max(total_chunks, 1), 1.0)
            progress_container.progress(percent, 
                                      text=f"Converting data... Chunk {current_chunk} of {total_chunks}")
        
        # Convert the file
        metadata = data_converter.convert_file(
            input_path=st.session_state.uploaded_file.name,
            output_path=st.session_state.output_path,
            column_mappings=st.session_state.column_mappings,
            progress_callback=update_progress,
            file_obj=st.session_state.uploaded_file
        )
        
        # Check if conversion was successful
        if metadata.get('success', False):
            # Verify file exists and isn't empty
            import os
            if not os.path.exists(metadata['output_path']):
                raise FileNotFoundError(f"Output file not found: {metadata['output_path']}")
                
            if os.path.getsize(metadata['output_path']) == 0:
                raise ValueError(f"Output file is empty: {metadata['output_path']}")
                
            # Update session state with the converted file details
            st.session_state.converted_file_path = metadata['output_path']
            st.session_state.data_source_path = metadata['output_path']
            
            # Show success message
            progress_container.success(f"Conversion complete! File saved to: {metadata['output_path']}")
            
            # Try to manually read the converted file to verify it's valid
            try:
                import pandas as pd
                test_df = pd.read_csv(metadata['output_path'])
                
                if test_df.empty:
                    st.warning("Converted file appears to be empty. No data was extracted.")
                    return
                    
                # Verify required columns are present
                missing_cols = []
                for col in ['incident_id', 'created_date', 'priority', 'status']:
                    if col not in test_df.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    st.warning(f"Converted file is missing required columns: {', '.join(missing_cols)}")
                    st.info("The file was converted but may not be usable for analysis.")
                    return
                
                # Everything looks good - attempt to load the data
                from ui.app import load_data
                success = load_data()
                
                if success:
                    st.success("Data loaded successfully!")
                    # Continue to analysis section
                else:
                    st.warning("File converted successfully, but there were issues loading it for analysis.")
                    
                    # If standard loading failed, try manual loading
                    st.info("Attempting to load data manually...")
                    
                    # Make sure date columns are properly converted to datetime format
                    for col in test_df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            try:
                                test_df[col] = pd.to_datetime(test_df[col], errors='coerce')
                            except:
                                pass
                    
                    # Store data in session state directly
                    st.session_state.raw_data = test_df.copy()
                    st.session_state.processed_data = test_df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.current_page = "Dashboard"  # Ensure right page is selected
                    
                    # Set data sufficient flag
                    if len(test_df) >= 5:  # Reduced threshold for quick test
                        st.session_state.data_sufficient = True
                    else:
                        st.session_state.data_sufficient = False
                        st.warning("The data contains very few records. Some analyses may be limited.")
                    
                    # Add some error handling and monitoring in session state
                    st.session_state.error_message = None
                    st.session_state.data_issues = []
                    st.session_state.debug_info = {
                        "loaded_manually": True,
                        "record_count": len(test_df),
                        "columns": list(test_df.columns)
                    }
                    
                    st.success("Data loaded manually!")
                
                # Display a clear separation
                st.markdown("---")
                
                # Create a prominent "Start Data Analysis" section
                st.subheader("🚀 Data Successfully Loaded!")
                
                # Provide a brief summary of loaded data
                st.write(f"Loaded {len(test_df)} records with {len(test_df.columns)} columns.")
                
                # Show key columns that were identified
                key_cols = []
                if 'incident_id' in test_df.columns: key_cols.append('incident_id')
                if 'created_date' in test_df.columns: key_cols.append('created_date')
                if 'priority' in test_df.columns: key_cols.append('priority')
                if 'status' in test_df.columns: key_cols.append('status')
                if 'category' in test_df.columns: key_cols.append('category')
                
                if key_cols:
                    st.write(f"Key identified columns: {', '.join(key_cols)}")
                
                # Show a data preview
                st.subheader("Data Preview")
                st.dataframe(test_df.head(5), use_container_width=True)
                
                # Highlight next steps for user
                st.info("The data has been loaded successfully. Now you can view insights and analysis in the tabs above.")
                
                # Create a single prominent button to start analysis (no fallback mode)
                if st.button("▶️ Start Data Analysis", type="primary", use_container_width=True):
                    # Set flags for processing
                    st.session_state.data_visible_in_dashboard = True  # This is crucial
                    st.session_state.analysis_triggered = True
                    st.session_state.refresh_triggered = True
                    
                    # Add a timestamp to track when analysis was started
                    import datetime
                    st.session_state.analysis_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add any needed preprocessing for analysis
                    try:
                        # Initialize key data for filters
                        if 'created_date' in test_df.columns:
                            valid_dates = test_df['created_date'].dropna()
                            if not valid_dates.empty:
                                start_date = valid_dates.min().date()
                                end_date = valid_dates.max().date()
                                st.session_state.date_range = (start_date, end_date)
                                
                        # Extract unique categories for filters
                        if 'category' in test_df.columns:
                            categories = test_df['category'].dropna().unique().tolist()
                            st.session_state.selected_categories = categories[:min(5, len(categories))]
                        
                        # Extract unique priorities for filters
                        if 'priority' in test_df.columns:
                            priorities = test_df['priority'].dropna().unique().tolist()
                            st.session_state.selected_priorities = priorities
                            
                        # Extract unique statuses for filters
                        if 'status' in test_df.columns:
                            statuses = test_df['status'].dropna().unique().tolist()
                            st.session_state.selected_statuses = statuses
                            
                        # Log that we're about to trigger analysis
                        import logging
                        logging.info("*** Starting data analysis - button clicked ***")
                        logging.info(f"Session state before rerun: data_visible_in_dashboard={st.session_state.get('data_visible_in_dashboard', False)}")
                    except Exception as e:
                        import logging
                        logging.warning(f"Error preparing dashboard data: {str(e)}")
                    
                    # Add a message to show the user something is happening
                    st.success("Analysis started! The dashboard will refresh momentarily...")
                    st.info("If the dashboard doesn't update automatically, please click the 'Refresh Analysis' button in the sidebar.")
                    
                    # Refresh the app to show analysis results
                    st.rerun()

            except Exception as read_err:
                import traceback
                st.error(f"Failed to validate converted file: {str(read_err)}")
                st.info("The conversion process completed, but the resulting file may have issues.")
                import logging
                logging.error(f"Convert validation error: {str(read_err)}")
                logging.error(traceback.format_exc())
                
                # Try a more direct approach to loading as fallback
                try:
                    test_df = pd.read_csv(metadata['output_path'], encoding='latin1')
                    if not test_df.empty:
                        st.success("Fallback loading succeeded!")
                        st.session_state.raw_data = test_df
                        st.session_state.processed_data = test_df
                        st.session_state.data_loaded = True
                        st.session_state.data_sufficient = len(test_df) >= 5
                        st.dataframe(test_df.head(5), use_container_width=True)
                        
                        st.info("The data has been loaded. Click the button below to start analysis.")
                        
                        # Add a single Start Data Analysis button (no fallback mode)
                        if st.button("▶️ Start Data Analysis", type="primary", use_container_width=True):
                            st.session_state.data_visible_in_dashboard = True
                            st.session_state.analysis_triggered = True
                            st.session_state.refresh_triggered = True
                            st.rerun()
                except Exception as fallback_err:
                    logging.error(f"Fallback loading failed: {str(fallback_err)}")
        else:
            # Display conversion error
            error_msg = metadata.get('error', 'Unknown error during conversion')
            error_diagnosis = metadata.get('error_diagnosis', '')
            
            progress_container.error(f"Error converting data: {error_msg}")
            
            if error_diagnosis:
                st.error(f"Error diagnosis: {error_diagnosis}")
            
            # Provide specific guidance based on error
            if "excel" in error_msg.lower():
                st.info("Try saving your Excel file in CSV format first, then upload the CSV.")
            elif "ambiguous" in error_msg.lower():
                st.info("There's an issue with the data format. Try simplifying your column mappings.")
            else:
                st.error("The conversion process encountered errors. Please check:")
                st.write("1. Make sure your file is not corrupted")
                st.write("2. Verify that your column mappings match the data")
                st.write("3. Try with a smaller file or fewer columns")
    
    except Exception as e:
        import traceback
        progress_container.error(f"Error converting data: {str(e)}")
        st.error("An unexpected error occurred during conversion.")
        
        # Log detailed error but show simplified message to user
        import logging
        logging.error(f"Convert and save data error: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Give the user some actionable steps
        st.info("Try the following:")
        st.write("1. Check if your file format is supported (CSV, Excel, JSON)")
        st.write("2. Verify your column mappings make sense for your data")
        st.write("3. Make sure the output directory is writable")
        st.write("4. Try with a smaller or less complex file first")

def apply_column_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the column mappings to create a standardized dataframe.
    
    Args:
        df: Source DataFrame with original columns
        
    Returns:
        DataFrame with standardized columns based on mappings
    """
    # Create a new dataframe with the mapped columns
    result_df = pd.DataFrame()
    
    # Apply each column mapping
    for target_col, source_col in st.session_state.column_mappings.items():
        if source_col in df.columns:
            # Apply special handling for certain column types
            if target_col == 'created_date' or target_col == 'resolved_date':
                # Try to convert to datetime
                try:
                    result_df[target_col] = pd.to_datetime(df[source_col], errors='coerce')
                except:
                    result_df[target_col] = df[source_col]
            elif target_col == 'priority':
                # Standardize priority values
                try:
                    result_df[target_col] = standardize_priority(df[source_col])
                except:
                    result_df[target_col] = df[source_col]
            elif target_col == 'status':
                # Standardize status values
                try:
                    result_df[target_col] = standardize_status(df[source_col])
                except:
                    result_df[target_col] = df[source_col]
            else:
                # Direct copy for other columns
                result_df[target_col] = df[source_col]
    
    return result_df


def standardize_priority(priority_series: pd.Series) -> pd.Series:
    """
    Standardize priority values to a consistent format.
    
    Args:
        priority_series: Series containing priority values
        
    Returns:
        Series with standardized priority values
    """
    # Convert to string and lowercase
    priority_series = priority_series.astype(str).str.lower()
    
    # Define mapping dictionary for common priority terms
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
    
    # Apply mapping
    def map_priority(x):
        x_str = str(x).lower()
        for k, v in priority_mapping.items():
            if k in x_str:
                return v
        return x
    
    return priority_series.apply(map_priority)


def standardize_status(status_series: pd.Series) -> pd.Series:
    """
    Standardize status values to a consistent format.
    
    Args:
        status_series: Series containing status values
        
    Returns:
        Series with standardized status values
    """
    # Convert to string and lowercase
    status_series = status_series.astype(str).str.lower()
    
    # Define mapping dictionary for common status terms
    status_mapping = {
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
    
    # Apply mapping
    def map_status(x):
        x_str = str(x).lower()
        for k, v in status_mapping.items():
            if k in x_str:
                return v
        return x
    
    return status_series.apply(map_status)


def display_data_guidance(data: pd.DataFrame, key_columns: Dict[str, str]) -> None:
    """
    Display guidance on data structure and missing fields to help users improve their data.
    
    Args:
        data: DataFrame containing the incident data
        key_columns: Dictionary of detected key columns
    """
    st.subheader("Data Structure Guidance")
    
    # Check which important columns are missing
    important_columns = {
        "timestamp": ["timestamp", "created_date", "date", "time", "created_at"],
        "incident_id": ["incident_id", "id", "ticket_id", "case_number"],
        "priority": ["priority", "severity", "urgency"],
        "status": ["status", "state"],
        "category": ["category", "type"],
        "resolution_time": ["resolution_time", "time_to_resolve", "resolution_duration"],
        "assignee": ["assignee", "assigned_to", "owner"]
    }
    
    missing_columns = []
    found_columns = {}
    
    for col_type, patterns in important_columns.items():
        found = False
        for pattern in patterns:
            matching_cols = [col for col in data.columns if pattern.lower() in col.lower()]
            if matching_cols:
                found = True
                found_columns[col_type] = matching_cols[0]
                break
        
        if not found:
            missing_columns.append(col_type)
    
    # Display found and missing columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Available Key Columns:**")
        if found_columns:
            for col_type, col_name in found_columns.items():
                st.write(f"- {col_type}: `{col_name}`")
        else:
            st.write("No key columns detected.")
    
    with col2:
        st.markdown("**❌ Missing Recommended Columns:**")
        if missing_columns:
            for col_type in missing_columns:
                st.write(f"- {col_type} (e.g., `{important_columns[col_type][0]}`)")
        else:
            st.write("All recommended columns are available!")
    
    # Display current data sample
    with st.expander("Your Current Data Sample", expanded=False):
        st.dataframe(data.head(5), use_container_width=True)
    
    # Show sample data with all recommended columns
    with st.expander("Recommended Data Structure", expanded=True):
        st.markdown("""
        For the best analysis results, your data should include the following columns:
        """)
        
        sample_data = {
            "incident_id": ["INC001", "INC002", "INC003", "INC004", "INC005"],
            "created_date": ["2023-01-01 10:30:00", "2023-01-02 14:15:00", "2023-01-03 09:45:00", "2023-01-04 11:20:00", "2023-01-05 16:00:00"],
            "resolved_date": ["2023-01-01 15:45:00", "2023-01-03 09:30:00", "2023-01-03 14:15:00", "2023-01-05 10:30:00", "2023-01-06 11:15:00"],
            "status": ["Closed", "Closed", "Closed", "Closed", "Closed"],
            "priority": ["P1", "P3", "P2", "P2", "P1"],
            "category": ["Network", "Application", "Database", "Hardware", "Security"],
            "subcategory": ["Router", "Login", "Performance", "Server", "Access"],
            "assigned_to": ["Team A", "Team B", "Team A", "Team C", "Team B"],
            "resolution_time": [5.25, 43.25, 4.5, 23.17, 19.25],
            "description": [
                "Network outage in data center",
                "Application login failure",
                "Database performance issue",
                "Server hardware failure",
                "Unauthorized access attempt"
            ]
        }
        
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        st.markdown("""
        ### Recommendations for Improving Your Data:
        
        1. **Add Missing Columns**: Include the missing columns identified above.
        2. **Date Formats**: Ensure date fields are in a standard format (e.g., YYYY-MM-DD HH:MM:SS).
        3. **Categorical Consistency**: Use consistent values for categories, statuses, and priorities.
        4. **Resolution Time**: Calculate time between created and resolved dates (in hours).
        5. **Sufficient History**: Include at least 30 days of incident data for trend analysis.
        6. **Complete Records**: Avoid missing values in key fields.
        """)


def render_insights_section(data: pd.DataFrame, key_columns: Dict[str, str], is_data_sufficient: bool) -> None:
    """
    Render the AI-generated insights section with better feedback.
    
    Args:
        data: DataFrame containing the incident data
        key_columns: Dictionary of detected key columns
        is_data_sufficient: Whether data is sufficient for full analysis
    """
    st.subheader("AI-Generated Insights")
    
    if not is_data_sufficient:
        st.warning(
            "⚠️ Data quality issues are preventing full insight generation. See below for how to improve your data."
        )
        
        # Display guidance on how to improve data
        display_data_guidance(data, key_columns)
        
        # Display limited insights if possible
        st.markdown("### Limited Insights Based on Available Data")
        try:
            # Attempt to generate basic insights with what we have
            basic_insights = generate_insights(data, key_columns, limited_mode=True)
            
            if basic_insights and len(basic_insights) > 0:
                for insight in basic_insights:
                    if 'title' in insight and 'content' in insight:
                        st.info(f"**{insight['title']}**  \n{insight['content']}")
            else:
                st.info("No insights could be generated with the current data. Please improve your data quality using the guidance above.")
        except Exception as e:
            st.error(f"Could not generate even limited insights: {str(e)}")
    else:
        # Generate full insights with sufficient data
        with st.spinner("Generating insights..."):
            try:
                insights = generate_insights(data, key_columns)
                
                if insights and len(insights) > 0:
                    # Display each insight in a card
                    for insight in insights:
                        if 'title' in insight and 'content' in insight:
                            st.info(f"**{insight['title']}**  \n{insight['content']}")
                else:
                    st.info("No significant insights detected in the current data selection.")
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                st.info("Try adjusting your filters or providing more data for better insights.")


def render_landing_page(
    data: pd.DataFrame,
    config: Any,
    is_data_sufficient: bool,
    data_loader: Any = None
) -> None:
    """
    Renders the main dashboard landing page with enhanced error handling and feedback.
    
    Args:
        data: DataFrame containing the incident data
        config: Application configuration
        is_data_sufficient: Whether data is sufficient for full analysis
        data_loader: Data loading utility (optional)
    """
    # Render page header
    page_header(
        "Incident Analytics Dashboard",
        "Overview of incident metrics and key insights",
        icon="📊"
    )
    
    # Split into tabs for data preparation and analytics
    tab1, tab2 = st.tabs(["Data Preparation", "Analytics Dashboard"])
    
    with tab1:
        # Render the data conversion section
        render_data_conversion_section()
    
    with tab2:
        # Check if data is available
        if data is None or data.empty:
            st.warning("No data available for analysis. Please prepare your data using the Data Preparation tab.")
            return
        
        # Detect key columns for flexibility with different data schemas
        key_columns = detect_key_columns(data)
        
        # Generate data profile to understand characteristics
        data_profile = generate_data_profile(data)
        
        # Create date filter for the entire dashboard
        st.subheader("Dashboard Filters")
        date_col = key_columns.get("timestamp")
        
        if date_col:
            col1, col2 = st.columns([2, 1])
            with col1:
                start_date, end_date = create_date_filter(key_prefix="dashboard")
            
            # Apply date filter to data with better error handling
            try:
                # Make a copy of the data first
                filtered_data = data.copy()
                
                # Check if the column exists (redundant but safe)
                if date_col not in filtered_data.columns:
                    raise ValueError(f"Date column '{date_col}' not found in data")
                    
                # Check if the column is a DataFrame instead of a Series
                column_data = filtered_data[date_col]
                if hasattr(column_data, 'columns'):
                    # It's a DataFrame - convert to Series by using first column
                    if len(column_data.columns) > 0:
                        filtered_data[date_col] = column_data.iloc[:, 0]
                    else:
                        raise ValueError(f"Date column '{date_col}' is empty")
                
                # Convert to datetime with better error handling
                if not pd.api.types.is_datetime64_any_dtype(filtered_data[date_col]):
                    filtered_data[date_col] = pd.to_datetime(filtered_data[date_col], errors='coerce')
                
                # Count valid dates
                valid_dates_mask = ~filtered_data[date_col].isna()
                valid_date_count = valid_dates_mask.sum()
                
                if valid_date_count == 0:
                    raise ValueError(f"No valid dates found in column '{date_col}'")
                    
                # Apply filter only to valid dates
                date_filtered = filtered_data[
                    (filtered_data[date_col].dt.date >= start_date) & 
                    (filtered_data[date_col].dt.date <= end_date)
                ]
                
                # Check if we have any data after filtering
                if not date_filtered.empty:
                    filtered_data = date_filtered
                    with col2:
                        st.info(f"Filtered data: {len(filtered_data)} incidents")
                        if len(filtered_data) < 10:
                            st.warning("Limited data after filtering. Some visualizations may not be available.")
                else:
                    with col2:
                        st.warning(f"No data found in the selected date range. Showing all data with valid dates.")
                    # Keep all data but with converted dates
                    filtered_data = filtered_data[valid_dates_mask]
                    
            except Exception as e:
                filtered_data = data
                with col2:
                    st.warning(f"Date filtering failed: {str(e)} - showing all data")
        else:
            filtered_data = data
            # Try to find alternative date columns
            date_like_cols = [col for col in data.columns if any(term in col.lower() for term in ['date', 'time', 'created'])]
            if date_like_cols:
                st.warning(f"No standard date column detected. Consider using '{date_like_cols[0]}' as your timestamp column.")
            else:
                st.warning("No date column detected - showing all data without time filtering")
        
        # Check if filtered data is sufficient
        if filtered_data.empty:
            st.error(
                "No data available after filtering. Please adjust your date filters."
            )
            # Show the available date range if possible
            if date_col and not data[date_col].isna().all():
                min_date = data[date_col].min().date()
                max_date = data[date_col].max().date()
                st.info(f"Available date range in data: {min_date} to {max_date}")
            return
        elif len(filtered_data) < 5:
            st.warning(
                f"Limited data after filtering ({len(filtered_data)} records). Some visualizations may not be available."
            )
        
        # Render data summary section
        data_summary_section(filtered_data)
        
        # Create top KPIs section
        st.subheader("Key Performance Indicators")
        
        # Calculate KPIs based on available data
        kpis = {}
        
        # Total incidents
        kpis["Total Incidents"] = len(filtered_data)
        
        # Average resolution time if available
        resolution_col = key_columns.get("resolution_time")
        if resolution_col and resolution_col in filtered_data.columns:
            try:
                avg_resolution = filtered_data[resolution_col].mean()
                if avg_resolution > 0:
                    kpis["Avg Resolution Time"] = format_duration(avg_resolution)
            except Exception as e:
                st.info(f"Could not calculate Average Resolution Time: {str(e)}")
        
        # Incident distribution by priority if available
        priority_col = key_columns.get("priority")
        if priority_col and priority_col in filtered_data.columns:
            try:
                # Get highest priority incidents count
                priority_counts = filtered_data[priority_col].value_counts()
                if not priority_counts.empty:
                    highest_priority = priority_counts.index[0]
                    kpis[f"{highest_priority} Incidents"] = priority_counts.iloc[0]
            except Exception as e:
                st.info(f"Could not calculate Priority Distribution: {str(e)}")
        
        # SLA compliance if available
        sla_col = key_columns.get("sla")
        if sla_col and sla_col in filtered_data.columns:
            try:
                # Fix the problematic line with the Series boolean ambiguity issue
                is_bool_type = pd.api.types.is_bool_dtype(filtered_data[sla_col])
                is_binary_values = filtered_data[sla_col].isin([0, 1, True, False]).all()
                
                if is_bool_type or is_binary_values:
                    sla_compliance = filtered_data[sla_col].mean() * 100
                    kpis["SLA Compliance"] = f"{sla_compliance:.1f}%"
            except Exception as e:
                st.info(f"Could not calculate SLA Compliance: {str(e)}")
        
        # Display KPIs
        if kpis:
            interactive_kpi_cards(kpis, columns=4)
        else:
            st.error("No KPIs could be calculated from the available data. Check data format and column names.")
            # Provide guidance on expected columns for KPIs
            st.info("""
            To display Key Performance Indicators, your data should include:
            - For Resolution Time: A column with time measurements (usually in hours)
            - For Priority Distribution: A column with priority values (like P1, P2, P3)
            - For SLA Compliance: A boolean or 0/1 column indicating if SLAs were met
            """)
        
        # Create tabs for different dashboard sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Trends & Forecasting", "Distribution Analysis", 
            "Resolution Performance", "Anomaly Detection"
        ])
        
        with tab1:
            # Incident trend analysis
            if date_col:
                # Create incident trend visualization
                st.subheader("Incident Trend Analysis")
                
                # Split into columns for filters
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Category filter if available
                    category_col = key_columns.get("category")
                    selected_categories = []
                    
                    if category_col and category_col in filtered_data.columns:
                        selected_categories = create_multiselect_filter(
                            filtered_data, 
                            category_col, 
                            f"Filter by {category_col.capitalize()}"
                        )
                
                with col2:
                    # Create rolling window control
                    rolling_window = st.slider(
                        "Smoothing (days)",
                        min_value=1,
                        max_value=30,
                        value=1,
                        help="Apply rolling average to smooth the trend line"
                    )
                
                # Apply category filter if selected
                trend_data = filtered_data
                if category_col and selected_categories:
                    trend_data = trend_data[trend_data[category_col].isin(selected_categories)]
                
                # Create incident trend visualization with better error handling
                if trend_data.empty:
                    st.warning("No data available for trend analysis after applying filters.")
                elif len(trend_data) < 3:  # Reduced minimum threshold
                    st.warning(f"Insufficient data for trend analysis after filtering. Only {len(trend_data)} incidents remain.")
                else:
                    try:
                        fig = create_interactive_time_series(
                            trend_data,
                            date_col,
                            "incident_id",  # Using incident_id for counting
                            title="Incident Volume Over Time",
                            aggregation="count",
                            rolling_window=rolling_window if rolling_window > 1 else None
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating trend visualization: {str(e)}")
                        st.info("Try adjusting your filters or check your date column format.")
                
                # Incident forecasting
                st.subheader("Incident Volume Forecast")
                
                # Check if we have enough data for forecasting with more informative messages
                if len(trend_data) < 30:
                    st.warning(f"Limited historical data for reliable forecasting. Found {len(trend_data)} data points, but 30+ are recommended.")
                    st.info("Forecasting will still be attempted, but results may be less reliable.")
                
                # Create time series data for forecasting with better error handling
                try:
                    # Create time series data for forecasting
                    time_series_data = trend_data.groupby(
                        pd.Grouper(key=date_col, freq='D')
                    ).size().reset_index(name='count')
                    
                    # Check if we have enough unique dates
                    unique_dates = time_series_data[date_col].nunique()
                    if unique_dates < 5:  # Reduced minimum threshold
                        st.warning(f"Only {unique_dates} unique dates found. Forecasting requires at least 5 unique dates.")
                        st.info("Consider using a wider date range or check your date column format.")
                    else:
                        # Generate forecast
                        forecast_days = st.slider(
                            "Forecast Horizon (days)",
                            min_value=7,
                            max_value=90,
                            value=30,
                            step=7
                        )
                        
                        # Generate forecast using the forecasting model
                        with st.spinner("Generating forecast..."):
                            try:
                                forecast_result = generate_forecast(
                                    time_series_data,
                                    date_column=date_col,
                                    value_column='count',
                                    forecast_horizon=forecast_days
                                )
                            
                                if forecast_result is not None and not forecast_result.empty:
                                    # Create visualization
                                    fig = go.Figure()
                                    
                                    # Add historical data
                                    fig.add_trace(go.Scatter(
                                        x=time_series_data[date_col],
                                        y=time_series_data['count'],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Add forecast data
                                    fig.add_trace(go.Scatter(
                                        x=forecast_result['ds'],
                                        y=forecast_result['yhat'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='red', dash='dash')
                                    ))
                                    
                                    # Add confidence intervals
                                    if 'yhat_lower' in forecast_result.columns and 'yhat_upper' in forecast_result.columns:
                                        fig.add_trace(go.Scatter(
                                            x=forecast_result['ds'],
                                            y=forecast_result['yhat_upper'],
                                            mode='lines',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=forecast_result['ds'],
                                            y=forecast_result['yhat_lower'],
                                            mode='lines',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            fillcolor='rgba(255, 0, 0, 0.2)',
                                            name='95% Confidence'
                                        ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title="Incident Volume Forecast",
                                        xaxis_title="Date",
                                        yaxis_title="Incident Count",
                                        height=500
                                    )
                                    
                                    # Apply theme
                                    fig = apply_theme_to_figure(fig)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display forecast insights
                                    if 'yhat' in forecast_result.columns:
                                        # Calculate average forecasted incidents
                                        avg_forecast = forecast_result['yhat'].mean()
                                        
                                        # Calculate trend (comparing last 30 days to next 30 days)
                                        historical_avg = time_series_data['count'].tail(min(30, len(time_series_data))).mean()
                                        forecast_avg = forecast_result['yhat'].head(min(30, len(forecast_result))).mean()
                                        
                                        trend_pct = ((forecast_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                                        
                                        # Display forecast metrics
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Avg Daily Incidents (Forecast)", 
                                                f"{avg_forecast:.1f}",
                                                f"{trend_pct:.1f}%"
                                            )
                                        
                                        with col2:
                                            peak_date = forecast_result.loc[forecast_result['yhat'].idxmax(), 'ds']
                                            peak_value = forecast_result['yhat'].max()
                                            st.metric(
                                                "Peak Volume Date", 
                                                peak_date.strftime('%Y-%m-%d'),
                                                f"{peak_value:.1f} incidents"
                                            )
                                        
                                        with col3:
                                            total_forecast = forecast_result['yhat'].sum()
                                            st.metric(
                                                f"Total Incidents ({forecast_days} days)", 
                                                f"{total_forecast:.0f}"
                                            )
                                else:
                                    st.error("Failed to generate forecast. Please try adjusting your data filters.")
                            except Exception as e:
                                st.error(f"Error generating forecast: {str(e)}")
                                st.info("The forecasting algorithm may require more data or a different data format.")
                except Exception as e:
                    st.error(f"Error preparing data for forecasting: {str(e)}")
                    st.info("Check that your date column is properly formatted and contains valid dates.")
            else:
                st.warning("Timestamp data is required for trend analysis and forecasting.")
                # Provide guidance on timestamp columns
                st.info("""
                To enable trend analysis and forecasting:
                1. Ensure your data includes a date/time column (e.g., 'created_date', 'timestamp')
                2. Make sure the date format is consistent (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
                3. Most date values should be valid (not empty or incorrectly formatted)
                """)
        
        with tab2:
            # Display Distribution Analysis tab content
            # (Existing code for this tab can be kept unchanged)
            incident_distribution_section(filtered_data)
        
        with tab3:
            # Display Resolution Performance tab content
            # (Existing code for this tab can be kept unchanged)
            resolution_analysis_section(filtered_data)
            
        with tab4:
            # Display Anomaly Detection tab content
            # (Existing code for this tab can be kept unchanged)
            st.subheader("Anomaly Detection")
            
            if not date_col:
                st.warning("Timestamp data is required for anomaly detection.")
            else:
                # Create anomaly detection controls
                sensitivity = st.slider(
                    "Detection Sensitivity",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    help="Higher values detect more anomalies but may include false positives"
                )
                
                try:
                    # Create time series data for anomaly detection
                    time_series_data = filtered_data.groupby(
                        pd.Grouper(key=date_col, freq='D')
                    ).size().reset_index(name='count')
                    
                    # Check if we have enough data points
                    if len(time_series_data) < 7:  # Reduced minimum threshold
                        st.warning(f"Limited data for reliable anomaly detection. Found {len(time_series_data)} data points, but 14+ are recommended.")
                        if len(time_series_data) < 3:
                            st.error("Insufficient data for anomaly detection. Please provide more data or adjust your date filter.")
                        else:
                            st.info("Anomaly detection will still be attempted, but results may be less reliable.")
                    
                    if len(time_series_data) >= 3:  # Minimum required for basic detection
                        # Detect anomalies
                        with st.spinner("Detecting anomalies..."):
                            try:
                                anomalies = detect_anomalies(
                                    time_series_data,
                                    date_column=date_col,
                                    value_column='count',
                                    sensitivity=sensitivity
                                )
                            
                                if anomalies is not None:
                                    # Create visualization
                                    fig = go.Figure()
                                    
                                    # Add normal points
                                    normal_data = time_series_data[~anomalies]
                                    fig.add_trace(go.Scatter(
                                        x=normal_data[date_col],
                                        y=normal_data['count'],
                                        mode='lines+markers',
                                        name='Normal',
                                        line=dict(color='blue'),
                                        marker=dict(size=6)
                                    ))
                                    
                                    # Add anomaly points
                                    anomaly_data = time_series_data[anomalies]
                                    if not anomaly_data.empty:
                                        fig.add_trace(go.Scatter(
                                            x=anomaly_data[date_col],
                                            y=anomaly_data['count'],
                                            mode='markers',
                                            name='Anomaly',
                                            marker=dict(
                                                color='red',
                                                size=10,
                                                symbol='circle',
                                                line=dict(width=2, color='black')
                                            )
                                        ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title="Incident Volume Anomalies",
                                        xaxis_title="Date",
                                        yaxis_title="Incident Count",
                                        height=500
                                    )
                                    
                                    # Apply theme
                                    fig = apply_theme_to_figure(fig)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display anomaly insights
                                    anomaly_count = anomalies.sum()
                                    
                                    if anomaly_count > 0:
                                        st.success(f"Detected {anomaly_count} anomalies in the incident data.")
                                        
                                        # List anomaly dates and values
                                        st.subheader("Anomaly Details")
                                        anomaly_details = time_series_data[anomalies].copy()
                                        
                                        # Calculate expected values and deviation
                                        try:
                                            # Simple moving average for expected values
                                            window_size = min(7, len(time_series_data))
                                            time_series_data_sorted = time_series_data.sort_values(date_col)
                                            time_series_data_sorted['expected'] = time_series_data_sorted['count'].rolling(window=window_size, center=True).mean().fillna(time_series_data_sorted['count'].mean())
                                            
                                            # Join expected values to anomalies
                                            anomaly_details = pd.merge(
                                                anomaly_details,
                                                time_series_data_sorted[[date_col, 'expected']],
                                                on=date_col,
                                                how='left'
                                            )
                                            
                                            # Calculate deviation
                                            anomaly_details['deviation'] = ((anomaly_details['count'] - anomaly_details['expected']) / anomaly_details['expected'] * 100).round(1)
                                            anomaly_details['deviation'] = anomaly_details['deviation'].astype(str) + '%'
                                            
                                            # Format for display
                                            display_cols = [date_col, 'count', 'expected', 'deviation']
                                            anomaly_details = anomaly_details[display_cols]
                                            anomaly_details.columns = ['Date', 'Actual Count', 'Expected Count', 'Deviation']
                                        except:
                                            # If calculation fails, show simpler table
                                            display_cols = [date_col, 'count']
                                            anomaly_details = anomaly_details[display_cols]
                                            anomaly_details.columns = ['Date', 'Incident Count']
                                        
                                        st.dataframe(anomaly_details, use_container_width=True)
                                    else:
                                        st.info("No anomalies detected with the current sensitivity setting. Try increasing the sensitivity or checking more data.")
                                else:
                                    st.error("Failed to detect anomalies. Please try adjusting your data filters or sensitivity.")
                            except Exception as e:
                                st.error(f"Error detecting anomalies: {str(e)}")
                                st.info("The anomaly detection algorithm may require more data or a different data format.")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    
        # Add AI insights section
        st.divider()
        render_insights_section(filtered_data, key_columns, is_data_sufficient)