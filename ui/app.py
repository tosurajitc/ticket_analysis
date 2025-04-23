"""
Main Streamlit application entry point for the incident management analytics dashboard.
This version includes updates to support the new data conversion workflow and
fixes for data visibility issues.
"""

import streamlit as st
import os
import sys
import logging
import inspect
import traceback
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config.config import Config
from config.constants import APP_TITLE, APP_DESCRIPTION

# Import data management
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.data_validator import DataValidator
from data.data_converter import DataConverter  # Import the new data converter

# Import UI modules
from ui.pages.landing_page import render_landing_page
from ui.pages.insights_page import render_insights_page
from ui.pages.automation_page import render_automation_page
from ui.pages.resource_page import render_resource_page
from ui.pages.conversation_page import render_conversation_page
from ui.utils.session_state import initialize_session_state
from ui.utils.ui_helpers import display_data_status

# Define pages and their render functions
PAGES = {
    "Dashboard": render_landing_page,
    "AI Insights": render_insights_page,
    "Automation Opportunities": render_automation_page,
    "Resource Optimization": render_resource_page,
    "Conversational Analytics": render_conversation_page
}

# Lowered threshold for data sufficiency
DATA_SUFFICIENCY_THRESHOLD = 5  # Reduced from previous higher value

def initialize_app() -> None:
    """
    Initialize the application, settings, configurations, and session state.
    """
    # Set page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state if not already done
    initialize_session_state()
    
    # Load configuration
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    # Check if data is loaded
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Add tracking for data visibility in dashboard
    if 'data_visible_in_dashboard' not in st.session_state:
        st.session_state.data_visible_in_dashboard = False
    
    # Initialize data_issues tracking
    if 'data_issues' not in st.session_state:
        st.session_state.data_issues = []
    
    # Initialize data_loader for the application
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader(st.session_state.config)
    
    # Initialize error tracking
    if 'error_details' not in st.session_state:
        st.session_state.error_details = {
            "last_error": None,
            "error_count": 0,
            "error_components": []
        }
    
    # Set up debugging flag
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def load_data() -> bool:
    """
    Load and process incident data from the converted file path.
    Fixed to prevent preprocessing failures from resulting in empty data.
    
    Returns:
        bool: True if data was loaded successfully, False otherwise
    """
    try:
        # Log start of data loading
        logging.info("Starting data loading process")
        logging.info(f"Data source path: {st.session_state.get('data_source_path', 'Not set')}")
        
        # Initialize data loader
        data_loader = DataLoader(st.session_state.config)
        
        # Reset data issues list
        st.session_state.data_issues = []
        
        # Check if data source is provided
        if not st.session_state.data_source_path:
            error_msg = "No data source path provided. Please convert your data first."
            st.session_state.data_issues.append(error_msg)
            logging.error(error_msg)
            return False
        
        # Ensure the file exists
        if not os.path.exists(st.session_state.data_source_path):
            error_msg = f"File not found: {st.session_state.data_source_path}"
            st.session_state.data_issues.append(error_msg)
            logging.error(error_msg)
            return False
            
        # Load data from path only
        try:
            # Load from path
            data = data_loader.load_from_path(st.session_state.data_source_path)
            logging.info(f"Attempting to load data from path: {st.session_state.data_source_path}")
            
            # If the loading failed, try formatting it
            if data is None:
                logging.info("Initial load failed. Attempting to format file before loading again")
                
                try:
                    # Import formatter
                    from data.data_formatter import DataFormatter
                    formatter = DataFormatter()
                    
                    # Format the source file
                    formatted_path = f"{st.session_state.data_source_path}.formatted"
                    format_metadata = formatter.format_file(st.session_state.data_source_path, formatted_path)
                    
                    if format_metadata.get('success', False):
                        # Update source path and try loading again
                        st.session_state.data_source_path = formatted_path
                        data = data_loader.load_from_path(formatted_path)
                        
                        if data is not None:
                            logging.info(f"Successfully loaded data after formatting: {formatted_path}")
                except Exception as format_err:
                    logging.error(f"Formatting recovery attempt failed: {str(format_err)}")
        except Exception as load_error:
            error_msg = f"Error loading data: {str(load_error)}"
            st.session_state.data_issues.append(error_msg)
            logging.error(error_msg, exc_info=True)
            return False
            
        if data is None or data.empty:
            error_msg = "Failed to load data. The file might be empty or in an unsupported format."
            st.session_state.error_message = error_msg
            st.session_state.data_issues.append(error_msg)
            logging.error(error_msg)
            return False
            
        # Log data characteristics for debugging
        logging.info(f"Loaded data shape: {data.shape}")
        logging.info(f"Loaded data columns: {list(data.columns)}")
            
        # Validate data with more lenient approach
        validator = DataValidator()
        validation_result = validator.validate(data)
        
        if not validation_result["is_valid"]:
            validation_error = f"Data validation failed: {validation_result['errors']}"
            st.session_state.error_message = validation_error
            st.session_state.data_issues.append(validation_error)
            # Continue processing anyway with a warning
            st.session_state.data_issues.append("Proceeding with available data, but some analyses may be limited.")
            logging.warning(validation_error)
        
        # FIXED APPROACH: Process data with DataProcessor but keep original data if processing fails
        processor = DataProcessor(st.session_state.config)
        
        # Create a combined preprocessing approach using multiple analysis types
        processed_data = data.copy()  # Start with a copy of the original data
        
        # Keep track of successful preprocessing steps
        successful_preprocessing = False
        
        # Try each preprocessing step independently and only apply if successful
        preprocessing_steps = [
            ("time_analysis", "Time analysis preprocessing"),
            ("category_analysis", "Category analysis preprocessing"),
            ("priority_analysis", "Priority analysis preprocessing")
        ]
        
        for analysis_type, log_message in preprocessing_steps:
            try:
                result = processor.prepare_data_for_analysis(processed_data.copy(), analysis_type)
                if result[0] is not None and not result[0].empty:
                    # Only update the processed data if this step succeeded
                    processed_data = result[0]
                    successful_preprocessing = True
                    logging.info(log_message)
                else:
                    logging.warning(f"{analysis_type} returned None or empty DataFrame, skipping this step")
            except Exception as e:
                logging.warning(f"{log_message} failed: {str(e)}")
                # Continue with other preprocessing steps
        
        # If all preprocessing failed, log a warning but continue with original data
        if not successful_preprocessing:
            logging.warning("All preprocessing steps failed. Using original data without preprocessing.")
            st.session_state.data_issues.append("Data preprocessing had issues. Using original data for analysis.")
            processed_data = data.copy()  # Fallback to original data
        
        # Check if we still have data after processing
        if processed_data is None or processed_data.empty:
            logging.error("No data after preprocessing, falling back to original data")
            processed_data = data.copy()  # Fallback to original data
            st.session_state.data_issues.append("Preprocessing resulted in empty data. Using original data instead.")
            
        # Log processed data characteristics
        logging.info(f"Processed data shape: {processed_data.shape}")
        logging.info(f"Processed data columns: {list(processed_data.columns)}")
            
        # Store data in session state
        st.session_state.raw_data = data
        st.session_state.processed_data = processed_data
        st.session_state.data_loaded = True
        st.session_state.error_message = None
        
        # When data is loaded, automatically set data_visible_in_dashboard to True
        st.session_state.data_visible_in_dashboard = True
        
        # Check data sufficiency with more details
        sufficiency_result = is_data_sufficient(processed_data)
        st.session_state.data_sufficient = sufficiency_result["is_sufficient"]
        
        # Store sufficiency details for UI feedback
        st.session_state.sufficiency_details = sufficiency_result
        
        if not sufficiency_result["is_sufficient"]:
            st.session_state.data_issues.extend(sufficiency_result["issues"])
        
        logging.info("Data loading and processing completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Unexpected error loading data: {str(e)}"
        st.session_state.error_message = error_msg
        st.session_state.data_issues.append(error_msg)
        logging.error(error_msg, exc_info=True)
        return False

def is_data_sufficient(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced version that checks if the data is sufficient for generating insights,
    with detailed feedback on what's missing or insufficient.
    
    Args:
        data: DataFrame containing the processed data
        
    Returns:
        Dict with sufficiency status and details:
        {
            "is_sufficient": bool,
            "issues": List of issues found,
            "recommended_columns": List of recommended columns,
            "missing_columns": List of important missing columns
        }
    """
    result = {
        "is_sufficient": False,
        "issues": [],
        "recommended_columns": [],
        "missing_columns": []
    }
    
    if data is None or data.empty:
        result["issues"].append("No data available for analysis.")
        return result
        
    # Check for minimum row count
    if len(data) < DATA_SUFFICIENCY_THRESHOLD:
        result["issues"].append(f"Insufficient data points. Found {len(data)} incidents, but at least {DATA_SUFFICIENCY_THRESHOLD} are recommended.")
        return result
    
    # Define important column patterns with alternatives
    recommended_columns = {
        "timestamp": ["timestamp", "created_date", "date", "time", "created", "opened", "reported_time"],
        "incident_id": ["incident_id", "id", "ticket_id", "case_number", "reference"],
        "priority": ["priority", "severity", "urgency", "importance"],
        "status": ["status", "state", "condition"]
    }
    
    # Check for required columns with flexible matching
    missing_columns = []
    found_columns = {}
    
    for col_type, patterns in recommended_columns.items():
        found = False
        for pattern in patterns:
            matching_cols = [col for col in data.columns if pattern.lower() in col.lower()]
            if matching_cols:
                found = True
                found_columns[col_type] = matching_cols[0]
                break
        
        if not found:
            missing_columns.append(col_type)
            result["issues"].append(f"Missing {col_type} information. Consider adding a column like {patterns[0]}.")
    
    result["missing_columns"] = missing_columns
    result["recommended_columns"] = list(recommended_columns.keys())
    
    # Check for date range if timestamp is available
    if "timestamp" in found_columns:
        timestamp_col = found_columns["timestamp"]
        try:
            dates = pd.to_datetime(data[timestamp_col], errors='coerce')
            valid_dates = dates.dropna()
            
            if len(valid_dates) < DATA_SUFFICIENCY_THRESHOLD:
                result["issues"].append(f"Too few valid date values in {timestamp_col}. Found {len(valid_dates)} valid dates.")
            else:
                date_range = (valid_dates.max() - valid_dates.min()).days
                if date_range < 1:
                    result["issues"].append(f"Date range is too small ({date_range} days). At least 1 day of data is recommended.")
        except Exception as e:
            result["issues"].append(f"Could not process dates in {timestamp_col}: {str(e)}")
    
    # Set sufficiency based on issues
    if len(result["issues"]) == 0:
        result["is_sufficient"] = True
    elif len(missing_columns) == 0 and len(data) >= DATA_SUFFICIENCY_THRESHOLD:
        # If we have all important columns and enough data, consider it sufficient
        # despite other minor issues
        result["is_sufficient"] = True
    
    return result

def render_sidebar() -> None:
    """
    Render the application sidebar with navigation and data controls.
    """
    with st.sidebar:
        st.title(APP_TITLE)
        st.markdown(APP_DESCRIPTION)
        
        st.divider()
        
        # If a converted file path exists, display it
        if 'converted_file_path' in st.session_state and st.session_state.converted_file_path:
            st.success(f"Converted File: {os.path.basename(st.session_state.converted_file_path)}")
        
        # Application capabilities information (moved from landing page)
        st.markdown("""
        This application provides:
        - AI-driven root cause analysis
        - Predictive incident forecasting
        - Resource optimization recommendations
        - Automation opportunity detection
        - Conversational analytics interface
        """)
        
        st.divider()
        
        # Refresh analysis button
        if st.button("Refresh Analysis"):
            st.session_state.refresh_triggered = True
            # Force data visibility to be True when refreshing
            st.session_state.data_visible_in_dashboard = True
            st.rerun()
            
        # Application info
        st.divider()
        st.info(
            "All insights are generated based on the data provided, with no predefined templates."
        )

def safe_render_page(page_func, page_name):
    """
    Safely render pages with proper error handling and parameter passing.
    
    Args:
        page_func: Function to render the page
        page_name: Name of the page being rendered
    """
    try:
        # Check if refresh was triggered
        refresh_triggered = st.session_state.get('refresh_triggered', False)
        
        # If data is loaded, ensure data is visible in dashboard
        if st.session_state.get('data_loaded', False):
            st.session_state.data_visible_in_dashboard = True
            
            if refresh_triggered:
                st.session_state.refresh_triggered = False  # Reset the trigger
        
        # Analyze the function signature to determine required arguments
        sig = inspect.signature(page_func)
        param_names = list(sig.parameters.keys())
        
        # Prepare arguments based on the function signature
        kwargs = {}
        
        # Common parameters
        if 'data' in param_names:
            kwargs['data'] = st.session_state.get('processed_data')
        
        if 'config' in param_names:
            kwargs['config'] = st.session_state.config
            
        if 'is_data_sufficient' in param_names:
            kwargs['is_data_sufficient'] = st.session_state.get('data_sufficient', False)
            
        if 'data_loader' in param_names:
            kwargs['data_loader'] = st.session_state.data_loader
        
        # Call the page function with the appropriate arguments
        page_func(**kwargs)
        
    except TypeError as e:
        # This specifically catches missing argument errors
        st.error(f"Error rendering {page_name} page: {str(e)}")
        st.info("This could be due to a mismatch in function parameters. Please check the function signature.")
        
        # Try a fallback rendering with minimal parameters
        try:
            # For debugging - show what happened
            sig = inspect.signature(page_func)
            st.write(f"Expected parameters: {list(sig.parameters.keys())}")
            
            # Try with a simpler approach - just pass the data_loader
            if 'data_loader' in st.session_state:
                page_func(data_loader=st.session_state.data_loader)
            else:
                st.warning(f"Cannot render {page_name} page: data_loader not available")
        except Exception as fallback_err:
            st.error(f"Fallback rendering also failed: {str(fallback_err)}")
            
    except Exception as e:
        st.error(f"Error rendering {page_name} page: {str(e)}")
        # Log the error with more details
        logging.error(f"Error rendering {page_name}: {str(e)}")
        logging.error(traceback.format_exc())

def main() -> None:
    """
    Main application entry point.
    """
    # Initialize application
    initialize_app()
    
    # Render sidebar
    render_sidebar()
    
    # Create tabs for page navigation
    tab_names = list(PAGES.keys())
    tabs = st.tabs(tab_names)
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        for tab, page_name in zip(tabs, tab_names):
            with tab:
                st.header("Welcome to Incident Analytics")
                st.markdown(
                    """
                    Get started by preparing your incident data in the Dashboard tab.
                    """
                )
                
                # If we're on the Dashboard tab, render data preparation
                if page_name == "Dashboard":
                    render_func = PAGES[page_name]
                    try:
                        # Pass data_loader to the Dashboard page
                        render_func(
                            data=None,
                            config=st.session_state.config,
                            is_data_sufficient=False,
                            data_loader=st.session_state.data_loader
                        )
                    except TypeError:
                        # Fallback if data_loader isn't accepted
                        render_func(
                            data=None,
                            config=st.session_state.config,
                            is_data_sufficient=False
                        )
    else:
        # Once data is loaded, automatically set data_visible_in_dashboard to True
        if not st.session_state.get('data_visible_in_dashboard', False):
            st.session_state.data_visible_in_dashboard = True
            
        # Render tabs with data
        for tab, page_name in zip(tabs, tab_names):
            with tab:
                # Check data sufficiency for the selected page
                if not st.session_state.get('data_sufficient', False) and page_name != "Dashboard":
                    st.warning(
                        "The provided data may not be sufficient for comprehensive analysis. "
                        "Some features may be limited or unavailable."
                    )
                
                # Render the page with safe rendering
                safe_render_page(PAGES[page_name], page_name)

if __name__ == "__main__":
    main()