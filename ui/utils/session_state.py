"""
Session state management for the incident management analytics dashboard.

This module handles the initialization and management of Streamlit's session state,
ensuring consistent state management across the application. It provides functions
to initialize, update, and reset session state variables.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta


def initialize_session_state() -> None:
    """
    Initializes the session state with default values for all necessary variables.
    This function should be called at the start of the application to ensure all
    required state variables are available.
    """
    # Data source configuration
    if "data_source_path" not in st.session_state:
        st.session_state.data_source_path = ""
    
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    
    if "data_source_option" not in st.session_state:
        st.session_state.data_source_option = "Upload File"
    
    # Data state
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if "data_sufficient" not in st.session_state:
        st.session_state.data_sufficient = False
    
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    
    # Analysis results (dynamically generated - not predefined)
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Filter states
    if "date_range" not in st.session_state:
        # Default to last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        st.session_state.date_range = (start_date, end_date)
    
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = []
    
    if "selected_priorities" not in st.session_state:
        st.session_state.selected_priorities = []
    
    if "selected_statuses" not in st.session_state:
        st.session_state.selected_statuses = []
    
    # Page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Chat history for conversational analytics
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Analysis cache to avoid recomputing (structure will be data-driven)
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    
    # Refresh trigger
    if "refresh_triggered" not in st.session_state:
        st.session_state.refresh_triggered = False
        
    # Data visibility tracking
    if "data_visible_in_dashboard" not in st.session_state:
        st.session_state.data_visible_in_dashboard = st.session_state.get('data_loaded', False)
        
    # Analysis tracking
    if "analysis_triggered" not in st.session_state:
        st.session_state.analysis_triggered = False
    
    if "analysis_start_time" not in st.session_state:
        st.session_state.analysis_start_time = None
        
    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = {
            "time_series_complete": False,
            "distribution_complete": False,
            "forecast_complete": False,
            "automation_complete": False,
            "resource_complete": False
        }
    if "analysis_triggered" not in st.session_state:
        st.session_state.analysis_triggered = False

    if "analysis_start_time" not in st.session_state:
        st.session_state.analysis_start_time = None


def get_filtered_data(
    date_column: Optional[str] = None,
    category_column: Optional[str] = None,
    priority_column: Optional[str] = None,
    status_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Returns a filtered version of the processed data based on current filters in session state.
    
    Args:
        date_column: Column containing date information
        category_column: Column containing category information
        priority_column: Column containing priority information
        status_column: Column containing status information
        
    Returns:
        Filtered DataFrame
    """
    if "processed_data" not in st.session_state or st.session_state.processed_data is None:
        return None
    
    filtered_data = st.session_state.processed_data.copy()
    
    # Apply date filter if specified
    if date_column and date_column in filtered_data.columns and "date_range" in st.session_state:
        start_date, end_date = st.session_state.date_range
        
        # Convert column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(filtered_data[date_column]):
            try:
                filtered_data[date_column] = pd.to_datetime(filtered_data[date_column], errors='coerce')
            except:
                # If conversion fails, skip date filtering
                pass
        
        # Filter by date range
        try:
            filtered_data = filtered_data[
                (filtered_data[date_column].dt.date >= start_date) & 
                (filtered_data[date_column].dt.date <= end_date)
            ]
        except:
            # If filtering fails, skip it
            pass
    
    # Apply category filter if specified
    if (category_column and category_column in filtered_data.columns and 
        "selected_categories" in st.session_state and st.session_state.selected_categories):
        filtered_data = filtered_data[filtered_data[category_column].isin(st.session_state.selected_categories)]
    
    # Apply priority filter if specified
    if (priority_column and priority_column in filtered_data.columns and 
        "selected_priorities" in st.session_state and st.session_state.selected_priorities):
        filtered_data = filtered_data[filtered_data[priority_column].isin(st.session_state.selected_priorities)]
    
    # Apply status filter if specified
    if (status_column and status_column in filtered_data.columns and 
        "selected_statuses" in st.session_state and st.session_state.selected_statuses):
        filtered_data = filtered_data[filtered_data[status_column].isin(st.session_state.selected_statuses)]
    
    return filtered_data


def update_date_filter(start_date: datetime, end_date: datetime) -> None:
    """
    Updates the date filter in session state.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
    """
    st.session_state.date_range = (start_date, end_date)
    
    # Clear analysis cache for date-dependent analyses
    if "analysis_cache" in st.session_state:
        # Only clear cache entries that depend on date filtering
        date_dependent_keys = [
            key for key in st.session_state.analysis_cache.keys()
            if "trend" in key or "time_series" in key or "forecast" in key
        ]
        
        for key in date_dependent_keys:
            if key in st.session_state.analysis_cache:
                del st.session_state.analysis_cache[key]


def update_category_filter(selected_categories: List[str]) -> None:
    """
    Updates the category filter in session state.
    
    Args:
        selected_categories: List of selected categories
    """
    st.session_state.selected_categories = selected_categories
    
    # Clear analysis cache for category-dependent analyses
    if "analysis_cache" in st.session_state:
        # Only clear cache entries that depend on category filtering
        category_dependent_keys = [
            key for key in st.session_state.analysis_cache.keys()
            if "category" in key or "distribution" in key
        ]
        
        for key in category_dependent_keys:
            if key in st.session_state.analysis_cache:
                del st.session_state.analysis_cache[key]


def update_priority_filter(selected_priorities: List[str]) -> None:
    """
    Updates the priority filter in session state.
    
    Args:
        selected_priorities: List of selected priorities
    """
    st.session_state.selected_priorities = selected_priorities
    
    # Clear analysis cache for priority-dependent analyses
    if "analysis_cache" in st.session_state:
        # Only clear cache entries that depend on priority filtering
        priority_dependent_keys = [
            key for key in st.session_state.analysis_cache.keys()
            if "priority" in key or "severity" in key
        ]
        
        for key in priority_dependent_keys:
            if key in st.session_state.analysis_cache:
                del st.session_state.analysis_cache[key]


def update_status_filter(selected_statuses: List[str]) -> None:
    """
    Updates the status filter in session state.
    
    Args:
        selected_statuses: List of selected statuses
    """
    st.session_state.selected_statuses = selected_statuses
    
    # Clear analysis cache for status-dependent analyses
    if "analysis_cache" in st.session_state:
        # Only clear cache entries that depend on status filtering
        status_dependent_keys = [
            key for key in st.session_state.analysis_cache.keys()
            if "status" in key or "state" in key
        ]
        
        for key in status_dependent_keys:
            if key in st.session_state.analysis_cache:
                del st.session_state.analysis_cache[key]


def cache_analysis_result(key: str, result: Any) -> None:
    """
    Caches an analysis result to avoid recomputing it.
    
    Args:
        key: Unique key for the analysis result
        result: Analysis result to cache
    """
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    
    st.session_state.analysis_cache[key] = {
        "result": result,
        "timestamp": datetime.now()
    }


def get_cached_analysis_result(
    key: str, 
    max_age_minutes: Optional[int] = 5
) -> Optional[Any]:
    """
    Retrieves a cached analysis result if it exists and is not expired.
    
    Args:
        key: Unique key for the analysis result
        max_age_minutes: Maximum age of the cached result in minutes
        
    Returns:
        Cached result if available and not expired, None otherwise
    """
    if ("analysis_cache" not in st.session_state or 
        key not in st.session_state.analysis_cache):
        return None
    
    cache_entry = st.session_state.analysis_cache[key]
    
    # Check if cache is expired
    if max_age_minutes is not None:
        cache_age = datetime.now() - cache_entry["timestamp"]
        if cache_age > timedelta(minutes=max_age_minutes):
            # Cache is expired
            return None
    
    return cache_entry["result"]


def clear_analysis_cache() -> None:
    """
    Clears all cached analysis results.
    """
    st.session_state.analysis_cache = {}


def clear_chat_history() -> None:
    """
    Clears the chat history for conversational analytics.
    """
    st.session_state.chat_history = []


def add_analysis_result(key: str, result: Any) -> None:
    """
    Adds an analysis result to the session state.
    
    Args:
        key: Unique key for the analysis result
        result: Analysis result to add
    """
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    
    st.session_state.analysis_results[key] = result


def get_analysis_result(key: str) -> Optional[Any]:
    """
    Retrieves an analysis result from the session state.
    
    Args:
        key: Unique key for the analysis result
        
    Returns:
        Analysis result if available, None otherwise
    """
    if "analysis_results" not in st.session_state:
        return None
    
    return st.session_state.analysis_results.get(key)


def reset_session_state() -> None:
    """
    Resets all session state variables to their default values.
    """
    # Save current data source settings
    data_source_path = st.session_state.get("data_source_path", "")
    uploaded_file = st.session_state.get("uploaded_file", None)
    data_source_option = st.session_state.get("data_source_option", "Upload File")
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize with defaults
    initialize_session_state()
    
    # Restore data source settings
    st.session_state.data_source_path = data_source_path
    st.session_state.uploaded_file = uploaded_file
    st.session_state.data_source_option = data_source_option