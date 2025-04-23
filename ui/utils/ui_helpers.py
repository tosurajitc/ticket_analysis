"""
UI helper functions for the incident management analytics dashboard.

This module provides utility functions to assist with UI rendering, formatting,
and display operations across the application. These helpers ensure consistent
UI presentation while maintaining the principle that all insights should be
derived from actual data, not predefined content.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re


def format_duration(duration: float, unit: str = "hours") -> str:
    """
    Formats a duration value into a human-readable string.
    
    Args:
        duration: Duration value
        unit: Unit of the duration ('hours', 'days', 'minutes', 'seconds')
        
    Returns:
        Formatted duration string
    """
    if pd.isna(duration):
        return "N/A"
    
    if duration < 0:
        return "Invalid duration"
    
    if unit == "hours":
        if duration < 1:
            # Convert to minutes
            minutes = duration * 60
            return f"{minutes:.1f} minutes"
        elif duration < 24:
            return f"{duration:.1f} hours"
        else:
            # Convert to days
            days = duration / 24
            return f"{days:.1f} days"
    elif unit == "days":
        if duration < 1:
            # Convert to hours
            hours = duration * 24
            return f"{hours:.1f} hours"
        else:
            return f"{duration:.1f} days"
    elif unit == "minutes":
        if duration < 1:
            # Convert to seconds
            seconds = duration * 60
            return f"{seconds:.1f} seconds"
        elif duration < 60:
            return f"{duration:.1f} minutes"
        else:
            # Convert to hours
            hours = duration / 60
            return f"{hours:.1f} hours"
    elif unit == "seconds":
        if duration < 60:
            return f"{duration:.1f} seconds"
        else:
            # Convert to minutes
            minutes = duration / 60
            return f"{minutes:.1f} minutes"
    
    return f"{duration:.1f} {unit}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Formats a value as a percentage string.
    
    Args:
        value: Value to format (0-1 range)
        decimal_places: Number of decimal places to include
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    
    # If value is not in 0-1 range, convert it
    if value > 1:
        value = value / 100
    
    return f"{value * 100:.{decimal_places}f}%"


def format_count(count: Union[int, float]) -> str:
    """
    Formats a count value with appropriate suffixes for readability.
    
    Args:
        count: Count value to format
        
    Returns:
        Formatted count string
    """
    if pd.isna(count):
        return "N/A"
    
    # Ensure count is an integer
    count = int(count)
    
    if count < 1000:
        return str(count)
    elif count < 1000000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1000000:.1f}M"


def format_timestamp(timestamp: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M") -> str:
    """
    Formats a timestamp into a human-readable string.
    
    Args:
        timestamp: Timestamp to format
        format_str: Format string for the output
        
    Returns:
        Formatted timestamp string
    """
    if pd.isna(timestamp):
        return "N/A"
    
    if isinstance(timestamp, str):
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            return timestamp
    
    try:
        return timestamp.strftime(format_str)
    except:
        return "Invalid timestamp"


def apply_theme_to_figure(fig: go.Figure) -> go.Figure:
    """
    Applies a consistent theme to a Plotly figure.
    
    Args:
        fig: Plotly figure to style
        
    Returns:
        Styled Plotly figure
    """
    # Apply consistent theming to the figure
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="rgba(240, 240, 240, 0.8)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3",
                  "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    )
    
    # Style axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(211, 211, 211, 0.5)",
        showline=True,
        linewidth=1,
        linecolor="rgba(0, 0, 0, 0.3)"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(211, 211, 211, 0.5)",
        showline=True,
        linewidth=1,
        linecolor="rgba(0, 0, 0, 0.3)"
    )
    
    return fig


def display_data_status(data: pd.DataFrame, is_sufficient: bool) -> None:
    """
    Displays data loading status information in the sidebar.
    
    Args:
        data: DataFrame containing the loaded data
        is_sufficient: Whether the data is sufficient for analysis
    """
    if data is None or data.empty:
        st.sidebar.error("No data loaded")
        return
    
    # Display data summary
    st.sidebar.success(f"Data loaded: {len(data)} incidents")
    
    # Find date range if timestamp column exists
    date_col = next((col for col in ["timestamp", "created_at", "date", "incident_date"] 
                      if col in data.columns), None)
    
    if date_col:
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                dates = pd.to_datetime(data[date_col], errors='coerce')
            else:
                dates = data[date_col]
            
            # Filter out NaT values
            dates = dates.dropna()
            
            if not dates.empty:
                min_date = dates.min()
                max_date = dates.max()
                date_range = (max_date - min_date).days
                
                st.sidebar.info(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range} days)")
        except:
            pass
    
    # Display data sufficiency status
    if is_sufficient:
        st.sidebar.success("Data is sufficient for analysis")
    else:
        st.sidebar.warning("Limited data available - some analyses may be restricted")


def generate_column_profile(data, column):
    """
    Generate a profile for a specific column.
    
    Args:
        data: DataFrame containing the incident data
        column: Column name to profile
        
    Returns:
        Dictionary with column profile
    """
    profile = {
        "name": column,
        "type": str(data[column].dtype),
        "non_null_count": int(data[column].count()),
        "null_count": int(data[column].isna().sum()),
        "null_percentage": float(data[column].isna().mean() * 100),
        "unique_count": int(data[column].nunique())
    }
    
    # Handle different data types
    
    # First check if it's a datetime column
    try:
        if pd.api.types.is_datetime64_any_dtype(data[column]) or pd.to_datetime(data[column], errors='coerce').notna().any():
            # For datetime columns
            date_column = data[column]
            if not pd.api.types.is_datetime64_any_dtype(date_column):
                date_column = pd.to_datetime(date_column, errors='coerce')
            
            valid_dates = date_column.dropna()
            if not valid_dates.empty:
                profile["min_value"] = valid_dates.min()
                profile["max_value"] = valid_dates.max()
                profile["date_range_days"] = (profile["max_value"] - profile["min_value"]).days
                profile["is_datetime"] = True
    except:
        pass
    
    # Check numeric columns
    if pd.api.types.is_numeric_dtype(data[column]):
        # FIX: Split the boolean check into two separate variables
        is_bool_type = pd.api.types.is_bool_dtype(data[column])
        is_binary_values = data[column].isin([0, 1, True, False]).all()
        
        if is_bool_type:
            profile["is_boolean"] = True
            profile["true_count"] = int(data[column].sum())
            profile["false_count"] = int(len(data[column]) - data[column].sum())
            profile["true_percentage"] = float(data[column].mean() * 100)
        
        # Then check if it contains only 0 and 1 (separate from the boolean check)
        elif is_binary_values:
            profile["is_binary"] = True
            profile["true_count"] = int(data[column].sum())
            profile["false_count"] = int(len(data[column]) - data[column].sum())
            profile["true_percentage"] = float(data[column].mean() * 100)
        else:
            # Regular numeric column
            profile["min_value"] = float(data[column].min())
            profile["max_value"] = float(data[column].max())
            profile["mean"] = float(data[column].mean())
            profile["median"] = float(data[column].median())
            profile["std"] = float(data[column].std())
    
    # For categorical columns
    elif pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_categorical_dtype(data[column]):
        value_counts = data[column].value_counts()
        
        # Store most common values
        top_values = value_counts.head(5).to_dict()
        profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        
        if not value_counts.empty:
            profile["most_common_value"] = str(value_counts.index[0])
            profile["most_common_count"] = int(value_counts.iloc[0])
            
            # Calculate diversity
            profile["diversity"] = float(data[column].nunique() / len(data))
    
    return profile


def generate_data_profile(data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generates a comprehensive profile of the dataset to inform
    dynamic visualizations and analyses.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        Dictionary containing dataset profile information
    """
    if data is None or data.empty:
        return {}
    
    # Create profile for each column
    profile = {}
    for column in data.columns:
        try:
            profile[column] = generate_column_profile(data, column)
        except Exception as e:
            # Add graceful error handling
            profile[column] = {
                "name": column,
                "error": f"Error profiling column: {str(e)}",
                "type": "unknown"
            }
    
    return profile


def detect_key_columns(data: pd.DataFrame) -> Dict[str, str]:
    """
    Detects key columns in the dataset based on column names.
    This allows the application to adapt to different dataset schemas.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        Dictionary mapping column roles to column names
    """
    if data is None or data.empty:
        return {}
    
    key_columns = {}
    
    # Incident ID column
    id_candidates = ["incident_id", "id", "ticket_id", "case_id", "issue_id"]
    for candidate in id_candidates:
        if candidate in data.columns:
            key_columns["incident_id"] = candidate
            break
    
    # Timestamp column
    timestamp_candidates = ["timestamp", "created_at", "date", "incident_date", "opened_at", "created_date"]
    for candidate in timestamp_candidates:
        if candidate in data.columns:
            key_columns["timestamp"] = candidate
            break
    
    # Priority column
    priority_candidates = ["priority", "severity", "impact", "urgency"]
    for candidate in priority_candidates:
        if candidate in data.columns:
            key_columns["priority"] = candidate
            break
    
    # Status column
    status_candidates = ["status", "state", "incident_status"]
    for candidate in status_candidates:
        if candidate in data.columns:
            key_columns["status"] = candidate
            break
    
    # Category column
    category_candidates = ["category", "type", "incident_type", "classification"]
    for candidate in category_candidates:
        if candidate in data.columns:
            key_columns["category"] = candidate
            break
    
    # Resolution time column
    resolution_candidates = ["resolution_time", "time_to_resolve", "duration", "resolution_duration"]
    for candidate in resolution_candidates:
        if candidate in data.columns:
            key_columns["resolution_time"] = candidate
            break
    
    # Assignment column
    assignment_candidates = ["assigned_to", "owner", "assignee", "resolver", "team"]
    for candidate in assignment_candidates:
        if candidate in data.columns:
            key_columns["assigned_to"] = candidate
            break
    
    # Description column
    description_candidates = ["description", "summary", "details", "incident_description"]
    for candidate in description_candidates:
        if candidate in data.columns:
            key_columns["description"] = candidate
            break
    
    # SLA column
    sla_candidates = ["sla_compliance", "sla_met", "within_sla", "breached"]
    for candidate in sla_candidates:
        if candidate in data.columns:
            key_columns["sla"] = candidate
            break
    
    return key_columns


def sanitize_column_name(name: str) -> str:
    """
    Sanitizes a column name for use in IDs and keys.
    
    Args:
        name: Column name to sanitize
        
    Returns:
        Sanitized column name
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = '_' + sanitized
    
    return sanitized.lower()


def format_insight_text(text: str, data: Optional[pd.DataFrame] = None) -> str:
    """
    Formats insight text, potentially replacing placeholders with actual values from data.
    This ensures insights are data-driven, not hardcoded.
    
    Args:
        text: Text to format
        data: Optional DataFrame containing data for placeholder replacement
        
    Returns:
        Formatted text
    """
    if data is None or data.empty:
        return text
    
    # Replace placeholders with actual values if they exist in the format {column_name:aggregate}
    # For example: {resolution_time:mean} will be replaced with the mean of the resolution_time column
    
    # Find all placeholders with regex
    placeholders = re.findall(r'\{([^{}:]+):([^{}]+)\}', text)
    
    # Process each placeholder
    for column, aggregate in placeholders:
        if column in data.columns:
            try:
                if aggregate == 'mean':
                    value = data[column].mean()
                elif aggregate == 'median':
                    value = data[column].median()
                elif aggregate == 'min':
                    value = data[column].min()
                elif aggregate == 'max':
                    value = data[column].max()
                elif aggregate == 'sum':
                    value = data[column].sum()
                elif aggregate == 'count':
                    value = len(data)
                elif aggregate == 'nunique':
                    value = data[column].nunique()
                elif aggregate.startswith('percentile_'):
                    # Extract percentile value (e.g., percentile_95 -> 95)
                    percentile = int(aggregate.split('_')[1])
                    value = np.percentile(data[column].dropna(), percentile)
                else:
                    # Unknown aggregate, leave placeholder as is
                    continue
                
                # Format the value based on type
                if isinstance(value, (int, np.integer)):
                    formatted_value = format_count(value)
                elif isinstance(value, (float, np.floating)):
                    # Check if it's likely a percentage
                    if column.lower().endswith('rate') or column.lower().endswith('percentage'):
                        formatted_value = format_percentage(value)
                    # Check if it's likely a duration
                    elif any(dur in column.lower() for dur in ['time', 'duration', 'period']):
                        formatted_value = format_duration(value)
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                # Replace placeholder with formatted value
                text = text.replace(f"{{{column}:{aggregate}}}", formatted_value)
            except:
                # If any error occurs, leave placeholder as is
                continue
    
    return text


def get_color_for_trend(trend: float) -> str:
    """
    Determines the appropriate color for a trend value.
    
    Args:
        trend: Trend value (positive or negative)
        
    Returns:
        Color string (red, green, or gray)
    """
    if pd.isna(trend):
        return "gray"
    
    if trend > 0:
        return "green"
    elif trend < 0:
        return "red"
    else:
        return "gray"


def get_icon_for_trend(trend: float, positive_is_good: bool = True) -> str:
    """
    Determines the appropriate icon for a trend value.
    
    Args:
        trend: Trend value (positive or negative)
        positive_is_good: Whether a positive trend is good (green) or bad (red)
        
    Returns:
        Icon string (↑, ↓, or -)
    """
    if pd.isna(trend):
        return "-"
    
    if trend > 0:
        return "↑" if positive_is_good else "↓"
    elif trend < 0:
        return "↓" if positive_is_good else "↑"
    else:
        return "-"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncates text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def render_help_tooltip(text: str, icon: str = "ℹ️") -> None:
    """
    Renders a help tooltip with the given text.
    
    Args:
        text: Help text to display
        icon: Icon to use for the tooltip
    """
    st.markdown(f"<span title='{text}'>{icon}</span>", unsafe_allow_html=True)