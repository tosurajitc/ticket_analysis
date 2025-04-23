"""
Interactive visualization elements for the incident management analytics dashboard.

This module provides reusable interactive components that can be used across
different pages of the application. These components are designed to be
data-driven, ensuring that insights are generated based on the actual ticket data
rather than predefined templates.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import numpy as np
from datetime import datetime, timedelta


def create_date_filter(key_prefix: str = "date") -> Tuple[datetime, datetime]:
    """
    Creates a date range filter widget.
    
    Args:
        key_prefix: Prefix for the widget keys to ensure uniqueness
        
    Returns:
        Tuple of selected start and end dates
    """
    col1, col2 = st.columns(2)
    
    # Default to last 30 days if no data is available
    default_end = datetime.now()
    default_start = default_end - timedelta(days=30)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            key=f"{key_prefix}_start"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            key=f"{key_prefix}_end"
        )
    
    return start_date, end_date


def create_multiselect_filter(
    data: pd.DataFrame,
    column: str,
    label: Optional[str] = None,
    key_suffix: str = "",
    max_selections: Optional[int] = None
) -> List:
    """
    Creates a multiselect filter for a dataframe column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to filter on
        label: Display label for the filter (defaults to column name if None)
        key_suffix: Suffix for the widget key to ensure uniqueness
        max_selections: Maximum number of items that can be selected
        
    Returns:
        List of selected values
    """
    if data is None or data.empty:
        st.warning(f"No data available for {label or column} filter")
        return []
    
    if column not in data.columns:
        st.warning(f"Column '{column}' not found in the data")
        return []
    
    # Get unique values and sort them
    options = sorted(data[column].dropna().unique().tolist())
    
    if not options:
        st.warning(f"No values available for {label or column} filter")
        return []
    
    # Set a reasonable default for max_selections if not specified
    if max_selections is None:
        max_selections = min(len(options), 10)
    
    # Create the multiselect with a default of no selection
    selected = st.multiselect(
        label or column,
        options=options,
        default=[],
        key=f"multiselect_{column}_{key_suffix}",
        max_selections=max_selections
    )
    
    return selected


def create_interactive_time_series(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Time Series Analysis",
    height: int = 400,
    category_column: Optional[str] = None,
    aggregation: str = "count",
    rolling_window: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive time series visualization.
    
    Args:
        data: DataFrame containing time series data
        date_column: Column containing date values
        value_column: Column to measure
        title: Chart title
        height: Chart height in pixels
        category_column: Optional column for grouping data
        aggregation: Aggregation function ('count', 'sum', 'mean', etc.)
        rolling_window: Optional rolling window size for smoothing
    
    Returns:
        Plotly figure object
    """
    if data is None or data.empty:
        st.warning("Insufficient data for time series visualization")
        # Return an empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available for time series visualization",
            height=height
        )
        return fig
    
    # Ensure date column is datetime type
    if date_column not in data.columns:
        st.warning(f"Date column '{date_column}' not found in data")
        return go.Figure()
    
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        
        # Drop rows with NA dates
        data = data.dropna(subset=[date_column])
        
        if data.empty:
            st.warning("No valid dates found in the data")
            return go.Figure()
            
        # Set up time series DataFrame
        if category_column and category_column in data.columns:
            # Group by date and category
            if aggregation == "count":
                df_grouped = data.groupby([
                    pd.Grouper(key=date_column, freq='D'),
                    category_column
                ]).size().reset_index(name='value')
            else:
                if value_column not in data.columns:
                    st.warning(f"Value column '{value_column}' not found in data")
                    return go.Figure()
                    
                df_grouped = data.groupby([
                    pd.Grouper(key=date_column, freq='D'),
                    category_column
                ])[value_column].agg(aggregation).reset_index(name='value')
            
            # Apply rolling window if specified
            if rolling_window:
                # We need to apply rolling window per category
                categories = df_grouped[category_column].unique()
                dfs = []
                
                for category in categories:
                    df_cat = df_grouped[df_grouped[category_column] == category].copy()
                    df_cat['value'] = df_cat['value'].rolling(window=rolling_window, min_periods=1).mean()
                    dfs.append(df_cat)
                
                df_grouped = pd.concat(dfs)
            
            fig = px.line(
                df_grouped,
                x=date_column,
                y='value',
                color=category_column,
                title=title
            )
        else:
            # Group by just date
            if aggregation == "count":
                df_grouped = data.groupby(
                    pd.Grouper(key=date_column, freq='D')
                ).size().reset_index(name='value')
            else:
                if value_column not in data.columns:
                    st.warning(f"Value column '{value_column}' not found in data")
                    return go.Figure()
                    
                df_grouped = data.groupby(
                    pd.Grouper(key=date_column, freq='D')
                )[value_column].agg(aggregation).reset_index(name='value')
            
            # Apply rolling window if specified
            if rolling_window:
                df_grouped['value'] = df_grouped['value'].rolling(
                    window=rolling_window,
                    min_periods=1
                ).mean()
            
            fig = px.line(
                df_grouped,
                x=date_column,
                y='value',
                title=title
            )
        
        # Enhance the figure
        fig.update_layout(
            height=height,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=f"{aggregation.capitalize()} of {value_column}",
            legend_title=category_column if category_column else None
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating time series: {str(e)}")
        return go.Figure()


def create_interactive_distribution(
    data: pd.DataFrame,
    column: str,
    title: str = "Distribution Analysis",
    height: int = 400,
    chart_type: str = "histogram",
    color_column: Optional[str] = None,
    bin_size: Optional[int] = None,
    top_n: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive distribution visualization (histogram or bar chart).
    
    Args:
        data: DataFrame containing data
        column: Column to visualize
        title: Chart title
        height: Chart height in pixels
        chart_type: Either "histogram" or "bar"
        color_column: Optional column for grouping/coloring
        bin_size: Optional bin size for histogram
        top_n: Optional limit to top N categories for bar charts

    Returns:
        Plotly figure object
    """
    if data is None or data.empty:
        st.warning("Insufficient data for distribution visualization")
        # Return an empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available for distribution visualization",
            height=height
        )
        return fig
    
    if column not in data.columns:
        st.warning(f"Column '{column}' not found in data")
        return go.Figure()
    
    try:
        if chart_type == "histogram":
            # Histogram for numerical data
            fig = px.histogram(
                data,
                x=column,
                color=color_column if color_column in data.columns else None,
                title=title,
                nbins=bin_size
            )

        elif chart_type == "bar":
            # For categorical data, count occurrences
            if color_column and color_column in data.columns:
                # Group by both columns
                grouped = data.groupby([column, color_column]).size().reset_index(name='count')
                
                # Option to limit to top N categories
                if top_n:
                    # Identify top categories
                    top_categories = data[column].value_counts().nlargest(top_n).index
                    grouped = grouped[grouped[column].isin(top_categories)]
                
                fig = px.bar(
                    grouped,
                    x=column,
                    y='count',
                    color=color_column,
                    title=title
                )
            else:
                # Simple count by single column
                counted = data[column].value_counts().reset_index()
                counted.columns = [column, 'count']
                
                # Option to limit to top N categories
                if top_n:
                    counted = counted.nlargest(top_n, 'count')
                
                fig = px.bar(
                    counted,
                    x=column,
                    y='count',
                    title=title
                )
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return go.Figure()
        
        # Enhance the figure
        fig.update_layout(
            height=height,
            xaxis_title=column,
            yaxis_title="Count",
            legend_title=color_column if color_column else None
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")
        return go.Figure()


def create_correlation_heatmap(
    data: pd.DataFrame,
    columns: List[str] = None,
    title: str = "Correlation Analysis",
    height: int = 500,
    colorscale: str = "RdBu_r"
) -> go.Figure:
    """
    Creates a correlation heatmap for numerical columns.
    
    Args:
        data: DataFrame containing data
        columns: List of columns to include (None for all numerical)
        title: Chart title
        height: Chart height in pixels
        colorscale: Plotly colorscale to use
    
    Returns:
        Plotly figure object
    """
    if data is None or data.empty:
        st.warning("Insufficient data for correlation analysis")
        fig = go.Figure()
        fig.update_layout(
            title="No data available for correlation analysis",
            height=height
        )
        return fig
    
    try:
        # Select only numerical columns if columns not specified
        if columns is None:
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) < 2:
                st.warning("Not enough numerical columns for correlation analysis")
                return go.Figure()
            df_corr = data[numeric_columns].corr()
        else:
            # Filter only existing columns
            valid_columns = [col for col in columns if col in data.columns]
            if len(valid_columns) < 2:
                st.warning("Not enough valid columns for correlation analysis")
                return go.Figure()
            
            # Convert to numeric where possible
            df_numeric = data[valid_columns].apply(pd.to_numeric, errors='coerce')
            df_corr = df_numeric.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns,
            y=df_corr.columns,
            colorscale=colorscale,
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()


def create_data_table(
    data: pd.DataFrame, 
    columns: List[str] = None,
    page_size: int = 10,
    title: str = "Data Table"
) -> None:
    """
    Creates an interactive data table with pagination.
    
    Args:
        data: DataFrame to display
        columns: List of columns to include (None for all)
        page_size: Number of rows per page
        title: Table title
        
    Returns:
        None (displays table directly)
    """
    if data is None or data.empty:
        st.warning("No data available for display")
        return
    
    st.subheader(title)
    
    # Filter columns if specified
    if columns:
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            st.warning("No valid columns specified")
            return
        display_data = data[valid_columns]
    else:
        display_data = data
    
    # Add pagination controls
    total_rows = len(display_data)
    total_pages = (total_rows + page_size - 1) // page_size
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if "page_number" not in st.session_state:
                st.session_state.page_number = 0
                
            if st.button("Previous", key=f"prev_{title}", 
                         disabled=(st.session_state.page_number <= 0)):
                st.session_state.page_number -= 1
                
        with col2:
            st.write(f"Page {st.session_state.page_number + 1} of {total_pages}")
            
        with col3:
            if st.button("Next", key=f"next_{title}", 
                         disabled=(st.session_state.page_number >= total_pages - 1)):
                st.session_state.page_number += 1
        
        # Get current page data
        start_idx = st.session_state.page_number * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_data = display_data.iloc[start_idx:end_idx]
        
        st.dataframe(page_data, use_container_width=True)
    else:
        # Show all data if it fits on one page
        st.dataframe(display_data, use_container_width=True)


def create_download_button(
    data: pd.DataFrame,
    filename: str = "data_export.csv",
    button_text: str = "Download Data",
    key: str = "download_button"
) -> None:
    """
    Creates a download button for exporting data.
    
    Args:
        data: DataFrame to download
        filename: Name of the downloaded file
        button_text: Text to display on the button
        key: Unique key for the button
        
    Returns:
        None (displays button directly)
    """
    if data is None or data.empty:
        st.warning("No data available for download")
        return
    
    try:
        # Determine file type and prepare accordingly
        file_ext = filename.split('.')[-1].lower()
        
        if file_ext == 'csv':
            output = data.to_csv(index=False)
            mime = "text/csv"
        elif file_ext in ['xlsx', 'xls']:
            # For Excel files, we need to use BytesIO
            import io
            buffer = io.BytesIO()
            data.to_excel(buffer, index=False)
            buffer.seek(0)
            output = buffer
            mime = "application/vnd.ms-excel"
        elif file_ext == 'json':
            output = data.to_json(orient='records')
            mime = "application/json"
        else:
            # Default to CSV
            output = data.to_csv(index=False)
            mime = "text/csv"
            filename = f"{filename.split('.')[0]}.csv"
        
        st.download_button(
            label=button_text,
            data=output,
            file_name=filename,
            mime=mime,
            key=key
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")


def create_anomaly_chart(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    anomalies_mask: np.ndarray,
    title: str = "Anomaly Detection",
    height: int = 400
) -> go.Figure:
    """
    Creates a visualization highlighting anomalies in time series data.
    
    Args:
        data: DataFrame containing time series data
        date_column: Column containing date values
        value_column: Column with values to analyze
        anomalies_mask: Boolean mask indicating anomaly points (True for anomalies)
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure object
    """
    if data is None or data.empty:
        st.warning("Insufficient data for anomaly detection visualization")
        fig = go.Figure()
        fig.update_layout(
            title="No data available for anomaly detection",
            height=height
        )
        return fig
    
    if date_column not in data.columns or value_column not in data.columns:
        st.warning(f"Required columns not found in data")
        return go.Figure()
    
    if len(anomalies_mask) != len(data):
        st.warning("Anomaly mask length doesn't match data length")
        return go.Figure()
    
    try:
        # Create a copy of data with anomaly flag
        plot_data = data.copy()
        plot_data['is_anomaly'] = anomalies_mask
        
        # Create figure with two traces
        fig = go.Figure()
        
        # Add normal points
        normal_data = plot_data[~plot_data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data[date_column],
            y=normal_data[value_column],
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue'),
            marker=dict(size=5)
        ))
        
        # Add anomaly points
        anomaly_data = plot_data[plot_data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_data[date_column],
            y=anomaly_data[value_column],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='red',
                size=8,
                symbol='circle',
                line=dict(width=1, color='black')
            )
        ))
        
        # Enhance the figure
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Date",
            yaxis_title=value_column,
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating anomaly chart: {str(e)}")
        return go.Figure()


def create_trend_comparison(
    data: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: str = "Trend Comparison",
    height: int = 400,
    normalize: bool = False
) -> go.Figure:
    """
    Creates a visualization comparing multiple trends over time.
    
    Args:
        data: DataFrame containing time series data
        date_column: Column containing date values
        value_columns: List of columns to compare
        title: Chart title
        height: Chart height in pixels
        normalize: Whether to normalize values for better comparison
    
    Returns:
        Plotly figure object
    """
    if data is None or data.empty:
        st.warning("Insufficient data for trend comparison")
        fig = go.Figure()
        fig.update_layout(
            title="No data available for trend comparison",
            height=height
        )
        return fig
    
    if date_column not in data.columns:
        st.warning(f"Date column '{date_column}' not found in data")
        return go.Figure()
    
    # Validate value columns
    valid_columns = [col for col in value_columns if col in data.columns]
    if not valid_columns:
        st.warning("No valid value columns found in data")
        return go.Figure()
    
    try:
        # Create plot data
        plot_data = data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(plot_data[date_column]):
            plot_data[date_column] = pd.to_datetime(plot_data[date_column], errors='coerce')
        
        # Drop NA dates
        plot_data = plot_data.dropna(subset=[date_column])
        
        if normalize:
            # Normalize each column to 0-1 range for comparison
            for col in valid_columns:
                min_val = plot_data[col].min()
                max_val = plot_data[col].max()
                if max_val > min_val:  # Avoid division by zero
                    plot_data[f"{col}_norm"] = (plot_data[col] - min_val) / (max_val - min_val)
                else:
                    plot_data[f"{col}_norm"] = 0
            
            # Update column list to use normalized columns
            valid_columns = [f"{col}_norm" for col in valid_columns]
        
        # Create figure
        fig = go.Figure()
        
        for col in valid_columns:
            display_name = col.replace('_norm', '') if normalize else col
            fig.add_trace(go.Scatter(
                x=plot_data[date_column],
                y=plot_data[col],
                mode='lines',
                name=display_name
            ))
        
        # Enhance the figure
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Date",
            yaxis_title="Value" if normalize else "Values",
            legend_title="Metrics"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trend comparison: {str(e)}")
        return go.Figure()


def interactive_kpi_cards(
    metrics: Dict[str, Union[int, float, str]],
    columns: int = 3,
    delta_columns: Optional[Dict[str, Union[int, float]]] = None,
    prefix: Optional[Dict[str, str]] = None,
    suffix: Optional[Dict[str, str]] = None
) -> None:
    """
    Displays KPI metrics as interactive cards.
    
    Args:
        metrics: Dictionary of metric name and value pairs
        columns: Number of columns in the grid
        delta_columns: Optional dictionary of metric name and delta value pairs
        prefix: Optional dictionary of metric name and prefix string pairs
        suffix: Optional dictionary of metric name and suffix string pairs
        
    Returns:
        None (displays cards directly)
    """
    if not metrics:
        st.warning("No metrics available to display")
        return
    
    # Default empty dictionaries
    delta_columns = delta_columns or {}
    prefix = prefix or {}
    suffix = suffix or {}
    
    # Create columns
    cols = st.columns(columns)
    
    # Distribute metrics across columns
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            delta = delta_columns.get(name)
            pre = prefix.get(name, "")
            suf = suffix.get(name, "")
            
            # If delta is provided, use metric with delta
            if delta is not None:
                st.metric(
                    label=name,
                    value=f"{pre}{value}{suf}",
                    delta=delta
                )
            else:
                st.metric(
                    label=name,
                    value=f"{pre}{value}{suf}"
                )


def resource_allocation_chart(
    resources: Dict[str, Dict[str, Union[int, float]]],
    title: str = "Resource Allocation",
    height: int = 400
) -> go.Figure:
    """
    Creates a stacked bar chart for resource allocation visualization.
    
    Args:
        resources: Dictionary of resource types and their allocation
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure object
    """
    if not resources:
        st.warning("No resource allocation data available")
        fig = go.Figure()
        fig.update_layout(
            title="No data available for resource allocation",
            height=height
        )
        return fig
    
    try:
        # Convert to dataframe for plotting
        data_list = []
        for resource_type, allocations in resources.items():
            for category, value in allocations.items():
                data_list.append({
                    'Resource': resource_type,
                    'Category': category,
                    'Value': value
                })
        
        df = pd.DataFrame(data_list)
        
        # Create stacked bar chart
        fig = px.bar(
            df,
            x='Resource',
            y='Value',
            color='Category',
            title=title,
            height=height
        )
        
        # Enhance the figure
        fig.update_layout(
            xaxis_title="Resource Type",
            yaxis_title="Allocation",
            legend_title="Category"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating resource allocation chart: {str(e)}")
        return go.Figure()