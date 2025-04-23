"""
Reusable UI components for the incident management analytics dashboard.

This module provides a set of UI components that can be used across different
pages of the application. Components are designed to be data-driven, ensuring
that insights are generated based on the actual ticket data rather than using
predefined templates or insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
from datetime import datetime, timedelta
import json

# Import visualization components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.interactive_elements import (
    create_date_filter,
    create_multiselect_filter,
    create_interactive_time_series,
    create_data_table,
    interactive_kpi_cards
)


def page_header(
    title: str,
    description: str = "",
    icon: str = None
) -> None:
    """
    Creates a standardized page header with title, description, and optional icon.

    Args:
        title: Page title
        description: Page description
        icon: Optional emoji icon to display
    """
    if icon:
        st.header(f"{icon} {title}")
    else:
        st.header(title)
    
    if description:
        st.markdown(description)
    
    st.divider()


def data_insufficiency_message(
    component_name: str = "This component",
    min_rows: Optional[int] = None,
    required_columns: Optional[List[str]] = None
) -> None:
    """
    Displays a standardized message when data is insufficient for analysis.

    Args:
        component_name: Name of the component requiring data
        min_rows: Optional minimum number of rows needed
        required_columns: Optional list of required columns
    """
    message = f"{component_name} requires more data to generate meaningful insights."
    
    if min_rows is not None:
        message += f" At least {min_rows} data points are needed."
    
    if required_columns is not None and len(required_columns) > 0:
        columns_str = ", ".join([f"`{col}`" for col in required_columns])
        message += f" The following columns are required: {columns_str}."
    
    st.warning(message)


def error_card(
    message: str,
    suggestion: Optional[str] = None
) -> None:
    """
    Displays an error message with optional suggestion.

    Args:
        message: Error message to display
        suggestion: Optional suggestion to resolve the error
    """
    st.error(message)
    
    if suggestion:
        st.info(f"Suggestion: {suggestion}")


def info_card(
    title: str,
    content: str,
    icon: str = "ℹ️"
) -> None:
    """
    Displays an information card with title and content.

    Args:
        title: Card title
        content: Card content
        icon: Icon to display
    """
    st.info(f"{icon} **{title}**  \n{content}")


def insight_card(
    title: str,
    content: str,
    metrics: Optional[Dict[str, Union[int, float, str]]] = None,
    severity: str = "info",
    collapsible: bool = False
) -> None:
    """
    Displays an insight card with title, content, and optional metrics.
    
    Args:
        title: Insight title
        content: Insight content
        metrics: Optional dictionary of metrics to display
        severity: Card severity/color (info, warning, success, error)
        collapsible: Whether the card should be collapsible
    """
    if collapsible:
        with st.expander(title):
            if severity == "info":
                st.info(content)
            elif severity == "warning":
                st.warning(content)
            elif severity == "success":
                st.success(content)
            elif severity == "error":
                st.error(content)
            
            # Display metrics if provided
            if metrics and len(metrics) > 0:
                st.divider()
                cols = st.columns(min(len(metrics), 3))
                for i, (name, value) in enumerate(metrics.items()):
                    with cols[i % len(cols)]:
                        st.metric(name, value)
    else:
        if severity == "info":
            st.info(f"**{title}**  \n{content}")
        elif severity == "warning":
            st.warning(f"**{title}**  \n{content}")
        elif severity == "success":
            st.success(f"**{title}**  \n{content}")
        elif severity == "error":
            st.error(f"**{title}**  \n{content}")
        
        # Display metrics if provided
        if metrics and len(metrics) > 0:
            cols = st.columns(min(len(metrics), 3))
            for i, (name, value) in enumerate(metrics.items()):
                with cols[i % len(cols)]:
                    st.metric(name, value)


def data_summary_section(data: pd.DataFrame) -> None:
    """
    Creates a summary section for the provided data.
    
    Args:
        data: DataFrame to summarize
    """
    if data is None or data.empty:
        data_insufficiency_message("Data summary section")
        return
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Incidents", len(data))
    
    with col2:
        # Calculate date range if timestamp is available
        if "timestamp" in data.columns:
            try:
                dates = pd.to_datetime(data["timestamp"])
                date_range = (dates.max() - dates.min()).days
                st.metric("Date Range", f"{date_range} days")
            except:
                st.metric("Date Range", "Unknown")
        else:
            st.metric("Date Range", "N/A")
    
    with col3:
        # Count unique values in a category column if available
        category_col = next((col for col in ["category", "type", "incident_type"] 
                            if col in data.columns), None)
        if category_col:
            unique_categories = data[category_col].nunique()
            st.metric("Unique Categories", unique_categories)
        else:
            st.metric("Unique Categories", "N/A")
    
    with col4:
        # Calculate average resolution time if available
        resolution_col = next((col for col in ["resolution_time", "time_to_resolve", "duration"] 
                              if col in data.columns), None)
        if resolution_col and resolution_col in data.columns:
            try:
                avg_resolution = data[resolution_col].mean()
                if avg_resolution < 1:
                    # Convert to hours if less than 1 day
                    avg_resolution = avg_resolution * 24
                    st.metric("Avg Resolution Time", f"{avg_resolution:.1f} hours")
                else:
                    st.metric("Avg Resolution Time", f"{avg_resolution:.1f} days")
            except:
                st.metric("Avg Resolution Time", "Unknown")
        else:
            st.metric("Avg Resolution Time", "N/A")


def incident_trend_section(
    data: pd.DataFrame,
    title: str = "Incident Trend Analysis",
    description: str = "Analysis of incident trends over time"
) -> None:
    """
    Creates a section for analyzing incident trends over time.
    
    Args:
        data: DataFrame containing incident data
        title: Section title
        description: Section description
    """
    if data is None or data.empty:
        data_insufficiency_message(
            "Incident trend analysis",
            min_rows=10,
            required_columns=["timestamp"]
        )
        return
    
    # Check if timestamp column exists
    date_col = next((col for col in ["timestamp", "created_at", "date", "incident_date"] 
                     if col in data.columns), None)
    
    if not date_col:
        error_card(
            "Timestamp column not found in data",
            "Please ensure your data includes a column with timestamp information"
        )
        return
    
    # Convert to datetime if needed
    try:
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data = data.copy()
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
        # Remove rows with invalid dates
        data = data.dropna(subset=[date_col])
    except Exception as e:
        error_card(
            f"Error converting timestamps: {str(e)}",
            "Please ensure your timestamp data is in a valid format"
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Create filters row
    col1, col2 = st.columns(2)
    
    with col1:
        # Date filter
        start_date, end_date = create_date_filter()
        
    with col2:
        # Category filter if available
        category_col = next((col for col in ["category", "type", "incident_type"] 
                            if col in data.columns), None)
        
        selected_categories = []
        if category_col:
            selected_categories = create_multiselect_filter(
                data, category_col, f"Filter by {category_col.capitalize()}"
            )
    
    # Filter data based on selections
    filtered_data = data.copy()
    
    # Apply date filter
    filtered_data = filtered_data[
        (filtered_data[date_col].dt.date >= start_date) & 
        (filtered_data[date_col].dt.date <= end_date)
    ]
    
    # Apply category filter if selected
    if category_col and selected_categories:
        filtered_data = filtered_data[filtered_data[category_col].isin(selected_categories)]
    
    # Check if we have sufficient data after filtering
    if len(filtered_data) < 5:
        data_insufficiency_message(
            "Trend analysis",
            min_rows=5
        )
        return
    
    # Create tabs for different trend views
    tab1, tab2, tab3 = st.tabs(["Overall Trend", "By Priority", "By Status"])
    
    with tab1:
        # Overall incident trend
        fig = create_interactive_time_series(
            filtered_data,
            date_col,
            "incident_id",  # Using incident_id for counting
            title="Incident Volume Over Time",
            aggregation="count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Trend by priority if available
        priority_col = next((col for col in ["priority", "severity", "impact"] 
                           if col in data.columns), None)
        
        if priority_col:
            fig = create_interactive_time_series(
                filtered_data,
                date_col,
                "incident_id",  # Using incident_id for counting
                title="Incidents by Priority",
                category_column=priority_col,
                aggregation="count"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            info_card(
                "Missing Priority Information",
                "Priority data is not available in the dataset."
            )
    
    with tab3:
        # Trend by status if available
        status_col = next((col for col in ["status", "state", "incident_status"] 
                          if col in data.columns), None)
        
        if status_col:
            fig = create_interactive_time_series(
                filtered_data,
                date_col,
                "incident_id",  # Using incident_id for counting
                title="Incidents by Status",
                category_column=status_col,
                aggregation="count"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            info_card(
                "Missing Status Information",
                "Status data is not available in the dataset."
            )


def incident_distribution_section(
    data: pd.DataFrame,
    title: str = "Incident Distribution Analysis",
    description: str = "Analysis of incident distribution across different dimensions"
) -> None:
    """
    Creates a section for analyzing incident distribution across dimensions.
    
    Args:
        data: DataFrame containing incident data
        title: Section title
        description: Section description
    """
    if data is None or data.empty:
        data_insufficiency_message(
            "Incident distribution analysis",
            min_rows=10
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Find available categorical columns
    categorical_columns = []
    for col in ["category", "type", "priority", "status", "assigned_to", "team", "source"]:
        if col in data.columns and data[col].nunique() < 50:  # Limit to columns with reasonable cardinality
            categorical_columns.append(col)
    
    if not categorical_columns:
        info_card(
            "No Categorical Columns Found",
            "This analysis requires categorical columns such as category, priority, status, etc."
        )
        return
    
    # Create selection for dimension
    selected_dimension = st.selectbox(
        "Select Dimension for Analysis",
        options=categorical_columns,
        index=0 if categorical_columns else None
    )
    
    if not selected_dimension:
        return
    
    # Create tabs for different distribution views
    tab1, tab2 = st.tabs(["Count Distribution", "Relative Distribution"])
    
    with tab1:
        # Count distribution
        try:
            counts = data[selected_dimension].value_counts()
            
            if len(counts) > 15:
                # If too many categories, limit to top 15
                counts = counts.nlargest(15)
                fig = px.bar(
                    counts,
                    x=counts.index,
                    y=counts.values,
                    title=f"Incident Count by {selected_dimension.capitalize()} (Top 15)"
                )
            else:
                fig = px.bar(
                    counts,
                    x=counts.index,
                    y=counts.values,
                    title=f"Incident Count by {selected_dimension.capitalize()}"
                )
            
            fig.update_layout(
                xaxis_title=selected_dimension.capitalize(),
                yaxis_title="Number of Incidents"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            error_card(
                f"Error creating distribution chart: {str(e)}",
                "Please try a different dimension"
            )
    
    with tab2:
        # Relative distribution (percentage)
        try:
            percentages = data[selected_dimension].value_counts(normalize=True) * 100
            
            if len(percentages) > 15:
                # If too many categories, limit to top 15
                percentages = percentages.nlargest(15)
                fig = px.pie(
                    values=percentages.values,
                    names=percentages.index,
                    title=f"Incident Distribution by {selected_dimension.capitalize()} (Top 15)"
                )
            else:
                fig = px.pie(
                    values=percentages.values,
                    names=percentages.index,
                    title=f"Incident Distribution by {selected_dimension.capitalize()}"
                )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            error_card(
                f"Error creating distribution chart: {str(e)}",
                "Please try a different dimension"
            )


def resolution_analysis_section(
    data: pd.DataFrame,
    title: str = "Resolution Time Analysis",
    description: str = "Analysis of incident resolution times across different dimensions"
) -> None:
    """
    Creates a section for analyzing incident resolution times.
    
    Args:
        data: DataFrame containing incident data
        title: Section title
        description: Section description
    """
    # Find resolution time column
    resolution_col = next((col for col in ["resolution_time", "time_to_resolve", "duration"] 
                           if col in data.columns), None)
    
    if not resolution_col:
        info_card(
            "Missing Resolution Time Data",
            "This analysis requires resolution time data which is not available in the dataset."
        )
        return
    
    # Check if we have sufficient data
    if data is None or data.empty or len(data) < 10:
        data_insufficiency_message(
            "Resolution time analysis",
            min_rows=10,
            required_columns=[resolution_col]
        )
        return
    
    # Create a clean copy of data for analysis
    analysis_data = data.copy()
    
    # Ensure resolution time is numeric
    try:
        analysis_data[resolution_col] = pd.to_numeric(analysis_data[resolution_col], errors='coerce')
        analysis_data = analysis_data.dropna(subset=[resolution_col])
    except Exception as e:
        error_card(
            f"Error processing resolution time data: {str(e)}",
            "Please ensure resolution time data is in a valid numeric format"
        )
        return
    
    if len(analysis_data) < 5:
        data_insufficiency_message(
            "Resolution time analysis",
            min_rows=5
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Create tabs for different resolution time views
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Distribution", "By Dimension"])
    
    with tab1:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_resolution = analysis_data[resolution_col].mean()
            st.metric("Average", f"{avg_resolution:.2f}")
        
        with col2:
            median_resolution = analysis_data[resolution_col].median()
            st.metric("Median", f"{median_resolution:.2f}")
        
        with col3:
            min_resolution = analysis_data[resolution_col].min()
            st.metric("Minimum", f"{min_resolution:.2f}")
        
        with col4:
            max_resolution = analysis_data[resolution_col].max()
            st.metric("Maximum", f"{max_resolution:.2f}")
        
        # Show percentiles
        st.subheader("Resolution Time Percentiles")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(analysis_data[resolution_col].dropna(), percentiles)
        
        # Create a DataFrame for the percentiles
        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Resolution Time': percentile_values
        })
        
        # Create a bar chart for percentiles
        fig = px.bar(
            percentile_df,
            x='Percentile',
            y='Resolution Time',
            title="Resolution Time Percentiles"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Resolution time distribution
        fig = px.histogram(
            analysis_data,
            x=resolution_col,
            nbins=20,
            title="Resolution Time Distribution"
        )
        
        fig.update_layout(
            xaxis_title="Resolution Time",
            yaxis_title="Number of Incidents"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show box plot for better visualization of distribution
        fig = px.box(
            analysis_data,
            y=resolution_col,
            title="Resolution Time Box Plot"
        )
        
        fig.update_layout(
            yaxis_title="Resolution Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Resolution time by dimension
        # Find available categorical columns
        categorical_columns = []
        for col in ["category", "type", "priority", "status", "assigned_to", "team", "source"]:
            if col in analysis_data.columns and analysis_data[col].nunique() < 50:
                categorical_columns.append(col)
        
        if not categorical_columns:
            info_card(
                "No Categorical Columns Found",
                "This analysis requires categorical columns such as category, priority, status, etc."
            )
            return
        
        # Create selection for dimension
        selected_dimension = st.selectbox(
            "Select Dimension",
            options=categorical_columns,
            index=0 if categorical_columns else None,
            key="resolution_dimension"
        )
        
        if not selected_dimension:
            return
        
        # Resolution time by selected dimension
        fig = px.box(
            analysis_data,
            x=selected_dimension,
            y=resolution_col,
            title=f"Resolution Time by {selected_dimension.capitalize()}"
        )
        
        fig.update_layout(
            xaxis_title=selected_dimension.capitalize(),
            yaxis_title="Resolution Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show average resolution time by dimension
        avg_by_dimension = analysis_data.groupby(selected_dimension)[resolution_col].mean().reset_index()
        avg_by_dimension = avg_by_dimension.sort_values(resolution_col, ascending=False)
        
        fig = px.bar(
            avg_by_dimension,
            x=selected_dimension,
            y=resolution_col,
            title=f"Average Resolution Time by {selected_dimension.capitalize()}"
        )
        
        fig.update_layout(
            xaxis_title=selected_dimension.capitalize(),
            yaxis_title="Average Resolution Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def dynamic_insight_card(
    data: pd.DataFrame,
    insight_generator: Callable,
    title: str = "AI-Generated Insight",
    input_columns: Optional[List[str]] = None
) -> None:
    """
    Creates a card with dynamically generated insights based on the data.
    
    Args:
        data: DataFrame containing incident data
        insight_generator: Function that generates insights from the data
        title: Card title
        input_columns: List of required columns for the insight generator
    """
    if data is None or data.empty:
        data_insufficiency_message("AI insight generation")
        return
    
    # Check if required columns are available
    if input_columns:
        missing_columns = [col for col in input_columns if col not in data.columns]
        if missing_columns:
            columns_str = ", ".join([f"`{col}`" for col in missing_columns])
            error_card(
                f"Missing required columns: {columns_str}",
                "This insight requires additional data fields"
            )
            return
    
    try:
        # Generate insight
        insight = insight_generator(data)
        
        if not insight:
            info_card(
                "No Insight Available",
                "Insufficient patterns found in the data for this insight type."
            )
            return
        
        # Display insight
        st.subheader(title)
        
        if isinstance(insight, dict):
            # Handle structured insight
            if "title" in insight and "content" in insight:
                severity = insight.get("severity", "info")
                metrics = insight.get("metrics", {})
                
                insight_card(
                    insight["title"],
                    insight["content"],
                    metrics=metrics,
                    severity=severity
                )
            else:
                # Handle dictionary with multiple insights
                for key, value in insight.items():
                    if isinstance(value, dict) and "title" in value and "content" in value:
                        severity = value.get("severity", "info")
                        metrics = value.get("metrics", {})
                        
                        insight_card(
                            value["title"],
                            value["content"],
                            metrics=metrics,
                            severity=severity,
                            collapsible=True
                        )
                    else:
                        st.markdown(f"**{key}**: {value}")
        elif isinstance(insight, str):
            # Handle simple string insight
            st.markdown(insight)
        else:
            # Handle other types
            st.json(insight)
    except Exception as e:
        error_card(
            f"Error generating insight: {str(e)}",
            "Please try again or check the data format"
        )


def resource_allocation_section(
    data: pd.DataFrame,
    resource_column: str,
    metric_column: Optional[str] = None,
    title: str = "Resource Allocation Analysis",
    description: str = "Analysis of incident allocation across resources"
) -> None:
    """
    Creates a section for analyzing incident allocation across resources.
    
    Args:
        data: DataFrame containing incident data
        resource_column: Column containing resource information
        metric_column: Optional column for measuring (default is count)
        title: Section title
        description: Section description
    """
    if data is None or data.empty:
        data_insufficiency_message(
            "Resource allocation analysis",
            min_rows=10,
            required_columns=[resource_column]
        )
        return
    
    if resource_column not in data.columns:
        error_card(
            f"Resource column '{resource_column}' not found in data",
            "Please ensure your data includes resource allocation information"
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Create clean data for analysis
    analysis_data = data.dropna(subset=[resource_column]).copy()
    
    if len(analysis_data) < 5:
        data_insufficiency_message(
            "Resource allocation analysis",
            min_rows=5
        )
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Allocation Distribution", "Time Series"])
    
    with tab1:
        # Resource allocation distribution
        if metric_column and metric_column in analysis_data.columns:
            # Group by resource and sum the metric
            grouped = analysis_data.groupby(resource_column)[metric_column].sum().reset_index()
            grouped = grouped.sort_values(metric_column, ascending=False)
            
            # Limit to top 15 if too many
            if len(grouped) > 15:
                grouped = grouped.head(15)
                title_suffix = " (Top 15)"
            else:
                title_suffix = ""
            
            fig = px.bar(
                grouped,
                x=resource_column,
                y=metric_column,
                title=f"Resource Allocation by {metric_column.capitalize()}{title_suffix}"
            )
        else:
            # Count incidents per resource
            counts = analysis_data[resource_column].value_counts()
            
            # Limit to top 15 if too many
            if len(counts) > 15:
                counts = counts.head(15)
                title_suffix = " (Top 15)"
            else:
                title_suffix = ""
            
            fig = px.bar(
                counts,
                x=counts.index,
                y=counts.values,
                title=f"Resource Allocation by Count{title_suffix}"
            )
        
        fig.update_layout(
            xaxis_title=resource_column.capitalize(),
            yaxis_title=metric_column.capitalize() if metric_column else "Incident Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show as pie chart
        if metric_column and metric_column in analysis_data.columns:
            fig = px.pie(
                grouped,
                values=metric_column,
                names=resource_column,
                title=f"Resource Allocation Distribution{title_suffix}"
            )
        else:
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=f"Resource Allocation Distribution{title_suffix}"
            )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Resource allocation over time
        date_col = next((col for col in ["timestamp", "created_at", "date", "incident_date"] 
                       if col in data.columns), None)
        
        if not date_col:
            info_card(
                "Missing Timestamp Information",
                "Time series analysis requires timestamp data which is not available in the dataset."
            )
            return
        
        # Convert to datetime if needed
        try:
            if not pd.api.types.is_datetime64_any_dtype(analysis_data[date_col]):
                analysis_data[date_col] = pd.to_datetime(analysis_data[date_col], errors='coerce')
                
            # Remove rows with invalid dates
            analysis_data = analysis_data.dropna(subset=[date_col])
            
            if len(analysis_data) < 5:
                data_insufficiency_message(
                    "Resource allocation time series analysis",
                    min_rows=5
                )
                return
            
            # Find top resources to avoid cluttered chart
            top_resources = analysis_data[resource_column].value_counts().nlargest(5).index.tolist()
            
            # Filter to top resources
            filtered_data = analysis_data[analysis_data[resource_column].isin(top_resources)]
            
            # Create time series chart
            fig = create_interactive_time_series(
                filtered_data,
                date_col,
                metric_column if metric_column and metric_column in filtered_data.columns else "incident_id",
                title=f"Resource Allocation Over Time (Top 5 {resource_column.capitalize()})",
                category_column=resource_column,
                aggregation="sum" if metric_column and metric_column in filtered_data.columns else "count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow selecting specific resources
            selected_resources = st.multiselect(
                f"Select specific {resource_column}",
                options=sorted(analysis_data[resource_column].unique()),
                default=[]
            )
            
            if selected_resources:
                # Filter to selected resources
                selected_data = analysis_data[analysis_data[resource_column].isin(selected_resources)]
                
                # Create time series chart for selected resources
                fig = create_interactive_time_series(
                    selected_data,
                    date_col,
                    metric_column if metric_column and metric_column in selected_data.columns else "incident_id",
                    title=f"Resource Allocation Over Time (Selected {resource_column.capitalize()})",
                    category_column=resource_column,
                    aggregation="sum" if metric_column and metric_column in selected_data.columns else "count"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            error_card(
                f"Error creating time series chart: {str(e)}",
                "Please check your data format"
            )
        
        
def sla_performance_section(
    data: pd.DataFrame,
    sla_column: Optional[str] = None,
    sla_target: Optional[float] = None,
    title: str = "SLA Performance Analysis",
    description: str = "Analysis of SLA compliance for incidents"
) -> None:
    """
    Creates a section for analyzing SLA performance.
    
    Args:
        data: DataFrame containing incident data
        sla_column: Column indicating SLA compliance or time
        sla_target: Optional target value for SLA if using resolution time
        title: Section title
        description: Section description
    """
    if data is None or data.empty:
        data_insufficiency_message(
            "SLA performance analysis",
            min_rows=10
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Find SLA column if not provided
    if not sla_column:
        sla_column = next((col for col in ["sla_compliance", "sla_met", "within_sla", "resolution_time"] 
                          if col in data.columns), None)
        
    if not sla_column:
        info_card(
            "Missing SLA Information",
            "SLA analysis requires data about SLA compliance or resolution times which is not available in the dataset."
        )
        return
    
    # Create a clean copy of data for analysis
    analysis_data = data.copy()
    
    # Process SLA data based on column type
    is_boolean_sla = False
    
    if sla_column in analysis_data.columns:
        # Check if it's a boolean SLA compliance column
        if pd.api.types.is_bool_dtype(analysis_data[sla_column]) or analysis_data[sla_column].isin([0, 1, True, False]).all():
            is_boolean_sla = True
            
            # Ensure boolean format
            analysis_data[sla_column] = analysis_data[sla_column].astype(bool)
        elif sla_target:
            # It's a time-based column that we need to compare against a target
            try:
                analysis_data[sla_column] = pd.to_numeric(analysis_data[sla_column], errors='coerce')
                analysis_data = analysis_data.dropna(subset=[sla_column])
                
                # Create boolean SLA compliance column
                analysis_data["sla_met"] = analysis_data[sla_column] <= sla_target
                is_boolean_sla = True
                sla_column = "sla_met"
            except Exception as e:
                error_card(
                    f"Error processing SLA data: {str(e)}",
                    "Please ensure SLA data is in a valid format"
                )
                return
        else:
            # It's a numeric column but no target provided
            try:
                analysis_data[sla_column] = pd.to_numeric(analysis_data[sla_column], errors='coerce')
                analysis_data = analysis_data.dropna(subset=[sla_column])
            except Exception as e:
                error_card(
                    f"Error processing SLA data: {str(e)}",
                    "Please ensure SLA data is in a valid format"
                )
                return
    else:
        error_card(
            f"SLA column '{sla_column}' not found in data",
            "Please specify a valid SLA column"
        )
        return
    
    if len(analysis_data) < 5:
        data_insufficiency_message(
            "SLA performance analysis",
            min_rows=5
        )
        return
    
    # Create tabs for different SLA views
    if is_boolean_sla:
        # Boolean SLA compliance analysis
        tab1, tab2, tab3 = st.tabs(["Overall Compliance", "By Dimension", "Trend"])
        
        with tab1:
            # Overall SLA compliance
            compliance_rate = analysis_data[sla_column].mean() * 100
            
            # Create gauge chart for compliance rate
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=compliance_rate,
                title={"text": "SLA Compliance Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "red"},
                        {"range": [50, 80], "color": "orange"},
                        {"range": [80, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 80
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw counts
            col1, col2 = st.columns(2)
            
            with col1:
                met_count = analysis_data[sla_column].sum()
                st.metric("SLA Met", met_count)
                
            with col2:
                missed_count = len(analysis_data) - met_count
                st.metric("SLA Missed", missed_count)
        
        with tab2:
            # SLA compliance by dimension
            # Find available categorical columns
            categorical_columns = []
            for col in ["category", "type", "priority", "status", "assigned_to", "team", "source"]:
                if col in analysis_data.columns and analysis_data[col].nunique() < 50:
                    categorical_columns.append(col)
            
            if not categorical_columns:
                info_card(
                    "No Categorical Columns Found",
                    "This analysis requires categorical columns such as category, priority, etc."
                )
            else:
                # Create selection for dimension
                selected_dimension = st.selectbox(
                    "Select Dimension",
                    options=categorical_columns,
                    index=0,
                    key="sla_dimension"
                )
                
                # Group by dimension and calculate compliance rate
                grouped = analysis_data.groupby(selected_dimension)[sla_column].agg(
                    ["mean", "count"]
                ).reset_index()
                
                grouped["mean"] = grouped["mean"] * 100  # Convert to percentage
                grouped = grouped.sort_values("mean", ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    grouped,
                    x=selected_dimension,
                    y="mean",
                    text="count",
                    title=f"SLA Compliance Rate by {selected_dimension.capitalize()}",
                    labels={"mean": "Compliance Rate (%)", "count": "Incident Count"}
                )
                
                fig.update_layout(
                    xaxis_title=selected_dimension.capitalize(),
                    yaxis_title="Compliance Rate (%)"
                )
                
                fig.update_traces(texttemplate='%{text} incidents', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # SLA compliance trend over time
            date_col = next((col for col in ["timestamp", "created_at", "date", "incident_date"] 
                           if col in data.columns), None)
            
            if not date_col:
                info_card(
                    "Missing Timestamp Information",
                    "Trend analysis requires timestamp data which is not available in the dataset."
                )
            else:
                # Convert to datetime if needed
                try:
                    if not pd.api.types.is_datetime64_any_dtype(analysis_data[date_col]):
                        analysis_data[date_col] = pd.to_datetime(analysis_data[date_col], errors='coerce')
                        
                    # Remove rows with invalid dates
                    analysis_data = analysis_data.dropna(subset=[date_col])
                    
                    if len(analysis_data) < 5:
                        data_insufficiency_message(
                            "SLA trend analysis",
                            min_rows=5
                        )
                        return
                    
                    # Group by date and calculate compliance rate
                    analysis_data["month"] = analysis_data[date_col].dt.strftime("%Y-%m")
                    grouped = analysis_data.groupby("month")[sla_column].agg(
                        ["mean", "count"]
                    ).reset_index()
                    
                    grouped["mean"] = grouped["mean"] * 100  # Convert to percentage
                    
                    # Create line chart
                    fig = px.line(
                        grouped,
                        x="month",
                        y="mean",
                        title="SLA Compliance Rate Over Time",
                        labels={"mean": "Compliance Rate (%)", "month": "Month"}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Compliance Rate (%)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show incident count over time for context
                    fig = px.bar(
                        grouped,
                        x="month",
                        y="count",
                        title="Incident Count Over Time",
                        labels={"count": "Incident Count", "month": "Month"}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Incident Count"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    error_card(
                        f"Error creating trend chart: {str(e)}",
                        "Please check your data format"
                    )
    else:
        # Numeric SLA analysis (e.g., resolution time)
        tab1, tab2 = st.tabs(["Distribution", "By Dimension"])
        
        with tab1:
            # Resolution time distribution
            fig = px.histogram(
                analysis_data,
                x=sla_column,
                nbins=20,
                title=f"{sla_column.replace('_', ' ').title()} Distribution"
            )
            
            fig.update_layout(
                xaxis_title=sla_column.replace('_', ' ').title(),
                yaxis_title="Number of Incidents"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show box plot for better visualization of distribution
            fig = px.box(
                analysis_data,
                y=sla_column,
                title=f"{sla_column.replace('_', ' ').title()} Box Plot"
            )
            
            fig.update_layout(
                yaxis_title=sla_column.replace('_', ' ').title()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_val = analysis_data[sla_column].mean()
                st.metric("Average", f"{avg_val:.2f}")
            
            with col2:
                median_val = analysis_data[sla_column].median()
                st.metric("Median", f"{median_val:.2f}")
            
            with col3:
                min_val = analysis_data[sla_column].min()
                st.metric("Minimum", f"{min_val:.2f}")
            
            with col4:
                max_val = analysis_data[sla_column].max()
                st.metric("Maximum", f"{max_val:.2f}")
        
        with tab2:
            # By dimension
            # Find available categorical columns
            categorical_columns = []
            for col in ["category", "type", "priority", "status", "assigned_to", "team", "source"]:
                if col in analysis_data.columns and analysis_data[col].nunique() < 50:
                    categorical_columns.append(col)
            
            if not categorical_columns:
                info_card(
                    "No Categorical Columns Found",
                    "This analysis requires categorical columns such as category, priority, etc."
                )
            else:
                # Create selection for dimension
                selected_dimension = st.selectbox(
                    "Select Dimension",
                    options=categorical_columns,
                    index=0,
                    key="sla_numeric_dimension"
                )
                
                # Create box plot by dimension
                fig = px.box(
                    analysis_data,
                    x=selected_dimension,
                    y=sla_column,
                    title=f"{sla_column.replace('_', ' ').title()} by {selected_dimension.capitalize()}"
                )
                
                fig.update_layout(
                    xaxis_title=selected_dimension.capitalize(),
                    yaxis_title=sla_column.replace('_', ' ').title()
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show average by dimension
                grouped = analysis_data.groupby(selected_dimension)[sla_column].mean().reset_index()
                grouped = grouped.sort_values(sla_column, ascending=False)
                
                fig = px.bar(
                    grouped,
                    x=selected_dimension,
                    y=sla_column,
                    title=f"Average {sla_column.replace('_', ' ').title()} by {selected_dimension.capitalize()}"
                )
                
                fig.update_layout(
                    xaxis_title=selected_dimension.capitalize(),
                    yaxis_title=f"Average {sla_column.replace('_', ' ').title()}"
                )
                
                st.plotly_chart(fig, use_container_width=True)


def conversation_interface(
    data: pd.DataFrame,
    llm_handler: Optional[Any] = None,
    title: str = "Ask Questions About Your Data",
    description: str = "Use natural language to query and analyze your incident data"
) -> None:
    """
    Creates a conversational interface for querying incident data.
    
    Args:
        data: DataFrame containing incident data
        llm_handler: Optional handler for LLM processing
        title: Section title
        description: Section description
    """
    if data is None or data.empty:
        data_insufficiency_message(
            "Conversational analytics",
            min_rows=10
        )
        return
    
    # Create subheader
    st.subheader(title)
    st.markdown(description)
    
    # Initialize chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])
    
    # Input for user query
    user_query = st.chat_input("Ask a question about your incident data...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        st.chat_message("user").markdown(user_query)
        
        # If no LLM handler is provided, show a placeholder response
        if llm_handler is None:
            assistant_response = (
                "I'll analyze your incident data to answer this question. "
                "However, the LLM integration is not configured. "
                "Please check the configuration or try again later."
            )
        else:
            # Process query using LLM handler
            try:
                with st.spinner("Analyzing your data..."):
                    assistant_response = llm_handler.process_query(user_query, data)
            except Exception as e:
                assistant_response = f"Error processing your query: {str(e)}"
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        
        # Display assistant response
        st.chat_message("assistant").markdown(assistant_response)