"""
Insights page for the incident management analytics dashboard.

This module renders the AI Insights page, which provides data-driven insights
about incident patterns, root causes, and trends. All insights are dynamically
generated based on the actual incident data, with no predefined content.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visualization.interactive_elements import (
    create_date_filter,
    create_multiselect_filter,
    create_data_table,
    interactive_kpi_cards
)

from ui.components import (
    page_header,
    data_insufficiency_message,
    insight_card,
    dynamic_insight_card
)

from ui.utils.ui_helpers import (
    detect_key_columns,
    generate_data_profile,
    format_duration,
    format_percentage,
    get_color_for_trend
)

from analysis.insights_generator import (
    generate_overview_insights,
    generate_insights  # Use the existing generate_insights for trend insights
)

from analysis.root_cause_analyzer import RootCauseAnalyzer


def render_insights_page(
    data: pd.DataFrame,
    config: Any,
    is_data_sufficient: bool
) -> None:
    """
    Renders the AI-generated insights page.
    
    Args:
        data: DataFrame containing the incident data
        config: Application configuration
        is_data_sufficient: Whether data is sufficient for full analysis
    """
    # Render page header
    page_header(
        "AI-Generated Insights",
        "Data-driven analysis of incident patterns and trends",
        icon="🧠"
    )
    
    # Check if data is available
    if data is None or data.empty:
        st.warning("No data available for analysis. Please load data using the sidebar controls.")
        return
    
    # Check if data is sufficient for AI analysis
    if not is_data_sufficient:
        st.warning(
            "The current dataset may not be sufficient for comprehensive AI analysis. "
            "Some insights may be limited or unavailable. Consider providing more data "
            "for better results."
        )
    
    # Detect key columns for flexibility with different data schemas
    key_columns = detect_key_columns(data)
    
    # Create date filter for the insights page
    date_col = key_columns.get("timestamp")
    
    if date_col:
        col1, col2 = st.columns([2, 1])
        with col1:
            start_date, end_date = create_date_filter(key_prefix="insights")
        
        # Apply date filter to data
        try:
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                date_data = pd.to_datetime(data[date_col], errors='coerce')
                data = data.copy()
                data[date_col] = date_data
            
            filtered_data = data[
                (data[date_col].dt.date >= start_date) & 
                (data[date_col].dt.date <= end_date)
            ]
            
            with col2:
                st.info(f"Filtered data: {len(filtered_data)} incidents")
        except:
            filtered_data = data
            with col2:
                st.warning("Date filtering failed - showing all data")
    else:
        filtered_data = data
        st.info("No date column detected - showing all data")
    
    # Check if filtered data is sufficient
    if filtered_data.empty or len(filtered_data) < 5:
        st.warning(
            "Insufficient data after filtering. Please adjust your filters or load more data."
        )
        return
    
    # Create tabs for different types of insights
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview Insights", "Root Cause Analysis", 
        "Trend Insights", "Seasonal Patterns"
    ])
    
    # Initialize RootCauseAnalyzer
    analyzer = RootCauseAnalyzer()
    
    with tab1:
        st.subheader("Key Incident Insights")
        
        # Generate overview insights
        with st.spinner("Analyzing incident data..."):
            try:
                insights = generate_overview_insights(filtered_data, key_columns)
                
                if insights and len(insights) > 0:
                    # Display each insight in a card
                    for i, insight in enumerate(insights):
                        if 'title' in insight and 'content' in insight:
                            # Determine severity based on insight content
                            severity = insight.get('severity', 'info')
                            
                            # Extract metrics if available
                            metrics = insight.get('metrics', {})
                            
                            # Display the insight card
                            insight_card(
                                insight['title'],
                                insight['content'],
                                metrics=metrics,
                                severity=severity,
                                collapsible=False
                            )
                else:
                    st.info(
                        "No significant insights detected in the current data selection. "
                        "Try expanding your date range or providing more incident data."
                    )
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
        # Show categorical distribution insights
        st.subheader("Category Distribution Insights")
        
        # Find categorical columns for analysis
        category_col = key_columns.get("category")
        if not category_col:
            # Try to find alternative categorical columns
            for col in ["type", "incident_type", "classification"]:
                if col in filtered_data.columns:
                    category_col = col
                    break
        
        if category_col and category_col in filtered_data.columns:
            try:
                # Calculate category distribution
                category_counts = filtered_data[category_col].value_counts()
                total_incidents = len(filtered_data)
                
                if len(category_counts) > 0:
                    # Create visualization
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title=f"Incident Distribution by {category_col.capitalize()}"
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights about category distribution
                    st.subheader("Top Categories")
                    
                    # Show top categories and their statistics
                    top_n = min(5, len(category_counts))
                    top_categories = category_counts.head(top_n)
                    
                    for category, count in top_categories.items():
                        percentage = (count / total_incidents) * 100
                        
                        # Calculate additional statistics for this category
                        category_data = filtered_data[filtered_data[category_col] == category]
                        
                        # Calculate average resolution time if available
                        resolution_text = ""
                        resolution_col = key_columns.get("resolution_time")
                        if resolution_col and resolution_col in category_data.columns:
                            try:
                                avg_resolution = category_data[resolution_col].mean()
                                if not pd.isna(avg_resolution) and avg_resolution > 0:
                                    # Compare to overall average
                                    overall_avg = filtered_data[resolution_col].mean()
                                    pct_diff = ((avg_resolution / overall_avg) - 1) * 100
                                    
                                    diff_text = ""
                                    if abs(pct_diff) > 5:  # Only show significant differences
                                        diff_text = f" ({pct_diff:+.1f}% vs. overall)"
                                    
                                    resolution_text = f"Average resolution time: {format_duration(avg_resolution)}{diff_text}"
                            except:
                                pass
                        
                        # Create insight card for each category
                        insight_content = (
                            f"Represents {percentage:.1f}% of all incidents ({count} incidents). "
                            f"{resolution_text}"
                        )
                        
                        insight_card(
                            category,
                            insight_content,
                            severity="info"
                        )
                else:
                    st.info(f"No {category_col} data available for analysis.")
            except Exception as e:
                st.error(f"Error analyzing category distribution: {str(e)}")
        else:
            st.info("Category data is not available for analysis.")
    
    with tab2:
        st.subheader("Root Cause Analysis")
        
        # Check if we have description data for analysis
        description_col = key_columns.get("description")
        category_col = key_columns.get("category")
        
        if description_col and description_col in filtered_data.columns:
            # Ensure we have enough textual data for analysis
            valid_descriptions = filtered_data[description_col].dropna()
            
            if len(valid_descriptions) < 10:
                st.warning(
                    "Insufficient description data for root cause analysis. "
                    "Please provide more incident data with detailed descriptions."
                )
            else:
                # Perform root cause analysis using RootCauseAnalyzer methods
                with st.spinner("Analyzing root causes..."):
                    try:
                        # Extract topics
                        topics_result = analyzer.extract_topics(
                            filtered_data,
                            text_columns=[description_col]
                        )
                        
                        # Find correlations
                        correlations_result = analyzer.find_correlations(
                            filtered_data,
                            target_col='resolution_time' if 'resolution_time' in filtered_data.columns else None,
                            feature_cols=[category_col] if category_col else None
                        )
                        
                        # Analyze recurring patterns
                        patterns_result = analyzer.analyze_recurring_patterns(
                            filtered_data,
                            timestamp_col=date_col,
                            text_columns=[description_col],
                            category_col=category_col
                        )
                        
                        # Get root cause insights
                        root_cause_insights = analyzer.get_root_cause_insights(
                            topics_result, 
                            correlations_result, 
                            patterns_result
                        )
                        
                        if root_cause_insights:
                            for insight in root_cause_insights:
                                # Determine severity
                                severity = "info"
                                if insight.get('type') == 'error':
                                    severity = "error"
                                elif insight.get('data', {}).get('frequency', 0) > 25:
                                    severity = "warning"
                                
                                # Create insight card
                                insight_card(
                                    insight.get('title', 'Root Cause Insight'),
                                    insight.get('message', ''),
                                    severity=severity
                                )
                        else:
                            st.info(
                                "No significant root causes identified in the current data selection. "
                                "Try expanding your date range or providing more detailed incident descriptions."
                            )
                    except Exception as e:
                        st.error(f"Error analyzing root causes: {str(e)}")
        else:
            st.warning(
                "Root cause analysis requires incident description data which is not available in the dataset. "
                "Please provide incident data with description fields."
            )
    
    with tab3:
        st.subheader("Trend Insights")
        
        # Check if we have timestamp data for trend analysis
        if date_col and date_col in filtered_data.columns:
            # Create trend analysis period selector
            trend_period = st.radio(
                "Trend Analysis Period",
                ["Weekly", "Monthly", "Quarterly"],
                horizontal=True
            )
            
            # Convert period to pandas frequency string
            if trend_period == "Weekly":
                freq = "W"
                min_periods = 4  # Need at least 4 weeks
            elif trend_period == "Monthly":
                freq = "M"
                min_periods = 3  # Need at least 3 months
            else:  # Quarterly
                freq = "Q"
                min_periods = 2  # Need at least 2 quarters
            
            # Create trend data
            try:
                # Ensure timestamp is datetime
                trend_data = filtered_data.copy()
                if not pd.api.types.is_datetime64_any_dtype(trend_data[date_col]):
                    trend_data[date_col] = pd.to_datetime(trend_data[date_col], errors='coerce')
                
                # Group by period
                try:
                    # Create time series data with reset_index to avoid duplicate index issues
                    period_counts = trend_data.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='count')
                    
                    # Convert back to Series if needed for existing code
                    period_series = pd.Series(period_counts['count'].values, index=period_counts[date_col])
                    period_counts = period_series
                except Exception as e:
                    # Alternative approach if the first one fails
                    # Create date-only column to avoid time component issues
                    trend_data = trend_data.copy()
                    trend_data['_date_only'] = trend_data[date_col].dt.floor(freq)
                    
                    # Use value_counts which doesn't rely on index
                    date_counts = trend_data['_date_only'].value_counts().sort_index()
                    period_counts = date_counts.resample(freq).sum()
                
                # Check if we have enough periods for trend analysis
                if len(period_counts) < min_periods:
                    st.warning(
                        f"Insufficient data for {trend_period.lower()} trend analysis. "
                        f"Need at least {min_periods} {trend_period.lower()} periods."
                    )
                else:
                    # Create trend visualization
                    fig = px.line(
                        x=period_counts.index,
                        y=period_counts.values,
                        markers=True,
                        labels={"x": "Period", "y": "Incident Count"},
                        title=f"{trend_period} Incident Count Trend"
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate trend insights using the existing generate_insights function
                    trend_insights = generate_insights(
                        trend_data,
                        key_columns
                    )
                    
                    if trend_insights and len(trend_insights) > 0:
                        # Display trend insights
                        for insight in trend_insights:
                            if 'title' in insight and 'content' in insight:
                                # Determine severity based on trend direction and magnitude
                                severity = "info"
                                
                                # Display the insight card
                                insight_card(
                                    insight['title'],
                                    insight['content'],
                                    severity=severity
                                )
                    else:
                        st.info(
                            "No significant trends detected in the current data selection. "
                            "Try expanding your date range or providing more incident data."
                        )
                    
                    # Show category trends if category data is available
                    if category_col and category_col in filtered_data.columns:
                        st.subheader(f"{trend_period} Category Trends")
                        
                        # Get top categories
                        top_categories = filtered_data[category_col].value_counts().head(5).index.tolist()
                        
                        # Create dataframe for visualization
                        category_trend_data = []
                        for category in top_categories:
                            # Filter data for this category
                            cat_data = trend_data[trend_data[category_col] == category]
                            
                            # Group by period
                            cat_counts = cat_data.groupby(pd.Grouper(key=date_col, freq=freq)).size()
                            
                            # Add to visualization data
                            for period, count in cat_counts.items():
                                category_trend_data.append({
                                    'Period': period,
                                    'Category': category,
                                    'Count': count
                                })
                        
                        # Create dataframe
                        cat_trend_df = pd.DataFrame(category_trend_data)
                        
                        if not cat_trend_df.empty:
                            # Create visualization
                            fig = px.line(
                                cat_trend_df,
                                x='Period',
                                y='Count',
                                color='Category',
                                markers=True,
                                title=f"Top 5 Categories - {trend_period} Trends"
                            )
                            
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error performing trend analysis: {str(e)}")
        else:
            st.warning("Trend analysis requires timestamp data which is not available in the dataset.")
    
    with tab4:
        st.subheader("Seasonal Patterns")
        
        # Check if we have timestamp data for seasonal analysis
        if date_col and date_col in filtered_data.columns:
            # Check if we have enough data for seasonal analysis
            try:
                # Ensure timestamp is datetime
                seasonal_data = filtered_data.copy()
                if not pd.api.types.is_datetime64_any_dtype(seasonal_data[date_col]):
                    seasonal_data[date_col] = pd.to_datetime(seasonal_data[date_col], errors='coerce')
                
                # Check date range
                min_date = seasonal_data[date_col].min()
                max_date = seasonal_data[date_col].max()
                date_range = (max_date - min_date).days
                
                if date_range < 60:  # Need at least 60 days for seasonal analysis
                    st.warning(
                        "Insufficient data for seasonal analysis. "
                        "Need at least 60 days of data for meaningful seasonal patterns."
                    )
                else:
                    # Create tabs for different seasonal views
                    season_tab1, season_tab2, season_tab3 = st.tabs([
                        "Day of Week", "Time of Day", "Monthly"
                    ])
                    

                    with season_tab1:
                        # Day of week analysis
                        try:
                            # Extract day of week
                            seasonal_data['day_of_week'] = seasonal_data[date_col].dt.day_name()
                            
                            # Order days correctly
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            
                            # Count incidents by day of week - with error handling for duplicate keys
                            dow_counts_raw = seasonal_data['day_of_week'].value_counts()
                            
                            # Create a properly ordered Series with zeros for missing days
                            dow_counts = pd.Series(0, index=day_order)
                            for day, count in dow_counts_raw.items():
                                if day in dow_counts.index:
                                    dow_counts[day] = count
                            
                            # Create visualization
                            fig = px.bar(
                                x=dow_counts.index,
                                y=dow_counts.values,
                                labels={"x": "Day of Week", "y": "Incident Count"},
                                title="Incident Distribution by Day of Week"
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Generate insights
                            with st.spinner("Analyzing day of week patterns..."):
                                # Find peak day
                                peak_day = dow_counts.idxmax()
                                peak_count = dow_counts.max()
                                peak_pct = (peak_count / dow_counts.sum()) * 100
                                
                                # Find lowest day
                                min_day = dow_counts.idxmin()
                                min_count = dow_counts.min()
                                min_pct = (min_count / dow_counts.sum()) * 100
                                
                                # Calculate weekday vs weekend
                                weekday_mask = seasonal_data['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                                weekday_count = weekday_mask.sum()
                                weekend_count = (~weekday_mask).sum()
                                
                                weekday_daily_avg = weekday_count / 5  # 5 weekdays
                                weekend_daily_avg = weekend_count / 2  # 2 weekend days
                                
                                # Create insight cards
                                insight_card(
                                    "Peak Incident Day",
                                    f"{peak_day} has the highest incident volume with {peak_count} incidents ({peak_pct:.1f}% of total).",
                                    severity="info"
                                )
                                
                                # Only show weekday vs weekend if there's a significant difference
                                if abs(weekday_daily_avg - weekend_daily_avg) / weekday_daily_avg > 0.2:  # 20% difference
                                    ratio = weekday_daily_avg / weekend_daily_avg if weekend_daily_avg > 0 else 0
                                    insight_card(
                                        "Weekday vs Weekend",
                                        f"Weekdays average {weekday_daily_avg:.1f} incidents per day, compared to {weekend_daily_avg:.1f} " +
                                        f"for weekends. There are {ratio:.1f}x more incidents on weekdays than weekends.",
                                        severity="info"
                                    )
                        except Exception as e:
                            st.error(f"Error analyzing day of week patterns: {str(e)}")
                    
                    with season_tab2:
                        # Time of day analysis
                        try:
                            # Extract hour of day if timestamp includes time
                            if pd.api.types.is_datetime64_any_dtype(seasonal_data[date_col]):
                                # Check if we have time information (not just dates)
                                times = seasonal_data[date_col].dt.time
                                all_midnight = all(t.hour == 0 and t.minute == 0 for t in times if not pd.isna(t))
                                
                                if not all_midnight:
                                    # We have actual time data
                                    seasonal_data['hour'] = seasonal_data[date_col].dt.hour
                                    
                                    # Group by hour
                                    hour_counts = seasonal_data.groupby('hour').size()
                                    
                                    # Create visualization
                                    fig = px.bar(
                                        x=hour_counts.index,
                                        y=hour_counts.values,
                                        labels={"x": "Hour of Day (24h)", "y": "Incident Count"},
                                        title="Incident Distribution by Hour of Day"
                                    )
                                    
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Generate insights
                                    with st.spinner("Analyzing time of day patterns..."):
                                        # Define periods
                                        morning = hour_counts.loc[6:11].sum()
                                        afternoon = hour_counts.loc[12:17].sum()
                                        evening = hour_counts.loc[18:23].sum()
                                        night = hour_counts.loc[0:5].sum()
                                        
                                        # Calculate percentages
                                        total = hour_counts.sum()
                                        morning_pct = (morning / total) * 100
                                        afternoon_pct = (afternoon / total) * 100
                                        evening_pct = (evening / total) * 100
                                        night_pct = (night / total) * 100
                                        
                                        # Create period distribution visualization
                                        periods = ['Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)', 'Night (0-5)']
                                        values = [morning, afternoon, evening, night]
                                        
                                        fig = px.pie(
                                            values=values,
                                            names=periods,
                                            title="Incident Distribution by Time Period"
                                        )
                                        
                                        fig.update_traces(textposition='inside', textinfo='percent+label')
                                        fig.update_layout(height=400)
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Find peak period
                                        period_data = {
                                            'Morning (6-11)': morning_pct,
                                            'Afternoon (12-17)': afternoon_pct,
                                            'Evening (18-23)': evening_pct,
                                            'Night (0-5)': night_pct
                                        }
                                        
                                        peak_period = max(period_data, key=period_data.get)
                                        peak_period_pct = period_data[peak_period]
                                        
                                        # Create insight card
                                        insight_card(
                                            "Peak Incident Period",
                                            f"The {peak_period} hours have the highest incident volume with {peak_period_pct:.1f}% of total incidents.",
                                            severity="info"
                                        )
                                else:
                                    st.info("Time of day analysis is not available because timestamps only contain dates without time information.")
                            else:
                                st.info("Time of day analysis is not available because timestamps only contain dates without time information.")
                        except Exception as e:
                            st.error(f"Error analyzing time of day patterns: {str(e)}")
                    

                    with season_tab3:
                        # Monthly analysis
                        try:
                            # Extract month
                            seasonal_data['month'] = seasonal_data[date_col].dt.month_name()
                            
                            # Order months correctly
                            month_order = [
                                'January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 'December'
                            ]
                            
                            # Count incidents by month - with error handling for duplicate keys
                            month_counts_raw = seasonal_data['month'].value_counts()
                            
                            # Create a properly ordered Series with zeros for missing months
                            month_counts = pd.Series(0, index=month_order)
                            for month, count in month_counts_raw.items():
                                if month in month_counts.index:
                                    month_counts[month] = count
                            
                            # Check if we have enough months
                            unique_months = month_counts[month_counts > 0].count()
                            
                            if unique_months < 3:
                                st.info(
                                    f"Monthly analysis requires data spanning at least 3 months. " +
                                    f"Current data only spans {unique_months} months."
                                )
                            else:
                                # Create visualization
                                fig = px.bar(
                                    x=month_counts.index,
                                    y=month_counts.values,
                                    labels={"x": "Month", "y": "Incident Count"},
                                    title="Incident Distribution by Month"
                                )
                                
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Generate seasonal insights
                                with st.spinner("Analyzing seasonal patterns..."):
                                    try:
                                        seasonal_insights = generate_insights(
                                            seasonal_data,
                                            key_columns
                                        )
                                        
                                        if seasonal_insights and len(seasonal_insights) > 0:
                                            # Display seasonal insights
                                            for insight in seasonal_insights:
                                                if 'title' in insight and 'content' in insight:
                                                    # Determine severity based on insight content
                                                    severity = insight.get('severity', 'info')
                                                    
                                                    # Display the insight card
                                                    insight_card(
                                                        insight['title'],
                                                        insight['content'],
                                                        severity=severity
                                                    )
                                        else:
                                            st.info(
                                                "No significant seasonal patterns detected in the current data selection. "
                                                "Try providing data with a longer time span for better seasonal analysis."
                                            )
                                    except Exception as e:
                                        st.error(f"Error generating seasonal insights: {str(e)}")
                        except Exception as e:
                            st.error(f"Error analyzing monthly patterns: {str(e)}")

            except Exception as e:
                st.error(f"Error performing seasonal analysis: {str(e)}")
        else:
            st.warning("Seasonal analysis requires timestamp data which is not available in the dataset.")
    
    # Display additional information
    st.divider()
    st.info(
        "This page provides AI-generated insights based on your incident data. "
        "All insights are derived dynamically from the data patterns, with no predefined content. "
        "The quality and depth of insights will improve with more comprehensive incident data."
    )