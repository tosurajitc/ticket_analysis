# visualization/dashboard_builder.py
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import datetime
import json

class DashboardBuilder:
    """
    Builds interactive dashboard components for the incident management analytics application.
    Organizes multiple visualizations, metrics, and insights into a cohesive dashboard view.
    """
    
    def __init__(self):
        """Initialize the dashboard builder."""
        self.logger = logging.getLogger(__name__)
        self.dashboard_components = {}
        self.insight_cache = {}
        self.metrics_cache = {}
    
    def _validate_data(self, df: pd.DataFrame, min_records: int = 10) -> bool:
        """
        Validate if the data is sufficient for dashboard components.
        
        Args:
            df: Incident dataframe
            min_records: Minimum number of records required
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for dashboard builder")
            return False
        
        if len(df) < min_records:
            self.logger.warning(
                f"Insufficient data for meaningful dashboard. Got {len(df)} incidents, "
                f"need at least {min_records}."
            )
            return False
            
        return True
    
    def build_kpi_component(self, df: pd.DataFrame, 
                          timestamp_col: str,
                          metrics: List[Dict] = None) -> Dict:
        """
        Build a component with Key Performance Indicators.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            metrics: List of metric configurations to calculate
                [{'name': 'Metric Name', 'type': 'count|avg|sum', 'column': 'col_name', 'filter': {}}, ...]
            
        Returns:
            Dictionary containing KPI metrics and insights
        """
        if not self._validate_data(df):
            return {
                'success': False,
                'message': 'Insufficient data for KPI metrics',
                'component': None
            }
        
        try:
            # Ensure timestamp is in datetime format
            df_metrics = df.copy()
            if timestamp_col in df_metrics.columns and df_metrics[timestamp_col].dtype != 'datetime64[ns]':
                df_metrics[timestamp_col] = pd.to_datetime(df_metrics[timestamp_col], errors='coerce')
            
            # If no metrics specified, use some sensible defaults based on available columns
            if not metrics:
                metrics = []
                
                # Total count of incidents
                metrics.append({
                    'name': 'Total Incidents',
                    'type': 'count',
                    'column': None
                })
                
                # Average resolution time if available
                resolution_cols = [col for col in df_metrics.columns 
                                 if 'resolution' in col.lower() or 'time_to_resolve' in col.lower()]
                if resolution_cols:
                    resolution_col = resolution_cols[0]
                    # Convert to numeric if needed
                    if not pd.api.types.is_numeric_dtype(df_metrics[resolution_col]):
                        try:
                            df_metrics[resolution_col] = pd.to_numeric(df_metrics[resolution_col], errors='coerce')
                            metrics.append({
                                'name': 'Avg Resolution Time',
                                'type': 'avg',
                                'column': resolution_col,
                                'format': ':.2f'
                            })
                        except:
                            pass
                
                # Count of open incidents if status column available
                status_cols = [col for col in df_metrics.columns if 'status' in col.lower()]
                if status_cols:
                    status_col = status_cols[0]
                    # Look for values that might indicate open status
                    open_values = [val for val in df_metrics[status_col].unique() 
                                 if any(term in str(val).lower() 
                                       for term in ['open', 'new', 'active', 'in progress'])]
                    if open_values:
                        metrics.append({
                            'name': 'Open Incidents',
                            'type': 'count',
                            'column': None,
                            'filter': {status_col: open_values}
                        })
                
                # Count high priority incidents if priority column available
                priority_cols = [col for col in df_metrics.columns 
                               if 'priority' in col.lower() or 'severity' in col.lower()]
                if priority_cols:
                    priority_col = priority_cols[0]
                    # Look for values that might indicate high priority
                    high_priority_values = [val for val in df_metrics[priority_col].unique() 
                                         if any(term in str(val).lower() 
                                               for term in ['p1', 'high', 'critical', '1', 'sev1'])]
                    if high_priority_values:
                        metrics.append({
                            'name': 'High Priority',
                            'type': 'count',
                            'column': None,
                            'filter': {priority_col: high_priority_values}
                        })
            
            # Calculate each metric
            calculated_metrics = []
            for metric in metrics:
                metric_type = metric.get('type', 'count')
                column = metric.get('column')
                metric_filter = metric.get('filter', {})
                format_spec = metric.get('format', '')
                
                # Apply filters if any
                filtered_df = df_metrics
                if metric_filter:
                    for filter_col, filter_values in metric_filter.items():
                        if filter_col in filtered_df.columns:
                            if isinstance(filter_values, list):
                                filtered_df = filtered_df[filtered_df[filter_col].isin(filter_values)]
                            else:
                                filtered_df = filtered_df[filtered_df[filter_col] == filter_values]
                
                # Calculate metric value
                if metric_type == 'count':
                    value = len(filtered_df)
                elif metric_type == 'avg' and column and column in filtered_df.columns:
                    if pd.api.types.is_numeric_dtype(filtered_df[column]):
                        value = filtered_df[column].mean()
                    else:
                        try:
                            numeric_vals = pd.to_numeric(filtered_df[column], errors='coerce')
                            value = numeric_vals.mean()
                        except:
                            value = None
                elif metric_type == 'sum' and column and column in filtered_df.columns:
                    if pd.api.types.is_numeric_dtype(filtered_df[column]):
                        value = filtered_df[column].sum()
                    else:
                        try:
                            numeric_vals = pd.to_numeric(filtered_df[column], errors='coerce')
                            value = numeric_vals.sum()
                        except:
                            value = None
                elif metric_type == 'max' and column and column in filtered_df.columns:
                    if pd.api.types.is_numeric_dtype(filtered_df[column]):
                        value = filtered_df[column].max()
                    else:
                        try:
                            numeric_vals = pd.to_numeric(filtered_df[column], errors='coerce')
                            value = numeric_vals.max()
                        except:
                            value = None
                else:
                    value = None
                
                # Format value
                display_value = value
                if value is not None:
                    if format_spec:
                        try:
                            display_value = format(value, format_spec)
                        except:
                            display_value = str(value)
                    else:
                        if isinstance(value, (int, np.integer)):
                            display_value = str(value)
                        elif isinstance(value, (float, np.floating)):
                            display_value = f"{value:.2f}"
                        else:
                            display_value = str(value)
                else:
                    display_value = "N/A"
                
                # Add to calculated metrics
                calculated_metrics.append({
                    'name': metric.get('name', f"Metric {len(calculated_metrics)+1}"),
                    'value': value,
                    'display_value': display_value,
                    'type': metric_type,
                    'column': column
                })
            
            # Generate some time-based insights
            insights = []
            
            # Check if we have timestamp data
            if timestamp_col in df_metrics.columns:
                # Calculate incidents per day
                df_metrics['date'] = df_metrics[timestamp_col].dt.date
                incidents_per_day = df_metrics.groupby('date').size()
                
                if len(incidents_per_day) > 1:
                    avg_daily = incidents_per_day.mean()
                    latest_date = incidents_per_day.index.max()
                    latest_count = incidents_per_day.loc[latest_date]
                    
                    # Compare latest to average
                    if latest_count > avg_daily * 1.2:  # 20% above average
                        insights.append({
                            'type': 'volume_alert',
                            'severity': 'high',
                            'metric': 'daily_incidents',
                            'message': f"Today's incident volume ({latest_count:.0f}) is {(latest_count/avg_daily - 1) * 100:.0f}% above average"
                        })
                    elif latest_count < avg_daily * 0.5:  # 50% below average
                        insights.append({
                            'type': 'volume_alert',
                            'severity': 'low',
                            'metric': 'daily_incidents',
                            'message': f"Today's incident volume ({latest_count:.0f}) is {(1 - latest_count/avg_daily) * 100:.0f}% below average"
                        })
                
                # Look at recent trend (last 7 days vs previous 7 days)
                if len(incidents_per_day) >= 14:
                    dates_sorted = sorted(incidents_per_day.index, reverse=True)
                    last_7_days = sum(incidents_per_day.loc[date] for date in dates_sorted[:7])
                    prev_7_days = sum(incidents_per_day.loc[date] for date in dates_sorted[7:14])
                    
                    if last_7_days > prev_7_days * 1.2:  # 20% increase
                        change_pct = (last_7_days / prev_7_days - 1) * 100
                        insights.append({
                            'type': 'trend_alert',
                            'severity': 'high',
                            'metric': 'weekly_trend',
                            'message': f"Incident volume increased {change_pct:.0f}% in the last 7 days"
                        })
                    elif last_7_days < prev_7_days * 0.8:  # 20% decrease
                        change_pct = (1 - last_7_days / prev_7_days) * 100
                        insights.append({
                            'type': 'trend_alert',
                            'severity': 'info',
                            'metric': 'weekly_trend',
                            'message': f"Incident volume decreased {change_pct:.0f}% in the last 7 days"
                        })
            
            # Add insights for specific metrics where available
            for metric in calculated_metrics:
                if metric['name'] == 'Avg Resolution Time' and metric['value'] is not None:
                    # Check for resolution time trends if timestamp available
                    if timestamp_col in df_metrics.columns and metric['column'] in df_metrics.columns:
                        # Add month column
                        df_metrics['month'] = df_metrics[timestamp_col].dt.to_period('M')
                        
                        # Calculate average resolution time by month
                        monthly_res_time = df_metrics.groupby('month')[metric['column']].mean()
                        
                        if len(monthly_res_time) > 1:
                            latest_month = monthly_res_time.index.max()
                            latest_res_time = monthly_res_time.loc[latest_month]
                            
                            # Get previous month if available
                            if len(monthly_res_time) > 1:
                                prev_months = [m for m in monthly_res_time.index if m < latest_month]
                                if prev_months:
                                    prev_month = max(prev_months)
                                    prev_res_time = monthly_res_time.loc[prev_month]
                                    
                                    # Compare current to previous month
                                    change_pct = (latest_res_time / prev_res_time - 1) * 100
                                    if abs(change_pct) > 10:  # 10% change
                                        direction = "increased" if change_pct > 0 else "decreased"
                                        insights.append({
                                            'type': 'resolution_time_trend',
                                            'severity': 'high' if change_pct > 0 else 'info',
                                            'metric': 'resolution_time',
                                            'message': f"Average resolution time {direction} by {abs(change_pct):.0f}% compared to previous month"
                                        })
            
            # If no insights were found, add a general message
            if not insights:
                insights.append({
                    'type': 'general',
                    'severity': 'info',
                    'message': "All metrics are within normal ranges"
                })
            
            # Store in cache
            self.metrics_cache = {metric['name']: metric for metric in calculated_metrics}
            
            return {
                'success': True,
                'message': 'KPI metrics calculated successfully',
                'component': {
                    'type': 'kpi',
                    'metrics': calculated_metrics,
                    'insights': insights
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building KPI component: {str(e)}")
            return {
                'success': False,
                'message': f'Error building KPI component: {str(e)}',
                'component': None
            }
    
    def build_trend_component(self, df: pd.DataFrame,
                            timestamp_col: str,
                            value_cols: List[str] = None,
                            category_col: str = None,
                            time_period: str = 'month') -> Dict:
        """
        Build a component showing trends over time.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            value_cols: Columns to track over time
            category_col: Column to use for grouping by category
            time_period: Time period for grouping ('day', 'week', 'month')
            
        Returns:
            Dictionary containing trend data and insights
        """
        if not self._validate_data(df) or timestamp_col not in df.columns:
            return {
                'success': False,
                'message': 'Insufficient data for trend analysis',
                'component': None
            }
        
        try:
            # Ensure timestamp is in datetime format
            df_trend = df.copy()
            if df_trend[timestamp_col].dtype != 'datetime64[ns]':
                df_trend[timestamp_col] = pd.to_datetime(df_trend[timestamp_col], errors='coerce')
            
            # Remove rows with invalid timestamp
            df_trend = df_trend.dropna(subset=[timestamp_col])
            
            if len(df_trend) < 10:
                return {
                    'success': False,
                    'message': 'Insufficient data for trend analysis after cleaning',
                    'component': None
                }
            
            # Determine appropriate time period based on data span
            date_range = (df_trend[timestamp_col].max() - df_trend[timestamp_col].min()).days
            if time_period == 'day' and date_range > 90:
                time_period = 'week'
            elif time_period == 'week' and date_range > 365:
                time_period = 'month'
            
            # Create time period column
            if time_period == 'day':
                df_trend['period'] = df_trend[timestamp_col].dt.date
            elif time_period == 'week':
                df_trend['period'] = df_trend[timestamp_col].dt.to_period('W').apply(lambda x: x.start_time.date())
            else:  # month
                df_trend['period'] = df_trend[timestamp_col].dt.to_period('M').apply(lambda x: x.start_time.date())
            
            # If value_cols not specified, use incident count
            if not value_cols:
                # Check for numeric columns that might be interesting to track
                numeric_cols = [col for col in df_trend.columns 
                              if pd.api.types.is_numeric_dtype(df_trend[col]) and col != timestamp_col]
                
                # Use resolution time if available
                resolution_cols = [col for col in numeric_cols 
                                 if 'resolution' in col.lower() or 'time_to_resolve' in col.lower()]
                if resolution_cols:
                    value_cols = resolution_cols[:1]  # Use first resolution column
            
            trend_data = {
                'periods': [],
                'values': {},
                'categories': {}
            }
            
            # Get sorted periods
            periods_sorted = sorted(df_trend['period'].unique())
            trend_data['periods'] = [p.strftime('%Y-%m-%d') for p in periods_sorted]
            
            # Calculate values for each period
            # Base count of incidents
            trend_data['values']['Incident Count'] = []
            period_counts = df_trend.groupby('period').size()
            
            for period in periods_sorted:
                trend_data['values']['Incident Count'].append(
                    int(period_counts.get(period, 0))
                )
            
            # Additional value columns if specified
            if value_cols:
                for col in value_cols:
                    if col in df_trend.columns and pd.api.types.is_numeric_dtype(df_trend[col]):
                        col_name = col.replace('_', ' ').title()
                        trend_data['values'][col_name] = []
                        
                        # Calculate average value per period
                        period_avgs = df_trend.groupby('period')[col].mean()
                        
                        for period in periods_sorted:
                            trend_data['values'][col_name].append(
                                float(period_avgs.get(period, 0))
                            )
            
            # Category breakdown if specified
            if category_col and category_col in df_trend.columns:
                categories = df_trend[category_col].unique()
                
                # Limit to top N categories if there are too many
                if len(categories) > 7:
                    top_cats = df_trend[category_col].value_counts().nlargest(6).index.tolist()
                    if len(df_trend[~df_trend[category_col].isin(top_cats)]) > 0:
                        # Group remaining as "Other"
                        df_trend.loc[~df_trend[category_col].isin(top_cats), category_col] = 'Other'
                        categories = top_cats + ['Other']
                    else:
                        categories = top_cats
                
                for category in categories:
                    cat_name = str(category)
                    trend_data['categories'][cat_name] = []
                    
                    # Get data for this category
                    cat_df = df_trend[df_trend[category_col] == category]
                    cat_counts = cat_df.groupby('period').size()
                    
                    for period in periods_sorted:
                        trend_data['categories'][cat_name].append(
                            int(cat_counts.get(period, 0))
                        )
            
            # Generate insights
            insights = []
            
            # Overall trend insights
            if len(periods_sorted) >= 3:
                # Calculate trend for incident count
                count_values = trend_data['values']['Incident Count']
                
                if len(count_values) >= 3:
                    first_half = sum(count_values[:len(count_values)//2]) / (len(count_values)//2)
                    second_half = sum(count_values[len(count_values)//2:]) / (len(count_values) - len(count_values)//2)
                    
                    if first_half > 0:
                        percent_change = (second_half - first_half) / first_half * 100
                        
                        if abs(percent_change) >= 10:
                            direction = "increasing" if percent_change > 0 else "decreasing"
                            insights.append({
                                'type': 'overall_trend',
                                'metric': 'incident_count',
                                'change_percent': f"{percent_change:.1f}%",
                                'direction': direction,
                                'message': f"Overall {direction} trend in incident volume ({abs(percent_change):.1f}%)"
                            })
            
            # Category-specific insights
            if 'categories' in trend_data and trend_data['categories']:
                for category, values in trend_data['categories'].items():
                    if len(values) >= 3:
                        first_half = sum(values[:len(values)//2]) / (len(values)//2) if len(values)//2 > 0 else 0
                        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2) if len(values) - len(values)//2 > 0 else 0
                        
                        if first_half > 0:
                            percent_change = (second_half - first_half) / first_half * 100
                            
                            if abs(percent_change) >= 20:  # Higher threshold for categories
                                direction = "increasing" if percent_change > 0 else "decreasing"
                                insights.append({
                                    'type': 'category_trend',
                                    'category': category,
                                    'change_percent': f"{percent_change:.1f}%",
                                    'direction': direction,
                                    'message': f"'{category}' incidents show {direction} trend ({abs(percent_change):.1f}%)"
                                })
            
            # Value column insights
            for value_name, values in trend_data['values'].items():
                if value_name != 'Incident Count' and len(values) >= 3:
                    first_half = sum(values[:len(values)//2]) / (len(values)//2) if len(values)//2 > 0 else 0
                    second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2) if len(values) - len(values)//2 > 0 else 0
                    
                    if first_half > 0:
                        percent_change = (second_half - first_half) / first_half * 100
                        
                        if abs(percent_change) >= 15:
                            direction = "increasing" if percent_change > 0 else "decreasing"
                            insights.append({
                                'type': 'value_trend',
                                'metric': value_name,
                                'change_percent': f"{percent_change:.1f}%",
                                'direction': direction,
                                'message': f"{value_name} shows {direction} trend ({abs(percent_change):.1f}%)"
                            })
            
            # If no insights were found, add a general message
            if not insights:
                insights.append({
                    'type': 'general',
                    'message': "No significant trends detected in the current data"
                })
            
            return {
                'success': True,
                'message': 'Trend analysis completed successfully',
                'component': {
                    'type': 'trend',
                    'time_period': time_period,
                    'data': trend_data,
                    'insights': insights
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building trend component: {str(e)}")
            return {
                'success': False,
                'message': f'Error building trend component: {str(e)}',
                'component': None
            }
    
    def build_distribution_component(self, df: pd.DataFrame,
                                   category_cols: List[str] = None,
                                   value_col: str = None) -> Dict:
        """
        Build a component showing the distribution of incidents across categories.
        
        Args:
            df: Incident dataframe
            category_cols: Columns to use for categorization
            value_col: Column to use for values (if None, will count incidents)
            
        Returns:
            Dictionary containing distribution data and insights
        """
        if not self._validate_data(df):
            return {
                'success': False,
                'message': 'Insufficient data for distribution analysis',
                'component': None
            }
        
        try:
            # If category columns not specified, try to find appropriate columns
            if not category_cols:
                # Look for columns that might be categorical
                potential_cols = []
                
                # Check for columns with common category names
                category_keywords = ['category', 'type', 'status', 'priority', 'severity', 'source']
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in category_keywords):
                        potential_cols.append(col)
                
                # If still no columns, look for columns with low cardinality (< 20 unique values)
                if not potential_cols:
                    for col in df.columns:
                        if col in df.select_dtypes(include=['object', 'category']).columns:
                            if len(df[col].unique()) <= 20:
                                potential_cols.append(col)
                
                # Use the columns with the most promising cardinality (not too many, not too few)
                if potential_cols:
                    # Sort columns by cardinality (prefer 3-10 unique values)
                    sorted_cols = []
                    for col in potential_cols:
                        unique_count = len(df[col].unique())
                        if 3 <= unique_count <= 10:
                            # Ideal cardinality - give it high score
                            sorted_cols.append((col, abs(unique_count - 5)))
                        elif unique_count < 3:
                            # Too few values - less useful
                            sorted_cols.append((col, 10 + (3 - unique_count)))
                        else:
                            # Too many values - less useful
                            sorted_cols.append((col, unique_count))
                    
                    sorted_cols.sort(key=lambda x: x[1])
                    category_cols = [col for col, _ in sorted_cols[:3]]  # Take top 3
            
            # Ensure we have at least one category column
            if not category_cols or not all(col in df.columns for col in category_cols):
                return {
                    'success': False,
                    'message': 'No suitable category columns found for distribution analysis',
                    'component': None
                }
            
            # Prepare value column if specified
            if value_col and value_col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[value_col]):
                    try:
                        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                    except:
                        value_col = None  # Fall back to counting if conversion fails
            
            distribution_data = {}
            
            # Generate distribution for each category column
            for col in category_cols:
                if col not in df.columns:
                    continue
                
                # Calculate distribution
                if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
                    # Sum or average values by category
                    totals = df.groupby(col)[value_col].agg(['sum', 'mean', 'count'])
                    
                    # Convert to more readable format
                    values = []
                    for category, (total, average, count) in totals.iterrows():
                        if pd.notna(category) and pd.notna(total) and pd.notna(average):
                            values.append({
                                'category': str(category),
                                'total': float(total),
                                'average': float(average),
                                'count': int(count)
                            })
                else:
                    # Count incidents by category
                    counts = df[col].value_counts()
                    
                    # Convert to more readable format
                    values = []
                    for category, count in counts.items():
                        if pd.notna(category):
                            values.append({
                                'category': str(category),
                                'count': int(count),
                                'percentage': float(count / len(df) * 100)
                            })
                
                # Sort by count in descending order
                values.sort(key=lambda x: x.get('count', 0), reverse=True)
                
                # Limit to top N categories if there are too many
                if len(values) > 10:
                    top_values = values[:9]  # Top 9
                    other_count = sum(v.get('count', 0) for v in values[9:])
                    other_total = sum(v.get('total', 0) for v in values[9:] if 'total' in v)
                    other_percentage = sum(v.get('percentage', 0) for v in values[9:] if 'percentage' in v)
                    
                    other_entry = {'category': 'Other', 'count': int(other_count)}
                    if 'percentage' in values[0]:
                        other_entry['percentage'] = float(other_percentage)
                    if 'total' in values[0]:
                        other_entry['total'] = float(other_total)
                        if other_count > 0:
                            other_entry['average'] = float(other_total / other_count)
                    
                    values = top_values + [other_entry]
                
                # Add to distribution data
                distribution_data[col] = values
            
            # Generate insights
            insights = []
            
            # Analyze distribution for insights
            for col, values in distribution_data.items():
                if not values:
                    continue
                
                # Top category insight
                top_category = values[0]['category']
                top_count = values[0]['count']
                total_count = sum(v['count'] for v in values)
                top_percentage = top_count / total_count * 100
                
                insights.append({
                    'type': 'top_category',
                    'column': col,
                    'category': top_category,
                    'count': top_count,
                    'percentage': f"{top_percentage:.1f}%",
                    'message': f"'{top_category}' is the most common {col} ({top_percentage:.1f}% of incidents)"
                })
                
                # Concentration insight
                if len(values) >= 3:
                    top_3_count = sum(values[i]['count'] for i in range(min(3, len(values))))
                    top_3_percentage = top_3_count / total_count * 100
                    

                    if top_3_percentage > 80:
                        insights.append({
                            'type': 'concentration',
                            'column': col,
                            'top_categories': [values[i]['category'] for i in range(min(3, len(values)))],
                            'percentage': f"{top_3_percentage:.1f}%",
                            'message': f"Top 3 {col} categories account for {top_3_percentage:.1f}% of all incidents"
                        })
                
                # Value insights (if applicable)
                if 'average' in values[0]:
                    # Find category with highest average
                    sorted_by_avg = sorted(values, key=lambda x: x.get('average', 0), reverse=True)
                    if sorted_by_avg:
                        high_avg_cat = sorted_by_avg[0]['category']
                        high_avg_val = sorted_by_avg[0]['average']
                        
                        # Only include if this category has a reasonable sample size
                        if sorted_by_avg[0]['count'] >= 5:
                            insights.append({
                                'type': 'high_average',
                                'column': col,
                                'category': high_avg_cat,
                                'average': high_avg_val,
                                'message': f"'{high_avg_cat}' has the highest average {value_col} ({high_avg_val:.2f})"
                            })
            
            # If no insights were found, add a general message
            if not insights:
                insights.append({
                    'type': 'general',
                    'message': "Incidents are relatively evenly distributed across categories"
                })
            
            return {
                'success': True,
                'message': 'Distribution analysis completed successfully',
                'component': {
                    'type': 'distribution',
                    'data': distribution_data,
                    'insights': insights
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building distribution component: {str(e)}")
            return {
                'success': False,
                'message': f'Error building distribution component: {str(e)}',
                'component': None
            }
    
    def build_insights_summary(self, components: List[Dict]) -> Dict:
        """
        Compile key insights from multiple dashboard components into a summary.
        
        Args:
            components: List of dashboard components with their insights
            
        Returns:
            Dictionary containing compiled insights
        """
        try:
            all_insights = []
            
            # Collect insights from all components
            for comp in components:
                if comp.get('success') and comp.get('component') and 'insights' in comp['component']:
                    all_insights.extend(comp['component']['insights'])
            
            if not all_insights:
                return {
                    'success': False,
                    'message': 'No insights available for summary',
                    'component': None
                }
            
            # Categorize insights
            categorized_insights = {
                'critical': [],
                'important': [],
                'informational': []
            }
            
            for insight in all_insights:
                insight_type = insight.get('type', '')
                severity = insight.get('severity', 'informational')
                
                # Categorize by type
                if insight_type in ['volume_alert', 'resolution_time_trend'] and severity == 'high':
                    categorized_insights['critical'].append(insight)
                elif insight_type in ['trend_alert', 'category_trend'] and 'increasing' in insight.get('message', '').lower():
                    categorized_insights['important'].append(insight)
                elif insight_type in ['concentration', 'top_category'] and float(insight.get('percentage', '0').strip('%')) > 70:
                    categorized_insights['important'].append(insight)
                elif insight_type != 'general':
                    categorized_insights['informational'].append(insight)
            
            # Limit each category to top N insights
            for category in categorized_insights:
                if len(categorized_insights[category]) > 5:
                    categorized_insights[category] = categorized_insights[category][:5]
            
            # Store in cache
            self.insight_cache = {
                'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'insights': categorized_insights
            }
            
            return {
                'success': True,
                'message': 'Insights summary generated successfully',
                'component': {
                    'type': 'insights_summary',
                    'insights': categorized_insights
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building insights summary: {str(e)}")
            return {
                'success': False,
                'message': f'Error building insights summary: {str(e)}',
                'component': None
            }
    
    def build_dashboard(self, df: pd.DataFrame, config: Dict = None) -> Dict:
        """
        Build a complete dashboard with multiple components.
        
        Args:
            df: Incident dataframe
            config: Dashboard configuration (if None, will auto-detect)
            
        Returns:
            Dictionary containing all dashboard components
        """
        if not self._validate_data(df):
            return {
                'success': False,
                'message': 'Insufficient data for dashboard',
                'dashboard': None
            }
        
        try:
            # Auto-detect configuration if not provided
            if not config:
                config = self._auto_detect_config(df)
            
            # Build components
            components = []
            
            # KPI metrics
            if 'kpi' in config:
                kpi_result = self.build_kpi_component(
                    df,
                    timestamp_col=config['kpi'].get('timestamp_col'),
                    metrics=config['kpi'].get('metrics')
                )
                if kpi_result['success']:
                    components.append(kpi_result)
            
            # Trend analysis
            if 'trend' in config:
                trend_result = self.build_trend_component(
                    df,
                    timestamp_col=config['trend'].get('timestamp_col'),
                    value_cols=config['trend'].get('value_cols'),
                    category_col=config['trend'].get('category_col'),
                    time_period=config['trend'].get('time_period', 'month')
                )
                if trend_result['success']:
                    components.append(trend_result)
            
            # Distribution analysis
            if 'distribution' in config:
                dist_result = self.build_distribution_component(
                    df,
                    category_cols=config['distribution'].get('category_cols'),
                    value_col=config['distribution'].get('value_col')
                )
                if dist_result['success']:
                    components.append(dist_result)
            
            # Compile insights summary
            insights_result = self.build_insights_summary(components)
            if insights_result['success']:
                components.append(insights_result)
            
            # Build dashboard object
            dashboard = {
                'title': config.get('title', 'Incident Management Dashboard'),
                'components': [comp['component'] for comp in components if comp['success']],
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'record_count': len(df)
            }
            
            return {
                'success': True,
                'message': 'Dashboard built successfully',
                'dashboard': dashboard
            }
            
        except Exception as e:
            self.logger.error(f"Error building dashboard: {str(e)}")
            return {
                'success': False,
                'message': f'Error building dashboard: {str(e)}',
                'dashboard': None
            }
    
    def _auto_detect_config(self, df: pd.DataFrame) -> Dict:
        """
        Auto-detect appropriate configuration for dashboard based on available data.
        
        Args:
            df: Incident dataframe
            
        Returns:
            Dictionary with dashboard configuration
        """
        config = {
            'title': 'Incident Management Dashboard',
            'kpi': {},
            'trend': {},
            'distribution': {}
        }
        
        # Look for timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            # Test each column to see if it can be converted to datetime
            for col in timestamp_cols:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    timestamp_col = col
                    config['kpi']['timestamp_col'] = timestamp_col
                    config['trend']['timestamp_col'] = timestamp_col
                    break
                except:
                    continue
        
        # Look for numeric columns for metrics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            # Try to find columns that can be converted to numeric
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                except:
                    continue
        
        # Look for resolution time columns
        resolution_cols = [col for col in numeric_cols if 'resolution' in col.lower() or 'time_to' in col.lower()]
        if resolution_cols:
            resolution_col = resolution_cols[0]
            config['trend']['value_cols'] = [resolution_col]
            config['distribution']['value_col'] = resolution_col
        
        # Look for category columns
        category_cols = []
        category_keywords = ['category', 'type', 'status', 'priority', 'severity', 'source']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in category_keywords):
                category_cols.append(col)
        
        if not category_cols:
            # Look for columns with reasonable cardinality
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if 2 < len(df[col].unique()) <= 20:
                    category_cols.append(col)
        
        if category_cols:
            config['trend']['category_col'] = category_cols[0]
            config['distribution']['category_cols'] = category_cols[:3]
        
        return config


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 300
    
    # Create timestamps with weekly patterns
    base_date = datetime.datetime(2023, 1, 1)
    dates = []
    for i in range(n_samples):
        # Add a random number of days (0-180 days)
        random_days = np.random.randint(0, 180)
        random_hours = np.random.randint(0, 24)
        dates.append(base_date + datetime.timedelta(days=random_days, hours=random_hours))
    
    # Sort dates
    dates.sort()
    
    # Create synthetic data
    categories = ['Network', 'Server', 'Application', 'Database', 'Security']
    priorities = ['P1', 'P2', 'P3', 'P4']
    
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(n_samples)],
        'created_at': dates,
        'category': np.random.choice(categories, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'priority': np.random.choice(priorities, n_samples, p=[0.1, 0.2, 0.4, 0.3]),
        'resolution_time': np.random.exponential(4, n_samples),  # Hours to resolve
        'status': np.random.choice(['Open', 'Closed', 'In Progress'], n_samples, p=[0.2, 0.7, 0.1])
    }
    
    # Add correlation between priority and resolution time
    for i in range(n_samples):
        if data['priority'][i] == 'P1':
            data['resolution_time'][i] *= 2  # P1 takes longer to resolve
        elif data['priority'][i] == 'P4':
            data['resolution_time'][i] *= 0.5  # P4 resolves faster
    
    # Add trend over time - incidents increasing over time
    for i in range(n_samples):
        day_factor = (dates[i] - base_date).days / 180  # Normalized time factor
        if np.random.random() < day_factor:
            # More likely to be high priority and network issues in later periods
            data['priority'][i] = np.random.choice(['P1', 'P2'], p=[0.6, 0.4])
            data['category'][i] = 'Network'
    
    df = pd.DataFrame(data)
    
    # Test the dashboard builder
    builder = DashboardBuilder()
    
    # Create full dashboard
    dashboard_result = builder.build_dashboard(df)
    
    if dashboard_result['success']:
        print("DASHBOARD COMPONENTS:")
        for i, component in enumerate(dashboard_result['dashboard']['components']):
            print(f"\nComponent {i+1}: {component['type']}")
            print("Insights:")
            if 'insights' in component:
                if isinstance(component['insights'], dict):
                    # For categorized insights
                    for category, insights in component['insights'].items():
                        if insights:
                            print(f"  {category.upper()}:")
                            for insight in insights:
                                print(f"  - {insight['message']}")
                else:
                    # For regular insights
                    for insight in component['insights']:
                        print(f"  - {insight['message']}")
    else:
        print(f"Dashboard generation failed: {dashboard_result['message']}")                            