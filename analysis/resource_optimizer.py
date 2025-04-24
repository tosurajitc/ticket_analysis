"""
Resource Optimizer module for Incident Management Analytics.

This module provides analysis and optimization insights for resource allocation,
focusing on workload distribution, staffing needs, and skill development opportunities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class ResourceOptimizer:
    """
    Analyzes incident data to provide resource optimization insights and recommendations.
    
    This class focuses on analyzing workload distribution, identifying staffing needs,
    and recommending skill development opportunities based on incident data patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ResourceOptimizer with application configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config or {}
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', self.config.get('CHUNK_SIZE', 1000)))
        self.logger = logging.getLogger(__name__)
        
        # Store the last analysis results for reuse
        self._last_analysis = None
        self._analysis_timestamp = None
    
    def analyze_workload_distribution(self, 
                                    df: pd.DataFrame,
                                    timestamp_col: str = None,
                                    category_col: str = None,
                                    priority_col: str = None,
                                    assignee_col: str = None) -> Dict[str, Any]:
        """
        Analyze workload distribution across time, categories, priorities, and assignees.
        
        Args:
            df: DataFrame containing incident data
            timestamp_col: Column name for incident timestamp/creation date
            category_col: Column name for incident category/type
            priority_col: Column name for incident priority/severity
            assignee_col: Column name for incident assignee/owner
            
        Returns:
            Dictionary with workload distribution analysis results
        """
        # Validate input data
        if df is None or df.empty:
            return {
                'success': False,
                'error': 'No data provided for workload distribution analysis',
            }
        
        # Find timestamp column if not provided
        if timestamp_col is None:
            timestamp_candidates = ['created_date', 'timestamp', 'creation_date', 'open_date', 'reported_date']
            for col in timestamp_candidates:
                if col in df.columns:
                    timestamp_col = col
                    break
        
        # Check if we have timestamp data for temporal analysis
        if timestamp_col is None or timestamp_col not in df.columns:
            return {
                'success': False,
                'error': 'No timestamp column found or provided for workload distribution analysis',
            }
        
        try:
            # Create a deep copy to avoid modifying the original DataFrame
            analysis_df = df.copy()
            
            # Ensure timestamp column is datetime type with better error handling
            try:
                if not pd.api.types.is_datetime64_any_dtype(analysis_df[timestamp_col]):
                    analysis_df[timestamp_col] = pd.to_datetime(analysis_df[timestamp_col], errors='coerce')
                    
                # Remove rows with invalid dates after conversion
                analysis_df = analysis_df.dropna(subset=[timestamp_col])
                
                # Check if we still have data after cleaning
                if analysis_df.empty:
                    return {
                        'success': False,
                        'error': 'All timestamp values were invalid or could not be converted to datetime',
                    }
            except Exception as date_error:
                # Log the error but try to continue with a minimal result
                self.logger.error(f"Timestamp conversion error: {str(date_error)}")
                return {
                    'success': False,
                    'error': f'Error converting timestamps: {str(date_error)}',
                }
            
            # Initialize results dictionary with essential structure
            result = {
                'success': True,
                'temporal': {},
                'categorical': {},
                'assignment': {},
                'coefficient_of_variation': None,
                'min': None,
                'max': None
            }
            
            # Process temporal data with explicit error handling
            try:
                # Extract temporal components with error handling
                try:
                    analysis_df['hour'] = analysis_df[timestamp_col].dt.hour
                    analysis_df['day_of_week'] = analysis_df[timestamp_col].dt.dayofweek
                    analysis_df['day_name'] = analysis_df[timestamp_col].dt.day_name()
                    analysis_df['month'] = analysis_df[timestamp_col].dt.month
                    analysis_df['month_name'] = analysis_df[timestamp_col].dt.month_name()
                    analysis_df['year'] = analysis_df[timestamp_col].dt.year
                except Exception as e:
                    self.logger.warning(f"Error extracting temporal components: {str(e)}")
                    # Continue with what we have
                
                # Analyze hourly distribution with error handling
                try:
                    hourly_counts = analysis_df.groupby('hour').size()
                    total_incidents = len(analysis_df)
                    hourly_pct = (hourly_counts / total_incidents * 100).round(1)
                    
                    peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else 0
                    peak_hour_count = hourly_counts.max() if not hourly_counts.empty else 0
                    peak_hour_pct = hourly_pct.max() if not hourly_pct.empty else 0
                    
                    # Format hour for display (24-hour format with leading zero)
                    peak_hour_formatted = f"{peak_hour:02d}:00" if not pd.isna(peak_hour) else "00:00"
                    
                    result['temporal']['hourly'] = {
                        'distribution': hourly_counts.to_dict(),
                        'percentages': hourly_pct.to_dict(),
                        'peak_hour': peak_hour if not pd.isna(peak_hour) else 0,
                        'peak_hour_formatted': peak_hour_formatted,
                        'peak_hour_count': int(peak_hour_count) if not pd.isna(peak_hour_count) else 0,
                        'peak_hour_percentage': float(peak_hour_pct) if not pd.isna(peak_hour_pct) else 0
                    }
                except Exception as e:
                    self.logger.warning(f"Error analyzing hourly distribution: {str(e)}")
                    # Create a minimal hourly result with default values
                    result['temporal']['hourly'] = {
                        'distribution': {},
                        'percentages': {},
                        'peak_hour': 0,
                        'peak_hour_formatted': "00:00",
                        'peak_hour_count': 0,
                        'peak_hour_percentage': 0
                    }
                
                # Analyze daily distribution with error handling
                try:
                    day_mapping = {
                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                    }
                    
                    daily_counts = analysis_df.groupby('day_of_week').size()
                    daily_pct = (daily_counts / total_incidents * 100).round(1)
                    
                    peak_day_idx = daily_counts.idxmax() if not daily_counts.empty else 0
                    peak_day = day_mapping.get(peak_day_idx, 'Unknown')
                    peak_day_count = daily_counts.max() if not daily_counts.empty else 0
                    peak_day_pct = daily_pct.max() if not daily_pct.empty else 0
                    
                    # Create readable daily distribution
                    daily_dist = {day_mapping.get(day_idx, 'Unknown'): count 
                                for day_idx, count in daily_counts.to_dict().items()}
                    daily_pct_dist = {day_mapping.get(day_idx, 'Unknown'): pct 
                                    for day_idx, pct in daily_pct.to_dict().items()}
                    
                    result['temporal']['daily'] = {
                        'distribution': daily_dist,
                        'percentages': daily_pct_dist,
                        'peak_day': peak_day,
                        'peak_day_index': int(peak_day_idx) if not pd.isna(peak_day_idx) else 0,
                        'peak_day_count': int(peak_day_count) if not pd.isna(peak_day_count) else 0,
                        'peak_day_percentage': float(peak_day_pct) if not pd.isna(peak_day_pct) else 0
                    }
                except Exception as e:
                    self.logger.warning(f"Error analyzing daily distribution: {str(e)}")
                    # Create a minimal daily result
                    result['temporal']['daily'] = {
                        'distribution': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0},
                        'percentages': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0},
                        'peak_day': 'Monday',
                        'peak_day_index': 0,
                        'peak_day_count': 0,
                        'peak_day_percentage': 0
                    }
                
                # Process category data safely
                if category_col and category_col in analysis_df.columns:
                    try:
                        # Ensure categorical data is string type
                        analysis_df[category_col] = analysis_df[category_col].astype(str)
                        
                        # Calculate basic category distribution
                        cat_counts = analysis_df[category_col].value_counts()
                        cat_pct = (cat_counts / len(analysis_df) * 100).round(1)
                        
                        result['categorical']['category'] = {
                            'column': category_col,
                            'distribution': cat_counts.to_dict(),
                            'percentages': cat_pct.to_dict(),
                            'total_categories': len(cat_counts),
                            'top_category': cat_counts.index[0] if not cat_counts.empty else 'Unknown',
                            'top_category_count': int(cat_counts.iloc[0]) if not cat_counts.empty else 0,
                            'top_category_percentage': float(cat_pct.iloc[0]) if not cat_pct.empty else 0
                        }
                    except Exception as e:
                        self.logger.warning(f"Error analyzing category distribution: {str(e)}")
                        # Create minimal category data
                        result['categorical']['category'] = {
                            'column': category_col,
                            'distribution': {},
                            'percentages': {},
                            'total_categories': 0,
                            'top_category': 'Unknown',
                            'top_category_count': 0,
                            'top_category_percentage': 0
                        }
                
                # Process assignee data safely
                if assignee_col and assignee_col in analysis_df.columns:
                    try:
                        # Ensure assignee data is string type
                        analysis_df[assignee_col] = analysis_df[assignee_col].astype(str)
                        
                        # Calculate basic assignee metrics
                        assignee_counts = analysis_df[assignee_col].value_counts()
                        total_assignees = len(assignee_counts)
                        
                        if not assignee_counts.empty:
                            result['coefficient_of_variation'] = assignee_counts.std() / assignee_counts.mean() if assignee_counts.mean() > 0 else 0
                            result['min'] = int(assignee_counts.min())
                            result['max'] = int(assignee_counts.max())
                            
                            result['assignment']['assignee'] = {
                                'column': assignee_col,
                                'total_assignees': total_assignees,
                                'average_workload': float(assignee_counts.mean()),
                                'median_workload': float(assignee_counts.median()),
                                'min_workload': int(assignee_counts.min()),
                                'max_workload': int(assignee_counts.max())
                            }
                    except Exception as e:
                        self.logger.warning(f"Error analyzing assignee distribution: {str(e)}")
                        # Create minimal assignee data
                        result['assignment']['assignee'] = {
                            'column': assignee_col,
                            'total_assignees': 0,
                            'average_workload': 0,
                            'median_workload': 0,
                            'min_workload': 0,
                            'max_workload': 0
                        }
                        
                        # Set default coefficient values
                        result['coefficient_of_variation'] = 0
                        result['min'] = 0
                        result['max'] = 0
                
            except Exception as temporal_error:
                self.logger.error(f"Error in temporal analysis: {str(temporal_error)}")
                
                # Create minimal temporal data to ensure structure
                result['temporal'] = {
                    'hourly': {
                        'distribution': {},
                        'percentages': {},
                        'peak_hour': 0,
                        'peak_hour_formatted': "00:00",
                        'peak_hour_count': 0,
                        'peak_hour_percentage': 0
                    },
                    'daily': {
                        'distribution': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0},
                        'percentages': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0},
                        'peak_day': 'Monday',
                        'peak_day_index': 0,
                        'peak_day_count': 0,
                        'peak_day_percentage': 0
                    },
                    'monthly': {
                        'distribution': {},
                        'percentages': {},
                        'peak_month': 'January',
                        'peak_month_count': 0,
                        'peak_month_percentage': 0
                    },
                    'weekday_vs_weekend': {
                        'weekday_count': 0,
                        'weekend_count': 0,
                        'weekday_percentage': 0,
                        'weekend_percentage': 0,
                        'weekday_daily_average': 0,
                        'weekend_daily_average': 0,
                        'weekday_to_weekend_ratio': 1.0
                    }
                }
                
            # Store the analysis results for reuse
            self._last_analysis = result
            self._analysis_timestamp = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing workload distribution: {str(e)}")
            
            # Return a minimal working result structure with error information
            return {
                'success': False,
                'error': f'Error during workload distribution analysis: {str(e)}',
                'temporal': {
                    'hourly': {
                        'distribution': {},
                        'peak_hour': 0,
                        'peak_hour_formatted': "00:00"
                    },
                    'daily': {
                        'distribution': {},
                        'peak_day': 'Unknown',
                        'peak_day_index': 0
                    }
                },
                'categorical': {},
                'assignment': {}
            }
        

    
    def _analyze_temporal_distribution(self, 
                                     df: pd.DataFrame, 
                                     timestamp_col: str,
                                     result: Dict[str, Any]) -> None:
        """
        Analyze temporal distribution of incidents.
        
        Args:
            df: DataFrame containing incident data
            timestamp_col: Column name for incident timestamp
            result: Dictionary to update with analysis results
        """
        # Extract temporal components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_name'] = df[timestamp_col].dt.day_name()
        df['month'] = df[timestamp_col].dt.month
        df['month_name'] = df[timestamp_col].dt.month_name()
        df['year'] = df[timestamp_col].dt.year
        
        # Analyze hourly distribution
        hourly_counts = df.groupby('hour').size()
        total_incidents = len(df)
        hourly_pct = (hourly_counts / total_incidents * 100).round(1)
        
        peak_hour = hourly_counts.idxmax()
        peak_hour_count = hourly_counts.max()
        peak_hour_pct = hourly_pct.max()
        
        # Format hour for display (24-hour format with leading zero)
        peak_hour_formatted = f"{peak_hour:02d}:00"
        
        result['temporal']['hourly'] = {
            'distribution': hourly_counts.to_dict(),
            'percentages': hourly_pct.to_dict(),
            'peak_hour': peak_hour,
            'peak_hour_formatted': peak_hour_formatted,
            'peak_hour_count': int(peak_hour_count),
            'peak_hour_percentage': float(peak_hour_pct)
        }
        
        # Analyze daily distribution
        day_mapping = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        
        daily_counts = df.groupby('day_of_week').size()
        daily_pct = (daily_counts / total_incidents * 100).round(1)
        
        peak_day_idx = daily_counts.idxmax()
        peak_day = day_mapping.get(peak_day_idx, 'Unknown')
        peak_day_count = daily_counts.max()
        peak_day_pct = daily_pct.max()
        
        # Create readable daily distribution
        daily_dist = {day_mapping.get(day_idx, 'Unknown'): count 
                     for day_idx, count in daily_counts.to_dict().items()}
        daily_pct_dist = {day_mapping.get(day_idx, 'Unknown'): pct 
                         for day_idx, pct in daily_pct.to_dict().items()}
        
        result['temporal']['daily'] = {
            'distribution': daily_dist,
            'percentages': daily_pct_dist,
            'peak_day': peak_day,
            'peak_day_index': int(peak_day_idx),
            'peak_day_count': int(peak_day_count),
            'peak_day_percentage': float(peak_day_pct)
        }
        
        # Analyze monthly distribution
        monthly_counts = df.groupby('month_name').size()
        monthly_pct = (monthly_counts / total_incidents * 100).round(1)
        
        result['temporal']['monthly'] = {
            'distribution': monthly_counts.to_dict(),
            'percentages': monthly_pct.to_dict(),
            'peak_month': monthly_counts.idxmax(),
            'peak_month_count': int(monthly_counts.max()),
            'peak_month_percentage': float(monthly_pct.max())
        }
        
        # Add weekday vs weekend comparison
        weekday_mask = df['day_of_week'].isin([0, 1, 2, 3, 4])  # Monday to Friday
        weekday_count = weekday_mask.sum()
        weekend_count = (~weekday_mask).sum()
        
        weekday_daily_avg = weekday_count / 5 if weekday_count > 0 else 0  # 5 weekdays
        weekend_daily_avg = weekend_count / 2 if weekend_count > 0 else 0  # 2 weekend days
        
        weekday_ratio = weekday_daily_avg / weekend_daily_avg if weekend_daily_avg > 0 else float('inf')
        
        result['temporal']['weekday_vs_weekend'] = {
            'weekday_count': int(weekday_count),
            'weekend_count': int(weekend_count),
            'weekday_percentage': float((weekday_count / total_incidents * 100).round(1)),
            'weekend_percentage': float((weekend_count / total_incidents * 100).round(1)),
            'weekday_daily_average': float(weekday_daily_avg),
            'weekend_daily_average': float(weekend_daily_avg),
            'weekday_to_weekend_ratio': float(weekday_ratio)
        }
    
    def _analyze_categorical_distribution(self, 
                                        df: pd.DataFrame, 
                                        category_col: str,
                                        result: Dict[str, Any]) -> None:
        """
        Analyze incident distribution by category.
        
        Args:
            df: DataFrame containing incident data
            category_col: Column name for incident category
            result: Dictionary to update with analysis results
        """
        # Ensure category values are strings
        df[category_col] = df[category_col].astype(str)
        
        # Calculate category distribution
        category_counts = df[category_col].value_counts()
        total_incidents = len(df)
        category_pct = (category_counts / total_incidents * 100).round(1)
        
        # Get top categories (top 5)
        top_categories = category_counts.nlargest(5)
        top_categories_pct = category_pct.loc[top_categories.index]
        
        result['categorical']['category'] = {
            'column': category_col,
            'distribution': category_counts.to_dict(),
            'percentages': category_pct.to_dict(),
            'total_categories': len(category_counts),
            'top_categories': top_categories.to_dict(),
            'top_categories_percentages': top_categories_pct.to_dict(),
            'top_category': category_counts.idxmax(),
            'top_category_count': int(category_counts.max()),
            'top_category_percentage': float(category_pct.max())
        }
        
        # Analyze temporal patterns by category (if timestamp data available)
        if 'hour' in df.columns and 'day_of_week' in df.columns:
            # Get peak hours by category
            peak_hours_by_category = {}
            peak_days_by_category = {}
            
            for category in top_categories.index:
                category_df = df[df[category_col] == category]
                
                if not category_df.empty:
                    # Peak hour
                    cat_hour_counts = category_df.groupby('hour').size()
                    if not cat_hour_counts.empty:
                        peak_hour = cat_hour_counts.idxmax()
                        peak_hours_by_category[category] = {
                            'peak_hour': int(peak_hour),
                            'peak_hour_formatted': f"{peak_hour:02d}:00",
                            'count': int(cat_hour_counts.max())
                        }
                    
                    # Peak day
                    cat_day_counts = category_df.groupby('day_of_week').size()
                    if not cat_day_counts.empty:
                        day_mapping = {
                            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                        }
                        peak_day_idx = cat_day_counts.idxmax()
                        peak_days_by_category[category] = {
                            'peak_day_index': int(peak_day_idx),
                            'peak_day': day_mapping.get(peak_day_idx, 'Unknown'),
                            'count': int(cat_day_counts.max())
                        }
            
            result['categorical']['category']['peak_hours_by_category'] = peak_hours_by_category
            result['categorical']['category']['peak_days_by_category'] = peak_days_by_category
    
    def _analyze_priority_distribution(self, 
                                     df: pd.DataFrame, 
                                     priority_col: str,
                                     result: Dict[str, Any]) -> None:
        """
        Analyze incident distribution by priority.
        
        Args:
            df: DataFrame containing incident data
            priority_col: Column name for incident priority
            result: Dictionary to update with analysis results
        """
        # Ensure priority values are strings
        df[priority_col] = df[priority_col].astype(str)
        
        # Calculate priority distribution
        priority_counts = df[priority_col].value_counts()
        total_incidents = len(df)
        priority_pct = (priority_counts / total_incidents * 100).round(1)
        
        result['categorical']['priority'] = {
            'column': priority_col,
            'distribution': priority_counts.to_dict(),
            'percentages': priority_pct.to_dict(),
            'total_priorities': len(priority_counts),
            'top_priority': priority_counts.idxmax(),
            'top_priority_count': int(priority_counts.max()),
            'top_priority_percentage': float(priority_pct.max())
        }
        
        # Calculate high-priority percentage
        # First standardize priority values
        priority_mapping = {
            'p1': 'critical',
            '1': 'critical',
            'critical': 'critical',
            'high': 'high',
            'p2': 'high',
            '2': 'high',
            'medium': 'medium',
            'normal': 'medium',
            'p3': 'medium',
            '3': 'medium',
            'low': 'low',
            'p4': 'low',
            '4': 'low',
            'p5': 'low',
            '5': 'low'
        }
        
        df['standard_priority'] = df[priority_col].str.lower().map(priority_mapping)
        
        # Calculate high priority incidents (critical + high)
        high_priority_mask = df['standard_priority'].isin(['critical', 'high'])
        high_priority_count = high_priority_mask.sum()
        high_priority_pct = (high_priority_count / total_incidents * 100).round(1)
        
        result['categorical']['priority']['high_priority_count'] = int(high_priority_count)
        result['categorical']['priority']['high_priority_percentage'] = float(high_priority_pct)

    def _analyze_assignee_distribution(self, 
                                     df: pd.DataFrame, 
                                     assignee_col: str,
                                     result: Dict[str, Any]) -> None:
        """
        Analyze incident distribution by assignee.
        
        Args:
            df: DataFrame containing incident data
            assignee_col: Column name for incident assignee
            result: Dictionary to update with analysis results
        """
        # Ensure assignee values are strings
        df[assignee_col] = df[assignee_col].astype(str)
        
        # Calculate assignee distribution
        assignee_counts = df[assignee_col].value_counts()
        total_incidents = len(df)
        total_assignees = len(assignee_counts)
        
        # Get top assignees (top 10)
        top_assignees = assignee_counts.nlargest(10)
        
        # Calculate workload statistics
        avg_workload = assignee_counts.mean()
        median_workload = assignee_counts.median()
        min_workload = assignee_counts.min()
        max_workload = assignee_counts.max()
        
        result['assignment']['assignee'] = {
            'column': assignee_col,
            'total_assignees': total_assignees,
            'top_assignees': top_assignees.to_dict(),
            'average_workload': float(avg_workload),
            'median_workload': float(median_workload),
            'min_workload': int(min_workload),
            'max_workload': int(max_workload),
            'workload_range': int(max_workload - min_workload)
        }
        
        # Calculate workload distribution percentiles
        workload_percentiles = {
            'p10': float(assignee_counts.quantile(0.1)),
            'p25': float(assignee_counts.quantile(0.25)),
            'p50': float(assignee_counts.quantile(0.5)),  # median
            'p75': float(assignee_counts.quantile(0.75)),
            'p90': float(assignee_counts.quantile(0.9))
        }
        
        result['assignment']['assignee']['workload_percentiles'] = workload_percentiles
        
        # Analyze cross-distribution with categories if available
        if 'category' in result['categorical']:
            category_col = result['categorical']['category']['column']
            
            # Get category distribution for top assignees
            assignee_category_matrix = {}
            
            for assignee in top_assignees.index:
                assignee_df = df[df[assignee_col] == assignee]
                if not assignee_df.empty:
                    category_counts = assignee_df[category_col].value_counts()
                    category_pct = (category_counts / len(assignee_df) * 100).round(1)
                    
                    # Find primary category (highest percentage)
                    primary_category = category_pct.idxmax()
                    primary_category_pct = category_pct.max()
                    
                    # Determine if specialist or generalist
                    is_specialist = primary_category_pct > 70  # More than 70% in one category
                    
                    assignee_category_matrix[assignee] = {
                        'incident_count': int(len(assignee_df)),
                        'category_distribution': category_counts.to_dict(),
                        'category_percentages': category_pct.to_dict(),
                        'primary_category': primary_category,
                        'primary_category_percentage': float(primary_category_pct),
                        'is_specialist': bool(is_specialist)
                    }
            
            result['assignment']['category_specialization'] = assignee_category_matrix
            
            # Count specialists and generalists
            specialists = [assignee for assignee, data in assignee_category_matrix.items() 
                          if data['is_specialist']]
            
            result['assignment']['specialists_count'] = len(specialists)
            result['assignment']['generalists_count'] = len(assignee_category_matrix) - len(specialists)
            
            # Calculate specialist distribution by category
            specialist_by_category = {}
            for assignee, data in assignee_category_matrix.items():
                if data['is_specialist']:
                    category = data['primary_category']
                    if category not in specialist_by_category:
                        specialist_by_category[category] = 0
                    specialist_by_category[category] += 1
            
            result['assignment']['specialist_by_category'] = specialist_by_category

    """
    Resource Optimizer module for Incident Management Analytics.

    This module provides analysis and optimization insights for resource allocation,
    focusing on workload distribution, staffing needs, and skill development opportunities.
    It generates qualitative insights for better resource management and optimization.
    """

    def _generate_skill_insights(self, skill_result: Dict[str, Any]) -> List[Dict]:
        """
        Generate insights from skill recommendation analysis.
        
        Args:
            skill_result: Result from get_skill_recommendations method
            
        Returns:
            List of dictionaries with skill optimization insights
        """
        insights = []
        
        # Extract recommendations
        recommendations = skill_result.get('recommendations', {})
        
        # Skills gaps
        if 'skill_gaps' in recommendations:
            for gap in recommendations['skill_gaps']:
                if gap.get('issue') != 'no_clear_gaps':
                    insights.append({
                        'type': 'skill_gap',
                        'data': gap,
                        'message': gap['recommendation']
                    })
        
        # Team composition insights
        if 'team_composition' in recommendations:
            composition = recommendations['team_composition']
            
            if 'specialist_performance' in recommendations:
                # Find category with biggest specialist advantage
                specialist_perf = recommendations['specialist_performance']
                best_categories = []
                
                for category, perf in specialist_perf.items():
                    if perf.get('significant_improvement'):
                        best_categories.append((category, perf))
                
                if best_categories:
                    # Sort by improvement percentage
                    best_categories.sort(
                        key=lambda x: float(x[1]['improvement_percentage'].rstrip('%')), 
                        reverse=True
                    )
                    top_category, top_perf = best_categories[0]
                    
                    insights.append({
                        'type': 'specialist_impact',
                        'data': {
                            'category': top_category,
                            'improvement': top_perf['improvement_percentage']
                        },
                        'message': f"Specialists resolve {top_category} incidents {top_perf['improvement_percentage']} faster than non-specialists"
                    })
        
        return insights
        
    # Part 3: Visualization Preparation and Utility Functions
    
    def prepare_visualization_data(self, 
                                  analysis_results: Dict[str, Any] = None, 
                                  viz_type: str = 'workload') -> Dict[str, Any]:
        """
        Prepare data for visualization based on analysis results.
        
        Args:
            analysis_results: Combined results from various analyses or None to use cached results
            viz_type: Type of visualization to prepare ('workload', 'staffing', 'skills')
            
        Returns:
            Dictionary with prepared visualization data
        """
        # If no analysis results provided, check for cached results
        if analysis_results is None:
            if self._last_analysis is None:
                return {
                    'success': False,
                    'error': 'No analysis results available for visualization'
                }
            analysis_results = self._last_analysis
        
        # Validate analysis results
        if not analysis_results.get('success', False):
            return {
                'success': False,
                'error': 'Invalid analysis results for visualization'
            }
        
        try:
            # Prepare visualization data based on type
            if viz_type == 'workload':
                return self._prepare_workload_visualization(analysis_results)
            elif viz_type == 'staffing':
                return self._prepare_staffing_visualization(analysis_results)
            elif viz_type == 'skills':
                return self._prepare_skills_visualization(analysis_results)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported visualization type: {viz_type}'
                }
        except Exception as e:
            self.logger.error(f"Error preparing visualization data: {str(e)}")
            return {
                'success': False,
                'error': f'Error preparing visualization data: {str(e)}'
            }
    
    def _prepare_workload_visualization(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare workload distribution visualization data.
        
        Args:
            analysis_results: Workload distribution analysis results
            
        Returns:
            Dictionary with prepared visualization data
        """
        # Extract temporal data
        temporal = analysis_results.get('temporal', {})
        
        viz_data = {
            'success': True,
            'workload': {
                'hourly': {},
                'daily': {},
                'monthly': {},
                'category': {},
                'assignee': {}
            }
        }
        
        # Prepare hourly data
        if 'hourly' in temporal:
            hourly = temporal['hourly']
            distribution = hourly.get('distribution', {})
            
            # Format data for chart
            hours = []
            counts = []
            
            for hour in range(24):  # Ensure all 24 hours are represented
                hours.append(f"{hour:02d}:00")
                counts.append(distribution.get(hour, 0))
            
            viz_data['workload']['hourly'] = {
                'labels': hours,
                'values': counts,
                'peak_hour': hourly.get('peak_hour_formatted', None),
                'title': 'Hourly Incident Distribution',
                'x_label': 'Hour of Day',
                'y_label': 'Incident Count'
            }
        
        # Prepare daily data
        if 'daily' in temporal:
            daily = temporal['daily']
            distribution = daily.get('distribution', {})
            
            # Format data for chart
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            counts = [distribution.get(day, 0) for day in days]
            
            viz_data['workload']['daily'] = {
                'labels': days,
                'values': counts,
                'peak_day': daily.get('peak_day', None),
                'title': 'Daily Incident Distribution',
                'x_label': 'Day of Week',
                'y_label': 'Incident Count'
            }
        
        # Prepare monthly data
        if 'monthly' in temporal:
            monthly = temporal['monthly']
            distribution = monthly.get('distribution', {})
            
            # Format data for chart
            months = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            counts = [distribution.get(month, 0) for month in months]
            
            viz_data['workload']['monthly'] = {
                'labels': months,
                'values': counts,
                'peak_month': monthly.get('peak_month', None),
                'title': 'Monthly Incident Distribution',
                'x_label': 'Month',
                'y_label': 'Incident Count'
            }
        
        # Prepare category data
        categorical = analysis_results.get('categorical', {})
        if 'category' in categorical:
            category = categorical['category']
            distribution = category.get('distribution', {})
            
            # Sort categories by count for better visualization
            sorted_categories = dict(sorted(
                distribution.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
            # Limit to top 10 categories for readability
            top_categories = dict(list(sorted_categories.items())[:10])
            
            # Format data for chart
            viz_data['workload']['category'] = {
                'labels': list(top_categories.keys()),
                'values': list(top_categories.values()),
                'title': 'Incident Distribution by Category',
                'x_label': 'Category',
                'y_label': 'Incident Count'
            }
        
        # Prepare assignee data
        assignment = analysis_results.get('assignment', {})
        if 'assignee' in assignment:
            assignee_data = assignment['assignee']
            
            if 'top_assignees' in assignee_data:
                top_assignees = assignee_data['top_assignees']
                
                # Format data for chart
                viz_data['workload']['assignee'] = {
                    'labels': list(top_assignees.keys()),
                    'values': list(top_assignees.values()),
                    'title': 'Incident Distribution by Assignee',
                    'x_label': 'Assignee',
                    'y_label': 'Incident Count'
                }
        
        return viz_data
    
    def _prepare_staffing_visualization(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare staffing needs visualization data.
        
        Args:
            analysis_results: Staffing prediction results
            
        Returns:
            Dictionary with prepared visualization data
        """
        # Ensure we have predictions data
        if 'predictions' not in analysis_results:
            return {
                'success': False,
                'error': 'No staffing predictions available for visualization'
            }
        
        predictions = analysis_results['predictions']
        
        viz_data = {
            'success': True,
            'staffing': {
                'daily': {},
                'category': {},
                'priority': {}
            }
        }
        
        # Prepare daily staffing data
        if 'by_day' in predictions:
            daily_predictions = predictions['by_day']
            
            # Format data for chart
            dates = []
            staff_needed = []
            incidents = []
            
            # Sort by date
            sorted_days = sorted(daily_predictions.items(), key=lambda x: x[0])
            
            for date_str, data in sorted_days:
                dates.append(date_str)
                staff_needed.append(data['staff_needed'])
                incidents.append(data['predicted_incidents'])
            
            viz_data['staffing']['daily'] = {
                'labels': dates,
                'staff_values': staff_needed,
                'incident_values': incidents,
                'title': 'Daily Staffing Needs Forecast',
                'x_label': 'Date',
                'y_label': 'Staff Needed'
            }
        
        # Prepare category staffing data
        if 'by_category' in predictions:
            category_predictions = predictions['by_category']
            
            # Sort categories by staff needed
            sorted_categories = dict(sorted(
                category_predictions.items(), 
                key=lambda item: item[1]['staff_needed'], 
                reverse=True
            ))
            
            # Format data for chart
            categories = []
            staff_needed = []
            resolution_times = []
            
            for category, data in sorted_categories.items():
                categories.append(category)
                staff_needed.append(data['staff_needed'])
                resolution_times.append(data.get('average_resolution_time', 0))
            
            viz_data['staffing']['category'] = {
                'labels': categories,
                'staff_values': staff_needed,
                'resolution_values': resolution_times,
                'title': 'Staffing Needs by Category',
                'x_label': 'Category',
                'y_label': 'Staff Needed'
            }
        
        # Prepare priority staffing data
        if 'by_priority' in predictions:
            priority_predictions = predictions['by_priority']
            
            # Sort priorities by staff needed
            sorted_priorities = dict(sorted(
                priority_predictions.items(), 
                key=lambda item: item[1]['staff_needed'], 
                reverse=True
            ))
            
            # Format data for chart
            priorities = []
            staff_needed = []
            
            for priority, data in sorted_priorities.items():
                priorities.append(priority)
                staff_needed.append(data['staff_needed'])
            
            viz_data['staffing']['priority'] = {
                'labels': priorities,
                'values': staff_needed,
                'title': 'Staffing Needs by Priority',
                'x_label': 'Priority',
                'y_label': 'Staff Needed'
            }
        
        return viz_data
    
    def _prepare_skills_visualization(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare skill recommendation visualization data.
        
        Args:
            analysis_results: Skill recommendation results
            
        Returns:
            Dictionary with prepared visualization data
        """
        # Ensure we have recommendations data
        if 'recommendations' not in analysis_results:
            return {
                'success': False,
                'error': 'No skill recommendations available for visualization'
            }
        
        recommendations = analysis_results['recommendations']
        
        viz_data = {
            'success': True,
            'skills': {
                'team_composition': {},
                'category_specialists': {},
                'performance_comparison': {}
            }
        }
        
        # Prepare team composition data
        if 'team_composition' in recommendations:
            composition = recommendations['team_composition']
            
            # Format data for pie chart
            viz_data['skills']['team_composition'] = {
                'labels': ['Specialists', 'Generalists'],
                'values': [
                    composition.get('specialists_count', 0),
                    composition.get('generalists_count', 0)
                ],
                'title': 'Team Composition',
                'subtitle': 'Specialists vs. Generalists'
            }
        
        # Prepare specialist by category data
        if 'team_composition' in recommendations and 'specialist_by_category' in recommendations['team_composition']:
            specialist_by_category = recommendations['team_composition']['specialist_by_category']
            
            # Sort categories by specialist count
            sorted_categories = dict(sorted(
                specialist_by_category.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
            # Format data for chart
            viz_data['skills']['category_specialists'] = {
                'labels': list(sorted_categories.keys()),
                'values': list(sorted_categories.values()),
                'title': 'Specialist Distribution by Category',
                'x_label': 'Category',
                'y_label': 'Number of Specialists'
            }
        
        # Prepare specialist performance comparison data
        if 'specialist_performance' in recommendations:
            specialist_performance = recommendations['specialist_performance']
            
            # Format data for chart
            categories = []
            specialist_times = []
            nonspecialist_times = []
            
            for category, perf in specialist_performance.items():
                categories.append(category)
                specialist_times.append(perf.get('specialist_resolution_time', 0))
                nonspecialist_times.append(perf.get('nonspecialist_resolution_time', 0))
            
            viz_data['skills']['performance_comparison'] = {
                'labels': categories,
                'specialist_values': specialist_times,
                'nonspecialist_values': nonspecialist_times,
                'title': 'Resolution Time: Specialists vs. Non-Specialists',
                'x_label': 'Category',
                'y_label': 'Resolution Time (hours)'
            }
        
        return viz_data
    
    def generate_chart_data(self,
                          chart_type: str,
                          analysis_results: Dict[str, Any] = None,
                          options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate data for specific chart types based on analysis results.
        
        Args:
            chart_type: Type of chart to generate ('workload_hourly', 'staffing_forecast', etc.)
            analysis_results: Analysis results to use for chart generation
            options: Additional options for chart generation
            
        Returns:
            Dictionary with chart data formatted for visualization library
        """
        # Default options
        default_options = {
            'title': None,
            'height': 400,
            'width': None,
            'color_scheme': 'default',
            'show_legend': True
        }
        
        # Merge provided options with defaults
        chart_options = {**default_options, **(options or {})}
        
        try:
            # Prepare visualization data if needed
            viz_category = None
            if chart_type.startswith('workload_'):
                viz_category = 'workload'
            elif chart_type.startswith('staffing_'):
                viz_category = 'staffing'
            elif chart_type.startswith('skills_'):
                viz_category = 'skills'
            
            if viz_category:
                viz_data = self.prepare_visualization_data(analysis_results, viz_category)
                if not viz_data.get('success', False):
                    return {
                        'success': False,
                        'error': viz_data.get('error', 'Error preparing visualization data')
                    }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported chart type: {chart_type}'
                }
            
            # Generate specific chart data
            if chart_type == 'workload_hourly':
                return self._generate_hourly_chart_data(viz_data, chart_options)
            elif chart_type == 'workload_daily':
                return self._generate_daily_chart_data(viz_data, chart_options)
            elif chart_type == 'workload_monthly':
                return self._generate_monthly_chart_data(viz_data, chart_options)
            elif chart_type == 'workload_category':
                return self._generate_category_chart_data(viz_data, chart_options)
            elif chart_type == 'workload_assignee':
                return self._generate_assignee_chart_data(viz_data, chart_options)
            elif chart_type == 'staffing_forecast':
                return self._generate_staffing_forecast_chart_data(viz_data, chart_options)
            elif chart_type == 'staffing_category':
                return self._generate_staffing_category_chart_data(viz_data, chart_options)
            elif chart_type == 'staffing_priority':
                return self._generate_staffing_priority_chart_data(viz_data, chart_options)
            elif chart_type == 'skills_composition':
                return self._generate_skills_composition_chart_data(viz_data, chart_options)
            elif chart_type == 'skills_specialists':
                return self._generate_skills_specialists_chart_data(viz_data, chart_options)
            elif chart_type == 'skills_performance':
                return self._generate_skills_performance_chart_data(viz_data, chart_options)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported chart type: {chart_type}'
                }
        except Exception as e:
            self.logger.error(f"Error generating chart data: {str(e)}")
            return {
                'success': False,
                'error': f'Error generating chart data: {str(e)}'
            }
    
    def _generate_hourly_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hourly workload distribution chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        hourly_data = viz_data.get('workload', {}).get('hourly', {})
        
        category_data = viz_data.get('staffing', {}).get('category', {})
        
        # Check if we have necessary data
        if not category_data or 'labels' not in category_data or 'staff_values' not in category_data:
            return {
                'success': False,
                'error': 'Insufficient data for staffing category chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': category_data['labels'],
                'datasets': [{
                    'label': 'Staff Needed',
                    'data': category_data['staff_values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(category_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or category_data.get('title', 'Staffing Needs by Category'),
                'x_label': category_data.get('x_label', 'Category'),
                'y_label': category_data.get('y_label', 'Staff Needed'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add resolution time if available
        if 'resolution_values' in category_data:
            chart_data['data']['datasets'].append({
                'label': 'Avg Resolution Time (hours)',
                'data': category_data['resolution_values'],
                'backgroundColor': 'rgba(0,0,0,0)',
                'borderColor': self._get_color_scheme(options['color_scheme'], 'secondary'),
                'type': 'line',
                'yAxisID': 'y1'  # Use secondary y-axis
            })
            
            # Add secondary y-axis
            chart_data['options']['y1_label'] = 'Resolution Time (hours)'
        
        return chart_data
    
    def _generate_staffing_priority_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate staffing by priority chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        priority_data = viz_data.get('staffing', {}).get('priority', {})
        
        # Check if we have necessary data
        if not priority_data or 'labels' not in priority_data or 'values' not in priority_data:
            return {
                'success': False,
                'error': 'Insufficient data for staffing priority chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': priority_data['labels'],
                'datasets': [{
                    'label': 'Staff Needed',
                    'data': priority_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(priority_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or priority_data.get('title', 'Staffing Needs by Priority'),
                'x_label': priority_data.get('x_label', 'Priority'),
                'y_label': priority_data.get('y_label', 'Staff Needed'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        return chart_data
    
    def _generate_skills_composition_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate team composition chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        composition_data = viz_data.get('skills', {}).get('team_composition', {})
        
        # Check if we have necessary data
        if not composition_data or 'labels' not in composition_data or 'values' not in composition_data:
            return {
                'success': False,
                'error': 'Insufficient data for team composition chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'pie',
            'data': {
                'labels': composition_data['labels'],
                'datasets': [{
                    'data': composition_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(composition_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or composition_data.get('title', 'Team Composition'),
                'subtitle': composition_data.get('subtitle', 'Specialists vs. Generalists'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights about team composition
        specialist_count = composition_data['values'][0] if len(composition_data['values']) > 0 else 0
        generalist_count = composition_data['values'][1] if len(composition_data['values']) > 1 else 0
        total = specialist_count + generalist_count
        
        if total > 0:
            specialist_ratio = specialist_count / total
            
            if specialist_ratio < 0.3:
                chart_data['insights'].append(
                    f"Low specialist ratio ({specialist_ratio:.0%}). Consider increasing specialization."
                )
            elif specialist_ratio > 0.7:
                chart_data['insights'].append(
                    f"High specialist ratio ({specialist_ratio:.0%}). Consider cross-training for better coverage."
                )
            else:
                chart_data['insights'].append(
                    f"Balanced specialist-to-generalist ratio ({specialist_ratio:.0%})."
                )
        
        return chart_data
    
    def _generate_skills_specialists_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specialists by category chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        specialists_data = viz_data.get('skills', {}).get('category_specialists', {})
        
        # Check if we have necessary data
        if not specialists_data or 'labels' not in specialists_data or 'values' not in specialists_data:
            return {
                'success': False,
                'error': 'Insufficient data for category specialists chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': specialists_data['labels'],
                'datasets': [{
                    'label': 'Number of Specialists',
                    'data': specialists_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(specialists_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or specialists_data.get('title', 'Specialist Distribution by Category'),
                'x_label': specialists_data.get('x_label', 'Category'),
                'y_label': specialists_data.get('y_label', 'Number of Specialists'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights about specialist distribution
        if len(specialists_data['labels']) > 0:
            zero_specialist_categories = sum(1 for count in specialists_data['values'] if count == 0)
            
            if zero_specialist_categories > 0:
                chart_data['insights'].append(
                    f"{zero_specialist_categories} categories have no dedicated specialists"
                )
        
        return chart_data
    
    def _generate_skills_performance_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specialist performance comparison chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        performance_data = viz_data.get('skills', {}).get('performance_comparison', {})
        
        # Check if we have necessary data
        if not performance_data or 'labels' not in performance_data or 'specialist_values' not in performance_data:
            return {
                'success': False,
                'error': 'Insufficient data for performance comparison chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': performance_data['labels'],
                'datasets': [
                    {
                        'label': 'Specialist Resolution Time',
                        'data': performance_data['specialist_values'],
                        'backgroundColor': self._get_color_scheme(options['color_scheme'], 'primary')
                    },
                    {
                        'label': 'Non-Specialist Resolution Time',
                        'data': performance_data.get('nonspecialist_values', []),
                        'backgroundColor': self._get_color_scheme(options['color_scheme'], 'secondary')
                    }
                ]
            },
            'options': {
                'title': options['title'] or performance_data.get('title', 'Resolution Time: Specialists vs. Non-Specialists'),
                'x_label': performance_data.get('x_label', 'Category'),
                'y_label': performance_data.get('y_label', 'Resolution Time (hours)'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights about performance comparison
        if len(performance_data['labels']) > 0 and 'nonspecialist_values' in performance_data:
            specialist_values = performance_data['specialist_values']
            nonspecialist_values = performance_data['nonspecialist_values']
            
            # Calculate average improvement percentage
            improvements = []
            
            for i in range(min(len(specialist_values), len(nonspecialist_values))):
                if specialist_values[i] > 0 and nonspecialist_values[i] > 0:
                    improvement = (nonspecialist_values[i] - specialist_values[i]) / nonspecialist_values[i]
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                
                if avg_improvement > 0.2:  # More than 20% improvement
                    chart_data['insights'].append(
                        f"Specialists are {avg_improvement:.0%} faster at resolving incidents than non-specialists"
                    )
        
        return chart_data
    
    def _get_color_scheme(self, scheme_name: str, color_type: str, count: int = 1) -> Union[str, List[str]]:
        """
        Get color scheme values based on scheme name and type.
        
        Args:
            scheme_name: Name of the color scheme
            color_type: Type of colors to return ('primary', 'secondary', 'categorical')
            count: Number of colors to return (for categorical schemes)
            
        Returns:
            String or list of color values
        """
        # Default color schemes
        color_schemes = {
            'default': {
                'primary': 'rgba(54, 162, 235, 0.8)',
                'secondary': 'rgba(255, 99, 132, 0.8)',
                'categorical': [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)',
                    'rgba(83, 102, 255, 0.8)',
                    'rgba(255, 159, 123, 0.8)',
                    'rgba(161, 161, 161, 0.8)'
                ]
            },
            'blue': {
                'primary': 'rgba(54, 162, 235, 0.8)',
                'secondary': 'rgba(153, 204, 255, 0.8)',
                'categorical': [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(51, 102, 255, 0.8)',
                    'rgba(153, 204, 255, 0.8)',
                    'rgba(0, 102, 204, 0.8)',
                    'rgba(102, 178, 255, 0.8)',
                    'rgba(0, 153, 204, 0.8)',
                    'rgba(51, 153, 255, 0.8)',
                    'rgba(0, 76, 153, 0.8)',
                    'rgba(0, 128, 255, 0.8)',
                    'rgba(0, 51, 102, 0.8)'
                ]
            },
            'green': {
                'primary': 'rgba(75, 192, 192, 0.8)',
                'secondary': 'rgba(142, 235, 163, 0.8)',
                'categorical': [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(102, 204, 0, 0.8)',
                    'rgba(142, 235, 163, 0.8)',
                    'rgba(0, 153, 0, 0.8)',
                    'rgba(102, 255, 102, 0.8)',
                    'rgba(0, 204, 102, 0.8)',
                    'rgba(51, 204, 51, 0.8)',
                    'rgba(0, 102, 51, 0.8)',
                    'rgba(0, 204, 0, 0.8)',
                    'rgba(0, 102, 0, 0.8)'
                ]
            }
        }
        
        # Get the requested scheme or default
        scheme = color_schemes.get(scheme_name, color_schemes['default'])
        
        # Return the requested color type
        if color_type == 'categorical':
            # Get requested number of colors
            colors = scheme['categorical']
            
            # If we need more colors than available, cycle through them
            if count > len(colors):
                expanded_colors = []
                for i in range(count):
                    expanded_colors.append(colors[i % len(colors)])
                return expanded_colors
            
            # Return requested number of colors
            return colors[:count]
        else:
            return scheme.get(color_type, scheme['primary'])
    
    # Utility functions
    
    def format_duration(self, hours: float) -> str:
        """
        Format duration in hours to a human-readable string.
        
        Args:
            hours: Duration in hours
            
        Returns:
            Formatted duration string
        """
        if hours is None or pd.isna(hours):
            return "N/A"
        
        if hours < 0:
            return "Invalid duration"
        
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} min"
        elif hours < 24:
            hours_int = int(hours)
            minutes = int((hours - hours_int) * 60)
            
            if minutes > 0:
                return f"{hours_int}h {minutes}m"
            else:
                return f"{hours_int}h"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            
            if remaining_hours > 0:
                return f"{days}d {remaining_hours}h"
            else:
                return f"{days}d"
    
    def format_percentage(self, value: float, include_sign: bool = False) -> str:
        """
        Format percentage value to a human-readable string.
        
        Args:
            value: Percentage value (0-100)
            include_sign: Whether to include + sign for positive values
            
        Returns:
            Formatted percentage string
        """
        if value is None or pd.isna(value):
            return "N/A"
        
        # Format with 1 decimal place and % sign
        sign = "+" if include_sign and value > 0 else ""
        return f"{sign}{value:.1f}%"
    
    def get_color_for_trend(self, value: float) -> str:
        """
        Get appropriate color for a trend value.
        
        Args:
            value: Trend value (positive or negative)
            
        Returns:
            CSS color string
        """
        if value is None or pd.isna(value):
            return "gray"
        
        if value > 10:
            return "red"  # Significant increase
        elif value > 0:
            return "orange"  # Moderate increase
        elif value < -10:
            return "green"  # Significant decrease
        elif value < 0:
            return "lightgreen"  # Moderate decrease
        else:
            return "gray"  # No change
        if not hourly_data or 'labels' not in hourly_data or 'values' not in hourly_data:
            return {
                'success': False,
                'error': 'Insufficient data for hourly chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': hourly_data['labels'],
                'datasets': [{
                    'label': 'Incident Count',
                    'data': hourly_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'primary')
                }]
            },
            'options': {
                'title': options['title'] or hourly_data.get('title', 'Hourly Incident Distribution'),
                'x_label': hourly_data.get('x_label', 'Hour of Day'),
                'y_label': hourly_data.get('y_label', 'Incident Count'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights if peak hour is available
        if 'peak_hour' in hourly_data:
            chart_data['insights'].append(
                f"Peak incident volume occurs at {hourly_data['peak_hour']}"
            )
        
        return chart_data
    
    def _generate_daily_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate daily workload distribution chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        daily_data = viz_data.get('workload', {}).get('daily', {})
        
        # Check if we have necessary data
        if not daily_data or 'labels' not in daily_data or 'values' not in daily_data:
            return {
                'success': False,
                'error': 'Insufficient data for daily chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': daily_data['labels'],
                'datasets': [{
                    'label': 'Incident Count',
                    'data': daily_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'primary')
                }]
            },
            'options': {
                'title': options['title'] or daily_data.get('title', 'Daily Incident Distribution'),
                'x_label': daily_data.get('x_label', 'Day of Week'),
                'y_label': daily_data.get('y_label', 'Incident Count'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights if peak day is available
        if 'peak_day' in daily_data:
            chart_data['insights'].append(
                f"{daily_data['peak_day']} has the highest incident volume"
            )
        
        return chart_data
    
    def _generate_monthly_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate monthly workload distribution chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        monthly_data = viz_data.get('workload', {}).get('monthly', {})
        
        # Check if we have necessary data
        if not monthly_data or 'labels' not in monthly_data or 'values' not in monthly_data:
            return {
                'success': False,
                'error': 'Insufficient data for monthly chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': monthly_data['labels'],
                'datasets': [{
                    'label': 'Incident Count',
                    'data': monthly_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'primary')
                }]
            },
            'options': {
                'title': options['title'] or monthly_data.get('title', 'Monthly Incident Distribution'),
                'x_label': monthly_data.get('x_label', 'Month'),
                'y_label': monthly_data.get('y_label', 'Incident Count'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights if peak month is available
        if 'peak_month' in monthly_data:
            chart_data['insights'].append(
                f"{monthly_data['peak_month']} has the highest incident volume"
            )
        
        return chart_data
    
    def _generate_category_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate category distribution chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        category_data = viz_data.get('workload', {}).get('category', {})
        
        # Check if we have necessary data
        if not category_data or 'labels' not in category_data or 'values' not in category_data:
            return {
                'success': False,
                'error': 'Insufficient data for category chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',  # Can also be 'pie' or 'doughnut'
            'data': {
                'labels': category_data['labels'],
                'datasets': [{
                    'label': 'Incident Count',
                    'data': category_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(category_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or category_data.get('title', 'Incident Distribution by Category'),
                'x_label': category_data.get('x_label', 'Category'),
                'y_label': category_data.get('y_label', 'Incident Count'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights if there are multiple categories
        if len(category_data['labels']) > 0:
            top_category = category_data['labels'][0]
            chart_data['insights'].append(
                f"{top_category} is the most common incident category"
            )
        
        return chart_data
    
    def _generate_assignee_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate assignee distribution chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        assignee_data = viz_data.get('workload', {}).get('assignee', {})
        
        # Check if we have necessary data
        if not assignee_data or 'labels' not in assignee_data or 'values' not in assignee_data:
            return {
                'success': False,
                'error': 'Insufficient data for assignee chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': assignee_data['labels'],
                'datasets': [{
                    'label': 'Incident Count',
                    'data': assignee_data['values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(assignee_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or assignee_data.get('title', 'Incident Distribution by Assignee'),
                'x_label': assignee_data.get('x_label', 'Assignee'),
                'y_label': assignee_data.get('y_label', 'Incident Count'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add insights if there are multiple assignees
        if len(assignee_data['labels']) > 0:
            top_assignee = assignee_data['labels'][0]
            chart_data['insights'].append(
                f"{top_assignee} handles the highest volume of incidents"
            )
        
        return chart_data
    
    def _generate_staffing_forecast_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate staffing forecast chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        forecast_data = viz_data.get('staffing', {}).get('daily', {})
        
        # Check if we have necessary data
        if not forecast_data or 'labels' not in forecast_data or 'staff_values' not in forecast_data:
            return {
                'success': False,
                'error': 'Insufficient data for staffing forecast chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'line',
            'data': {
                'labels': forecast_data['labels'],
                'datasets': [{
                    'label': 'Staff Needed',
                    'data': forecast_data['staff_values'],
                    'borderColor': self._get_color_scheme(options['color_scheme'], 'primary'),
                    'backgroundColor': 'rgba(0, 0, 0, 0)',
                    'borderWidth': 2
                }]
            },
            'options': {
                'title': options['title'] or forecast_data.get('title', 'Daily Staffing Needs Forecast'),
                'x_label': forecast_data.get('x_label', 'Date'),
                'y_label': forecast_data.get('y_label', 'Staff Needed'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add incidents line if available
        if 'incident_values' in forecast_data:
            chart_data['data']['datasets'].append({
                'label': 'Predicted Incidents',
                'data': forecast_data['incident_values'],
                'borderColor': self._get_color_scheme(options['color_scheme'], 'secondary'),
                'backgroundColor': 'rgba(0, 0, 0, 0)',
                'borderWidth': 2,
                'yAxisID': 'y1'  # Use secondary y-axis
            })
            
            # Add secondary y-axis
            chart_data['options']['y1_label'] = 'Predicted Incidents'
        
        return chart_data
    
    def _generate_staffing_category_chart_data(self, viz_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate staffing by category chart data.
        
        Args:
            viz_data: Visualization data from prepare_visualization_data
            options: Chart options
            
        Returns:
            Dictionary with chart data
        """
        category_data = viz_data.get('staffing', {}).get('category', {})
        
        # Check if we have necessary data
        if not category_data or 'labels' not in category_data or 'staff_values' not in category_data:
            return {
                'success': False,
                'error': 'Insufficient data for staffing category chart'
            }
        
        # Create chart data
        chart_data = {
            'success': True,
            'chart_type': 'bar',
            'data': {
                'labels': category_data['labels'],
                'datasets': [{
                    'label': 'Staff Needed',
                    'data': category_data['staff_values'],
                    'backgroundColor': self._get_color_scheme(options['color_scheme'], 'categorical', len(category_data['labels']))
                }]
            },
            'options': {
                'title': options['title'] or category_data.get('title', 'Staffing Needs by Category'),
                'x_label': category_data.get('x_label', 'Category'),
                'y_label': category_data.get('y_label', 'Staff Needed'),
                'height': options['height'],
                'width': options['width'],
                'show_legend': options['show_legend']
            },
            'insights': []
        }
        
        # Add resolution time as a secondary line if available
        if 'resolution_values' in category_data:
            chart_data['data']['datasets'].append({
                'label': 'Avg Resolution Time (hours)',
                'data': category_data['resolution_values'],
                'backgroundColor': 'rgba(0,0,0,0)',
                'borderColor': self._get_color_scheme(options['color_scheme'], 'secondary'),
                'type': 'line',
                'yAxisID': 'y1'  # Use secondary y-axis
            })
            
            # Add secondary y-axis
            chart_data['options']['y1_label'] = 'Resolution Time (hours)'
        
        # Add insights if there are categories
        if len(category_data['labels']) > 0:
            top_category = category_data['labels'][0]
            chart_data['insights'].append(
                f"{top_category} requires the most staff for incident resolution"
            )
        
        return chart_data


    def get_color_for_trend(self, value: float) -> str:
        """
        Get appropriate color for a trend value.
        
        Args:
            value: Trend value (positive or negative)
            
        Returns:
            CSS color string
        """
        if value is None or pd.isna(value):
            return "gray"
        
        if value > 10:
            return "red"  # Significant increase
        elif value > 0:
            return "orange"  # Moderate increase
        elif value < -10:
            return "green"  # Significant decrease
        elif value < 0:
            return "lightgreen"  # Moderate decrease
        else:
            return "gray"  # No change
    
    # Part 4: Integration and Finalization
    
    def analyze_data_for_resource_page(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis for the resource optimization page.
        This method combines all analysis methods into a single call for easier integration.
        
        Args:
            df: DataFrame containing incident data
            
        Returns:
            Dictionary with all analysis results and visualizations
        """
        # Initialize results
        results = {
            'success': False,
            'workload': None,
            'staffing': None,
            'skills': None,
            'insights': [],
            'visualizations': {},
            'metrics': {}
        }
        
        try:
            # Skip analysis if no data
            if df is None or df.empty:
                results['error'] = 'No data provided for analysis'
                return results
            
            # Find key columns automatically
            key_columns = self._detect_key_columns(df)
            
            # Step 1: Analyze workload distribution
            workload_result = self.analyze_workload_distribution(
                df=df,
                timestamp_col=key_columns.get('timestamp'),
                category_col=key_columns.get('category'),
                priority_col=key_columns.get('priority'),
                assignee_col=key_columns.get('assignee')
            )
            
            # Continue only if workload analysis succeeded
            if not workload_result.get('success', False):
                results['error'] = workload_result.get('error', 'Workload analysis failed')
                return results
            
            # Step 2: Predict staffing needs
            staffing_result = self.predict_staffing_needs(
                df,
                timestamp_col=key_columns.get('timestamp'),
                resolution_time_col=key_columns.get('resolution_time'),
                category_col=key_columns.get('category'),
                priority_col=key_columns.get('priority')
            )
            
            # Step 3: Get skill recommendations
            skill_result = self.get_skill_recommendations(
                df,
                resolution_time_col=key_columns.get('resolution_time'),
                category_col=key_columns.get('category'),
                assignee_col=key_columns.get('assignee')
            )
            
            # Step 4: Generate resource optimization insights
            insights = self.get_resource_optimization_insights(
                workload_result,
                staffing_result,
                skill_result
            )
            
            # Step 5: Prepare visualization data
            visualizations = {}
            
            # Workload visualizations
            workload_viz = self.prepare_visualization_data(analysis_results={
                'success': True,
                'temporal': workload_result.get('temporal', {}),
                'categorical': workload_result.get('categorical', {}),
                'assignment': workload_result.get('assignment', {})
            }, viz_type='workload')
            
            if workload_viz.get('success', False):
                visualizations['workload'] = workload_viz.get('workload', {})
            
            # Staffing visualizations
            if staffing_result.get('success', False):
                staffing_viz = self.prepare_visualization_data(
                    analysis_results={'success': True, 'predictions': staffing_result.get('predictions', {})},
                    viz_type='staffing'
                )
                
                if staffing_viz.get('success', False):
                    visualizations['staffing'] = staffing_viz.get('staffing', {})
            
            # Skills visualizations
            if skill_result.get('success', False):
                skills_viz = self.prepare_visualization_data(
                    analysis_results={'success': True, 'recommendations': skill_result.get('recommendations', {})},
                    viz_type='skills'
                )
                
                if skills_viz.get('success', False):
                    visualizations['skills'] = skills_viz.get('skills', {})
            
            # Step 6: Calculate key metrics for resource page
            metrics = self._calculate_resource_page_metrics(
                df, 
                workload_result, 
                staffing_result, 
                skill_result,
                key_columns
            )
            
            # Combine all results
            results = {
                'success': True,
                'workload': workload_result,
                'staffing': staffing_result,
                'skills': skill_result,
                'insights': insights,
                'visualizations': visualizations,
                'metrics': metrics,
                'key_columns': key_columns
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing comprehensive analysis: {str(e)}")
            results['error'] = f'Error performing comprehensive analysis: {str(e)}'
            return results
    
    def _detect_key_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect key columns in the dataframe for analysis.
        
        Args:
            df: DataFrame containing incident data
            
        Returns:
            Dictionary mapping column types to actual column names
        """
        key_columns = {}
        
        # Define patterns for each column type
        column_patterns = {
            'timestamp': ['created_date', 'timestamp', 'creation_date', 'open_date', 'reported_date'],
            'incident_id': ['incident_id', 'id', 'ticket_id', 'number', 'reference'],
            'category': ['category', 'type', 'incident_type', 'classification', 'group'],
            'priority': ['priority', 'severity', 'urgency', 'importance'],
            'status': ['status', 'state', 'condition'],
            'assignee': ['assignee', 'assigned_to', 'owner', 'resolver', 'handler'],
            'resolution_time': ['resolution_time', 'resolution_time_hours', 'time_to_resolve', 'duration'],
            'resolved_date': ['resolved_date', 'closed_date', 'end_date', 'completion_date']
        }
        
        # Search for each column type
        for col_type, patterns in column_patterns.items():
            for pattern in patterns:
                matches = [col for col in df.columns if pattern.lower() in col.lower()]
                if matches:
                    key_columns[col_type] = matches[0]
                    break
        
        # If resolution_time not found, but we have created_date and resolved_date, we can calculate it
        if 'resolution_time' not in key_columns and 'timestamp' in key_columns and 'resolved_date' in key_columns:
            created_col = key_columns['timestamp']
            resolved_col = key_columns['resolved_date']
            
            try:
                # Create a copy to avoid modifying the original dataframe
                calc_df = df.copy()
                
                # Convert to datetime
                if not pd.api.types.is_datetime64_any_dtype(calc_df[created_col]):
                    calc_df[created_col] = pd.to_datetime(calc_df[created_col], errors='coerce')
                
                if not pd.api.types.is_datetime64_any_dtype(calc_df[resolved_col]):
                    calc_df[resolved_col] = pd.to_datetime(calc_df[resolved_col], errors='coerce')
                    
                # Calculate resolution time in hours
                calc_df['calculated_resolution_time'] = (
                    calc_df[resolved_col] - calc_df[created_col]
                ).dt.total_seconds() / 3600  # Convert to hours
                
                # Add as a new column if calculation successful
                df['calculated_resolution_time'] = calc_df['calculated_resolution_time']
                key_columns['resolution_time'] = 'calculated_resolution_time'
                
            except Exception as e:
                self.logger.warning(f"Could not calculate resolution time: {str(e)}")
        
        return key_columns
    
    def _calculate_resource_page_metrics(self,
                                       df: pd.DataFrame,
                                       workload_result: Dict[str, Any],
                                       staffing_result: Dict[str, Any],
                                       skill_result: Dict[str, Any],
                                       key_columns: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate key metrics for the resource optimization page.
        
        Args:
            df: DataFrame containing incident data
            workload_result: Workload distribution analysis results
            staffing_result: Staffing prediction results
            skill_result: Skill recommendation results
            key_columns: Dictionary mapping column types to actual column names
            
        Returns:
            Dictionary with key metrics for the resource page
        """
        metrics = {
            'workload_balance': {},
            'staffing_needs': {},
            'skill_distribution': {}
        }
        
        # Workload balance metrics
        if workload_result.get('success', False):
            # Get workload distribution coefficient
            cv = workload_result.get('coefficient_of_variation')
            
            if cv is not None:
                # Calculate workload balance score (0-100)
                if cv < 0.3:
                    balance_score = 80 + (0.3 - cv) * 66.7  # 80-100
                elif cv < 0.5:
                    balance_score = 60 + (0.5 - cv) * 100  # 60-80
                elif cv < 0.8:
                    balance_score = 40 + (0.8 - cv) * 66.7  # 40-60
                elif cv < 1.2:
                    balance_score = 20 + (1.2 - cv) * 50  # 20-40
                else:
                    balance_score = max(0, 20 - (cv - 1.2) * 10)  # 0-20
                
                # Determine balance level
                if cv < 0.3:
                    balance_level = "Well Balanced"
                elif cv < 0.5:
                    balance_level = "Moderately Balanced"
                elif cv < 0.8:
                    balance_level = "Somewhat Imbalanced"
                elif cv < 1.2:
                    balance_level = "Imbalanced"
                else:
                    balance_level = "Highly Imbalanced"
                
                metrics['workload_balance'] = {
                    'balance_score': min(100, max(0, round(balance_score))),
                    'balance_level': balance_level,
                    'variation_coefficient': round(cv, 2)
                }
                
                # Additional workload metrics
                min_workload = workload_result.get('min')
                max_workload = workload_result.get('max')
                
                if min_workload is not None and max_workload is not None:
                    metrics['workload_balance'].update({
                        'min_workload': min_workload,
                        'max_workload': max_workload,
                        'workload_ratio': round(max_workload / min_workload if min_workload > 0 else float('inf'), 1)
                    })
        
        # Staffing needs metrics
        if staffing_result and staffing_result.get('success', False):
            predictions = staffing_result.get('predictions', {})
            
            if 'overall' in predictions:
                overall = predictions['overall']
                
                metrics['staffing_needs'] = {
                    'average_daily_staff': overall.get('average_daily_staff'),
                    'total_predicted_incidents': overall.get('total_predicted_incidents'),
                    'total_predicted_hours': overall.get('total_predicted_hours'),
                    'forecast_period_days': overall.get('forecast_period_days')
                }
                
                # Check if peak staffing day is available
                if 'by_day' in predictions:
                    days = predictions['by_day']
                    
                    if days:
                        max_staff_day = max(days.items(), key=lambda x: x[1]['staff_needed'])
                        date = max_staff_day[0]
                        staff = max_staff_day[1]['staff_needed']
                        day_name = max_staff_day[1]['day_name']
                        
                        metrics['staffing_needs']['peak_staffing_day'] = {
                            'date': date,
                            'day_name': day_name,
                            'staff_needed': staff
                        }
        
        # Skill distribution metrics
        if skill_result and skill_result.get('success', False):
            recommendations = skill_result.get('recommendations', {})
            
            if 'team_composition' in recommendations:
                composition = recommendations['team_composition']
                
                metrics['skill_distribution'] = {
                    'total_assignees': composition.get('total_assignees', 0),
                    'specialists_count': composition.get('specialists_count', 0),
                    'generalists_count': composition.get('generalists_count', 0),
                    'specialist_ratio': composition.get('specialist_ratio', 0)
                }
                
                # Check for skill gaps and performance metrics
                skill_gaps = []
                
                if 'skill_gaps' in recommendations:
                    for gap in recommendations['skill_gaps']:
                        if gap.get('issue') != 'no_clear_gaps':
                            skill_gaps.append({
                                'category': gap.get('category', 'Unknown'),
                                'issue': gap.get('issue'),
                                'recommendation': gap.get('recommendation')
                            })
                
                metrics['skill_distribution']['skill_gaps'] = skill_gaps
                
                # Add specialist performance if available
                if 'specialist_performance' in recommendations:
                    perf_data = recommendations['specialist_performance']
                    
                    performance_metrics = []
                    for category, perf in perf_data.items():
                        if perf.get('significant_improvement', False):
                            performance_metrics.append({
                                'category': category,
                                'specialist_time': perf.get('specialist_resolution_time'),
                                'nonspecialist_time': perf.get('nonspecialist_resolution_time'),
                                'improvement': perf.get('improvement_percentage')
                            })
                    
                    metrics['skill_distribution']['performance_metrics'] = performance_metrics
        
        return metrics
    
    def get_resource_page_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get all data needed for the resource optimization page in a format ready for display.
        This is the main integration method to be called from resource_page.py.
        
        Args:
            df: DataFrame containing incident data
            
        Returns:
            Dictionary with all data for the resource optimization page
        """
        # Perform comprehensive analysis
        analysis_results = self.analyze_data_for_resource_page(df)
        
        # If analysis failed, return error
        if not analysis_results.get('success', False):
            return {
                'success': False,
                'error': analysis_results.get('error', 'Analysis failed')
            }
        
        # Format data for resource page display
        page_data = {
            'success': True,
            'metrics': analysis_results.get('metrics', {}),
            'insights': self._format_insights_for_display(analysis_results.get('insights', [])),
            'charts': self._prepare_charts_for_display(analysis_results.get('visualizations', {})),
            'recommendations': self._format_recommendations_for_display(
                analysis_results.get('workload', {}),
                analysis_results.get('staffing', {}),
                analysis_results.get('skills', {})
            )
        }
        
        return page_data
    
    def _format_insights_for_display(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format insights for display on the resource optimization page.
        
        Args:
            insights: List of raw insight dictionaries
            
        Returns:
            List of formatted insight dictionaries ready for display
        """
        formatted_insights = []
        
        for insight in insights:
            # Determine severity for UI display
            severity = 'info'
            if insight.get('type') == 'error':
                severity = 'error'
            elif insight.get('type') in ['workload_balance', 'temporal_peak'] and insight.get('data', {}).get('coefficient_of_variation', 0) > 0.5:
                severity = 'warning'
            elif insight.get('type') == 'priority_distribution':
                severity = 'warning'
            elif insight.get('type') == 'skill_gap':
                severity = 'warning'
            
            # Create formatted insight
            formatted_insight = {
                'title': self._get_insight_title(insight),
                'content': insight.get('message', ''),
                'severity': severity,
                'type': insight.get('type', 'general'),
                'data': insight.get('data', {})
            }
            
            formatted_insights.append(formatted_insight)
        
        return formatted_insights
    
    def _get_insight_title(self, insight: Dict[str, Any]) -> str:
        """
        Generate a title for an insight based on its type.
        
        Args:
            insight: Raw insight dictionary
            
        Returns:
            Title string for the insight
        """
        insight_type = insight.get('type', '')
        
        # Map insight types to titles
        type_titles = {
            'temporal_peak': 'Peak Volume Pattern',
            'weekday_weekend_difference': 'Weekday vs Weekend Pattern',
            'category_distribution': 'Category Distribution',
            'priority_distribution': 'Priority Distribution',
            'workload_balance': 'Workload Balance',
            'specialist_distribution': 'Team Composition',
            'staffing_prediction': 'Staffing Forecast',
            'peak_staffing_day': 'Peak Staffing Need',
            'category_staffing': 'Category Staffing',
            'priority_staffing': 'Priority Staffing',
            'skill_gap': 'Skill Gap Identified',
            'specialist_impact': 'Specialist Performance'
        }
        
        # Get title based on type, or use capitalized type as fallback
        return type_titles.get(insight_type, insight_type.replace('_', ' ').title())
    
    def _prepare_charts_for_display(self, visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare chart data for display on the resource optimization page.
        
        Args:
            visualizations: Raw visualization data
            
        Returns:
            Dictionary with chart data ready for display
        """
        charts = {}
        
        # Process workload charts
        if 'workload' in visualizations:
            workload = visualizations['workload']
            
            for chart_type in ['hourly', 'daily', 'monthly', 'category', 'assignee']:
                if chart_type in workload:
                    chart_data = self._generate_chart_data(
                        f'workload_{chart_type}', 
                        {'success': True, 'workload': workload}
                    )
                    
                    if chart_data.get('success', False):
                        charts[f'workload_{chart_type}'] = chart_data
        
        # Process staffing charts
        if 'staffing' in visualizations:
            staffing = visualizations['staffing']
            
            for chart_type in ['daily', 'category', 'priority']:
                if chart_type in staffing:
                    chart_data = self._generate_chart_data(
                        f'staffing_{chart_type}',
                        {'success': True, 'staffing': staffing}
                    )
                    
                    if chart_data.get('success', False):
                        charts[f'staffing_{chart_type}'] = chart_data
        
        # Process skills charts
        if 'skills' in visualizations:
            skills = visualizations['skills']
            
            for chart_type in ['team_composition', 'category_specialists', 'performance_comparison']:
                if chart_type in skills:
                    skill_chart_type = 'skills_composition' if chart_type == 'team_composition' else \
                                      'skills_specialists' if chart_type == 'category_specialists' else \
                                      'skills_performance'
                    
                    chart_data = self._generate_chart_data(
                        skill_chart_type,
                        {'success': True, 'skills': skills}
                    )
                    
                    if chart_data.get('success', False):
                        charts[skill_chart_type] = chart_data
        
        return charts
    
    def _format_recommendations_for_display(self,
                                          workload_result: Dict[str, Any],
                                          staffing_result: Dict[str, Any],
                                          skill_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format recommendations for display on the resource optimization page.
        
        Args:
            workload_result: Workload distribution analysis results
            staffing_result: Staffing prediction results
            skill_result: Skill recommendation results
            
        Returns:
            List of formatted recommendation dictionaries ready for display
        """
        recommendations = []
        
        # Add workload balance recommendations
        if workload_result.get('success', False):
            cv = workload_result.get('coefficient_of_variation')
            
            if cv is not None:
                if cv > 0.5:  # Imbalanced workload
                    recommendations.append({
                        'category': 'workload_balance',
                        'title': 'Rebalance Workload Distribution',
                        'priority': 'high' if cv > 0.8 else 'medium',
                        'description': (
                            f"Current workload distribution has a high variation coefficient ({cv:.2f}). "
                            f"Consider redistributing incidents more evenly across team members."
                        )
                    })
            
            # Add temporal recommendations
            if 'temporal' in workload_result:
                temporal = workload_result['temporal']
                
                if 'hourly' in temporal:
                    peak_hour = temporal['hourly'].get('peak_hour')
                    peak_hour_formatted = temporal['hourly'].get('peak_hour_formatted')
                    peak_hour_pct = temporal['hourly'].get('peak_hour_percentage')
                    
                    if peak_hour is not None and peak_hour_pct and peak_hour_pct > 20:
                        recommendations.append({
                            'category': 'temporal_staffing',
                            'title': 'Adjust Staffing for Peak Hours',
                            'priority': 'medium',
                            'description': (
                                f"Incident volume peaks at {peak_hour_formatted}, accounting for {peak_hour_pct:.1f}% of all incidents. "
                                f"Consider increasing staff during this time period."
                            )
                        })
                
                if 'daily' in temporal:
                    peak_day = temporal['daily'].get('peak_day')
                    peak_day_pct = temporal['daily'].get('peak_day_percentage')
                    
                    if peak_day and peak_day_pct and peak_day_pct > 25:
                        recommendations.append({
                            'category': 'temporal_staffing',
                            'title': 'Adjust Staffing for Peak Days',
                            'priority': 'medium',
                            'description': (
                                f"{peak_day} has the highest incident volume, accounting for {peak_day_pct:.1f}% of all incidents. "
                                f"Consider scheduling more staff on {peak_day}s."
                            )
                        })
        
        # Add staffing recommendations
        if staffing_result and staffing_result.get('success', False):
            predictions = staffing_result.get('predictions', {})
            
            if 'overall' in predictions:
                overall = predictions['overall']
                avg_staff = overall.get('average_daily_staff')
                forecast_days = overall.get('forecast_period_days')
                
                if avg_staff and forecast_days:
                    recommendations.append({
                        'category': 'staffing_needs',
                        'title': 'Staffing Forecast',
                        'priority': 'high',
                        'description': (
                            f"Based on historical patterns, an average of {avg_staff} staff members will be needed daily "
                            f"over the next {forecast_days} days to maintain service levels."
                        )
                    })
            
            # Add category-specific staffing recommendations
            if 'by_category' in predictions:
                categories = predictions['by_category']
                
                # Find top category by staff needs
                if categories:
                    top_category = max(categories.items(), key=lambda x: x[1]['staff_needed'])
                    category_name = top_category[0]
                    staff_needed = top_category[1]['staff_needed']
                    
                    recommendations.append({
                        'category': 'category_staffing',
                        'title': f'Specialized Staffing for {category_name}',
                        'priority': 'medium',
                        'description': (
                            f"{category_name} incidents require {staff_needed} dedicated staff members. "
                            f"Ensure sufficient specialists are available for this category."
                        )
                    })
        
        # Add skill recommendations
        if skill_result and skill_result.get('success', False):
            skill_recs = skill_result.get('recommendations', {})
            
            # Add skill gap recommendations
            if 'skill_gaps' in skill_recs:
                for gap in skill_recs['skill_gaps']:
                    if gap.get('issue') != 'no_clear_gaps':
                        recommendations.append({
                            'category': 'skill_development',
                            'title': f"Skill Development: {gap.get('category', 'Multiple Categories')}",
                            'priority': 'high',
                            'description': gap.get('recommendation', '')
                        })
            
            # Add team composition recommendations
            if 'team_composition' in skill_recs:
                composition = skill_recs['team_composition']
                specialist_ratio = composition.get('specialist_ratio', 0)
                
                if specialist_ratio < 0.3:  # Too few specialists
                    recommendations.append({
                        'category': 'team_composition',
                        'title': 'Increase Specialist Coverage',
                        'priority': 'medium',
                        'description': (
                            f"Current specialist-to-generalist ratio is low ({specialist_ratio:.0%}). "
                            f"Consider developing more specialized skills within the team."
                        )
                    })
                elif specialist_ratio > 0.7:  # Too many specialists
                    recommendations.append({
                        'category': 'team_composition',
                        'title': 'Improve Team Flexibility',
                        'priority': 'medium',
                        'description': (
                            f"Current specialist-to-generalist ratio is high ({specialist_ratio:.0%}). "
                            f"Consider cross-training specialists to improve flexibility and coverage."
                        )
                    })
        
        return recommendations   


    
    def predict_staffing_needs(self,
                        df: pd.DataFrame,
                        timestamp_col: str = None,
                        resolution_time_col: str = None,
                        category_col: str = None,
                        priority_col: str = None,
                        forecast_days: int = 30) -> Dict[str, Any]:
        """
        Predict future staffing needs based on historical incident patterns and resolution times.
        
        Args:
            df: DataFrame containing incident data
            timestamp_col: Column name for incident timestamp/creation date
            resolution_time_col: Column name for incident resolution time
            category_col: Column name for incident category/type
            priority_col: Column name for incident priority/severity
            forecast_days: Number of days to forecast staffing needs
            
        Returns:
            Dictionary with staffing needs predictions
        """
        # Validate input data - use our own validation if the function doesn't exist
        try:
            is_valid, error_message = self.validate_dataframe(df)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_message,
                }
        except AttributeError:
            # If validate_dataframe doesn't exist, do basic validation
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No data provided for staffing needs prediction',
                }
        
        # The rest of the original method remains unchanged...
        # Find timestamp column if not provided
        if timestamp_col is None:
            timestamp_candidates = ['created_date', 'timestamp', 'creation_date', 'open_date', 'reported_date']
            for col in timestamp_candidates:
                if col in df.columns:
                    timestamp_col = col
                    break
        
        # Find resolution time column if not provided
        if resolution_time_col is None:
            resolution_candidates = ['resolution_time', 'resolution_time_hours', 'time_to_resolve', 'duration']
            for col in resolution_candidates:
                if col in df.columns:
                    resolution_time_col = col
                    break
            
            # If still not found, try to calculate from resolved_date if available
            if resolution_time_col is None and timestamp_col is not None:
                resolve_date_candidates = ['resolved_date', 'closed_date', 'end_date', 'completion_date']
                for col in resolve_date_candidates:
                    if col in df.columns:
                        try:
                            # Create a copy to avoid modifying the original dataframe
                            calc_df = df.copy()
                            
                            # Convert to datetime
                            if not pd.api.types.is_datetime64_any_dtype(calc_df[timestamp_col]):
                                calc_df[timestamp_col] = pd.to_datetime(calc_df[timestamp_col], errors='coerce')
                            
                            if not pd.api.types.is_datetime64_any_dtype(calc_df[col]):
                                calc_df[col] = pd.to_datetime(calc_df[col], errors='coerce')
                                
                            # Calculate resolution time in hours
                            calc_df['calculated_resolution_time'] = (
                                calc_df[col] - calc_df[timestamp_col]
                            ).dt.total_seconds() / 3600  # Convert to hours
                            
                            # Filter out negative or unreasonably large values
                            valid_mask = (
                                (calc_df['calculated_resolution_time'] >= 0) & 
                                (calc_df['calculated_resolution_time'] < 10000)  # Less than ~1 year
                            )
                            
                            if valid_mask.sum() > len(calc_df) * 0.5:  # At least 50% valid
                                resolution_time_col = 'calculated_resolution_time'
                                df = calc_df  # Use the updated dataframe
                                break
                        except Exception as e:
                            self.logger.warning(f"Could not calculate resolution time: {str(e)}")
        
        # Check if we have the required columns
        if timestamp_col is None or timestamp_col not in df.columns:
            return {
                'success': False,
                'error': 'No timestamp column found or provided for staffing prediction',
            }
            
        if resolution_time_col is None or resolution_time_col not in df.columns:
            return {
                'success': False,
                'error': 'No resolution time column found or could be calculated',
            }
        
        try:
            # Create a deep copy to avoid modifying the original DataFrame
            analysis_df = df.copy()
            
            # Ensure timestamp column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(analysis_df[timestamp_col]):
                analysis_df[timestamp_col] = pd.to_datetime(analysis_df[timestamp_col], errors='coerce')
            
            # Ensure resolution time is numeric
            analysis_df[resolution_time_col] = pd.to_numeric(analysis_df[resolution_time_col], errors='coerce')
            
            # Handle missing or invalid resolution times
            mean_resolution_time = analysis_df[resolution_time_col].mean()
            if pd.isna(mean_resolution_time) or mean_resolution_time <= 0:
                mean_resolution_time = 4.0  # Default to 4 hours if no valid data
            
            # Filter out invalid resolution times for analysis
            analysis_df = analysis_df[
                (analysis_df[resolution_time_col] > 0) & 
                (analysis_df[resolution_time_col] < 1000)  # Reasonable upper bound
            ].copy()
            
            # Create basic prediction result with minimal viable data
            result = {
                'success': True,
                'predictions': {
                    'overall': {
                        'average_daily_staff': round(len(df) / 30, 1),  # Rough estimate
                        'total_predicted_incidents': len(df),
                        'total_predicted_hours': round(len(df) * mean_resolution_time, 1),
                        'forecast_period_days': forecast_days
                    },
                    'by_day': {},
                    'by_category': {},
                    'by_priority': {}
                }
            }
            
            # Skip remaining complex analysis if data is too limited
            if len(analysis_df) < 10:
                return result
                
            # Extract time components for daily predictions
            try:
                analysis_df['date'] = analysis_df[timestamp_col].dt.date
                analysis_df['day_of_week'] = analysis_df[timestamp_col].dt.dayofweek
                
                # Calculate daily incident counts
                daily_counts = analysis_df.groupby('date').size()
                
                # Calculate day-of-week averages
                dow_counts = analysis_df.groupby('day_of_week').size()
                day_counts = analysis_df['day_of_week'].value_counts()
                
                # Average incidents per specific day of week
                avg_incidents_by_day = {}
                for day in range(7):
                    if day in dow_counts.index and day in day_counts.index:
                        # Average incidents for this day of week
                        avg_incidents_by_day[day] = dow_counts[day] / day_counts[day]
                    else:
                        # Use overall average if no data for this day
                        avg_incidents_by_day[day] = daily_counts.mean() if not daily_counts.empty else 3
                        
                # Generate predictions for the next forecast_days
                start_date = (datetime.now() + timedelta(days=1)).date()
                day_mapping = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                for day_offset in range(forecast_days):
                    forecast_date = start_date + timedelta(days=day_offset)
                    day_of_week = forecast_date.weekday()
                    day_name = day_mapping[day_of_week]
                    
                    # Predict incidents for this day
                    predicted_incidents = avg_incidents_by_day.get(day_of_week, 3)  # Default to 3 if unknown
                    
                    # Calculate staff hours needed
                    staff_hours = predicted_incidents * mean_resolution_time
                    
                    # Calculate required staff (assuming 8-hour workday)
                    staff_needed = max(1, round(staff_hours / 8))
                    
                    # Store prediction
                    result['predictions']['by_day'][str(forecast_date)] = {
                        'date': str(forecast_date),
                        'day_of_week': day_of_week,
                        'day_name': day_name,
                        'predicted_incidents': round(predicted_incidents, 1),
                        'predicted_hours': round(staff_hours, 1),
                        'staff_needed': staff_needed
                    }
            except Exception as e:
                self.logger.warning(f"Error generating daily predictions: {str(e)}")
            
            # Process category data if available
            if category_col and category_col in analysis_df.columns:
                try:
                    # Calculate category distribution
                    category_counts = analysis_df[category_col].value_counts()
                    total_count = len(analysis_df)
                    
                    for category, count in category_counts.items():
                        # Calculate category's resolution time
                        cat_df = analysis_df[analysis_df[category_col] == category]
                        cat_resolution_time = cat_df[resolution_time_col].mean()
                        if pd.isna(cat_resolution_time) or cat_resolution_time <= 0:
                            cat_resolution_time = mean_resolution_time  # Use overall average if invalid
                        
                        # Calculate category's share of incidents
                        category_pct = count / total_count
                        category_incidents = total_count * category_pct * (forecast_days / 30)  # Scale to forecast period
                        
                        # Calculate staff hours and needs
                        category_hours = category_incidents * cat_resolution_time
                        category_staff = max(1, round(category_hours / (8 * forecast_days)))
                        
                        # Store category prediction
                        result['predictions']['by_category'][str(category)] = {
                            'percentage': round(category_pct * 100, 1),
                            'predicted_incidents': round(category_incidents, 1),
                            'average_resolution_time': round(cat_resolution_time, 1),
                            'predicted_hours': round(category_hours, 1),
                            'staff_needed': category_staff
                        }
                except Exception as e:
                    self.logger.warning(f"Error generating category predictions: {str(e)}")
            
            # Process priority data if available
            if priority_col and priority_col in analysis_df.columns:
                try:
                    # Calculate priority distribution
                    priority_counts = analysis_df[priority_col].value_counts()
                    total_count = len(analysis_df)
                    
                    for priority, count in priority_counts.items():
                        # Calculate priority's resolution time
                        pri_df = analysis_df[analysis_df[priority_col] == priority]
                        pri_resolution_time = pri_df[resolution_time_col].mean()
                        if pd.isna(pri_resolution_time) or pri_resolution_time <= 0:
                            pri_resolution_time = mean_resolution_time  # Use overall average if invalid
                        
                        # Calculate priority's share of incidents
                        priority_pct = count / total_count
                        priority_incidents = total_count * priority_pct * (forecast_days / 30)  # Scale to forecast period
                        
                        # Calculate staff hours and needs
                        priority_hours = priority_incidents * pri_resolution_time
                        priority_staff = max(1, round(priority_hours / (8 * forecast_days)))
                        
                        # Store priority prediction
                        result['predictions']['by_priority'][str(priority)] = {
                            'percentage': round(priority_pct * 100, 1),
                            'predicted_incidents': round(priority_incidents, 1),
                            'average_resolution_time': round(pri_resolution_time, 1),
                            'predicted_hours': round(priority_hours, 1),
                            'staff_needed': priority_staff
                        }
                except Exception as e:
                    self.logger.warning(f"Error generating priority predictions: {str(e)}")
            
            # Update overall predictions
            total_staff = 0
            total_incidents = 0
            
            for day_data in result['predictions']['by_day'].values():
                total_staff += day_data['staff_needed']
                total_incidents += day_data['predicted_incidents']
            
            result['predictions']['overall']['average_daily_staff'] = round(total_staff / forecast_days, 1)
            result['predictions']['overall']['total_predicted_incidents'] = round(total_incidents, 1)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting staffing needs: {str(e)}")
            
            # Return minimal result structure with error
            return {
                'success': False,
                'error': f'Error during staffing needs prediction: {str(e)}',
                'predictions': {
                    'overall': {
                        'average_daily_staff': round(len(df) / 30, 1),  # Rough estimate
                        'total_predicted_incidents': len(df),
                        'forecast_period_days': forecast_days
                    }
                }
            }

    def _predict_future_staffing(self,
                            avg_incidents_by_day: Dict[int, float],
                            mean_resolution_time: float,
                            category_metrics: Dict[str, Dict],
                            priority_metrics: Dict[str, Dict],
                            forecast_days: int) -> Dict[str, Any]:
        """
        Generate staffing predictions for future days.
        
        Args:
            avg_incidents_by_day: Average incidents by day of week
            mean_resolution_time: Mean resolution time in hours
            category_metrics: Metrics by incident category
            priority_metrics: Metrics by incident priority
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with staffing predictions
        """
        # Start from tomorrow
        start_date = (datetime.now() + timedelta(days=1)).date()
        
        # Day of week names mapping
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Generate daily predictions
        daily_predictions = {}
        total_incidents = 0
        total_hours = 0
        total_staff = 0
        
        for day_offset in range(forecast_days):
            forecast_date = start_date + timedelta(days=day_offset)
            day_of_week = forecast_date.weekday()
            day_name = day_names[day_of_week]
            
            # Predict incidents for this day
            predicted_incidents = avg_incidents_by_day.get(day_of_week, 3)  # Default to 3 if unknown
            
            # Calculate staff hours needed
            staff_hours = predicted_incidents * mean_resolution_time
            
            # Calculate required staff (assuming 8-hour workday)
            staff_needed = max(1, round(staff_hours / 8))
            
            # Store prediction
            prediction = {
                'date': str(forecast_date),
                'day_of_week': day_of_week,
                'day_name': day_name,
                'predicted_incidents': round(predicted_incidents, 1),
                'predicted_hours': round(staff_hours, 1),
                'staff_needed': staff_needed
            }
            
            daily_predictions[str(forecast_date)] = prediction
            
            # Update totals
            total_incidents += predicted_incidents
            total_hours += staff_hours
            total_staff += staff_needed
        
        # Create summary metrics
        overall = {
            'total_predicted_incidents': round(total_incidents, 1),
            'total_predicted_hours': round(total_hours, 1),
            'average_daily_staff': round(total_staff / forecast_days, 1),
            'forecast_period_days': forecast_days
        }
        
        # Generate category-specific predictions if available
        category_predictions = {}
        if category_metrics:
            for category, metrics in category_metrics.items():
                # Calculate category's share of incidents
                category_pct = metrics['percentage'] / 100
                category_incidents = total_incidents * category_pct
                
                # Get category-specific resolution time
                cat_resolution_time = metrics.get('avg_resolution_time', mean_resolution_time)
                
                # Calculate staff hours and needs
                category_hours = category_incidents * cat_resolution_time
                category_staff = max(1, round(category_hours / (8 * forecast_days)))
                
                category_predictions[category] = {
                    'percentage': f"{metrics['percentage']}%",
                    'predicted_incidents': round(category_incidents, 1),
                    'average_resolution_time': round(cat_resolution_time, 1),
                    'predicted_hours': round(category_hours, 1),
                    'staff_needed': category_staff
                }
        
        # Generate priority-specific predictions if available
        priority_predictions = {}
        if priority_metrics:
            for priority, metrics in priority_metrics.items():
                # Calculate priority's share of incidents
                priority_pct = metrics['percentage'] / 100
                priority_incidents = total_incidents * priority_pct
                
                # Get priority-specific resolution time
                pri_resolution_time = metrics.get('avg_resolution_time', mean_resolution_time)
                
                # Calculate staff hours and needs
                priority_hours = priority_incidents * pri_resolution_time
                priority_staff = max(1, round(priority_hours / (8 * forecast_days)))
                
                priority_predictions[priority] = {
                    'percentage': f"{metrics['percentage']}%",
                    'predicted_incidents': round(priority_incidents, 1),
                    'average_resolution_time': round(pri_resolution_time, 1),
                    'predicted_hours': round(priority_hours, 1),
                    'staff_needed': priority_staff
                }
        
        return {
            'by_day': daily_predictions,
            'overall': overall,
            'by_category': category_predictions,
            'by_priority': priority_predictions
        } 
    


    def get_skill_recommendations(self,
                            df: pd.DataFrame,
                            resolution_time_col: str = None,
                            category_col: str = None,
                            assignee_col: str = None) -> Dict[str, Any]:
        """
        Generate skill development recommendations based on incident resolution analysis.
        
        Args:
            df: DataFrame containing incident data
            resolution_time_col: Column name for incident resolution time
            category_col: Column name for incident category/type
            assignee_col: Column name for incident assignee/owner
            
        Returns:
            Dictionary with skill development recommendations
        """
        # Validate input data - use our own validation if the function doesn't exist
        try:
            is_valid, error_message = self.validate_dataframe(df)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_message,
                }
        except AttributeError:
            # If validate_dataframe doesn't exist, do basic validation
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No data provided for skill recommendations',
                }
        
        # Find resolution time column if not provided
        if resolution_time_col is None:
            resolution_candidates = ['resolution_time', 'resolution_time_hours', 'time_to_resolve', 'duration']
            for col in resolution_candidates:
                if col in df.columns:
                    resolution_time_col = col
                    break
        
        # Find category column if not provided
        if category_col is None:
            category_candidates = ['category', 'type', 'incident_type', 'classification', 'group']
            for col in category_candidates:
                if col in df.columns:
                    category_col = col
                    break
        
        # Find assignee column if not provided
        if assignee_col is None:
            assignee_candidates = ['assignee', 'assigned_to', 'owner', 'resolver', 'handler']
            for col in assignee_candidates:
                if col in df.columns:
                    assignee_col = col
                    break
        
        # Check if we have the required columns
        required_columns = []
        if category_col is None or category_col not in df.columns:
            required_columns.append('category')
            
        if assignee_col is None or assignee_col not in df.columns:
            required_columns.append('assignee')
        
        if required_columns:
            return {
                'success': False,
                'error': f'Missing required columns for skill recommendations: {", ".join(required_columns)}',
            }
        
        try:
            # Create a deep copy to avoid modifying the original DataFrame
            analysis_df = df.copy()
            
            # Initialize result
            result = {
                'success': True,
                'recommendations': {
                    'team_composition': {},
                    'skill_gaps': []
                }
            }
            
            # Create minimal team composition to ensure structure
            result['recommendations']['team_composition'] = {
                'total_assignees': 0,
                'specialists_count': 0,
                'generalists_count': 0,
                'specialist_ratio': 0,
                'specialist_by_category': {}
            }
            
            # Add minimal skill gap to ensure structure
            result['recommendations']['skill_gaps'] = [{
                'issue': 'no_clear_gaps',
                'recommendation': 'No specific skill gaps identified in the current data'
            }]
            
            # Create minimal structure in case resolution time analysis fails
            if resolution_time_col:
                result['recommendations']['specialist_performance'] = {}
            
            # Validate that assignee and category have sensible values
            if len(analysis_df[assignee_col].unique()) < 2 or len(analysis_df[category_col].unique()) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient variation in assignee or category data for skill analysis',
                    'recommendations': result['recommendations']  # Return the minimal structure
                }
            
            # Calculate resolution time if available
            has_resolution_data = False
            if resolution_time_col and resolution_time_col in analysis_df.columns:
                analysis_df[resolution_time_col] = pd.to_numeric(analysis_df[resolution_time_col], errors='coerce')
                has_resolution_data = len(analysis_df[resolution_time_col].dropna()) > 0
            
            # Analyze team composition
            try:
                # Ensure string values for categories
                analysis_df[category_col] = analysis_df[category_col].astype(str)
                analysis_df[assignee_col] = analysis_df[assignee_col].astype(str)
                
                # Calculate assignee distribution
                assignee_counts = analysis_df[assignee_col].value_counts()
                total_assignees = len(assignee_counts)
                result['recommendations']['team_composition']['total_assignees'] = total_assignees
                
                # Calculate category expertise
                # Create contingency table of assignee vs category
                try:
                    assignee_category = pd.crosstab(
                        analysis_df[assignee_col],
                        analysis_df[category_col],
                        normalize='index'
                    )
                    
                    # Identify specialists and generalists
                    specialists = []
                    specialist_categories = {}
                    
                    for assignee in assignee_category.index:
                        if len(assignee_category.columns) > 0:  # Make sure there are columns
                            max_category_pct = assignee_category.loc[assignee].max()
                            max_category = assignee_category.loc[assignee].idxmax()
                            
                            if max_category_pct > 0.7:  # 70% threshold for specialist
                                specialists.append(assignee)
                                
                                if max_category not in specialist_categories:
                                    specialist_categories[max_category] = []
                                    
                                specialist_categories[max_category].append(assignee)
                    
                    # Calculate specialist metrics
                    specialists_count = len(specialists)
                    generalists_count = total_assignees - specialists_count
                    
                    # Store team composition in results
                    result['recommendations']['team_composition'] = {
                        'total_assignees': total_assignees,
                        'specialists_count': specialists_count,
                        'generalists_count': generalists_count,
                        'specialist_ratio': round(specialists_count / total_assignees, 2) if total_assignees > 0 else 0,
                        'specialist_by_category': {
                            category: len(assignees) for category, assignees in specialist_categories.items()
                        }
                    }
                    
                    # Analyze resolution performance by specialist status if available
                    if has_resolution_data:
                        specialist_performance = {}
                        
                        for category, assignees in specialist_categories.items():
                            # Skip categories with too few specialists
                            if len(assignees) < 2:
                                continue
                            
                            # Get incidents for this category
                            category_incidents = analysis_df[analysis_df[category_col] == category]
                            
                            # Skip categories with too few incidents
                            if len(category_incidents) < 10:
                                continue
                            
                            # Split by specialist/non-specialist
                            specialist_incidents = category_incidents[category_incidents[assignee_col].isin(assignees)]
                            nonspecialist_incidents = category_incidents[~category_incidents[assignee_col].isin(assignees)]
                            
                            # Need enough data in both groups
                            if len(specialist_incidents) < 5 or len(nonspecialist_incidents) < 5:
                                continue
                            
                            # Calculate mean resolution times
                            specialist_mean = specialist_incidents[resolution_time_col].mean()
                            nonspecialist_mean = nonspecialist_incidents[resolution_time_col].mean()
                            
                            # Skip if either mean is invalid
                            if pd.isna(specialist_mean) or pd.isna(nonspecialist_mean) or specialist_mean <= 0 or nonspecialist_mean <= 0:
                                continue
                            
                            # Calculate improvement percentage
                            improvement = (nonspecialist_mean - specialist_mean) / nonspecialist_mean if nonspecialist_mean > 0 else 0
                            
                            specialist_performance[category] = {
                                'specialist_count': len(assignees),
                                'specialist_resolution_time': float(specialist_mean),
                                'nonspecialist_resolution_time': float(nonspecialist_mean),
                                'improvement_percentage': f"{improvement * 100:.1f}%",
                                'significant_improvement': bool(improvement > 0.2)  # 20% threshold
                            }
                        
                        result['recommendations']['specialist_performance'] = specialist_performance
                    
                    # Generate skill gap recommendations
                    skill_gaps = []
                    
                    # Check category coverage
                    for category, count in analysis_df[category_col].value_counts().items():
                        category_pct = count / len(analysis_df) * 100
                        specialists_for_category = len(specialist_categories.get(category, []))
                        
                        # High volume category with few specialists
                        if category_pct > 15 and specialists_for_category < 2:
                            skill_gaps.append({
                                'category': category,
                                'issue': 'specialist_shortage',
                                'current_specialists': specialists_for_category,
                                'volume_percentage': f"{category_pct:.1f}%",
                                'recommendation': f"Develop more specialists for {category} incidents, which represent {category_pct:.1f}% of all incidents."
                            })
                    
                    # Check performance gaps
                    if 'specialist_performance' in result['recommendations']:
                        for category, perf in result['recommendations']['specialist_performance'].items():
                            if perf.get('significant_improvement', False):
                                skill_gaps.append({
                                    'category': category,
                                    'issue': 'performance_gap',
                                    'improvement_potential': perf.get('improvement_percentage'),
                                    'recommendation': f"Train more team members in {category} skills, as specialists resolve these incidents {perf.get('improvement_percentage', '0%')} faster than non-specialists."
                                })
                    
                    # Check team composition balance
                    specialist_ratio = result['recommendations']['team_composition'].get('specialist_ratio', 0)
                    
                    if specialist_ratio < 0.3:  # Less than 30% specialists
                        skill_gaps.append({
                            'issue': 'low_specialization',
                            'recommendation': f"The team has a low specialist ratio ({specialist_ratio:.0%}). Consider developing more specialized skills within the team for common incident categories."
                        })
                    elif specialist_ratio > 0.7:  # More than 70% specialists
                        skill_gaps.append({
                            'issue': 'high_specialization',
                            'recommendation': f"The team has a high specialist ratio ({specialist_ratio:.0%}). Consider cross-training specialists to improve flexibility and coverage."
                        })
                    
                    # Update skill gaps in result
                    if skill_gaps:
                        result['recommendations']['skill_gaps'] = skill_gaps
                    
                except Exception as cross_tab_error:
                    self.logger.warning(f"Error analyzing assignee-category distribution: {str(cross_tab_error)}")
            
            except Exception as comp_error:
                self.logger.error(f"Error analyzing team composition: {str(comp_error)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating skill recommendations: {str(e)}")
            return {
                'success': False,
                'error': f'Error during skill recommendation generation: {str(e)}',
                'recommendations': {
                    'team_composition': {
                        'total_assignees': 0,
                        'specialists_count': 0,
                        'generalists_count': 0,
                        'specialist_ratio': 0
                    },
                    'skill_gaps': [{
                        'issue': 'error',
                        'recommendation': f"Could not analyze skills due to an error: {str(e)}"
                    }]
                }
            }

    def safe_percentage_calculation(
        series: pd.Series, 
        total_count: int, 
        decimal_places: int = 1
    ) -> float:
        """
        Safely calculate percentage with error handling.
        
        Args:
            series: Input Series
            total_count: Total count for percentage calculation
            decimal_places: Decimal places for rounding
        
        Returns:
            Calculated percentage
        """
        try:
            return round(float(series / total_count * 100), decimal_places) if total_count > 0 else 0.0
        except Exception:
            return 0.0


    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, str]:
        """
        Validate if the dataframe meets the basic requirements for analysis.
        
        Args:
            df: DataFrame to validate
            required_columns: List of column names that are required
            
        Returns:
            Tuple containing:
                - Boolean indicating if dataframe is valid
                - Error message if invalid, empty string if valid
        """
        # Check if dataframe is None or empty
        if df is None:
            return False, "No data provided for analysis"
        
        if df.empty:
            return False, "Empty dataframe provided for analysis"
        
        # Check for required columns if specified
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check if dataframe has reasonable size for analysis
        if len(df) < 5:
            return False, f"Insufficient data for analysis: only {len(df)} records"
        
        # All checks passed
        return True, ""
    


    def _generate_skill_gap_recommendations(self,
                                        df: pd.DataFrame,
                                        category_col: str,
                                        assignee_col: str,
                                        resolution_time_col: str = None,
                                        specialist_categories: Dict[str, List[str]] = None,
                                        specialist_performance: Dict[str, Dict] = None,
                                        result: Dict[str, Any] = None) -> None:
        """
        Generate skill gap recommendations based on incident data analysis.
        
        Args:
            df: DataFrame containing incident data
            category_col: Column name for incident category
            assignee_col: Column name for incident assignee
            resolution_time_col: Column name for incident resolution time
            specialist_categories: Dictionary mapping categories to specialist assignees
            specialist_performance: Performance comparison between specialists and non-specialists
            result: Dictionary to update with skill gap recommendations
        """
        skill_gaps = []
        
        # Calculate category distribution
        category_counts = df[category_col].value_counts()
        category_pct = (category_counts / len(df) * 100).round(1)
        
        # Calculate category complexity if resolution time is available
        category_complexity = {}
        if resolution_time_col:
            category_resolution = df.groupby(category_col)[resolution_time_col].agg(['mean', 'count'])
            overall_mean = df[resolution_time_col].mean()
            
            for category, data in category_resolution.iterrows():
                # Skip categories with too few incidents
                if data['count'] < 5:
                    continue
                
                # Calculate relative complexity
                relative_complexity = data['mean'] / overall_mean if overall_mean > 0 else 1.0
                
                category_complexity[category] = {
                    'mean_resolution_time': float(data['mean']),
                    'incident_count': int(data['count']),
                    'relative_complexity': float(relative_complexity),
                    'complexity_level': (
                        'High' if relative_complexity > 1.3 else
                        'Medium' if relative_complexity > 0.7 else
                        'Low'
                    )
                }
        
        # Identify skill gaps based on category complexity and specialist coverage
        if category_complexity and specialist_categories:
            for category, complexity in category_complexity.items():
                # Skip low-complexity categories
                if complexity['complexity_level'] == 'Low':
                    continue
                
                # Get category volume
                category_volume = category_counts.get(category, 0)
                category_percentage = category_pct.get(category, 0)
                
                # Check if this is a high-volume category
                is_high_volume = category_percentage > 15  # More than 15% of incidents
                
                # Check specialist coverage
                specialist_count = len(specialist_categories.get(category, []))
                has_specialist_gap = specialist_count < 2  # Need at least 2 specialists
                
                # Check performance gap if specialist performance data is available
                has_performance_gap = False
                performance_improvement = None
                if specialist_performance and category in specialist_performance:
                    perf_data = specialist_performance[category]
                    has_performance_gap = perf_data.get('significant_improvement', False)
                    performance_improvement = perf_data.get('improvement_percentage')
                
                # Create recommendation if there's a gap
                if has_specialist_gap and (complexity['complexity_level'] == 'High' or is_high_volume):
                    skill_gaps.append({
                        'category': category,
                        'issue': 'specialist_shortage',
                        'current_specialists': specialist_count,
                        'relative_complexity': complexity['relative_complexity'],
                        'complexity_level': complexity['complexity_level'],
                        'volume_percentage': f"{category_percentage}%",
                        'recommendation': (
                            f"Develop more specialists for {category} incidents, which have "
                            f"{complexity['relative_complexity']:.1f}x longer resolution times than average "
                            f"and represent {category_percentage}% of all incidents."
                        )
                    })
                    
                # Create recommendation for performance gap
                elif has_performance_gap and performance_improvement:
                    skill_gaps.append({
                        'category': category,
                        'issue': 'performance_gap',
                        'improvement_potential': performance_improvement,
                        'recommendation': (
                            f"Train more team members in {category} skills, as specialists resolve "
                            f"these incidents {performance_improvement} faster than non-specialists."
                        )
                    })
        
        # If no specific gaps were identified, provide general recommendation
        if not skill_gaps:
            # Check team composition for general recommendations
            team_composition = result['recommendations']['team_composition']
            specialist_ratio = team_composition.get('specialist_ratio', 0)
            
            if specialist_ratio < 0.3:  # Less than 30% specialists
                skill_gaps.append({
                    'issue': 'low_specialization',
                    'recommendation': (
                        f"The team has a low specialist ratio ({specialist_ratio:.0%}). Consider developing "
                        f"more specialized skills within the team for common incident categories."
                    )
                })
            elif specialist_ratio > 0.7:  # More than 70% specialists
                skill_gaps.append({
                    'issue': 'high_specialization',
                    'recommendation': (
                        f"The team has a high specialist ratio ({specialist_ratio:.0%}). Consider cross-training "
                        f"specialists to improve flexibility and coverage."
                    )
                })
            else:
                skill_gaps.append({
                    'issue': 'no_clear_gaps',
                    'recommendation': (
                        "The team has a balanced specialist-to-generalist ratio. Continue to monitor "
                        "category-specific performance and adjust specialization as needed."
                    )
                })
        
        # Add skill gaps to result
        result['recommendations']['skill_gaps'] = skill_gaps




    def get_resource_optimization_insights(
        self,
        workload_result: Dict[str, Any] = None,
        staffing_result: Dict[str, Any] = None,
        skill_result: Dict[str, Any] = None
    ) -> List[Dict]:
        # Validate input results
        if not workload_result or not workload_result.get('success'):
            return [{
                'type': 'error',
                'message': 'Invalid or unsuccessful workload analysis'
            }]
        
        try:
            insights = []
            
            # Process workload distribution insights
            if workload_result.get('data', {}).get('top_category'):
                insights.append({
                    'type': 'workload_distribution',
                    'message': f"Top category: {workload_result['data']['top_category']} " +
                            f"({workload_result['data']['top_category_percentage']:.1f}%)"
                })
            
            # Additional processing for staffing and skill results...
            
            return insights
        
        except Exception as e:
            return [{
                'type': 'error',
                'message': f'Resource optimization insights generation error: {str(e)}'
            }]

    def _generate_workload_insights(self, workload_result: Dict[str, Any]) -> List[Dict]:
        """
        Generate insights from workload distribution analysis.
        
        Args:
            workload_result: Result from analyze_workload_distribution method
            
        Returns:
            List of dictionaries with workload optimization insights
        """
        insights = []
        
        # Extract workload distribution components
        if 'temporal' in workload_result and workload_result['temporal']:
            temporal = workload_result['temporal']
            
            # Peak hour insight
            if 'hourly' in temporal:
                peak_hour = temporal['hourly'].get('peak_hour')
                peak_hour_formatted = temporal['hourly'].get('peak_hour_formatted')
                peak_hour_pct = temporal['hourly'].get('peak_hour_percentage')
                
                if peak_hour is not None and peak_hour_pct:
                    insights.append({
                        'type': 'temporal_peak',
                        'subtype': 'hour',
                        'data': {
                            'peak_hour': peak_hour,
                            'percentage': peak_hour_pct
                        },
                        'message': f"Peak incident volume occurs at {peak_hour_formatted}, accounting for {peak_hour_pct:.1f}% of all incidents"
                    })
            
            # Peak day insight
            if 'daily' in temporal:
                peak_day = temporal['daily'].get('peak_day')
                peak_day_pct = temporal['daily'].get('peak_day_percentage')
                
                if peak_day and peak_day_pct:
                    insights.append({
                        'type': 'temporal_peak',
                        'subtype': 'day',
                        'data': {
                            'peak_day': peak_day,
                            'percentage': peak_day_pct
                        },
                        'message': f"{peak_day} has the highest incident volume, accounting for {peak_day_pct:.1f}% of all incidents"
                    })
            
            # Weekday vs weekend insight
            if 'weekday_vs_weekend' in temporal:
                weekday_vs_weekend = temporal['weekday_vs_weekend']
                weekday_avg = weekday_vs_weekend.get('weekday_daily_average')
                weekend_avg = weekday_vs_weekend.get('weekend_daily_average')
                ratio = weekday_vs_weekend.get('weekday_to_weekend_ratio')
                
                if weekday_avg and weekend_avg and ratio and ratio > 1.5:
                    insights.append({
                        'type': 'weekday_weekend_difference',
                        'data': {
                            'weekday_avg': weekday_avg,
                            'weekend_avg': weekend_avg,
                            'ratio': ratio
                        },
                        'message': f"Weekdays average {weekday_avg:.1f} incidents per day, {ratio:.1f}x more than weekends ({weekend_avg:.1f} per day)"
                    })
        
        # Generate category insights
        if 'categorical' in workload_result and workload_result['categorical']:
            categorical = workload_result['categorical']
            
            # Top category insight
            if 'category' in categorical:
                category_data = categorical['category']
                top_category = category_data.get('top_category')
                top_category_pct = category_data.get('top_category_percentage')
                
                if top_category and top_category_pct:
                    insights.append({
                        'type': 'category_distribution',
                        'data': {
                            'top_category': top_category,
                            'percentage': top_category_pct
                        },
                        'message': f"{top_category} is the most common incident type, representing {top_category_pct:.1f}% of all incidents"
                    })
                
                # Category-specific peak times
                if 'peak_hours_by_category' in category_data:
                    for category, data in category_data['peak_hours_by_category'].items():
                        insights.append({
                            'type': 'temporal_category_peak',
                            'data': {
                                'category': category,
                                'peak_hour': data.get('peak_hour'),
                                'peak_hour_formatted': data.get('peak_hour_formatted')
                            },
                            'message': f"{category} incidents peak at {data.get('peak_hour_formatted', str(data.get('peak_hour')) + ':00')}"
                        })
            
            # Priority distribution insight
            if 'priority' in categorical:
                priority_data = categorical['priority']
                high_priority_pct = priority_data.get('high_priority_percentage')
                
                if high_priority_pct and high_priority_pct > 25:  # If more than 25% are high priority
                    insights.append({
                        'type': 'priority_distribution',
                        'data': {
                            'high_priority_percentage': high_priority_pct
                        },
                        'message': f"High-priority incidents comprise {high_priority_pct:.1f}% of total volume, requiring attention to resource allocation"
                    })
        
        # Generate workload balance insights
        if 'coefficient_of_variation' in workload_result:
            cv = workload_result.get('coefficient_of_variation')
            min_workload = workload_result.get('min')
            max_workload = workload_result.get('max')
            
            if cv and cv > 0.5 and min_workload is not None and max_workload is not None:
                insights.append({
                    'type': 'workload_balance',
                    'data': {
                        'coefficient_of_variation': cv,
                        'min_workload': min_workload,
                        'max_workload': max_workload
                    },
                    'message': f"Workload is unevenly distributed (variation coefficient: {cv:.2f}), with assignees handling between {min_workload} and {max_workload} incidents"
                })
        
        # Generate assignment insights
        if 'assignment' in workload_result and workload_result['assignment']:
            assignment = workload_result['assignment']
            
            # Specialist distribution
            if 'specialists_count' in assignment and 'generalists_count' in assignment:
                specialists = assignment.get('specialists_count', 0)
                generalists = assignment.get('generalists_count', 0)
                total = specialists + generalists
                
                if total > 0 and specialists > 0:
                    specialist_ratio = specialists / total
                    
                    if specialist_ratio < 0.3:  # Less than 30% specialists
                        insights.append({
                            'type': 'specialist_distribution',
                            'data': {
                                'specialist_count': specialists,
                                'generalist_count': generalists,
                                'specialist_ratio': specialist_ratio
                            },
                            'message': f"Team has a low proportion of specialists ({specialists} of {total} team members, {specialist_ratio:.0%}), which may impact resolution efficiency"
                        })
                    elif specialist_ratio > 0.7:  # More than 70% specialists
                        insights.append({
                            'type': 'specialist_distribution',
                            'data': {
                                'specialist_count': specialists,
                                'generalist_count': generalists,
                                'specialist_ratio': specialist_ratio
                            },
                            'message': f"Team has a high proportion of specialists ({specialists} of {total} team members, {specialist_ratio:.0%}), which may limit flexibility"
                        })
        
        return insights

    def _generate_staffing_insights(self, staffing_result: Dict[str, Any]) -> List[Dict]:
        """
        Generate insights from staffing needs prediction.
        
        Args:
            staffing_result: Result from predict_staffing_needs method
            
        Returns:
            List of dictionaries with staffing optimization insights
        """
        insights = []
        
        # Extract predictions
        predictions = staffing_result.get('predictions', {})
        
        # Overall staffing needs
        if 'overall' in predictions:
            overall = predictions['overall']
            avg_staff = overall.get('average_daily_staff')
            total_incidents = overall.get('total_predicted_incidents')
            forecast_days = overall.get('forecast_period_days')
            
            if avg_staff and total_incidents and forecast_days:
                insights.append({
                    'type': 'staffing_prediction',
                    'data': {
                        'avg_daily_staff': avg_staff,
                        'total_incidents': total_incidents,
                        'forecast_days': forecast_days
                    },
                    'message': f"Estimated {avg_staff} staff needed daily to handle {total_incidents:.0f} incidents over the next {forecast_days} days"
                })
        
        # Daily variations
        if 'by_day' in predictions:
            days = predictions['by_day']
            # Find day with highest staffing need
            if days:
                max_staff_day = max(days.items(), key=lambda x: x[1]['staff_needed'])
                date = max_staff_day[0]
                staff = max_staff_day[1]['staff_needed']
                day_name = max_staff_day[1]['day_name']
                
                insights.append({
                    'type': 'peak_staffing_day',
                    'data': {
                        'date': date,
                        'day': day_name,
                        'staff_needed': staff
                    },
                    'message': f"Highest staffing need is on {day_name} ({date}) with {staff} staff required"
                })
        
        # Category-specific staffing
        if 'by_category' in predictions:
            categories = predictions['by_category']
            # Find category with highest staffing need
            if categories:
                max_staff_category = max(categories.items(), key=lambda x: x[1]['staff_needed'])
                category = max_staff_category[0]
                staff = max_staff_category[1]['staff_needed']
                
                insights.append({
                    'type': 'category_staffing',
                    'data': {
                        'category': category,
                        'staff_needed': staff
                    },
                    'message': f"{category} incidents require the most resources with {staff} dedicated staff recommended"
                })
        
        # Priority-specific staffing
        if 'by_priority' in predictions:
            priorities = predictions['by_priority']
            # Find high priority staffing needs
            high_priority_entries = {k: v for k, v in priorities.items() 
                                if k.lower() in ['critical', 'high', 'p1', '1', 'p2', '2']}
            
            if high_priority_entries:
                total_high_priority_staff = sum(entry['staff_needed'] for entry in high_priority_entries.values())
                
                if total_high_priority_staff > 0:
                    insights.append({
                        'type': 'priority_staffing',
                        'data': {
                            'high_priority_staff': total_high_priority_staff
                        },
                        'message': f"High-priority incidents require at least {total_high_priority_staff} dedicated staff to maintain service levels"
                    })
        
        return insights

    def _generate_skill_insights(self, skill_result: Dict[str, Any]) -> List[Dict]:
        """
        Generate insights from skill recommendation analysis.
        
        Args:
            skill_result: Result from get_skill_recommendations method
            
        Returns:
            List of dictionaries with skill optimization insights
        """
        insights = []
        
        # Extract recommendations
        recommendations = skill_result.get('recommendations', {})
        
        # Skills gaps
        if 'skill_gaps' in recommendations:
            for gap in recommendations['skill_gaps']:
                if gap.get('issue') != 'no_clear_gaps':
                    insights.append({
                        'type': 'skill_gap',
                        'data': gap,
                        'message': gap['recommendation']
                    })
        
        # Team composition insights
        if 'team_composition' in recommendations:
            composition = recommendations['team_composition']
            
            if 'specialist_performance' in recommendations:
                # Find category with biggest specialist advantage
                specialist_perf = recommendations['specialist_performance']
                best_categories = []
                
                for category, perf in specialist_perf.items():
                    if perf.get('significant_improvement'):
                        best_categories.append((category, perf))
                
                if best_categories:
                    # Sort by improvement percentage
                    best_categories.sort(
                        key=lambda x: float(x[1]['improvement_percentage'].rstrip('%')), 
                        reverse=True
                    )
                    top_category, top_perf = best_categories[0]
                    
                    insights.append({
                        'type': 'specialist_impact',
                        'data': {
                            'category': top_category,
                            'improvement': top_perf['improvement_percentage']
                        },
                        'message': f"Specialists resolve {top_category} incidents {top_perf['improvement_percentage']} faster than non-specialists"
                    })
        
        return insights   