#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing module for the Incident Management Analytics application.
This module handles transformation, aggregation, and feature extraction
from incident data for various analyses.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from collections import Counter

from data.data_validator import check_data_sufficiency

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing incident data for various analyses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataProcessor with application configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
    
    def prepare_data_for_analysis(self, df: pd.DataFrame, analysis_type: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Prepare data for a specific type of analysis.
        
        Args:
            df: DataFrame with incident data
            analysis_type: Type of analysis to prepare for
            
        Returns:
            Tuple containing:
                - Processed DataFrame for the analysis (or None if insufficient)
                - Dictionary with metadata and processing results
        """
        # Check if data is sufficient for this analysis
        sufficiency_check = check_data_sufficiency(df, analysis_type)
        if not sufficiency_check["is_sufficient"]:
            logger.warning(f"Insufficient data for {analysis_type}: {sufficiency_check['reason']}")
            return None, {
                "success": False,
                "error": sufficiency_check["reason"],
                "recommendations": sufficiency_check["recommendations"],
                "analysis_type": analysis_type
            }
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Process data based on analysis type
        try:
            if analysis_type == "time_analysis":
                processed_df, metadata = self._prepare_time_analysis(processed_df)
            elif analysis_type == "resolution_time_analysis":
                processed_df, metadata = self._prepare_resolution_time_analysis(processed_df)
            elif analysis_type == "priority_analysis":
                processed_df, metadata = self._prepare_priority_analysis(processed_df)
            elif analysis_type == "resource_analysis":
                processed_df, metadata = self._prepare_resource_analysis(processed_df)
            elif analysis_type == "category_analysis":
                processed_df, metadata = self._prepare_category_analysis(processed_df)
            elif analysis_type == "system_analysis":
                processed_df, metadata = self._prepare_system_analysis(processed_df)
            elif analysis_type == "automation_opportunity":
                processed_df, metadata = self._prepare_automation_analysis(processed_df)
            elif analysis_type == "text_analysis":
                processed_df, metadata = self._prepare_text_analysis(processed_df)
            else:
                return None, {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}",
                    "analysis_type": analysis_type
                }
            
            metadata.update({
                "success": True,
                "analysis_type": analysis_type,
                "row_count": len(processed_df)
            })
            
            return processed_df, metadata
            
        except Exception as e:
            logger.error(f"Error preparing data for {analysis_type}: {str(e)}", exc_info=True)
            return None, {
                "success": False,
                "error": f"Error processing data: {str(e)}",
                "analysis_type": analysis_type
            }
    
    def _prepare_time_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for time-based analysis with robust date handling.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Ensure we have a datetime column
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        for date_col in date_columns:
            try:
                # Convert to datetime, ensuring we work with a Series
                current_column = df[date_col]
                
                # Check if already datetime
                if not pd.api.types.is_datetime64_any_dtype(current_column):
                    # Convert to datetime with coercion
                    df[date_col] = pd.to_datetime(current_column, errors='coerce')
                
                # Remove invalid dates
                df = df[pd.notna(df[date_col])]
            except Exception as e:
                logger.warning(f"Could not process date column {date_col}: {str(e)}")
        
        # Add year, month, day columns
        if 'created_date' in df.columns:
            df['year'] = df['created_date'].dt.year
            df['month'] = df['created_date'].dt.month
            df['day'] = df['created_date'].dt.day
            df['day_of_week'] = df['created_date'].dt.dayofweek
        
        # Calculate resolution time if possible
        if 'created_date' in df.columns and 'resolved_date' in df.columns:
            try:
                # Ensure both are datetime
                df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
                df['resolved_date'] = pd.to_datetime(df['resolved_date'], errors='coerce')
                
                # Calculate resolution time in hours
                df['resolution_time_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
                
                # Filter out unrealistic resolution times (e.g., negative or extremely long)
                df = df[
                    (df['resolution_time_hours'] >= 0) & 
                    (df['resolution_time_hours'] < 8760)  # Less than a year
                ]
            except Exception as e:
                logger.warning(f"Error calculating resolution time: {str(e)}")
        
        # Prepare metadata
        metadata = {
            'date_columns': date_columns,
            'total_records': len(df),
            'date_range': {
                'start': df['created_date'].min() if 'created_date' in df.columns else None,
                'end': df['created_date'].max() if 'created_date' in df.columns else None
            }
        }
        
        return df, metadata
    
    def _prepare_resolution_time_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for resolution time analysis.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Check if we have the necessary columns
        required_cols = ['created_date', 'resolved_date']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure date columns are datetime
        for col in required_cols:
            if df[col].dtype != 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Filter out rows with missing date values
        df = df.dropna(subset=required_cols)
        
        # Filter out rows with invalid dates (resolved < created)
        df['valid_dates'] = df['resolved_date'] >= df['created_date']
        valid_count = df['valid_dates'].sum()
        invalid_count = len(df) - valid_count
        
        if valid_count == 0:
            raise ValueError("No incidents with valid resolution dates found")
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} incidents with invalid dates (resolution before creation)")
            df = df[df['valid_dates']]
        
        df = df.drop(columns=['valid_dates'])
        
        # Calculate resolution time in different units
        df['resolution_time_seconds'] = (df['resolved_date'] - df['created_date']).dt.total_seconds()
        df['resolution_time_minutes'] = df['resolution_time_seconds'] / 60
        df['resolution_time_hours'] = df['resolution_time_minutes'] / 60
        df['resolution_time_days'] = df['resolution_time_hours'] / 24
        
        # Calculate business hours resolution time if possible
        if 'business_hours' in df.columns:
            # This is a simplified calculation and may not be accurate for long-running incidents
            business_day_seconds = 8 * 60 * 60  # 8 hours in seconds
            df['resolution_time_business_hours'] = df.apply(
                lambda x: min(x['resolution_time_seconds'], business_day_seconds) 
                if x['business_hours'] == 1 else 0, 
                axis=1
            )
        
        # Calculate MTTR (Mean Time To Resolve)
        mttr_seconds = df['resolution_time_seconds'].mean()
        mttr_hours = mttr_seconds / 3600
        
        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99]
        resolution_percentiles = {}
        for p in percentiles:
            resolution_percentiles[f'p{p}'] = df['resolution_time_hours'].quantile(p/100)
        
        # Group incidents by resolution time bands
        resolution_bands = [
            ('< 1 hour', df['resolution_time_hours'] < 1),
            ('1-4 hours', (df['resolution_time_hours'] >= 1) & (df['resolution_time_hours'] < 4)),
            ('4-8 hours', (df['resolution_time_hours'] >= 4) & (df['resolution_time_hours'] < 8)),
            ('8-24 hours', (df['resolution_time_hours'] >= 8) & (df['resolution_time_hours'] < 24)),
            ('1-3 days', (df['resolution_time_hours'] >= 24) & (df['resolution_time_hours'] < 72)),
            ('3-7 days', (df['resolution_time_hours'] >= 72) & (df['resolution_time_hours'] < 168)),
            ('> 7 days', df['resolution_time_hours'] >= 168)
        ]
        
        resolution_bands_counts = {}
        for band_name, band_condition in resolution_bands:
            count = band_condition.sum()
            resolution_bands_counts[band_name] = {
                'count': int(count),
                'percentage': (count / len(df)) * 100 if len(df) > 0 else 0
            }
        
        # Check for trends in resolution time if we have enough data
        resolution_trend = None
        if len(df) >= 20:
            # Group by month and calculate average resolution time
            df['month_year'] = df['created_date'].dt.to_period('M')
            monthly_avg = df.groupby('month_year')['resolution_time_hours'].mean()
            
            if len(monthly_avg) >= 3:
                # Check if there's a trend (using simple comparison of first vs last)
                first_month = monthly_avg.iloc[0]
                last_month = monthly_avg.iloc[-1]
                change_pct = ((last_month - first_month) / first_month * 100) if first_month > 0 else 0
                
                trend_direction = 'increasing' if change_pct > 10 else ('decreasing' if change_pct < -10 else 'stable')
                resolution_trend = {
                    'direction': trend_direction,
                    'change_percentage': change_pct
                }
        
        # Additional factors that might correlate with resolution time
        resolution_factors = self._analyze_resolution_factors(df)
        
        # Compile metadata
        metadata = {
            'incident_count': len(df),
            'mttr': {
                'seconds': mttr_seconds,
                'minutes': mttr_seconds / 60,
                'hours': mttr_hours,
                'days': mttr_hours / 24
            },
            'resolution_percentiles': resolution_percentiles,
            'resolution_bands': resolution_bands_counts,
            'resolution_trend': resolution_trend,
            'resolution_factors': resolution_factors,
            'invalid_dates_count': invalid_count
        }
        
        return df, metadata
    
    def _prepare_priority_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for priority-based analysis with robust handling.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Identify priority columns
        priority_columns = [col for col in df.columns if 'priority' in col.lower()]
        
        # Comprehensive priority mapping
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
            'low': 'low'
        }
        
        # Process each priority column
        for priority_col in priority_columns:
            try:
                # Ensure we're working with a Series
                current_column = df[priority_col]
                
                # Standardize priority values
                df[priority_col] = current_column.astype(str).str.lower().map(
                    lambda x: priority_mapping.get(x, 'unknown')
                )
            except Exception as e:
                logger.warning(f"Could not process priority column {priority_col}: {str(e)}")
        
        # Filter out unknown priorities
        df = df[df[priority_columns[0]] != 'unknown']
        
        # Prepare metadata
        metadata = {
            'priority_columns': priority_columns,
            'total_records': len(df),
            'priority_distribution': {}
        }
        
        # Calculate priority distribution
        if priority_columns:
            try:
                priority_dist = df[priority_columns[0]].value_counts()
                metadata['priority_distribution'] = {
                    'counts': priority_dist.to_dict(),
                    'percentages': (priority_dist / len(df) * 100).to_dict()
                }
            except Exception as e:
                logger.warning(f"Could not calculate priority distribution: {str(e)}")
        
        return df, metadata
    
    def _prepare_resource_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for resource-based analysis.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Determine which resource column to use
        resource_col = None
        if 'assignee' in df.columns:
            resource_col = 'assignee'
        elif 'assignment_group' in df.columns:
            resource_col = 'assignment_group'
        else:
            # Look for alternative columns
            alt_columns = [col for col in df.columns if any(term in col.lower() for term in 
                                                          ['assign', 'owner', 'resource', 'respons', 'handler'])]
            if not alt_columns:
                raise ValueError("No assignee or assignment group column found")
            
            resource_col = alt_columns[0]
            logger.info(f"Using {resource_col} for resource analysis")
        
        # Clean resource values and handle empty/null values
        df[resource_col] = df[resource_col].fillna('Unassigned').astype(str)
        df[resource_col] = df[resource_col].apply(lambda x: x.strip())
        df[resource_col] = df[resource_col].apply(lambda x: 'Unassigned' if x == '' else x)
        
        # Resource distribution
        resource_counts = df[resource_col].value_counts()
        
        # Limit to top resources for manageability
        top_n = min(20, len(resource_counts))
        top_resources = resource_counts.head(top_n).index.tolist()
        other_resources = resource_counts.tail(len(resource_counts) - top_n).index.tolist()
        
        # Create a simplified resource column with top resources and "Others"
        df['resource_simplified'] = df[resource_col].apply(
            lambda x: x if x in top_resources else 'Others'
        )
        
        # Calculate workload metrics
        resource_metrics = {}
        
        # Incident count per resource
        for resource in top_resources:
            resource_df = df[df[resource_col] == resource]
            resource_metrics[resource] = {
                'incident_count': len(resource_df),
                'percentage': (len(resource_df) / len(df)) * 100,
            }
        
        # Add metrics for "Others" category
        others_df = df[df[resource_col].isin(other_resources)]
        if len(others_df) > 0:
            resource_metrics['Others'] = {
                'incident_count': len(others_df),
                'percentage': (len(others_df) / len(df)) * 100,
                'includes_count': len(other_resources)
            }
        
        # Calculate resolution metrics if data available
        if all(col in df.columns for col in ['created_date', 'resolved_date']):
            # Ensure date columns are datetime
            for col in ['created_date', 'resolved_date']:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate resolution time
            df['resolution_time_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
            
            # Filter valid resolution times
            valid_df = df[(df['resolution_time_hours'] >= 0) & (df['resolution_time_hours'] < 8760)]  # < 1 year
            
            if len(valid_df) > 0:
                for resource in list(resource_metrics.keys()):
                    if resource == 'Others':
                        resource_valid_df = valid_df[valid_df[resource_col].isin(other_resources)]
                    else:
                        resource_valid_df = valid_df[valid_df[resource_col] == resource]
                        
                    if len(resource_valid_df) > 0:
                        resource_metrics[resource].update({
                            'mean_resolution_hours': resource_valid_df['resolution_time_hours'].mean(),
                            'median_resolution_hours': resource_valid_df['resolution_time_hours'].median(),
                            'resolved_count': len(resource_valid_df),
                            'resolution_efficiency': len(resource_valid_df) / resource_metrics[resource]['incident_count'] * 100
                        })
        
        # Calculate workload distribution metrics
        workload_distribution = {
            'std_dev': resource_counts.std(),
            'mean': resource_counts.mean(),
            'coefficient_of_variation': resource_counts.std() / resource_counts.mean() if resource_counts.mean() > 0 else 0,
            'min': resource_counts.min(),
            'max': resource_counts.max(),
            'range': resource_counts.max() - resource_counts.min()
        }
        
        # Check for resource trends if we have time data
        resource_trends = None
        if 'created_date' in df.columns:
            if df['created_date'].dtype != 'datetime64[ns]':
                df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
            
            # Only proceed if we have valid dates
            if not df['created_date'].isnull().all():
                # Add month-year column for trending
                df['month_year'] = df['created_date'].dt.to_period('M')
                
                # Get enough months for trending (at least 3)
                months = df['month_year'].unique()
                if len(months) >= 3:
                    # Sort months
                    months = sorted(months)
                    
                    # Calculate resource workload trends for top resources
                    workload_trends = {}
                    for resource in top_resources:
                        monthly_counts = []
                        for month in months:
                            month_df = df[df['month_year'] == month]
                            count = month_df[month_df[resource_col] == resource].shape[0]
                            monthly_counts.append({
                                'month': str(month),
                                'count': count
                            })
                        
                        # Calculate trend (simple first vs last comparison)
                        first_count = monthly_counts[0]['count']
                        last_count = monthly_counts[-1]['count']
                        
                        if first_count > 0:
                            change_pct = ((last_count - first_count) / first_count) * 100
                        else:
                            change_pct = float('inf') if last_count > 0 else 0
                        
                        trend_direction = 'increasing' if change_pct > 10 else ('decreasing' if change_pct < -10 else 'stable')
                        
                        workload_trends[resource] = {
                            'direction': trend_direction,
                            'change_percentage': change_pct if not np.isinf(change_pct) else 0,
                            'monthly_data': monthly_counts
                        }
                    
                    resource_trends = {
                        'first_month': str(months[0]),
                        'last_month': str(months[-1]),
                        'trends': workload_trends
                    }
        
        # Compile metadata
        metadata = {
            'resource_column': resource_col,
            'total_resources': len(resource_counts),
            'top_resources': top_resources,
            'resource_metrics': resource_metrics,
            'workload_distribution': workload_distribution,
            'resource_trends': resource_trends
        }
        
        return df, metadata
        
    def _prepare_category_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for category-based analysis with robust handling.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Identify category columns
        category_columns = [col for col in df.columns if 'category' in col.lower()]
        
        # Process each category column
        for category_col in category_columns:
            try:
                # Ensure we're working with a Series
                current_column = df[category_col]
                
                # Standardize category values
                df[category_col] = current_column.astype(str).str.lower().str.strip()
                
                # Remove empty or generic categories
                df = df[
                    (df[category_col] != '') & 
                    (df[category_col] != 'nan') & 
                    (df[category_col] != 'unknown')
                ]
            except Exception as e:
                logger.warning(f"Could not process category column {category_col}: {str(e)}")
        
        # Prepare metadata
        metadata = {
            'category_columns': category_columns,
            'total_records': len(df),
            'category_distribution': {}
        }
        
        # Calculate category distribution
        if category_columns:
            try:
                category_dist = df[category_columns[0]].value_counts()
                metadata['category_distribution'] = {
                    'counts': category_dist.to_dict(),
                    'percentages': (category_dist / len(df) * 100).to_dict(),
                    'unique_categories': len(category_dist)
                }
            except Exception as e:
                logger.warning(f"Could not calculate category distribution: {str(e)}")
        
        return df, metadata
    
    def _prepare_system_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for system-based analysis.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Determine which system column to use
        system_col = None
        if 'affected_system' in df.columns:
            system_col = 'affected_system'
        else:
            # Look for alternative columns
            alt_columns = [col for col in df.columns if any(term in col.lower() for term in 
                                                         ['system', 'application', 'service', 'component', 'config item'])]
            if not alt_columns:
                raise ValueError("No affected system or application column found")
            
            system_col = alt_columns[0]
            logger.info(f"Using {system_col} for system analysis")
        
        # Clean and standardize system values
        df[system_col] = df[system_col].fillna('Unknown').astype(str)
        df[system_col] = df[system_col].apply(lambda x: x.strip())
        df[system_col] = df[system_col].apply(lambda x: 'Unknown' if x == '' else x)
        
        # System distribution
        system_counts = df[system_col].value_counts()
        system_percentages = df[system_col].value_counts(normalize=True).multiply(100).to_dict()
        
        # Limit to top systems for manageability
        top_n = min(15, len(system_counts))
        top_systems = system_counts.head(top_n).index.tolist()
        other_systems = system_counts.tail(len(system_counts) - top_n).index.tolist()
        
        # Create a simplified system column with top systems and "Others"
        df['system_simplified'] = df[system_col].apply(
            lambda x: x if x in top_systems else 'Others'
        )
        
        # Calculate metrics for each system
        system_metrics = {}
        
        for system in top_systems:
            sys_df = df[df[system_col] == system]
            system_metrics[system] = {
                'incident_count': len(sys_df),
                'percentage': system_percentages.get(system, 0)
            }
        
        # Add metrics for "Others" category
        others_df = df[df[system_col].isin(other_systems)]
        if len(others_df) > 0:
            system_metrics['Others'] = {
                'incident_count': len(others_df),
                'percentage': sum(system_percentages.get(sys, 0) for sys in other_systems),
                'includes_count': len(other_systems)
            }
        
        # Calculate downtime and impact metrics if available
        if 'created_date' in df.columns and 'resolved_date' in df.columns:
            # Ensure date columns are datetime
            for col in ['created_date', 'resolved_date']:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate downtime in hours
            df['downtime_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
            
            # Filter valid downtime values
            valid_df = df[(df['downtime_hours'] >= 0) & (df['downtime_hours'] < 8760)]  # < 1 year
            
            if len(valid_df) > 0:
                for system in list(system_metrics.keys()):
                    if system == 'Others':
                        sys_valid_df = valid_df[valid_df[system_col].isin(other_systems)]
                    else:
                        sys_valid_df = valid_df[valid_df[system_col] == system]
                        
                    if len(sys_valid_df) > 0:
                        system_metrics[system].update({
                            'total_downtime_hours': sys_valid_df['downtime_hours'].sum(),
                            'mean_downtime_hours': sys_valid_df['downtime_hours'].mean(),
                            'median_downtime_hours': sys_valid_df['downtime_hours'].median(),
                            'resolved_count': len(sys_valid_df)
                        })
        
        # Calculate MTBF if we have enough time-series data
        if 'created_date' in df.columns:
            # Group incidents by system and date, sort by date
            mtbf_metrics = {}
            for system in top_systems:
                sys_df = df[df[system_col] == system].sort_values('created_date')
                
                if len(sys_df) >= 3:  # Need at least 3 incidents to calculate meaningful MTBF
                    # Calculate time between incidents
                    sys_df = sys_df.reset_index(drop=True)
                    sys_df['next_incident'] = sys_df['created_date'].shift(-1)
                    sys_df['time_between_hours'] = (sys_df['next_incident'] - sys_df['created_date']).dt.total_seconds() / 3600
                    
                    # Filter valid time gaps
                    valid_gaps = sys_df['time_between_hours'].dropna()
                    if len(valid_gaps) > 0:
                        mtbf_metrics[system] = {
                            'mtbf_hours': valid_gaps.mean(),
                            'min_time_between': valid_gaps.min(),
                            'max_time_between': valid_gaps.max()
                        }
        
        # Check for system trends if we have time data
        system_trends = None
        if 'created_date' in df.columns:
            if df['created_date'].dtype != 'datetime64[ns]':
                df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
            
            # Only proceed if we have valid dates
            if not df['created_date'].isnull().all():
                # Add month-year column for trending
                df['month_year'] = df['created_date'].dt.to_period('M')
                
                # Get enough months for trending (at least 3)
                months = df['month_year'].unique()
                if len(months) >= 3:
                    # Sort months
                    months = sorted(months)
                    
                    # Calculate system trends for top systems
                    trend_metrics = {}
                    for system in top_systems:
                        monthly_counts = []
                        for month in months:
                            month_df = df[df['month_year'] == month]
                            count = month_df[month_df[system_col] == system].shape[0]
                            monthly_counts.append({
                                'month': str(month),
                                'count': count
                            })
                        
                        # Calculate trend (simple first vs last comparison)
                        first_count = monthly_counts[0]['count']
                        last_count = monthly_counts[-1]['count']
                        
                        if first_count > 0:
                            change_pct = ((last_count - first_count) / first_count) * 100
                        else:
                            change_pct = float('inf') if last_count > 0 else 0
                        
                        trend_direction = 'increasing' if change_pct > 10 else ('decreasing' if change_pct < -10 else 'stable')
                        
                        trend_metrics[system] = {
                            'direction': trend_direction,
                            'change_percentage': change_pct if not np.isinf(change_pct) else 0,
                            'monthly_data': monthly_counts
                        }
                    
                    system_trends = {
                        'first_month': str(months[0]),
                        'last_month': str(months[-1]),
                        'trends': trend_metrics
                    }
        
        # Compile metadata
        metadata = {
            'system_column': system_col,
            'total_systems': len(system_counts),
            'top_systems': top_systems,
            'system_metrics': system_metrics,
            'mtbf_metrics': mtbf_metrics if 'mtbf_metrics' in locals() else None,
            'system_trends': system_trends
        }
        
        return df, metadata
    
    def _prepare_automation_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for automation opportunity analysis.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Determine which text columns to use for automation analysis
        text_cols = []
        for col in ['description', 'resolution_notes', 'summary', 'title', 'comments']:
            if col in df.columns:
                text_cols.append(col)
        
        if not text_cols:
            # Look for alternative columns with text content
            alt_columns = [col for col in df.columns if df[col].dtype == 'object' and 
                           df[col].astype(str).str.len().mean() > 20]  # Average length > 20 chars
            
            if alt_columns:
                text_cols = alt_columns[:2]  # Use at most 2 alternative text columns
                logger.info(f"Using {text_cols} for automation analysis")
            else:
                # If no text columns, try to use categorical columns
                cat_columns = [col for col in df.columns if df[col].dtype == 'object' and 
                              df[col].nunique() < df.shape[0] * 0.5]  # Less than 50% unique values
                
                if cat_columns:
                    text_cols = cat_columns[:3]  # Use at most 3 categorical columns
                    logger.info(f"No text columns found. Using categorical columns {text_cols} for automation analysis")
                else:
                    raise ValueError("No suitable text or categorical columns found for automation analysis")
        
        # Combine text fields for analysis
        df['combined_text'] = df[text_cols].astype(str).apply(lambda x: ' '.join(x), axis=1)
        
        # Remove very short texts
        df['text_length'] = df['combined_text'].str.len()
        df = df[df['text_length'] > 10]  # Require at least 10 characters
        
        if len(df) < 10:
            raise ValueError("Insufficient data with meaningful text content for automation analysis")
        
        # Find patterns for potential automation
        patterns = self._identify_automation_patterns(df)
        
        # Analyze frequent categories if available
        category_patterns = None
        category_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                        ['category', 'type', 'class', 'group'])]
        
        if category_cols:
            category_patterns = self._analyze_category_patterns(df, category_cols)
        
        # Analyze resolution time patterns if available
        resolution_patterns = None
        if all(col in df.columns for col in ['created_date', 'resolved_date']):
            resolution_patterns = self._analyze_resolution_patterns(df)
        
        # Analyze common phrases in text fields
        text_patterns = self._analyze_text_patterns(df)
        
        # Compile metadata
        metadata = {
            'text_columns_used': text_cols,
            'total_incidents': len(df),
            'identified_patterns': patterns,
            'category_patterns': category_patterns,
            'resolution_patterns': resolution_patterns,
            'text_patterns': text_patterns
        }
        
        return df, metadata
    
    def _prepare_text_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for text-based analysis.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Tuple containing:
                - Processed DataFrame
                - Dictionary with metadata
        """
        # Determine which text columns to use
        text_cols = []
        for col in ['description', 'resolution_notes', 'summary', 'title', 'comments']:
            if col in df.columns:
                text_cols.append(col)
        
        if not text_cols:
            # Look for alternative columns with text content
            alt_columns = [col for col in df.columns if df[col].dtype == 'object' and 
                           df[col].astype(str).str.len().mean() > 20]  # Average length > 20 chars
            
            if alt_columns:
                text_cols = alt_columns[:2]  # Use at most 2 alternative text columns
                logger.info(f"Using {text_cols} for text analysis")
            else:
                raise ValueError("No suitable text columns found for text analysis")
        
        # Process each text column
        text_metrics = {}
        
        for col in text_cols:
            # Basic text metrics
            df[f'{col}_length'] = df[col].astype(str).str.len()
            df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
            
            metrics = {
                'mean_length': df[f'{col}_length'].mean(),
                'median_length': df[f'{col}_length'].median(),
                'max_length': df[f'{col}_length'].max(),
                'mean_word_count': df[f'{col}_word_count'].mean(),
                'median_word_count': df[f'{col}_word_count'].median(),
                'empty_count': (df[f'{col}_length'] <= 3).sum(),  # Consider <= 3 chars as empty
                'empty_percentage': (df[f'{col}_length'] <= 3).mean() * 100
            }
            
            # Extract top keywords and phrases
            if df[f'{col}_length'].mean() > 20:  # Only analyze if average text is long enough
                keywords = self._extract_keywords(df[col].astype(str))
                metrics['top_keywords'] = keywords
            
            text_metrics[col] = metrics
        
        # Find common phrases or terms across incidents
        common_phrases = self._extract_common_phrases(df, text_cols)
        
        # Compile metadata
        metadata = {
            'text_columns': text_cols,
            'text_metrics': text_metrics,
            'common_phrases': common_phrases
        }
        
        return df, metadata
    
    def _detect_seasonality(self, df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Detect seasonality patterns in incident data.
        
        Args:
            df: DataFrame with incident data
            date_col: Name of the date column
            
        Returns:
            Dictionary with seasonality information
        """
        seasonality = {}
        
        # Only analyze if we have enough data
        if len(df) < 20:
            return {'has_seasonality': False, 'reason': 'Insufficient data for seasonality analysis'}
        
        # Analyze daily patterns
        if 'day_of_week' in df.columns:
            dow_counts = df['day_of_week'].value_counts().sort_index()
            
            # Check if weekends have significantly different incident counts
            weekday_mean = dow_counts.iloc[0:5].mean()  # Monday-Friday
            weekend_mean = dow_counts.iloc[5:7].mean() if len(dow_counts) >= 7 else 0  # Saturday-Sunday
            
            if weekday_mean > 0:
                weekend_ratio = weekend_mean / weekday_mean
                
                if weekend_ratio < 0.5:  # Much fewer incidents on weekends
                    seasonality['daily'] = {
                        'pattern': 'weekday_heavy',
                        'weekday_mean': weekday_mean,
                        'weekend_mean': weekend_mean,
                        'weekend_ratio': weekend_ratio
                    }
                elif weekend_ratio > 1.5:  # More incidents on weekends
                    seasonality['daily'] = {
                        'pattern': 'weekend_heavy',
                        'weekday_mean': weekday_mean,
                        'weekend_mean': weekend_mean,
                        'weekend_ratio': weekend_ratio
                    }
        
        # Analyze monthly patterns
        if 'month' in df.columns:
            month_counts = df.groupby('month').size()
            
            # Check if there's significant monthly variation
            month_std = month_counts.std()
            month_mean = month_counts.mean()
            
            if month_mean > 0:
                month_cv = month_std / month_mean  # Coefficient of variation
                
                if month_cv > 0.3:  # Arbitrary threshold for significant variation
                    # Identify peak months
                    peak_months = month_counts[month_counts > month_mean * 1.3].index.tolist()
                    low_months = month_counts[month_counts < month_mean * 0.7].index.tolist()
                    
                    seasonality['monthly'] = {
                        'variation_coefficient': month_cv,
                        'peak_months': peak_months,
                        'low_months': low_months
                    }
        
        # Analyze quarterly patterns
        if 'quarter' in df.columns:
            quarter_counts = df.groupby('quarter').size()
            
            # Check if there's significant quarterly variation
            quarter_std = quarter_counts.std()
            quarter_mean = quarter_counts.mean()
            
            if quarter_mean > 0 and len(quarter_counts) == 4:  # Need all 4 quarters
                quarter_cv = quarter_std / quarter_mean
                
                if quarter_cv > 0.2:  # Lower threshold for quarterly variation
                    # Identify peak quarters
                    peak_quarters = quarter_counts[quarter_counts > quarter_mean * 1.2].index.tolist()
                    low_quarters = quarter_counts[quarter_counts < quarter_mean * 0.8].index.tolist()
                    
                    seasonality['quarterly'] = {
                        'variation_coefficient': quarter_cv,
                        'peak_quarters': peak_quarters,
                        'low_quarters': low_quarters
                    }
        
        # Analyze hourly patterns if available
        if 'hour' in df.columns:
            hour_counts = df.groupby('hour').size()
            
            # Check if there's significant hourly variation
            hour_std = hour_counts.std()
            hour_mean = hour_counts.mean()
            
            if hour_mean > 0:
                hour_cv = hour_std / hour_mean
                
                if hour_cv > 0.5:  # Higher threshold for hourly variation
                    # Identify peak hours
                    peak_hours = hour_counts[hour_counts > hour_mean * 1.5].index.tolist()
                    low_hours = hour_counts[hour_counts < hour_mean * 0.5].index.tolist()
                    
                    seasonality['hourly'] = {
                        'variation_coefficient': hour_cv,
                        'peak_hours': peak_hours,
                        'low_hours': low_hours
                    }
        
        # Determine if seasonality exists
        seasonality['has_seasonality'] = len(seasonality) > 0
        
        return seasonality
    
    def _analyze_resolution_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze factors that correlate with resolution time.
        
        Args:
            df: DataFrame with incident data containing resolution_time_hours
            
        Returns:
            Dictionary with resolution factors
        """
        resolution_factors = {}
        
        # Only analyze if we have resolution time data
        if 'resolution_time_hours' not in df.columns or df['resolution_time_hours'].isnull().all():
            return resolution_factors
        
        # Filter out extreme values
        df_filtered = df[(df['resolution_time_hours'] >= 0) & (df['resolution_time_hours'] < 8760)]  # < 1 year
        
        if len(df_filtered) < 10:
            return resolution_factors
        
        # Analyze resolution time by priority if available
        if 'priority' in df.columns:
            priority_resolution = df_filtered.groupby('priority')['resolution_time_hours'].agg(['mean', 'median', 'count']).reset_index()
            if len(priority_resolution) > 1:  # Need at least 2 priority levels
                priority_resolution = priority_resolution.sort_values('mean')
                
                # Calculate correlation ratio
                highest_priority_time = priority_resolution['mean'].iloc[0]
                lowest_priority_time = priority_resolution['mean'].iloc[-1]
                
                if highest_priority_time > 0:
                    ratio = lowest_priority_time / highest_priority_time
                    
                    resolution_factors['priority'] = {
                        'correlation': 'strong' if ratio > 2 else ('moderate' if ratio > 1.3 else 'weak'),
                        'ratio': ratio,
                        'details': priority_resolution.to_dict('records')
                    }
        
        # Analyze resolution time by day of week if available
        if 'day_of_week' in df.columns:
            dow_resolution = df_filtered.groupby('day_name')['resolution_time_hours'].agg(['mean', 'median', 'count']).reset_index()
            
            # Calculate variation
            max_time = dow_resolution['mean'].max()
            min_time = dow_resolution['mean'].min()
            
            if min_time > 0:
                ratio = max_time / min_time
                
                resolution_factors['day_of_week'] = {
                    'correlation': 'strong' if ratio > 1.5 else ('moderate' if ratio > 1.2 else 'weak'),
                    'max_day': dow_resolution.loc[dow_resolution['mean'].idxmax(), 'day_name'],
                    'min_day': dow_resolution.loc[dow_resolution['mean'].idxmin(), 'day_name'],
                    'ratio': ratio,
                    'details': dow_resolution.to_dict('records')
                }
        
        # Analyze resolution time by hour if available
        if 'hour' in df.columns:
            # Group by 3-hour blocks for more meaningful analysis
            df_filtered['hour_block'] = (df_filtered['hour'] // 3) * 3
            hour_resolution = df_filtered.groupby('hour_block')['resolution_time_hours'].agg(['mean', 'median', 'count']).reset_index()
            
            # Format hour blocks for readability
            hour_resolution['hour_range'] = hour_resolution['hour_block'].apply(
                lambda x: f"{x:02d}:00 - {(x+3) % 24:02d}:00"
            )
            
            # Calculate variation
            max_time = hour_resolution['mean'].max()
            min_time = hour_resolution['mean'].min()
            
            if min_time > 0:
                ratio = max_time / min_time
                
                resolution_factors['hour'] = {
                    'correlation': 'strong' if ratio > 1.5 else ('moderate' if ratio > 1.2 else 'weak'),
                    'max_block': hour_resolution.loc[hour_resolution['mean'].idxmax(), 'hour_range'],
                    'min_block': hour_resolution.loc[hour_resolution['mean'].idxmin(), 'hour_range'],
                    'ratio': ratio,
                    'details': hour_resolution[['hour_range', 'mean', 'median', 'count']].to_dict('records')
                }
        
        # Analyze resolution time by category if available
        for cat_col in ['category', 'subcategory']:
            if cat_col in df.columns:
                # Get top categories by frequency
                top_cats = df_filtered[cat_col].value_counts().head(10).index
                
                # Filter to only include top categories
                cat_df = df_filtered[df_filtered[cat_col].isin(top_cats)]
                
                if len(cat_df) >= 10:  # Ensure we have enough data
                    cat_resolution = cat_df.groupby(cat_col)['resolution_time_hours'].agg(['mean', 'median', 'count']).reset_index()
                    cat_resolution = cat_resolution.sort_values('mean')
                    
                    # Calculate variation
                    max_time = cat_resolution['mean'].max()
                    min_time = cat_resolution['mean'].min()
                    
                    if min_time > 0 and len(cat_resolution) > 1:
                        ratio = max_time / min_time
                        
                        resolution_factors[cat_col] = {
                            'correlation': 'strong' if ratio > 2 else ('moderate' if ratio > 1.3 else 'weak'),
                            'fastest_category': cat_resolution.iloc[0][cat_col],
                            'slowest_category': cat_resolution.iloc[-1][cat_col],
                            'ratio': ratio,
                            'details': cat_resolution.to_dict('records')
                        }
        
        # Analyze resolution time by assignee/group if available
        for res_col in ['assignee', 'assignment_group']:
            if res_col in df.columns:
                # Get top resources by frequency
                top_resources = df_filtered[res_col].value_counts().head(10).index
                
                # Only include resources with at least 5 incidents
                resource_counts = df_filtered[res_col].value_counts()
                valid_resources = resource_counts[resource_counts >= 5].index
                top_resources = [r for r in top_resources if r in valid_resources]
                
                # Filter to only include top resources
                res_df = df_filtered[df_filtered[res_col].isin(top_resources)]
                
                if len(res_df) >= 10 and len(top_resources) >= 2:  # Ensure we have enough data
                    res_resolution = res_df.groupby(res_col)['resolution_time_hours'].agg(['mean', 'median', 'count']).reset_index()
                    res_resolution = res_resolution.sort_values('mean')
                    
                    # Calculate variation
                    max_time = res_resolution['mean'].max()
                    min_time = res_resolution['mean'].min()
                    
                    if min_time > 0:
                        ratio = max_time / min_time
                        
                        resolution_factors[res_col] = {
                            'correlation': 'strong' if ratio > 2 else ('moderate' if ratio > 1.3 else 'weak'),
                            'fastest_resource': res_resolution.iloc[0][res_col],
                            'slowest_resource': res_resolution.iloc[-1][res_col],
                            'ratio': ratio,
                            'details': res_resolution.to_dict('records')
                        }
        
        return resolution_factors
    
    def _identify_automation_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify patterns in incident data that might be suitable for automation.
        
        Args:
            df: DataFrame with incident data including combined_text
            
        Returns:
            List of identified automation patterns
        """
        # This is just a structure - real patterns will be identified from actual data
        automation_patterns = []
        
        # First look for repeating incident types
        if 'category' in df.columns and 'subcategory' in df.columns:
            # Group by category and subcategory
            category_counts = df.groupby(['category', 'subcategory']).size().reset_index(name='count')
            category_counts = category_counts.sort_values('count', ascending=False)
            
            # Find high-frequency categories (more than 5% of incidents)
            threshold = max(5, len(df) * 0.05)
            high_freq_categories = category_counts[category_counts['count'] >= threshold]
            
            for _, row in high_freq_categories.iterrows():
                cat_df = df[(df['category'] == row['category']) & (df['subcategory'] == row['subcategory'])]
                
                # Check if resolution time is consistent (low standard deviation)
                consistent_resolution = False
                if 'resolution_time_hours' in cat_df.columns:
                    resolution_std = cat_df['resolution_time_hours'].std()
                    resolution_mean = cat_df['resolution_time_hours'].mean()
                    
                    if resolution_mean > 0:
                        cv = resolution_std / resolution_mean  # Coefficient of variation
                        consistent_resolution = cv < 0.5  # Arbitrary threshold
                
                # Check if text descriptions are similar
                similar_descriptions = False
                if 'combined_text' in cat_df.columns:
                    # Simple approach: count common words/phrases
                    common_terms = self._extract_common_terms(cat_df['combined_text'])
                    similar_descriptions = len(common_terms) >= 3
                
                # If we have consistent patterns, add as automation candidate
                if consistent_resolution or similar_descriptions:
                    automation_patterns.append({
                        'category': row['category'],
                        'subcategory': row['subcategory'],
                        'frequency': row['count'],
                        'percentage': (row['count'] / len(df)) * 100,
                        'consistent_resolution': consistent_resolution,
                        'similar_descriptions': similar_descriptions,
                        'automation_potential': 'high' if (consistent_resolution and similar_descriptions) else 'medium'
                    })
        
        # If no categories, look for patterns in text
        elif 'combined_text' in df.columns:
            # Extract common phrases across incidents
            common_phrases = self._extract_common_phrases(df, ['combined_text'])
            
            # For each common phrase, check incidents containing it
            for phrase, count in common_phrases['frequent_phrases'].items():
                if count >= max(5, len(df) * 0.05):  # At least 5% of incidents or 5 incidents
                    phrase_df = df[df['combined_text'].str.contains(phrase, case=False, na=False)]
                    
                    # Check if resolution time is consistent
                    consistent_resolution = False
                    if 'resolution_time_hours' in phrase_df.columns:
                        resolution_std = phrase_df['resolution_time_hours'].std()
                        resolution_mean = phrase_df['resolution_time_hours'].mean()
                        
                        if resolution_mean > 0:
                            cv = resolution_std / resolution_mean
                            consistent_resolution = cv < 0.5
                    
                    automation_patterns.append({
                        'key_phrase': phrase,
                        'frequency': count,
                        'percentage': (count / len(df)) * 100,
                        'consistent_resolution': consistent_resolution,
                        'automation_potential': 'high' if consistent_resolution else 'medium'
                    })
        
        # Look for patterns in resolution notes if available
        if 'resolution_notes' in df.columns:
            # Find incidents with similar resolution steps
            resolution_phrases = self._extract_common_terms(df['resolution_notes'], min_count=3)
            
            for phrase, count in resolution_phrases.items():
                if count >= max(5, len(df) * 0.05) and len(phrase.split()) >= 2:  # Require at least 2 words
                    phrase_df = df[df['resolution_notes'].str.contains(phrase, case=False, na=False)]
                    
                    automation_patterns.append({
                        'resolution_phrase': phrase,
                        'frequency': count,
                        'percentage': (count / len(df)) * 100,
                        'automation_potential': 'medium',
                        'type': 'resolution_procedure'
                    })
        
        # Limit to top 10 patterns
        if len(automation_patterns) > 10:
            # Sort by potential and frequency
            automation_patterns.sort(key=lambda x: (
                0 if x.get('automation_potential') == 'high' else 
                (1 if x.get('automation_potential') == 'medium' else 2),
                -x.get('frequency', 0)
            ))
            automation_patterns = automation_patterns[:10]
        
        return automation_patterns
    
    def _analyze_category_patterns(self, df: pd.DataFrame, category_cols: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns in incident categories.
        
        Args:
            df: DataFrame with incident data
            category_cols: List of category column names
            
        Returns:
            Dictionary with category pattern information
        """
        category_patterns = {}
        
        for col in category_cols:
            # Get frequency of each category
            cat_counts = df[col].value_counts()
            
            # Find categories that appear frequently
            threshold = max(5, len(df) * 0.05)
            frequent_cats = cat_counts[cat_counts >= threshold].index.tolist()
            
            # Check resolution time consistency for each frequent category
            cat_consistency = {}
            
            if 'resolution_time_hours' in df.columns:
                for cat in frequent_cats:
                    cat_df = df[df[col] == cat]
                    
                    if len(cat_df) >= 5:  # Need enough data
                        resolution_std = cat_df['resolution_time_hours'].std()
                        resolution_mean = cat_df['resolution_time_hours'].mean()
                        
                        if resolution_mean > 0:
                            cv = resolution_std / resolution_mean
                            
                            cat_consistency[cat] = {
                                'mean_resolution_hours': resolution_mean,
                                'std_dev': resolution_std,
                                'coefficient_of_variation': cv,
                                'consistency': 'high' if cv < 0.3 else ('medium' if cv < 0.5 else 'low'),
                                'count': len(cat_df)
                            }
            
            category_patterns[col] = {
                'frequent_categories': frequent_cats,
                'resolution_consistency': cat_consistency
            }
        
        return category_patterns
    
    def _analyze_resolution_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in incident resolution times.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Dictionary with resolution pattern information
        """
        # Calculate resolution time if not already done
        if 'resolution_time_hours' not in df.columns:
            df['resolution_time_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
            
        # Filter valid resolution times
        valid_df = df[(df['resolution_time_hours'] >= 0) & (df['resolution_time_hours'] < 8760)]  # < 1 year
        
        if len(valid_df) < 10:
            return {'error': 'Insufficient valid resolution time data'}
        
        # Find incidents with very short resolution times (potential for automation)
        quick_resolution_threshold = valid_df['resolution_time_hours'].quantile(0.25)  # Bottom 25%
        quick_incidents = valid_df[valid_df['resolution_time_hours'] <= quick_resolution_threshold]
        
        # Find common characteristics in quick resolution incidents
        quick_patterns = {}
        
        # Check categories if available
        for cat_col in ['category', 'subcategory']:
            if cat_col in quick_incidents.columns:
                cat_counts = quick_incidents[cat_col].value_counts(normalize=True) * 100
                cat_counts_all = valid_df[cat_col].value_counts(normalize=True) * 100
                
                # Find categories overrepresented in quick incidents
                overrepresented = {}
                for cat, pct in cat_counts.items():
                    if cat in cat_counts_all and cat_counts_all[cat] > 0:
                        ratio = pct / cat_counts_all[cat]
                        
                        if ratio > 1.5 and cat_counts[cat] >= 5:  # At least 50% more common and at least 5%
                            overrepresented[cat] = {
                                'quick_percentage': pct,
                                'overall_percentage': cat_counts_all[cat],
                                'ratio': ratio
                            }
                
                if overrepresented:
                    quick_patterns[cat_col] = overrepresented
        
        # Check for common text patterns in quick resolution incidents
        if 'combined_text' in quick_incidents.columns:
            common_terms = self._extract_common_terms(quick_incidents['combined_text'])
            
            if common_terms:
                quick_patterns['common_terms'] = common_terms
        
        return {
            'quick_resolution_threshold_hours': quick_resolution_threshold,
            'quick_incidents_count': len(quick_incidents),
            'quick_incidents_percentage': (len(quick_incidents) / len(valid_df)) * 100,
            'patterns': quick_patterns
        }
    
    def _analyze_text_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in incident text fields.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Dictionary with text pattern information
        """
        text_patterns = {}
        
        if 'combined_text' in df.columns:
            # Extract common terms
            common_terms = self._extract_common_terms(df['combined_text'])
            
            # Extract frequently co-occurring terms
            term_pairs = self._extract_term_pairs(df['combined_text'])
            
            text_patterns = {
                'common_terms': common_terms,
                'term_pairs': term_pairs
            }
        
        return text_patterns
    
    def _extract_common_terms(self, text_series: pd.Series, min_count: int = 5) -> Dict[str, int]:
        """
        Extract common terms from a series of text.
        
        Args:
            text_series: Series of text strings
            min_count: Minimum frequency for terms to be included
            
        Returns:
            Dictionary of common terms and their frequencies
        """
        # Combine all text
        all_text = ' '.join(text_series.fillna('').astype(str))
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it',
            'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'have', 'was', 'an', 'will', 'can', 'all', 'has',
            'there', 'been', 'if', 'they', 'their', 'we', 'when', 'who', 'would', 'could', 'should', 'what', 'which',
            'were', 'been', 'had', 'his', 'her', 'our', 'my', 'any', 'but', 'do', 'some', 'up', 'out', 'so', 'no',
            'more', 'only', 'just', 'than', 'then', 'now', 'very', 'also', 'after', 'before', 'over', 'under', 'through'
        }
        
        # Split into words and count frequency
        words = all_text.lower().split()
        words = [word.strip('.,!?:;()[]{}"\'/\\') for word in words]
        words = [word for word in words if word and word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Find common phrases (2-3 words)
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        phrase_counts = Counter(phrases)
        
        # Combine words and phrases, filtering by minimum count
        common_terms = {term: count for term, count in word_counts.most_common(50) if count >= min_count}
        common_terms.update({term: count for term, count in phrase_counts.most_common(30) if count >= min_count})
        
        return common_terms
    
    def _extract_term_pairs(self, text_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Extract pairs of terms that frequently co-occur.
        
        Args:
            text_series: Series of text strings
            
        Returns:
            List of term pairs and their co-occurrence metrics
        """
        # Tokenize each document
        documents = []
        for text in text_series.fillna('').astype(str):
            # Simple tokenization: lowercase, remove punctuation, split by space
            tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
            # Remove stop words and very short words
            stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that'}
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            documents.append(tokens)
        
        # Count frequency of each term
        term_freq = Counter()
        for doc in documents:
            # Count each term only once per document
            term_freq.update(set(doc))
        
        # Only consider terms that appear in multiple documents
        min_doc_count = max(3, len(documents) * 0.05)  # At least 5% of documents or 3
        common_terms = {term for term, count in term_freq.items() if count >= min_doc_count}
        
        # Count co-occurrences
        cooccurrences = Counter()
        for doc in documents:
            # Only consider common terms
            doc_terms = [t for t in doc if t in common_terms]
            # Count each pair only once per document
            for i, term1 in enumerate(doc_terms):
                for term2 in doc_terms[i+1:]:
                    if term1 != term2:  # Avoid self-pairs
                        pair = tuple(sorted([term1, term2]))
                        cooccurrences[pair] += 1
        
        # Calculate association strength
        term_pairs = []
        for (term1, term2), count in cooccurrences.most_common(20):
            # Only include pairs that co-occur at least 3 times
            if count >= 3:
                # Calculate Jaccard similarity: intersection / union
                jaccard = count / (term_freq[term1] + term_freq[term2] - count)
                
                term_pairs.append({
                    'term1': term1,
                    'term2': term2,
                    'cooccurrence_count': count,
                    'term1_count': term_freq[term1],
                    'term2_count': term_freq[term2],
                    'jaccard_similarity': jaccard
                })
        
        return term_pairs
    
    def _extract_common_phrases(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, Any]:
        """
        Extract common phrases from text columns.
        
        Args:
            df: DataFrame with incident data
            text_cols: List of text column names
            
        Returns:
            Dictionary with common phrase information
        """
        # Combine text from all specified columns
        combined_text = ''
        for col in text_cols:
            combined_text += ' ' + ' '.join(df[col].fillna('').astype(str))
        
        # Simple phrase extraction
        phrases = []
        
        # Extract 2-4 word phrases
        words = combined_text.lower().split()
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        if len(words) >= 4:
            for i in range(len(words) - 3):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Remove phrases containing only stop words
        stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it'}
        filtered_phrases = {}
        for phrase, count in phrase_counts.most_common(100):
            words = phrase.split()
            if not all(word in stop_words for word in words) and count >= 3:
                filtered_phrases[phrase] = count
        
        return {
            'frequent_phrases': filtered_phrases
        }
    
    def _extract_keywords(self, text_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Extract important keywords from text.
        
        Args:
            text_series: Series of text strings
            
        Returns:
            List of keyword information
        """
        # Combine all text
        all_text = ' '.join(text_series.fillna('').astype(str))
        
        # Simple keyword extraction using frequency
        words = all_text.lower().split()
        words = [word.strip('.,!?:;()[]{}"\'/\\') for word in words]
        
        # Remove stop words and short words
        stop_words = {
            'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it',
            'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'have', 'was', 'an', 'will', 'can', 'all', 'has',
            'there', 'been', 'if', 'they', 'their', 'we', 'when', 'who', 'would', 'could', 'should', 'what', 'which'
        }
        
        filtered_words = [word for word in words if word and word not in stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Convert to list of dictionaries
        keywords = [{'word': word, 'count': count} for word, count in word_counts.most_common(30)]
        
        return keywords