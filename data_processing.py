
# Module for processing and transforming ticket data

import pandas as pd
import numpy as np
from datetime import datetime
import re
import streamlit as st

class DataProcessor:
    """
    Class for processing and transforming ticket data.
    Handles data cleaning, feature engineering, and preparing data for analysis.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def process(self, df):
        """
        Process the ticket data.
        
        Args:
            df (pandas.DataFrame): Raw ticket data
            
        Returns:
            pandas.DataFrame: Processed ticket data
        """
        # Make a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Clean column names
        processed_df = self._clean_column_names(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Convert date columns to datetime
        processed_df = self._convert_date_columns(processed_df)
        
        # Feature engineering
        processed_df = self._feature_engineering(processed_df)
        
        # Extract keywords from descriptions
        processed_df = self._extract_keywords(processed_df)
        
        return processed_df
    
    def _clean_column_names(self, df):
        """
        Clean column names by stripping whitespace and converting to lowercase.
        
        Args:
            df (pandas.DataFrame): DataFrame with columns to clean
            
        Returns:
            pandas.DataFrame: DataFrame with cleaned column names
        """
        df.columns = df.columns.str.strip().str.lower()
        
        # Map common variations of column names to standardized names
        column_mapping = {
            'ticket': 'number',
            'ticket number': 'number',
            'incident number': 'number',
            'case number': 'number',
            'issue': 'number',
            'issue number': 'number',
            'description': 'short description',
            'issue description': 'short description',
            'ticket description': 'short description',
            'summary': 'short description',
            'created': 'opened',
            'created date': 'opened',
            'open date': 'opened',
            'date opened': 'opened',
            'creation date': 'opened',
            'resolved date': 'closed',
            'resolution date': 'closed',
            'date closed': 'closed',
            'close date': 'closed',
            'completed date': 'closed',
            'assignee': 'assigned to',
            'owner': 'assigned to',
            'assigned user': 'assigned to',
            'team': 'assignment group',
            'group': 'assignment group',
            'support group': 'assignment group',
            'support team': 'assignment group',
            'status': 'state',
            'ticket status': 'state',
            'issue status': 'state',
            'current state': 'state',
            'requester': 'opened by',
            'creator': 'opened by',
            'reporter': 'opened by',
            'created by': 'opened by'
        }
        
        # Rename columns based on mapping
        for col in df.columns:
            if col in column_mapping:
                df = df.rename(columns={col: column_mapping[col]})
                
        return df
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame with missing values
            
        Returns:
            pandas.DataFrame: DataFrame with handled missing values
        """
        # Check for common ticket columns and fill missing values appropriately
        if 'priority' in df.columns:
            df['priority'] = df['priority'].fillna('Not Specified')
            
        if 'state' in df.columns:
            df['state'] = df['state'].fillna('Unknown')
            
        if 'assignment group' in df.columns:
            df['assignment group'] = df['assignment group'].fillna('Unassigned')
            
        if 'assigned to' in df.columns:
            df['assigned to'] = df['assigned to'].fillna('Unassigned')
            
        if 'short description' in df.columns:
            df['short description'] = df['short description'].fillna('No description provided')
        
        # For any remaining string columns, replace NaNs with 'Unknown'
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
            
        return df
    
    def _convert_date_columns(self, df):
        """
        Convert date columns to datetime format.
        
        Args:
            df (pandas.DataFrame): DataFrame with date columns
            
        Returns:
            pandas.DataFrame: DataFrame with converted date columns
        """
        # List of potential date columns
        date_columns = ['opened', 'closed', 'resolved', 'updated', 'due date', 'start date', 'end date']
        
        for col in df.columns:
            if col in date_columns or any(date_term in col.lower() for date_term in ['date', 'time']):
                try:
                    # Try to convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    # If conversion fails, keep the column as is
                    pass
        
        return df
    
    def _feature_engineering(self, df):
        """
        Create new features from existing data.
        
        Args:
            df (pandas.DataFrame): DataFrame to add features to
            
        Returns:
            pandas.DataFrame: DataFrame with new features
        """
        # Create resolution time feature if opened and closed columns exist
        if 'opened' in df.columns and 'closed' in df.columns:
            # Calculate resolution time in hours
            df['resolution_time_hours'] = (df['closed'] - df['opened']).dt.total_seconds() / 3600
            
            # Handle negative or NaN resolution times
            df.loc[df['resolution_time_hours'] < 0, 'resolution_time_hours'] = np.nan
            
            # Create resolution time categories
            bins = [0, 1, 4, 24, 72, float('inf')]
            labels = ['< 1 hour', '1-4 hours', '4-24 hours', '1-3 days', '> 3 days']
            df['resolution_time_category'] = pd.cut(df['resolution_time_hours'], bins=bins, labels=labels)
            
        # Create month and year columns from opened date if it exists
        if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
            df['opened_month'] = df['opened'].dt.month
            df['opened_year'] = df['opened'].dt.year
            df['opened_day_of_week'] = df['opened'].dt.dayofweek
            df['opened_hour'] = df['opened'].dt.hour
            
            # Create day type (weekend/weekday)
            df['is_weekend'] = df['opened_day_of_week'].isin([5, 6])  # 5 = Saturday, 6 = Sunday
            
            # Create business hours flag (assuming 9-5)
            df['is_business_hours'] = ((df['opened_hour'] >= 9) & (df['opened_hour'] < 17) & ~df['is_weekend'])
        
        return df
    
    def _extract_keywords(self, df):
        """
        Extract keywords from short descriptions or work notes.
        
        Args:
            df (pandas.DataFrame): DataFrame with text columns
            
        Returns:
            pandas.DataFrame: DataFrame with extracted keywords
        """
        # Check if short description column exists
        if 'short description' in df.columns:
            # Convert to string (in case it's not)
            df['short description'] = df['short description'].astype(str)
            
            # Extract common technical terms/issues
            common_issues = [
                'error', 'failed', 'failure', 'broken', 'bug', 'crash', 'issue',
                'password', 'reset', 'access', 'login', 'permission', 'account',
                'slow', 'performance', 'latency', 'timeout', 'hang',
                'install', 'update', 'upgrade', 'patch', 'deploy',
                'network', 'connection', 'wifi', 'internet', 'server',
                'print', 'printer', 'email', 'outlook', 'office'
            ]
            
            # Create issue type columns
            for issue in common_issues:
                col_name = f"contains_{issue}"
                df[col_name] = df['short description'].str.lower().str.contains(issue, regex=False)
            
            # Try to extract error codes (patterns like "Error XYZ123" or similar)
            df['error_code'] = df['short description'].str.extract(r'(?i)error[ :-]?([a-z0-9]{3,})')
            
        return df