
# Module for generating charts and visualizations from ticket data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import io
import base64

class ChartGenerator:
    """
    Class for generating charts and visualizations from ticket data.
    """
    
    def __init__(self):
        """Initialize the ChartGenerator."""
        # Set style for all charts
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def generate_charts(self, df):
        """
        Generate charts from the processed ticket data.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            dict: Dictionary of matplotlib figures
        """
        charts = {}
        
        # Generate various charts based on available data
        
        # Priority distribution chart
        if 'priority' in df.columns:
            charts['priority_chart'] = self._create_priority_chart(df)
        
        # State distribution chart
        if 'state' in df.columns:
            charts['state_chart'] = self._create_state_chart(df)
        
        # Time series chart if opened column exists and is datetime
        if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
            charts['time_series_chart'] = self._create_time_series_chart(df)
        
        # Assignment group chart
        if 'assignment group' in df.columns:
            charts['assignment_group_chart'] = self._create_assignment_group_chart(df)
        
        # Resolution time distribution chart
        if 'resolution_time_hours' in df.columns:
            charts['resolution_time_chart'] = self._create_resolution_time_chart(df)
        
        # Day of week chart
        if 'opened_day_of_week' in df.columns:
            charts['day_of_week_chart'] = self._create_day_of_week_chart(df)
        
        # Common issues chart
        issue_columns = [col for col in df.columns if col.startswith('contains_')]
        if issue_columns:
            charts['common_issues_chart'] = self._create_common_issues_chart(df, issue_columns)
        
        return charts
    
    def _create_priority_chart(self, df):
        """
        Create a chart showing the distribution of ticket priorities.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: Priority distribution chart
        """
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Get priority counts
        priority_counts = df['priority'].value_counts()
        
        # Define colors based on priority (if standard priorities are used)
        colors = []
        for priority in priority_counts.index:
            if 'critical' in str(priority).lower() or '1' in str(priority):
                colors.append('#d9534f')  # Red for critical/1
            elif 'high' in str(priority).lower() or '2' in str(priority):
                colors.append('#f0ad4e')  # Orange for high/2
            elif 'medium' in str(priority).lower() or '3' in str(priority):
                colors.append('#5bc0de')  # Blue for medium/3
            elif 'low' in str(priority).lower() or '4' in str(priority):
                colors.append('#5cb85c')  # Green for low/4
            else:
                colors.append('#777777')  # Gray for others
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            priority_counts, 
            labels=priority_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Adjust text properties
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
        
        # Add title
        plt.title('Ticket Distribution by Priority')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add legend with counts
        legend_labels = [f"{priority} ({count})" for priority, count in priority_counts.items()]
        plt.legend(wedges, legend_labels, title="Priority", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        return fig
    
    def _create_state_chart(self, df):
        """
        Create a chart showing the distribution of ticket states.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: State distribution chart
        """
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Get state counts
        state_counts = df['state'].value_counts()
        
        # If there are too many states, limit to top 10
        if len(state_counts) > 10:
            state_counts = state_counts.head(10)
            ax.set_title('Top 10 Ticket States')
        else:
            ax.set_title('Ticket Distribution by State')
        
        # Create horizontal bar chart
        bars = ax.barh(state_counts.index, state_counts.values, color='#5bc0de')
        
        # Add values at the end of each bar
        for i, v in enumerate(state_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Add labels and grid
        ax.set_xlabel('Number of Tickets')
        ax.set_ylabel('State')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _create_time_series_chart(self, df):
        """
        Create a time series chart showing ticket volume over time.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: Time series chart
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by day and count tickets
        daily_counts = df.resample('D', on='opened').size()
        
        # Calculate 7-day moving average
        rolling_avg = daily_counts.rolling(window=7).mean()
        
        # Plot data
        ax.plot(daily_counts.index, daily_counts.values, 'o-', alpha=0.6, label='Daily Tickets')
        ax.plot(rolling_avg.index, rolling_avg.values, 'r-', linewidth=2, label='7-Day Moving Average')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tickets')
        ax.set_title('Ticket Volume Over Time')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis to show dates nicely
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _create_assignment_group_chart(self, df):
        """
        Create a chart showing the distribution of tickets by assignment group.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: Assignment group chart
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get assignment group counts
        group_counts = df['assignment group'].value_counts()
        
        # Limit to top 10 groups
        if len(group_counts) > 10:
            group_counts = group_counts.head(10)
        
        # Create horizontal bar chart
        bars = ax.barh(group_counts.index, group_counts.values, color='#5cb85c')
        
        # Add values at the end of each bar
        for i, v in enumerate(group_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Add labels and title
        ax.set_xlabel('Number of Tickets')
        ax.set_ylabel('Assignment Group')
        ax.set_title('Top 10 Assignment Groups by Ticket Volume')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout to ensure all labels are visible
        plt.tight_layout()
        
        return fig
    
    def _create_resolution_time_chart(self, df):
        """
        Create a chart showing the distribution of resolution times.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: Resolution time chart
        """
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Remove outliers for better visualization (e.g., resolution times > 30 days)
        filtered_df = df[df['resolution_time_hours'] <= 720]  # 30 days = 720 hours
        
        # Create histogram
        sns.histplot(filtered_df['resolution_time_hours'], bins=30, kde=True, ax=ax)
        
        # Add vertical lines for mean and median
        plt.axvline(filtered_df['resolution_time_hours'].mean(), color='r', linestyle='--', label=f'Mean: {filtered_df["resolution_time_hours"].mean():.2f} hours')
        plt.axvline(filtered_df['resolution_time_hours'].median(), color='g', linestyle='--', label=f'Median: {filtered_df["resolution_time_hours"].median():.2f} hours')
        
        # Add labels and title
        ax.set_xlabel('Resolution Time (hours)')
        ax.set_ylabel('Number of Tickets')
        ax.set_title('Distribution of Ticket Resolution Times')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _create_day_of_week_chart(self, df):
        """
        Create a chart showing ticket volume by day of week.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            matplotlib.figure.Figure: Day of week chart
        """
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Map day numbers to names
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        # Get day of week counts
        day_counts = df['opened_day_of_week'].value_counts().sort_index()
        
        # Map index to day names
        day_counts.index = day_counts.index.map(day_map)
        
        # Create bar chart
        bars = ax.bar(day_counts.index, day_counts.values, color='#5bc0de')
        
        # Add values on top of each bar
        for i, v in enumerate(day_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Tickets')
        ax.set_title('Ticket Volume by Day of Week')
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _create_common_issues_chart(self, df, issue_columns):
        """
        Create a chart showing the most common issues.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            issue_columns (list): List of columns containing issue flags
            
        Returns:
            matplotlib.figure.Figure: Common issues chart
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Count occurrences of each issue
        issue_counts = {}
        for col in issue_columns:
            issue_name = col.replace('contains_', '')
            issue_counts[issue_name] = df[col].sum()
        
        # Convert to Series and sort
        issue_series = pd.Series(issue_counts).sort_values(ascending=False)
        
        # Limit to top 15 issues
        if len(issue_series) > 15:
            issue_series = issue_series.head(15)
        
        # Create horizontal bar chart
        bars = ax.barh(issue_series.index, issue_series.values, color='#f0ad4e')
        
        # Add values at the end of each bar
        for i, v in enumerate(issue_series.values):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Add labels and title
        ax.set_xlabel('Number of Tickets')
        ax.set_ylabel('Issue')
        ax.set_title('Top Common Issues Mentioned in Tickets')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig