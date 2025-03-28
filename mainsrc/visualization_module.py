# mainsrc/visualization_module.py
# Wrapper for chart_generator

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing module
from chart_generator import ChartGenerator

class VisualizationModule:
    """
    Module for generating visualizations from ticket data.
    Wraps around the existing ChartGenerator with additional features.
    """
    
    def __init__(self):
        """Initialize the visualization module."""
        self.chart_generator = ChartGenerator()
    
    def generate_charts(self, df):
        """
        Generate all charts for the dataset.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            dict: Dictionary of matplotlib figures
        """
        try:
            # Use the existing chart generator for base charts
            charts = self.chart_generator.generate_charts(df)
            
            # Add any additional charts not covered by the base generator
            if 'subcategory 3' in df.columns and 'subcategory 3' not in [chart.split('_')[0] for chart in charts.keys()]:
                geo_chart = self._create_geography_chart(df)
                if geo_chart:
                    charts['geography_chart'] = geo_chart
            
            return charts
        except Exception as e:
            st.error(f"Error generating charts: {str(e)}")
            return {}
    
    def _create_geography_chart(self, df):
        """
        Create a chart showing geographic distribution of tickets.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            matplotlib.figure.Figure: Geography chart or None if error
        """
        try:
            # Look for geography data in Subcategory 3 or similar columns
            geo_columns = []
            for col in df.columns:
                if 'subcategory' in col.lower() or any(term in col.lower() for term in ['region', 'country', 'location']):
                    geo_columns.append(col)
            
            if not geo_columns:
                return None
                
            # Use the first column that has clear geographic data
            for col in geo_columns:
                # Check if column contains AMER, APAC, EMEA, etc.
                if df[col].str.contains('AMER|APAC|EMEA|GLOBAL', case=False, regex=True).any():
                    # Create figure
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Get top 10 regions/countries
                    geo_counts = df[col].value_counts().head(10)
                    
                    # Create bar chart
                    bars = geo_counts.plot(kind='bar', ax=ax, color=plt.cm.Greens(0.6))
                    
                    # Add data labels
                    for i, value in enumerate(geo_counts):
                        ax.text(i, value + (geo_counts.max() * 0.02), f"{value}", ha='center')
                    
                    # Set title and labels
                    plt.title(f'Ticket Distribution by {col}', fontsize=14)
                    plt.ylabel('Number of Tickets')
                    plt.xlabel(col.capitalize())
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    return fig
            
            return None
        except Exception as e:
            st.error(f"Error creating geography chart: {str(e)}")
            return None
    
    def display_charts(self, charts):
        """
        Display charts in the Streamlit UI.
        
        Args:
            charts: Dictionary of matplotlib figures
        """
        if not charts:
            st.info("No charts to display.")
            return
        
        # First row: Pie charts side by side
        if 'priority_chart' in charts or 'subcategory_chart' in charts:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'priority_chart' in charts:
                    st.pyplot(charts['priority_chart'])
            
            with col2:
                if 'subcategory_chart' in charts:
                    st.pyplot(charts['subcategory_chart'])
        
        # Second row: Bar charts side by side
        if 'state_chart' in charts or 'assignment_group_chart' in charts:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'state_chart' in charts:
                    st.pyplot(charts['state_chart'])
            
            with col2:
                if 'assignment_group_chart' in charts:
                    st.pyplot(charts['assignment_group_chart'])
        
        # Geography chart if available
        if 'geography_chart' in charts:
            st.pyplot(charts['geography_chart'])
        
        # Time series chart (full width)
        if 'time_series_chart' in charts:
            st.pyplot(charts['time_series_chart'])