# visualization/chart_generator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import io
import base64
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import datetime
from collections import Counter

class ChartGenerator:
    """
    Generates visualizations for incident data analysis.
    Creates charts that represent various metrics, trends, and patterns
    in incident data to support data-driven decision making.
    """
    
    def __init__(self, theme: str = 'light', fig_size: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize the chart generator.
        
        Args:
            theme: Color theme for charts ('light' or 'dark')
            fig_size: Default figure size (width, height) in inches
            dpi: Resolution in dots per inch
        """
        self.theme = theme
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Set style based on theme
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#ff9e4a', '#66c5cc', '#f2d974', '#5cc8d7', '#79c36a', '#e5b5cd', '#ffcc99']
            self.cmap = 'viridis'
            self.bg_color = '#1f1f1f'
            self.text_color = '#e0e0e0'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            self.cmap = 'Blues'
            self.bg_color = '#ffffff'
            self.text_color = '#333333'
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate if the data is sufficient for visualization.
        
        Args:
            df: Incident dataframe
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for visualization")
            return False
        
        return True
    
    def fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string for embedding in HTML.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=self.dpi)
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_data
    
    def time_series_chart(self, df: pd.DataFrame, 
                         timestamp_col: str,
                         value_col: str = None,
                         category_col: str = None,
                         agg_func: str = 'count',
                         freq: str = 'D',
                         title: str = 'Incident Volume Over Time',
                         rolling_window: int = None) -> Optional[Dict]:
        """
        Create a time series chart showing incident volume or metrics over time.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            value_col: Column to aggregate (if None, will count rows)
            category_col: Column to use for grouping data by category
            agg_func: Aggregation function ('count', 'sum', 'mean', etc.)
            freq: Frequency for resampling ('H': hourly, 'D': daily, 'W': weekly, 'M': monthly)
            title: Chart title
            rolling_window: Size of rolling window for smoothing (if None, no smoothing)
            
        Returns:
            Dictionary with chart information and base64 encoded image,
            or None if visualization fails
        """
        if not self._validate_data(df):
            return None
        
        try:
            # Create a copy of the dataframe
            df_plot = df.copy()
            
            # Ensure timestamp column is datetime
            if df_plot[timestamp_col].dtype != 'datetime64[ns]':
                df_plot[timestamp_col] = pd.to_datetime(df_plot[timestamp_col], errors='coerce')
                
            # Remove rows with invalid timestamp
            df_plot = df_plot.dropna(subset=[timestamp_col])
            
            if len(df_plot) < 2:
                self.logger.warning("Insufficient data for time series visualization")
                return None
            
            # Set timestamp as index
            df_plot = df_plot.set_index(timestamp_col)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # Check for appropriate frequencies based on date range
            date_range = (df_plot.index.max() - df_plot.index.min()).days
            
            if freq == 'D' and date_range > 180:
                # Switch to weekly for longer date ranges
                freq = 'W'
            elif freq == 'H' and date_range > 30:
                # Switch to daily for longer date ranges
                freq = 'D'
            
            # Prepare data for plotting
            if category_col and category_col in df.columns:
                # Group by category and timestamp
                categories = df_plot[category_col].unique()
                
                # If too many categories, take top N by count
                if len(categories) > 7:
                    top_categories = df_plot[category_col].value_counts().nlargest(6).index.tolist()
                    other_mask = ~df_plot[category_col].isin(top_categories)
                    df_plot.loc[other_mask, category_col] = 'Other'
                    categories = top_categories + ['Other']
                
                for i, category in enumerate(categories):
                    category_data = df_plot[df_plot[category_col] == category]
                    
                    if len(category_data) < 2:
                        continue
                    
                    # Resample data
                    if value_col and value_col in df.columns:
                        if agg_func == 'count':
                            time_series = category_data.resample(freq).size()
                        else:
                            time_series = category_data[value_col].resample(freq).agg(agg_func)
                    else:
                        time_series = category_data.resample(freq).size()
                    
                    # Apply rolling window if specified
                    if rolling_window and len(time_series) > rolling_window:
                        time_series = time_series.rolling(window=rolling_window).mean()
                    
                    # Plot data
                    time_series.plot(
                        ax=ax,
                        label=str(category),
                        color=self.colors[i % len(self.colors)],
                        alpha=0.8
                    )
            else:
                # Resample data
                if value_col and value_col in df.columns:
                    if agg_func == 'count':
                        time_series = df_plot.resample(freq).size()
                    else:
                        time_series = df_plot[value_col].resample(freq).agg(agg_func)
                else:
                    time_series = df_plot.resample(freq).size()
                
                # Apply rolling window if specified
                if rolling_window and len(time_series) > rolling_window:
                    time_series = time_series.rolling(window=rolling_window).mean()
                
                # Plot data
                time_series.plot(
                    ax=ax,
                    color=self.colors[0],
                    alpha=0.8
                )
            
            # Set up chart formatting
            ax.set_title(title, fontsize=14, color=self.text_color)
            ax.set_xlabel('Time', fontsize=12, color=self.text_color)
            
            y_label = value_col if value_col else 'Count'
            if agg_func and agg_func != 'count':
                y_label = f"{agg_func.capitalize()} of {y_label}"
            
            ax.set_ylabel(y_label, fontsize=12, color=self.text_color)
            
            # Format x-axis based on frequency
            if freq == 'H':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
                plt.xticks(rotation=45)
            elif freq == 'D':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                plt.xticks(rotation=45)
            elif freq == 'W':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                plt.xticks(rotation=45)
            elif freq == 'M':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.xticks(rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend if categories are used
            if category_col and category_col in df.columns:
                ax.legend(title=category_col, fontsize=10)
            
            # Adjust layout
            fig.tight_layout()
            
            # Convert to base64
            img_data = self.fig_to_base64(fig)
            
            # Calculate some basic statistics
            stats = {
                'start_date': df_plot.index.min().strftime('%Y-%m-%d'),
                'end_date': df_plot.index.max().strftime('%Y-%m-%d'),
                'total_records': len(df),
                'frequency': freq
            }
            
            # Get peaks and trends if available
            insights = []
            
            if len(time_series) > 2:
                # Find peak
                peak_idx = time_series.idxmax()
                peak_value = time_series.max()
                
                insights.append({
                    'type': 'peak',
                    'date': peak_idx.strftime('%Y-%m-%d'),
                    'value': float(peak_value),
                    'message': f"Peak of {peak_value:.1f} on {peak_idx.strftime('%Y-%m-%d')}"
                })
                
                # Detect trend
                if len(time_series) >= 10:
                    first_half = time_series[:len(time_series)//2].mean()
                    second_half = time_series[len(time_series)//2:].mean()
                    
                    if first_half > 0:
                        percent_change = (second_half - first_half) / first_half * 100
                        
                        if abs(percent_change) > 10:
                            trend_direction = "increasing" if percent_change > 0 else "decreasing"
                            insights.append({
                                'type': 'trend',
                                'direction': trend_direction,
                                'change_percent': f"{percent_change:.1f}%",
                                'message': f"Overall {trend_direction} trend of {abs(percent_change):.1f}%"
                            })
            
            return {
                'chart_type': 'time_series',
                'title': title,
                'img_data': img_data,
                'stats': stats,
                'insights': insights
            }
            
        except Exception as e:
            self.logger.error(f"Error creating time series chart: {str(e)}")
            return None
    
    def distribution_chart(self, df: pd.DataFrame,
                         category_col: str,
                         value_col: str = None,
                         agg_func: str = 'count',
                         title: str = 'Incident Distribution',
                         sort_by: str = 'value',
                         limit: int = 10) -> Optional[Dict]:
        """
        Create a bar chart showing distribution of incidents by category.
        
        Args:
            df: Incident dataframe
            category_col: Column to use for categories
            value_col: Column to aggregate (if None, will count rows)
            agg_func: Aggregation function ('count', 'sum', 'mean', etc.)
            title: Chart title
            sort_by: How to sort bars ('value' or 'name')
            limit: Maximum number of categories to show
            
        Returns:
            Dictionary with chart information and base64 encoded image,
            or None if visualization fails
        """
        if not self._validate_data(df) or category_col not in df.columns:
            return None
        
        try:
            # Prepare data for plotting
            if value_col and value_col in df.columns and agg_func != 'count':
                # Aggregate by the value column
                grouped_data = df.groupby(category_col)[value_col].agg(agg_func)
            else:
                # Count occurrences
                grouped_data = df[category_col].value_counts()
            
            # Handle empty data
            if grouped_data.empty:
                self.logger.warning(f"No data to plot for category: {category_col}")
                return None
            
            # Sort data
            if sort_by == 'value':
                grouped_data = grouped_data.sort_values(ascending=False)
            else:
                grouped_data = grouped_data.sort_index()
            
            # Limit number of categories
            if len(grouped_data) > limit:
                top_categories = grouped_data.nlargest(limit-1).index.tolist()
                other_count = grouped_data[~grouped_data.index.isin(top_categories)].sum()
                grouped_data = grouped_data[grouped_data.index.isin(top_categories)]
                grouped_data['Other'] = other_count
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # Plot data
            ax.bar(
                grouped_data.index,
                grouped_data.values,
                color=self.colors[:len(grouped_data)],
                alpha=0.8
            )
            
            # Set up chart formatting
            ax.set_title(title, fontsize=14, color=self.text_color)
            ax.set_xlabel(category_col, fontsize=12, color=self.text_color)
            
            y_label = value_col if value_col else 'Count'
            if agg_func and agg_func != 'count':
                y_label = f"{agg_func.capitalize()} of {y_label}"
                
            ax.set_ylabel(y_label, fontsize=12, color=self.text_color)
            
            # Format x-axis for readability
            plt.xticks(rotation=45, ha='right')
            
            # Make sure bars have values on top
            for i, value in enumerate(grouped_data.values):
                ax.text(
                    i, value + (max(grouped_data.values) * 0.01), 
                    f"{value:.0f}" if value >= 10 else f"{value:.1f}", 
                    ha='center', va='bottom', 
                    fontsize=9, color=self.text_color
                )
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            fig.tight_layout()
            
            # Convert to base64
            img_data = self.fig_to_base64(fig)
            
            # Calculate statistics
            total = grouped_data.sum()
            top_category = grouped_data.idxmax()
            top_value = grouped_data.max()
            top_percentage = (top_value / total) * 100 if total > 0 else 0
            
            stats = {
                'total': float(total),
                'unique_categories': len(grouped_data),
                'top_category': str(top_category),
                'top_value': float(top_value),
                'top_percentage': f"{top_percentage:.1f}%"
            }
            
            # Generate insights
            insights = []
            
            # Add insight about top category
            insights.append({
                'type': 'top_category',
                'category': str(top_category),
                'value': float(top_value),
                'percentage': f"{top_percentage:.1f}%",
                'message': f"'{top_category}' accounts for {top_percentage:.1f}% ({top_value:.0f}) of incidents"
            })
            
            # Add insight about distribution if applicable
            if len(grouped_data) >= 3:
                top_3_pct = (grouped_data.nlargest(3).sum() / total) * 100
                if top_3_pct > 75:
                    insights.append({
                        'type': 'concentration',
                        'value': f"{top_3_pct:.1f}%",
                        'message': f"Top 3 categories account for {top_3_pct:.1f}% of all incidents"
                    })
            
            return {
                'chart_type': 'distribution',
                'title': title,
                'img_data': img_data,
                'stats': stats,
                'insights': insights
            }
            
        except Exception as e:
            self.logger.error(f"Error creating distribution chart: {str(e)}")
            return None
    
    def correlation_chart(self, df: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        category_col: str = None,
                        title: str = 'Correlation Analysis',
                        chart_type: str = 'scatter') -> Optional[Dict]:
        """
        Create a chart showing correlation between two variables.
        
        Args:
            df: Incident dataframe
            x_col: Column for x-axis
            y_col: Column for y-axis
            category_col: Column to use for color coding points
            title: Chart title
            chart_type: Type of chart ('scatter', 'heatmap', 'boxplot')
            
        Returns:
            Dictionary with chart information and base64 encoded image,
            or None if visualization fails
        """
        if not self._validate_data(df) or x_col not in df.columns or y_col not in df.columns:
            return None
        
        try:
            # Create a copy of the dataframe
            df_plot = df.copy()
            
            # Ensure numerical columns
            for col in [x_col, y_col]:
                if not pd.api.types.is_numeric_dtype(df_plot[col]):
                    try:
                        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
                    except:
                        self.logger.warning(f"Column {col} could not be converted to numeric")
                        return None
            
            # Remove rows with NaN values
            df_plot = df_plot.dropna(subset=[x_col, y_col])
            
            if len(df_plot) < 10:
                self.logger.warning("Insufficient data for correlation visualization")
                return None
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # Choose chart type
            if chart_type == 'scatter':
                # Scatter plot
                if category_col and category_col in df.columns:
                    categories = df_plot[category_col].unique()
                    
                    # If too many categories, take top N by count
                    if len(categories) > 7:
                        top_categories = df_plot[category_col].value_counts().nlargest(6).index.tolist()
                        other_mask = ~df_plot[category_col].isin(top_categories)
                        df_plot.loc[other_mask, category_col] = 'Other'
                        categories = top_categories + ['Other']
                    
                    for i, category in enumerate(categories):
                        category_data = df_plot[df_plot[category_col] == category]
                        
                        if len(category_data) < 2:
                            continue
                        
                        ax.scatter(
                            category_data[x_col],
                            category_data[y_col],
                            label=str(category),
                            color=self.colors[i % len(self.colors)],
                            alpha=0.7
                        )
                    
                    # Add legend
                    ax.legend(title=category_col, fontsize=10)
                else:
                    ax.scatter(
                        df_plot[x_col],
                        df_plot[y_col],
                        color=self.colors[0],
                        alpha=0.7
                    )
                
                # Calculate and plot trend line
                try:
                    z = np.polyfit(df_plot[x_col], df_plot[y_col], 1)
                    p = np.poly1d(z)
                    ax.plot(
                        np.sort(df_plot[x_col]),
                        p(np.sort(df_plot[x_col])),
                        linestyle='--',
                        color='gray',
                        alpha=0.8
                    )
                except:
                    pass
                
            elif chart_type == 'heatmap':
                # For heatmap, both variables should be categorical
                x_categorical = False
                y_categorical = False
                
                # Check if x_col is numerical
                if pd.api.types.is_numeric_dtype(df_plot[x_col]):
                    # Create bins for numerical data
                    x_bins = min(10, len(df_plot[x_col].unique()))
                    df_plot[f"{x_col}_bin"] = pd.cut(
                        df_plot[x_col], 
                        bins=x_bins, 
                        labels=[f"{i+1}" for i in range(x_bins)]
                    )
                    x_col_plot = f"{x_col}_bin"
                    x_categorical = True
                else:
                    x_col_plot = x_col
                    x_categorical = True
                
                # Check if y_col is numerical
                if pd.api.types.is_numeric_dtype(df_plot[y_col]):
                    # Create bins for numerical data
                    y_bins = min(10, len(df_plot[y_col].unique()))
                    df_plot[f"{y_col}_bin"] = pd.cut(
                        df_plot[y_col], 
                        bins=y_bins, 
                        labels=[f"{i+1}" for i in range(y_bins)]
                    )
                    y_col_plot = f"{y_col}_bin"
                    y_categorical = True
                else:
                    y_col_plot = y_col
                    y_categorical = True
                
                if x_categorical and y_categorical:
                    # Create a contingency table
                    contingency = pd.crosstab(
                        df_plot[y_col_plot], 
                        df_plot[x_col_plot],
                        normalize='all'
                    ) * 100  # Convert to percentage
                    
                    # Create heatmap
                    sns.heatmap(
                        contingency,
                        annot=True,
                        fmt='.1f',
                        cmap=self.cmap,
                        ax=ax,
                        cbar_kws={'label': 'Percentage (%)'}
                    )
                else:
                    self.logger.warning("Heatmap requires categorical variables or binnable numeric variables")
                    return None
                
            elif chart_type == 'boxplot':
                # For boxplot, x should be categorical and y should be numerical
                if pd.api.types.is_numeric_dtype(df_plot[y_col]):
                    # If x is numeric, convert to categorical
                    if pd.api.types.is_numeric_dtype(df_plot[x_col]):
                        # Create bins for numerical data
                        x_bins = min(10, len(df_plot[x_col].unique()))
                        df_plot[f"{x_col}_bin"] = pd.cut(
                            df_plot[x_col], 
                            bins=x_bins
                        )
                        x_col_plot = f"{x_col}_bin"
                    else:
                        x_col_plot = x_col
                    
                    # If too many categories, take top N by count
                    categories = df_plot[x_col_plot].unique()
                    if len(categories) > 10:
                        top_categories = df_plot[x_col_plot].value_counts().nlargest(9).index.tolist()
                        df_plot = df_plot[df_plot[x_col_plot].isin(top_categories)]
                    
                    # Create boxplot
                    sns.boxplot(
                        x=x_col_plot,
                        y=y_col,
                        data=df_plot,
                        palette=self.colors[:len(df_plot[x_col_plot].unique())],
                        ax=ax
                    )
                    
                    # Rotate x-axis labels for readability
                    plt.xticks(rotation=45, ha='right')
                else:
                    self.logger.warning("Boxplot requires y-axis to be numerical")
                    return None
            
            # Set up chart formatting
            ax.set_title(title, fontsize=14, color=self.text_color)
            ax.set_xlabel(x_col, fontsize=12, color=self.text_color)
            ax.set_ylabel(y_col, fontsize=12, color=self.text_color)
            
            # Add grid for scatter plot
            if chart_type == 'scatter':
                ax.grid(True, alpha=0.3)
            
            # Adjust layout
            fig.tight_layout()
            
            # Convert to base64
            img_data = self.fig_to_base64(fig)
            
            # Calculate statistics and insights
            stats = {
                'total_records': len(df_plot),
                'chart_type': chart_type
            }
            
            insights = []
            
            # Calculate correlation coefficient for scatter plot
            if chart_type == 'scatter':
                correlation = df_plot[x_col].corr(df_plot[y_col])
                correlation_type = "positive" if correlation > 0 else "negative"
                correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                
                stats['correlation'] = round(correlation, 3)
                
                insights.append({
                    'type': 'correlation',
                    'value': round(correlation, 3),
                    'strength': correlation_strength,
                    'direction': correlation_type,
                    'message': f"{correlation_strength.capitalize()} {correlation_type} correlation ({correlation:.3f}) between {x_col} and {y_col}"
                })
            
            # Add insights for heatmap or boxplot
            if chart_type == 'heatmap' and 'contingency' in locals():
                # Find cell with highest value
                max_val = contingency.max().max()
                max_indices = np.where(contingency.values == max_val)
                if len(max_indices[0]) > 0:
                    max_row = contingency.index[max_indices[0][0]]
                    max_col = contingency.columns[max_indices[1][0]]
                    
                    insights.append({
                        'type': 'association',
                        'value': f"{max_val:.1f}%",
                        'variables': [str(max_row), str(max_col)],
                        'message': f"Strongest association between {y_col}='{max_row}' and {x_col}='{max_col}' ({max_val:.1f}%)"
                    })
            
            if chart_type == 'boxplot':
                # Find category with highest median
                category_medians = df_plot.groupby(x_col_plot)[y_col].median().sort_values(ascending=False)
                if not category_medians.empty:
                    top_category = category_medians.index[0]
                    top_median = category_medians.iloc[0]
                    
                    insights.append({
                        'type': 'group_comparison',
                        'category': str(top_category),
                        'value': float(top_median),
                        'variable': y_col,
                        'message': f"'{top_category}' has the highest median {y_col} ({top_median:.1f})"
                    })
            
            return {
                'chart_type': 'correlation',
                'subtype': chart_type,
                'title': title,
                'img_data': img_data,
                'stats': stats,
                'insights': insights
            }
            
        except Exception as e:
            self.logger.error(f"Error creating correlation chart: {str(e)}")
            return None
    
    def generate_dashboard_charts(self, df: pd.DataFrame) -> Dict:
        """
        Generate a set of default charts for the incident dashboard.
        
        Args:
            df: Incident dataframe
            
        Returns:
            Dictionary containing multiple charts for dashboard display
        """
        if not self._validate_data(df):
            return {
                'success': False,
                'message': 'Insufficient data for dashboard visualization',
                'charts': None
            }
        
        try:
            dashboard_charts = []
            
            # Try to infer the column names based on common patterns
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            category_cols = [col for col in df.columns if 'type' in col.lower() or 'category' in col.lower() or 'status' in col.lower()]
            priority_cols = [col for col in df.columns if 'priority' in col.lower() or 'severity' in col.lower()]
            resolution_cols = [col for col in df.columns if 'resolution' in col.lower() or 'solve' in col.lower()]
            
            # For time series chart
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                time_chart = self.time_series_chart(
                    df,
                    timestamp_col=timestamp_col,
                    title='Incident Volume Over Time'
                )
                
                if time_chart:
                    dashboard_charts.append(time_chart)
                
                # If priority column exists, add time series by priority
                if priority_cols:
                    priority_col = priority_cols[0]
                    priority_time_chart = self.time_series_chart(
                        df,
                        timestamp_col=timestamp_col,
                        category_col=priority_col,
                        title=f'Incident Volume by {priority_col} Over Time'
                    )
                    
                    if priority_time_chart:
                        dashboard_charts.append(priority_time_chart)
            
            # For distribution charts
            if category_cols:
                category_col = category_cols[0]
                category_chart = self.distribution_chart(
                    df,
                    category_col=category_col,
                    title=f'Incident Distribution by {category_col}'
                )
                
                if category_chart:
                    dashboard_charts.append(category_chart)
            

            if priority_cols:
                priority_col = priority_cols[0]
                priority_chart = self.distribution_chart(
                    df,
                    category_col=priority_col,
                    title=f'Incident Distribution by {priority_col}'
                )
                
                if priority_chart:
                    dashboard_charts.append(priority_chart)
            
            # For correlation charts
            if resolution_cols and category_cols:
                resolution_col = resolution_cols[0]
                category_col = category_cols[0]
                
                # Ensure resolution time is numeric
                df_corr = df.copy()
                if not pd.api.types.is_numeric_dtype(df_corr[resolution_col]):
                    try:
                        df_corr[resolution_col] = pd.to_numeric(df_corr[resolution_col], errors='coerce')
                        
                        corr_chart = self.correlation_chart(
                            df_corr,
                            x_col=category_col,
                            y_col=resolution_col,
                            chart_type='boxplot',
                            title=f'{resolution_col} by {category_col}'
                        )
                        
                        if corr_chart:
                            dashboard_charts.append(corr_chart)
                    except:
                        pass
            
            # Generate weekly distribution chart if timestamp exists
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                df_weekly = df.copy()
                
                # Ensure timestamp is datetime
                if df_weekly[timestamp_col].dtype != 'datetime64[ns]':
                    df_weekly[timestamp_col] = pd.to_datetime(df_weekly[timestamp_col], errors='coerce')
                
                # Create day of week column
                df_weekly['day_of_week'] = df_weekly[timestamp_col].dt.day_name()
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_data = df_weekly['day_of_week'].value_counts().reindex(day_order).fillna(0)
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=self.fig_size)
                
                # Plot data
                ax.bar(
                    weekly_data.index,
                    weekly_data.values,
                    color=self.colors[:7],
                    alpha=0.8
                )
                
                # Set up chart formatting
                ax.set_title('Incident Distribution by Day of Week', fontsize=14, color=self.text_color)
                ax.set_xlabel('Day of Week', fontsize=12, color=self.text_color)
                ax.set_ylabel('Count', fontsize=12, color=self.text_color)
                
                # Add values on top of bars
                for i, value in enumerate(weekly_data.values):
                    ax.text(
                        i, value + (max(weekly_data.values) * 0.01), 
                        f"{value:.0f}", 
                        ha='center', va='bottom', 
                        fontsize=9, color=self.text_color
                    )
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adjust layout
                fig.tight_layout()
                
                # Convert to base64
                img_data = self.fig_to_base64(fig)
                
                # Find day with most incidents
                if not weekly_data.empty:
                    peak_day = weekly_data.idxmax()
                    peak_count = weekly_data.max()
                    total_incidents = weekly_data.sum()
                    peak_percentage = (peak_count / total_incidents) * 100 if total_incidents > 0 else 0
                    
                    # Generate insights
                    insights = [{
                        'type': 'peak_day',
                        'day': peak_day,
                        'count': float(peak_count),
                        'percentage': f"{peak_percentage:.1f}%",
                        'message': f"{peak_day} has the highest incident volume ({peak_percentage:.1f}% of incidents)"
                    }]
                    
                    # Check for weekend vs weekday distribution
                    weekday_data = sum(weekly_data[:5])
                    weekend_data = sum(weekly_data[5:])
                    weekday_percentage = (weekday_data / total_incidents) * 100 if total_incidents > 0 else 0
                    weekend_percentage = (weekend_data / total_incidents) * 100 if total_incidents > 0 else 0
                    
                    insights.append({
                        'type': 'weekday_comparison',
                        'weekday_percentage': f"{weekday_percentage:.1f}%",
                        'weekend_percentage': f"{weekend_percentage:.1f}%",
                        'message': f"Weekdays account for {weekday_percentage:.1f}% of incidents, weekends {weekend_percentage:.1f}%"
                    })
                    
                    weekly_chart = {
                        'chart_type': 'distribution',
                        'title': 'Incident Distribution by Day of Week',
                        'img_data': img_data,
                        'stats': {
                            'peak_day': peak_day,
                            'peak_count': float(peak_count),
                            'peak_percentage': f"{peak_percentage:.1f}%"
                        },
                        'insights': insights
                    }
                    
                    dashboard_charts.append(weekly_chart)
            
            # If we have timestamp, try to create hourly distribution
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                df_hourly = df.copy()
                
                # Ensure timestamp is datetime
                if df_hourly[timestamp_col].dtype != 'datetime64[ns]':
                    df_hourly[timestamp_col] = pd.to_datetime(df_hourly[timestamp_col], errors='coerce')
                
                # Create hour column
                df_hourly['hour_of_day'] = df_hourly[timestamp_col].dt.hour
                
                # Get hourly distribution
                hourly_data = df_hourly['hour_of_day'].value_counts().sort_index()
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=self.fig_size)
                
                # Plot data
                ax.bar(
                    hourly_data.index,
                    hourly_data.values,
                    color=self.colors[1],
                    alpha=0.8
                )
                
                # Set up chart formatting
                ax.set_title('Incident Distribution by Hour of Day', fontsize=14, color=self.text_color)
                ax.set_xlabel('Hour of Day', fontsize=12, color=self.text_color)
                ax.set_ylabel('Count', fontsize=12, color=self.text_color)
                
                # Format x-axis as hours
                ax.set_xticks(range(0, 24, 2))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
                
                # Add values on top of selected bars
                peak_hours = hourly_data.nlargest(3).index
                for i in hourly_data.index:
                    if i in peak_hours:
                        ax.text(
                            i, hourly_data[i] + (max(hourly_data.values) * 0.01), 
                            f"{hourly_data[i]:.0f}", 
                            ha='center', va='bottom', 
                            fontsize=9, color=self.text_color
                        )
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adjust layout
                fig.tight_layout()
                
                # Convert to base64
                img_data = self.fig_to_base64(fig)
                
                # Find hour with most incidents
                if not hourly_data.empty:
                    peak_hour = hourly_data.idxmax()
                    peak_count = hourly_data.max()
                    total_incidents = hourly_data.sum()
                    peak_percentage = (peak_count / total_incidents) * 100 if total_incidents > 0 else 0
                    
                    # Define business hours (8am-6pm)
                    business_hours = range(8, 18)
                    business_hour_data = sum(hourly_data.get(h, 0) for h in business_hours)
                    non_business_hour_data = sum(hourly_data.get(h, 0) for h in range(24) if h not in business_hours)
                    
                    business_percentage = (business_hour_data / total_incidents) * 100 if total_incidents > 0 else 0
                    non_business_percentage = (non_business_hour_data / total_incidents) * 100 if total_incidents > 0 else 0
                    
                    # Generate insights
                    insights = [{
                        'type': 'peak_hour',
                        'hour': int(peak_hour),
                        'hour_formatted': f"{peak_hour:02d}:00",
                        'count': float(peak_count),
                        'percentage': f"{peak_percentage:.1f}%",
                        'message': f"{peak_hour:02d}:00 has the highest incident volume ({peak_percentage:.1f}% of incidents)"
                    },
                    {
                        'type': 'business_hours',
                        'business_percentage': f"{business_percentage:.1f}%",
                        'non_business_percentage': f"{non_business_percentage:.1f}%",
                        'message': f"Business hours (8am-6pm) account for {business_percentage:.1f}% of incidents"
                    }]
                    
                    hourly_chart = {
                        'chart_type': 'distribution',
                        'title': 'Incident Distribution by Hour of Day',
                        'img_data': img_data,
                        'stats': {
                            'peak_hour': int(peak_hour),
                            'peak_hour_formatted': f"{peak_hour:02d}:00",
                            'peak_count': float(peak_count),
                            'peak_percentage': f"{peak_percentage:.1f}%",
                            'business_hours_percentage': f"{business_percentage:.1f}%"
                        },
                        'insights': insights
                    }
                    
                    dashboard_charts.append(hourly_chart)
            
            return {
                'success': True,
                'message': 'Dashboard charts generated successfully',
                'charts': dashboard_charts
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard charts: {str(e)}")
            return {
                'success': False,
                'message': f'Error generating dashboard charts: {str(e)}',
                'charts': None
            }
    
    def create_custom_chart(self, df: pd.DataFrame, chart_config: Dict) -> Optional[Dict]:
        """
        Create a custom chart based on the provided configuration.
        
        Args:
            df: Incident dataframe
            chart_config: Configuration for the chart
                {
                    'type': Chart type ('time_series', 'distribution', 'correlation'),
                    'x_col': Column for x-axis,
                    'y_col': Column for y-axis (optional),
                    'category_col': Column for categories (optional),
                    'title': Chart title,
                    'agg_func': Aggregation function (optional),
                    'chart_subtype': Subtype for correlation charts (optional),
                    ...
                }
            
        Returns:
            Dictionary with chart information and base64 encoded image,
            or None if visualization fails
        """
        if not self._validate_data(df):
            return None
        
        try:
            chart_type = chart_config.get('type', '')
            
            if chart_type == 'time_series':
                return self.time_series_chart(
                    df,
                    timestamp_col=chart_config.get('x_col'),
                    value_col=chart_config.get('y_col'),
                    category_col=chart_config.get('category_col'),
                    agg_func=chart_config.get('agg_func', 'count'),
                    freq=chart_config.get('freq', 'D'),
                    title=chart_config.get('title', 'Time Series Chart'),
                    rolling_window=chart_config.get('rolling_window')
                )
            elif chart_type == 'distribution':
                return self.distribution_chart(
                    df,
                    category_col=chart_config.get('x_col'),
                    value_col=chart_config.get('y_col'),
                    agg_func=chart_config.get('agg_func', 'count'),
                    title=chart_config.get('title', 'Distribution Chart'),
                    sort_by=chart_config.get('sort_by', 'value'),
                    limit=chart_config.get('limit', 10)
                )
            elif chart_type == 'correlation':
                return self.correlation_chart(
                    df,
                    x_col=chart_config.get('x_col'),
                    y_col=chart_config.get('y_col'),
                    category_col=chart_config.get('category_col'),
                    title=chart_config.get('title', 'Correlation Chart'),
                    chart_type=chart_config.get('chart_subtype', 'scatter')
                )
            else:
                self.logger.warning(f"Unsupported chart type: {chart_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating custom chart: {str(e)}")
            return None


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 200
    
    # Create timestamps with weekly patterns
    base_date = datetime.datetime(2023, 1, 1)
    dates = []
    for i in range(n_samples):
        # Add more incidents on weekdays, fewer on weekends
        if i % 7 >= 5:  # Weekend
            if np.random.random() < 0.5:  # 50% chance to skip
                continue
        
        # Add a random number of days (0-90 days)
        random_days = np.random.randint(0, 90)
        random_hours = np.random.randint(0, 24)
        dates.append(base_date + datetime.timedelta(days=random_days, hours=random_hours))
    
    # Create synthetic data
    categories = ['Network', 'Server', 'Application', 'Database', 'Security']
    priorities = ['P1', 'P2', 'P3', 'P4']
    
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(len(dates))],
        'created_at': dates,
        'category': np.random.choice(categories, len(dates)),
        'priority': np.random.choice(priorities, len(dates), p=[0.1, 0.2, 0.4, 0.3]),
        'resolution_time': np.random.exponential(4, len(dates)),  # Hours to resolve
    }
    
    # Add correlation between priority and resolution time
    for i in range(len(dates)):
        if data['priority'][i] == 'P1':
            data['resolution_time'][i] *= 2  # P1 takes longer to resolve
        elif data['priority'][i] == 'P4':
            data['resolution_time'][i] *= 0.5  # P4 resolves faster
    
    # Add correlation between category and resolution time
    for i in range(len(dates)):
        if data['category'][i] == 'Network':
            data['resolution_time'][i] *= 1.5  # Network issues take longer
        elif data['category'][i] == 'Database':
            data['resolution_time'][i] *= 1.3  # Database issues also take longer
    
    df = pd.DataFrame(data)
    
    # Test the chart generator
    generator = ChartGenerator()
    
    # Generate dashboard charts
    dashboard_result = generator.generate_dashboard_charts(df)
    
    if dashboard_result['success']:
        print("DASHBOARD CHARTS:")
        for i, chart in enumerate(dashboard_result['charts']):
            print(f"\nChart {i+1}: {chart['title']}")
            print("Insights:")
            for insight in chart.get('insights', []):
                print(f"- {insight['message']}")
    else:
        print(f"Dashboard chart generation failed: {dashboard_result['message']}")
    
    # Test custom chart
    time_chart = generator.create_custom_chart(df, {
        'type': 'time_series',
        'x_col': 'created_at',
        'category_col': 'priority',
        'title': 'Incidents by Priority Over Time'
    })
    
    if time_chart:
        print("\nCUSTOM CHART:")
        print(f"Title: {time_chart['title']}")
        print("Insights:")
        for insight in time_chart.get('insights', []):
            print(f"- {insight['message']}")                    