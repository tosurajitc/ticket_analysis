import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import io
from matplotlib.figure import Figure
from utils.json_utils import make_json_serializable

class VisualizationAgent:
    """
    Agent responsible for creating visualizations of ticket data.
    """
    
    def __init__(self, llm):
        self.llm = llm
        # Set up visualization style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10})
    
    def generate_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Figure]:
        """
        Generate relevant visualizations based on the data and analysis results
        """
        visualizations = []
        
        # Determine which visualizations to create based on the data
        category_cols = self._identify_category_columns(df, analysis_results)
        priority_cols = self._identify_priority_columns(df, analysis_results)
        status_cols = self._identify_status_columns(df, analysis_results)
        time_cols = self._identify_time_columns(df, analysis_results)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Print the selected columns for debugging
        print(f"Selected category columns: {category_cols}")
        print(f"Selected priority columns: {priority_cols}")
        print(f"Selected status columns: {status_cols}")
        
        # 1. Create pie charts for all appropriate categorical columns (mandatory)
        all_categorical = []
        all_categorical.extend(category_cols)
        all_categorical.extend(priority_cols)
        all_categorical.extend(status_cols)
        all_categorical = list(set(all_categorical))  # Remove duplicates
        
        # Sort by cardinality (columns with better distribution values first)
        all_categorical_with_counts = [(col, df[col].nunique()) for col in all_categorical if 2 <= df[col].nunique() <= 15]
        all_categorical_with_counts.sort(key=lambda x: abs(x[1] - 7))  # Optimal cardinality around 7
        
        # Generate pie charts for the best 3 categorical columns
        for col, _ in all_categorical_with_counts[:3]:
            pie_chart = self._create_pie_chart(df, col)
            if pie_chart:
                visualizations.append(pie_chart)
        
        # 2. Create category distribution bar charts for remaining categories
        for col in category_cols[:2]:  # Limit to top 2 category columns
            if col not in [c for c, _ in all_categorical_with_counts[:3]]:  # Skip if we already made a pie chart
                cat_chart = self._create_category_chart(df, col)
                if cat_chart:
                    visualizations.append(cat_chart)
        
        # 3. Create time-based charts
        if time_cols:
            time_chart = self._create_time_chart(df, time_cols[0])
            if time_chart:
                visualizations.append(time_chart)
        
        # 4. Create category by priority heatmap
        if category_cols and priority_cols:
            heatmap = self._create_category_priority_heatmap(df, category_cols[0], priority_cols[0])
            if heatmap:
                visualizations.append(heatmap)
        
        # 5. Create resolution time distribution
        resolution_time_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['time', 'hours', 'days', 'duration'])]
        if resolution_time_cols:
            time_dist_chart = self._create_resolution_time_chart(df, resolution_time_cols[0])
            if time_dist_chart:
                visualizations.append(time_dist_chart)
        
        return visualizations
    
    def _create_pie_chart(self, df: pd.DataFrame, category_col: str) -> Figure:
        """Create a pie chart for category distribution with improved label handling"""
        try:
            # Get value counts with limit to top 6 categories
            value_counts = df[category_col].value_counts().nlargest(6)
            
            # Add "Other" category if there are more categories
            if df[category_col].nunique() > 6:
                other_count = df[category_col].value_counts().iloc[6:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=['Other'])])
            
            # Calculate percentages
            total = value_counts.sum()
            percentages = (value_counts / total * 100).round(1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(9, 7))
            
            # Use a colorful palette
            colors = plt.cm.tab10(np.linspace(0, 1, len(value_counts)))
            
            # Create two lists - one for labels to display directly on the chart, 
            # and one for labels to only show in the legend
            pie_labels = []
            pie_autopct = []
            
            for i, (label, pct) in enumerate(percentages.items()):
                if pct >= 2.0:  # Only show labels for segments 3% or larger
                    pie_labels.append(label)
                    pie_autopct.append(f'{pct}%')
                else:
                    pie_labels.append('')  # Empty label for small segments
                    pie_autopct.append('')
            
            # Create pie chart
            wedges, texts = ax.pie(
                value_counts, 
                labels=pie_labels,
                colors=colors,
                startangle=90,
                wedgeprops={'width': 0.5, 'edgecolor': 'w'},
                textprops={'fontsize': 10}
            )
            
            # Add percentage annotations for segments >= 3%
            for i, p in enumerate(wedges):
                if percentages.iloc[i] >= 3.0:
                    ang = (p.theta2 - p.theta1)/2. + p.theta1
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    connectionstyle = f"angle,angleA=0,angleB={ang}"
                    ax.annotate(
                        f'{percentages.iloc[i]}%',
                        xy=(x, y), 
                        xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment,
                        arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle),
                    )
            
            # Add a comprehensive legend
            legend_labels = [f"{label} ({percentages.iloc[i]}%)" for i, label in enumerate(value_counts.index)]
            ax.legend(
                wedges, 
                legend_labels, 
                title=f"{category_col.title()}",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=9
            )
            
            # Add a title
            ax.set_title(f'{category_col.title()} Distribution', fontsize=12, fontweight='bold')
            
            # Add a white circle to create a donut chart effect
            centre_circle = plt.Circle((0, 0), 0.25, fc='white')
            ax.add_patch(centre_circle)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating pie chart for {category_col}: {str(e)}")
            return None



    def _identify_category_columns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify columns that represent meaningful categories in the data"""
        category_cols = []
        
        # First check analysis results
        if "category_analysis" in analysis_results:
            category_analysis = analysis_results["category_analysis"]
            if "category_columns" in category_analysis:
                category_cols = category_analysis["category_columns"]
        
        # If no columns found in analysis, try heuristics
        if not category_cols:
            potential_cols = []
            for col in df.columns:
                if any(term in col.lower() for term in ['category', 'type', 'group', 'class']):
                    if df[col].dtype == 'object' or df[col].dtype == 'category':
                        potential_cols.append(col)
            
            # If still no columns found, find columns with moderate cardinality
            if not potential_cols:
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    potential_cols.append(col)
            
            # Score columns based on their distribution properties
            scored_cols = []
            for col in potential_cols:
                # Skip columns with too many NaN values
                null_pct = df[col].isna().mean()
                if null_pct > 0.5:  # Skip if more than 50% values are missing
                    continue
                    
                unique_count = df[col].nunique()
                total_count = len(df)
                
                # Skip columns with only 1 unique value
                if unique_count <= 1:
                    continue
                    
                # Skip columns with too many unique values (likely IDs or free text)
                if unique_count > min(30, total_count * 0.5):
                    continue
                    
                # Calculate distribution score (higher is better)
                value_counts = df[col].value_counts(normalize=True)
                # Avoid columns where one value dominates completely (>95%)
                if value_counts.iloc[0] > 0.95:
                    continue
                    
                # Higher score for columns with good distribution (3-15 values is ideal)
                distribution_score = 0
                if 3 <= unique_count <= 15:
                    distribution_score = 10
                elif unique_count < 3:
                    distribution_score = 5
                else:
                    distribution_score = 7
                    
                # Prioritize columns with more meaningful names
                name_score = 0
                meaningful_terms = ['status', 'priority', 'category', 'severity', 'type', 'reason', 'source']
                if any(term in col.lower() for term in meaningful_terms):
                    name_score = 5
                    
                scored_cols.append((col, distribution_score + name_score))
            
            # Sort by score and get top columns
            scored_cols.sort(key=lambda x: x[1], reverse=True)
            category_cols = [col for col, score in scored_cols[:5]]  # Take top 5 highest scoring columns
        
        return category_cols
    
    def _identify_priority_columns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify columns that represent priority in the data"""
        priority_cols = []
        
        # First check analysis results
        if "priority_analysis" in analysis_results:
            priority_analysis = analysis_results["priority_analysis"]
            if "priority_columns" in priority_analysis:
                priority_cols = priority_analysis["priority_columns"]
        
        # If no columns found in analysis, try heuristics
        if not priority_cols:
            for col in df.columns:
                if any(term in col.lower() for term in ['priority', 'severity', 'urgency', 'importance']):
                    priority_cols.append(col)
        
        return priority_cols
    
    def _identify_status_columns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify columns that represent status in the data"""
        status_cols = []
        
        # First check analysis results
        if "status_analysis" in analysis_results:
            status_analysis = analysis_results["status_analysis"]
            if "status_columns" in status_analysis:
                status_cols = status_analysis["status_columns"]
        
        # If no columns found in analysis, try heuristics
        if not status_cols:
            for col in df.columns:
                if any(term in col.lower() for term in ['status', 'state', 'resolution']):
                    status_cols.append(col)
        
        return status_cols
    
    def _identify_time_columns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify columns that represent time dimensions in the data"""
        time_cols = []
        
        # First check analysis results
        if "time_analysis" in analysis_results:
            time_analysis = analysis_results["time_analysis"]
            if "date_columns" in time_analysis:
                time_cols = time_analysis["date_columns"]
        
        # If no columns found in analysis, try heuristics
        if not time_cols:
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'created', 'time', 'updated']):
                    try:
                        # Check if it can be converted to datetime
                        pd.to_datetime(df[col], errors='coerce')
                        time_cols.append(col)
                    except:
                        pass
        
        return time_cols
    
    def _create_category_chart(self, df: pd.DataFrame, category_col: str) -> Figure:
        """Create a bar chart for category distribution"""
        try:
            # Limit to top 10 categories
            value_counts = df[category_col].value_counts().nlargest(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='bar', ax=ax, color='steelblue')
            
            ax.set_title(f'Top 10 {category_col.title()} Distribution')
            ax.set_xlabel(category_col.title())
            ax.set_ylabel('Count')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for i, v in enumerate(value_counts):
                ax.text(i, v + 0.1, str(v), ha='center')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating category chart for {category_col}: {str(e)}")
            return None
    
    def _create_priority_chart(self, df: pd.DataFrame, priority_col: str) -> Figure:
        """Create a pie chart for priority distribution"""
        try:
            # Get priority distribution
            priority_counts = df[priority_col].value_counts()
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Use a colormap based on priority (assumes higher priority has lower value)
            try:
                # Try to sort priority values if they're numeric or have a natural order
                sorted_priorities = sorted(priority_counts.index, 
                                         key=lambda x: (
                                             # Try to convert to int if possible
                                             int(x) if str(x).isdigit() 
                                             # Otherwise use position in common priority terms
                                             else {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'p0': 0, 'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}.get(str(x).lower(), 999)
                                         ))
                priority_counts = priority_counts.reindex(sorted_priorities)
                colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(priority_counts)))
            except:
                # If sorting fails, use default colors
                colors = plt.cm.tab10(np.linspace(0, 1, len(priority_counts)))
            
            wedges, texts, autotexts = ax.pie(
                priority_counts, 
                labels=priority_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90)
            
            # Improve text visibility
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            ax.set_title(f'{priority_col.title()} Distribution')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating priority chart for {priority_col}: {str(e)}")
            return None
    
    def _create_status_chart(self, df: pd.DataFrame, status_col: str) -> Figure:
        """Create a horizontal bar chart for status distribution"""
        try:
            # Get status distribution
            status_counts = df[status_col].value_counts()
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            status_counts.plot(kind='barh', ax=ax, color='lightseagreen')
            
            ax.set_title(f'{status_col.title()} Distribution')
            ax.set_xlabel('Count')
            ax.set_ylabel(status_col.title())
            
            # Add value labels
            for i, v in enumerate(status_counts):
                ax.text(v + 0.1, i, str(v), va='center')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating status chart for {status_col}: {str(e)}")
            return None
    
    def _create_time_chart(self, df: pd.DataFrame, time_col: str) -> Figure:
        """Create a time series chart for ticket volume over time"""
        try:
            # Make a copy to avoid modifying the original dataframe
            plot_df = df.copy()
            
            # Convert to datetime, ensuring we use string format for consistent parsing
            # Removed infer_datetime_format as it's deprecated
            if plot_df[time_col].dtype != 'datetime64[ns]':
                plot_df[f'{time_col}_dt'] = pd.to_datetime(plot_df[time_col].astype(str), errors='coerce')
            else:
                plot_df[f'{time_col}_dt'] = plot_df[time_col].copy()
            
            # Drop rows with NaT values
            valid_dates = plot_df.dropna(subset=[f'{time_col}_dt'])
            
            if len(valid_dates) == 0:
                print(f"No valid dates found in column {time_col}")
                return None
            
            # Extract month and year
            valid_dates['month_year'] = valid_dates[f'{time_col}_dt'].dt.to_period('M')
            
            # Group by month and count
            monthly_counts = valid_dates.groupby('month_year').size()
            
            # Skip if no data
            if len(monthly_counts) == 0:
                return None
                
            # Convert period index to datetime for plotting
            monthly_counts.index = monthly_counts.index.to_timestamp()
            
            # Create time series plot
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_counts.plot(kind='line', marker='o', ax=ax, color='darkblue')
            
            ax.set_title(f'Ticket Volume by Month ({time_col})')
            ax.set_xlabel('Month')
            ax.set_ylabel('Ticket Count')
            
            # Format x-axis as dates
            plt.xticks(rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Error creating time chart for {time_col}: {str(e)}")
            return None
    
    def _create_category_priority_heatmap(self, df: pd.DataFrame, category_col: str, priority_col: str) -> Figure:
        """Create a heatmap showing category by priority"""
        try:
            # Get cross-tabulation of category and priority
            cross_tab = pd.crosstab(df[category_col], df[priority_col])
            
            # Limit to top 10 categories if there are more
            if len(cross_tab) > 10:
                # Get the top 10 categories by total count
                top_categories = cross_tab.sum(axis=1).nlargest(10).index
                cross_tab = cross_tab.loc[top_categories]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            heatmap = sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            
            ax.set_title(f'{category_col.title()} by {priority_col.title()}')
            ax.set_xlabel(priority_col.title())
            ax.set_ylabel(category_col.title())
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating category-priority heatmap: {str(e)}")
            return None
    
    def _create_resolution_time_chart(self, df: pd.DataFrame, time_col: str) -> Figure:
        """Create a histogram of resolution times"""
        try:
            # Filter out extreme outliers for better visualization
            q1 = df[time_col].quantile(0.25)
            q3 = df[time_col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            # Filter data within bounds
            filtered_data = df[df[time_col] <= upper_bound][time_col]
            
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_data, kde=True, ax=ax, color='mediumseagreen')
            
            ax.set_title(f'Distribution of {time_col.title()}')
            ax.set_xlabel(time_col.title())
            ax.set_ylabel('Frequency')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean and median lines
            mean_val = filtered_data.mean()
            median_val = filtered_data.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating resolution time chart for {time_col}: {str(e)}")
            return None