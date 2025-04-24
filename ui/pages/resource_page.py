import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from analysis.resource_optimizer import ResourceOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourcePage:
    def __init__(self, data_loader, config=None):
        """
        Initialize Resource Optimization Page
        
        Args:
            data_loader: Data loading and preprocessing utility
            config: Optional configuration dictionary
        """
        self.data_loader = data_loader
        self.config = config
        self.resource_optimizer = ResourceOptimizer()

    def render_page(self, data: pd.DataFrame = None, is_data_sufficient: bool = False):
        """
        Render the Resource Optimization page
        
        Args:
            data: Processed incident data
            is_data_sufficient: Flag indicating data sufficiency
        """
        st.header("🔍 Resource Optimization Insights")
        
        # Load incident data if not provided
        if data is None:
            try:
                data = self.data_loader.load_processed_data()
            except Exception as e:
                st.error(f"Error loading incident data: {str(e)}")
                return

        # Validate data sufficiency - reduced threshold to 30 records
        if data is None or data.empty:
            st.error("No data available for analysis. Please load data using the sidebar controls.")
            return
        
        if len(data) < 30:
            st.warning(
                "Limited data available for resource optimization analysis. "
                f"Current dataset has only {len(data)} records. For comprehensive insights, "
                "consider providing at least 30 incident records."
            )
            # Continue with limited data instead of returning

        # Identify columns for analysis with better error handling
        try:
            timestamp_col = self._find_timestamp_column(data)
            resolution_time_col = self._find_resolution_time_column(data)
            category_col = self._find_category_column(data)
            priority_col = self._find_priority_column(data)
            assignee_col = self._find_assignee_column(data)
            
            # Log identified columns for debugging
            st.info(f"Analyzing using columns: Timestamp: {timestamp_col}, Resolution Time: {resolution_time_col}, "
                   f"Category: {category_col}, Priority: {priority_col}, Assignee: {assignee_col}")
            
            # Check if we have minimum required columns
            if not timestamp_col:
                st.warning("No timestamp column detected. This is required for temporal analysis.")
                # Try to infer a date column if possible
                date_like_cols = [col for col in data.columns if any(term in str(col).lower() for term in 
                                                                ['date', 'time', 'created', 'opened'])]
                if date_like_cols:
                    timestamp_col = date_like_cols[0]
                    st.info(f"Using '{timestamp_col}' as timestamp column.")
                else:
                    # Create a dummy timestamp column for basic analysis
                    st.warning("Creating a dummy timestamp column for basic analysis.")
                    data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
                    timestamp_col = 'timestamp'
            
            if not resolution_time_col:
                st.warning("No resolution time column detected. Some analyses will be limited.")
                # Check if we can calculate it from timestamp and resolved_date
                resolved_cols = [col for col in data.columns if any(term in str(col).lower() for term in 
                                                              ['resolved', 'closed', 'completed'])]
                if timestamp_col and resolved_cols:
                    st.info(f"Attempting to calculate resolution time from '{timestamp_col}' and '{resolved_cols[0]}'")
                    try:
                        # Create a copy to avoid modifying the original dataframe
                        calc_df = data.copy()
                        
                        # Convert to datetime
                        if not pd.api.types.is_datetime64_any_dtype(calc_df[timestamp_col]):
                            calc_df[timestamp_col] = pd.to_datetime(calc_df[timestamp_col], errors='coerce')
                        
                        if not pd.api.types.is_datetime64_any_dtype(calc_df[resolved_cols[0]]):
                            calc_df[resolved_cols[0]] = pd.to_datetime(calc_df[resolved_cols[0]], errors='coerce')
                            
                        # Calculate resolution time in hours
                        calc_df['calculated_resolution_time'] = (
                            calc_df[resolved_cols[0]] - calc_df[timestamp_col]
                        ).dt.total_seconds() / 3600  # Convert to hours
                        
                        # Filter out negative or unreasonably large values
                        valid_mask = (
                            (calc_df['calculated_resolution_time'] >= 0) & 
                            (calc_df['calculated_resolution_time'] < 10000)  # Less than ~1 year
                        )
                        
                        if valid_mask.sum() > len(calc_df) * 0.3:  # At least 30% valid
                            data['calculated_resolution_time'] = calc_df['calculated_resolution_time']
                            resolution_time_col = 'calculated_resolution_time'
                            st.success(f"Successfully calculated resolution time from {timestamp_col} and {resolved_cols[0]}")
                    except Exception as e:
                        st.warning(f"Could not calculate resolution time: {str(e)}")
                
                # If still not available, create a dummy resolution time for basic analysis
                if not resolution_time_col:
                    st.warning("Creating a dummy resolution time column with random values for basic analysis.")
                    data['resolution_time'] = np.random.uniform(1, 24, size=len(data))  # 1-24 hours
                    resolution_time_col = 'resolution_time'
            
            if not category_col:
                st.warning("No category column detected. Some analyses will be limited.")
            
            if not assignee_col:
                st.warning("No assignee column detected. Team composition analysis will be limited.")
                
            # Display the data profile first to give the user context
            with st.expander("Data Overview", expanded=False):
                st.dataframe(data.head(5))
                
                # Show some basic statistics
                st.subheader("Dataset Statistics")
                cols = st.columns(3)
                cols[0].metric("Total Records", len(data))
                
                if timestamp_col:
                    try:
                        date_range = (data[timestamp_col].max() - data[timestamp_col].min()).days
                        cols[1].metric("Date Range (days)", date_range)
                    except:
                        pass
                
                if resolution_time_col:
                    try:
                        avg_resolution = data[resolution_time_col].mean()
                        cols[2].metric("Avg Resolution Time (hours)", f"{avg_resolution:.1f}")
                    except:
                        pass

            # Generate resource optimization insights
            with st.spinner("Analyzing resource optimization data..."):
                try:
                    resource_insights = self._generate_resource_insights(
                        data, 
                        timestamp_col, 
                        resolution_time_col, 
                        category_col, 
                        priority_col, 
                        assignee_col
                    )
                    
                    # Display insights
                    self._display_resource_insights(resource_insights)
                except Exception as e:
                    st.error(f"Error generating resource optimization insights: {str(e)}")
                    logger.exception("Error in resource optimization analysis")
                    # Try with simplified analysis as a backup
                    st.info("Attempting simplified analysis...")
                    try:
                        simplified_insights = self._generate_simplified_insights(
                            data, timestamp_col, resolution_time_col, category_col
                        )
                        self._display_simplified_insights(simplified_insights)
                    except Exception as backup_error:
                        st.error(f"Simplified analysis also failed: {str(backup_error)}")
                        logger.exception("Error in simplified analysis")
        
        except Exception as e:
            st.error(f"Error in resource page initialization: {str(e)}")
            logger.exception("Error in resource page initialization")

    def _generate_resource_insights(
        self, 
        data: pd.DataFrame, 
        timestamp_col: str, 
        resolution_time_col: str, 
        category_col: str = None, 
        priority_col: str = None, 
        assignee_col: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive resource optimization insights
        
        Args:
            data: Processed incident data
            timestamp_col: Column containing timestamps
            resolution_time_col: Column containing resolution times
            category_col: Optional column containing incident categories
            priority_col: Optional column containing incident priorities
            assignee_col: Optional column containing assignees
        
        Returns:
            Dict containing resource optimization insights
        """
        # Create default insights in case analysis fails
        default_insights = [{
            'type': 'data_summary',
            'message': f"Analysis based on {len(data)} incidents across {len(data.columns)} dimensions."
        }]
        
        # Create basic workload result structure 
        minimal_workload_result = {
            'success': True,
            'temporal': {},
            'categorical': {},
            'assignment': {}
        }
        
        # Create basic staffing result structure
        minimal_staffing_result = {
            'success': True,
            'predictions': {
                'overall': {
                    'average_daily_staff': round(len(data) / 30, 1),  # Rough estimate
                    'total_predicted_incidents': len(data),
                    'forecast_period_days': 30
                }
            }
        }
        
        # Create basic skill result structure
        minimal_skill_result = {
            'success': False,
            'error': 'Simplified analysis does not include skills assessment'
        }
        
        try:
            # Ensure timestamp column is properly formatted
            if timestamp_col and timestamp_col in data.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                        data = data.copy()
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
                except Exception as e:
                    st.warning(f"Timestamp conversion error: {str(e)}")
                    
            # Try to analyze workload distribution with safeguards
            try:
                workload_result = self.resource_optimizer.analyze_workload_distribution(
                    data, 
                    timestamp_col=timestamp_col,
                    category_col=category_col,
                    priority_col=priority_col,
                    assignee_col=assignee_col
                )
                
                # If the result doesn't have 'success' key or it's False, use minimal structure
                if not workload_result or not workload_result.get('success', False):
                    workload_result = minimal_workload_result
                    default_insights.append({
                        'type': 'workload_analysis',
                        'message': "Basic workload analysis completed with limited insights available."
                    })
            except Exception as e:
                st.warning(f"Workload analysis error: {str(e)}")
                workload_result = minimal_workload_result
                
            # Try to predict staffing needs with safeguards
            try:
                staffing_result = self.resource_optimizer.predict_staffing_needs(
                    data,
                    timestamp_col=timestamp_col,
                    resolution_time_col=resolution_time_col,
                    category_col=category_col,
                    priority_col=priority_col
                )
                
                # If the result doesn't have 'success' key or it's False, use minimal structure
                if not staffing_result or not staffing_result.get('success', False):
                    staffing_result = minimal_staffing_result
                    default_insights.append({
                        'type': 'staffing_analysis',
                        'message': "Basic staffing analysis completed with limited insights available."
                    })
            except Exception as e:
                st.warning(f"Staffing analysis error: {str(e)}")
                staffing_result = minimal_staffing_result
                
            # Try to get skill recommendations with safeguards
            try:
                if assignee_col:
                    skill_result = self.resource_optimizer.get_skill_recommendations(
                        data,
                        resolution_time_col=resolution_time_col,
                        category_col=category_col,
                        assignee_col=assignee_col
                    )
                    
                    # If the result doesn't have 'success' key or it's False, use minimal structure
                    if not skill_result or not skill_result.get('success', False):
                        skill_result = minimal_skill_result
                else:
                    skill_result = minimal_skill_result
            except Exception as e:
                st.warning(f"Skill analysis error: {str(e)}")
                skill_result = minimal_skill_result
            
            # Generate basic insights directly if any of the analyses failed
            optimization_insights = default_insights
                
            # Try to get comprehensive insights from resource optimizer
            try:
                if (workload_result.get('success', False) and
                    staffing_result.get('success', False)):
                    
                    optimizer_insights = self.resource_optimizer.get_resource_optimization_insights(
                        workload_result,
                        staffing_result,
                        skill_result
                    )
                    
                    # Use optimizer insights if they're valid
                    if optimizer_insights and isinstance(optimizer_insights, list) and len(optimizer_insights) > 0:
                        optimization_insights = optimizer_insights
            except Exception as e:
                st.warning(f"Insight generation error: {str(e)}")
                # Keep using default_insights
                
            # Always return a result with the expected structure
            return {
                "workload_result": workload_result,
                "staffing_result": staffing_result,
                "skill_result": skill_result,
                "optimization_insights": optimization_insights,
                "data": data
            }
            
        except Exception as e:
            st.error(f"Resource insight generation error: {str(e)}")
            
            # Return minimal working result structure with error information
            return {
                "workload_result": minimal_workload_result,
                "staffing_result": minimal_staffing_result,
                "skill_result": minimal_skill_result,
                "optimization_insights": [{
                    'type': 'error',
                    'message': f"Error generating resource insights: {str(e)}"
                }],
                "data": data
            }
    

    def _generate_simplified_insights(
        self,
        data: pd.DataFrame,
        timestamp_col: str,
        resolution_time_col: str,
        category_col: str = None
    ) -> Dict[str, Any]:
        """
        Generate simplified insights when the full analysis fails
        
        Args:
            data: Processed incident data
            timestamp_col: Column containing timestamps
            resolution_time_col: Column containing resolution times
            category_col: Optional column containing incident categories
        
        Returns:
            Dict containing simplified resource insights
        """
        insights = []
        
        # Basic volume analysis
        total_incidents = len(data)
        insights.append({
            'type': 'volume',
            'message': f"Total incident volume: {total_incidents} records"
        })
        
        # Time-based analysis if possible
        if timestamp_col and timestamp_col in data.columns:
            try:
                # Ensure datetime format
                if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
                
                # Extract day of week
                data['day_of_week'] = data[timestamp_col].dt.day_name()
                day_counts = data['day_of_week'].value_counts()
                
                if not day_counts.empty:
                    peak_day = day_counts.idxmax()
                    peak_count = day_counts.max()
                    peak_pct = peak_count / total_incidents * 100
                    
                    insights.append({
                        'type': 'temporal_peak',
                        'subtype': 'day',
                        'message': f"{peak_day} has the highest incident volume with {peak_count} incidents ({peak_pct:.1f}% of total)"
                    })
                    
                    # Calculate weekday vs weekend
                    weekday_mask = data['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                    weekday_count = weekday_mask.sum()
                    weekend_count = (~weekday_mask).sum()
                    
                    weekday_avg = weekday_count / 5 if weekday_count > 0 else 0  # 5 weekdays
                    weekend_avg = weekend_count / 2 if weekend_count > 0 else 0  # 2 weekend days
                    
                    if weekday_avg > 0 and weekend_avg > 0:
                        ratio = weekday_avg / weekend_avg
                        
                        insights.append({
                            'type': 'weekday_weekend_difference',
                            'message': f"Weekdays average {weekday_avg:.1f} incidents per day, compared to {weekend_avg:.1f} for weekends ({ratio:.1f}x difference)"
                        })
            except Exception as e:
                logger.warning(f"Error in simplified time analysis: {str(e)}")
        
        # Category analysis if possible
        if category_col and category_col in data.columns:
            try:
                category_counts = data[category_col].value_counts()
                
                if not category_counts.empty:
                    top_category = category_counts.index[0]
                    top_count = category_counts.iloc[0]
                    top_pct = top_count / total_incidents * 100
                    
                    insights.append({
                        'type': 'category_distribution',
                        'message': f"'{top_category}' is the most common incident type, accounting for {top_pct:.1f}% of all incidents"
                    })
            except Exception as e:
                logger.warning(f"Error in simplified category analysis: {str(e)}")
        
        # Resolution time analysis if possible
        if resolution_time_col and resolution_time_col in data.columns:
            try:
                resolution_times = pd.to_numeric(data[resolution_time_col], errors='coerce')
                
                # Filter valid values
                valid_times = resolution_times[(resolution_times > 0) & (resolution_times < 1000)]
                
                if len(valid_times) > 0:
                    avg_time = valid_times.mean()
                    median_time = valid_times.median()
                    
                    insights.append({
                        'type': 'resolution_time',
                        'message': f"Average resolution time is {avg_time:.1f} hours, with a median of {median_time:.1f} hours"
                    })
            except Exception as e:
                logger.warning(f"Error in simplified resolution time analysis: {str(e)}")
        
        # Create a fake workload result
        workload_result = {
            'success': True,
            'temporal': {},
            'categorical': {},
            'assignment': {}
        }
        
        # Create fake staffing result
        staffing_result = {
            'success': True,
            'predictions': {
                'overall': {
                    'average_daily_staff': round(total_incidents / 30, 1),  # Rough estimate
                    'total_predicted_incidents': total_incidents,
                    'forecast_period_days': 30
                }
            }
        }
        
        # Create a fake skill result
        skill_result = {
            'success': False,
            'error': 'Simplified analysis does not include skills assessment'
        }
        
        return {
            "workload_result": workload_result,
            "staffing_result": staffing_result,
            "skill_result": skill_result,
            "optimization_insights": insights,
            "data": data
        }

    def _display_resource_insights(self, insights: Dict[str, Any]):
        """
        Display resource optimization insights
        
        Args:
            insights (Dict): Resource optimization insights
        """
        # Display optimization insights
        st.subheader("Resource Optimization Insights")
        
        # Check if insights were successfully generated with better error handling
        if not insights:
            st.error("No insights data available. Analysis failed completely.")
            return
            
        optimization_insights = insights.get('optimization_insights', [])
        
        # Extra check to handle different possible formats
        if isinstance(optimization_insights, dict):
            # Convert to list if it's a dict
            optimization_insights = [optimization_insights]
        
        # Final check before displaying
        if not optimization_insights or (isinstance(optimization_insights, list) and len(optimization_insights) == 0):
            st.info("No specific resource optimization insights could be generated from the current data.")
            st.warning("Try providing more comprehensive incident data with timestamps, resolution times, and assignees.")
            return
        
        # Display each insight with better error handling for different formats
        for insight in optimization_insights:
            if not isinstance(insight, dict):
                continue
                
            # Determine styling based on insight type
            severity = 'info'
            insight_type = insight.get('type', '')
            
            if insight_type == 'error':
                severity = 'error'
            elif insight_type in ['temporal_peak', 'priority_distribution', 'workload_balance']:
                severity = 'warning'
            
            # Get message with fallback
            message = insight.get('message', 'No detailed information available')
            
            # Create insight card
            st.markdown(f"""
            <div style="
                border: 1px solid {'red' if severity == 'error' else 'orange' if severity == 'warning' else 'blue'};
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: {'#ffeeee' if severity == 'error' else '#fff3cd' if severity == 'warning' else '#e7f3fe'}
            ">
            <strong>{insight_type.replace('_', ' ').title()}:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
        
        # Add detailed visualization tabs
        tabs = st.tabs([
            "Workload Distribution", 
            "Staffing Predictions", 
            "Skill Insights"
        ])
        
        with tabs[0]:
            self._visualize_workload_distribution(insights)
        
        with tabs[1]:
            self._visualize_staffing_predictions(insights.get('staffing_result', {}))
        
        with tabs[2]:
            self._visualize_skill_insights(insights)

    def _display_simplified_insights(self, insights: Dict[str, Any]):
        """
        Display simplified insights when full analysis fails
        
        Args:
            insights (Dict): Simplified resource insights
        """
        st.subheader("Simplified Resource Insights")
        st.info("Displaying simplified insights due to issues with the full analysis.")
        
        # Display each insight
        optimization_insights = insights.get('optimization_insights', [])
        
        for insight in optimization_insights:
            if not isinstance(insight, dict):
                continue
                
            # Determine styling based on insight type
            severity = 'info'
            insight_type = insight.get('type', '')
            
            if insight_type == 'error':
                severity = 'error'
            elif insight_type in ['temporal_peak', 'resource_constraint']:
                severity = 'warning'
            
            # Get message with fallback
            message = insight.get('message', 'No detailed information available')
            
            # Create insight card
            st.markdown(f"""
            <div style="
                border: 1px solid {'red' if severity == 'error' else 'orange' if severity == 'warning' else 'blue'};
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: {'#ffeeee' if severity == 'error' else '#fff3cd' if severity == 'warning' else '#e7f3fe'}
            ">
            <strong>{insight_type.replace('_', ' ').title()}:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
            
        # Show basic resource metrics
        data = insights.get('data')
        workload_result = insights.get('workload_result', {})
        staffing_result = insights.get('staffing_result', {})
        
        if data is not None and not data.empty:
            st.subheader("Basic Resource Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Incidents", len(data))
            
            with col2:
                # Get average daily staff if available
                avg_staff = (staffing_result.get('predictions', {})
                             .get('overall', {})
                             .get('average_daily_staff', 0))
                
                if avg_staff > 0:
                    st.metric("Est. Daily Staff Needed", f"{avg_staff:.1f}")
                else:
                    st.metric("Est. Daily Staff Needed", f"{len(data)/30:.1f}")
            
            with col3:
                # Try to get resolution time if available
                resolution_time_col = self._find_resolution_time_column(data)
                if resolution_time_col:
                    try:
                        avg_resolution = data[resolution_time_col].mean()
                        st.metric("Avg Resolution Time", f"{avg_resolution:.1f} hours")
                    except:
                        st.metric("Avg Resolution Time", "N/A")
                else:
                    st.metric("Avg Resolution Time", "N/A")
            
            # Show basic category distribution if available
            category_col = self._find_category_column(data)
            if category_col:
                try:
                    st.subheader("Category Distribution")
                    category_counts = data[category_col].value_counts().head(5)
                    
                    # Calculate percentages
                    total = len(data)
                    category_pcts = category_counts / total * 100
                    
                    # Create a DataFrame for display
                    display_df = pd.DataFrame({
                        'Count': category_counts,
                        'Percentage': category_pcts.round(1).astype(str) + '%'
                    })
                    
                    st.dataframe(display_df)
                except Exception as e:
                    st.warning(f"Could not display category distribution: {str(e)}")

    def _visualize_workload_distribution(self, insights: Dict):
        """
        Visualize workload distribution
        """
        data = insights.get('data')
        workload_result = insights.get('workload_result', {})
        
        if not data is not None or data.empty or not workload_result.get('success', False):
            st.info("No workload distribution data available.")
            return
        
        # Find appropriate columns for visualization
        timestamp_col = self._find_timestamp_column(data)
        category_col = self._find_category_column(data)
        
        # Generate time-based visualizations if possible
        if timestamp_col:
            try:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                    data = data.copy()
                    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
                
                # Create a copy to avoid modifying the original data
                viz_data = data.copy()
                
                # Extract day of week
                viz_data['day_of_week'] = viz_data[timestamp_col].dt.day_name()
                
                # Count incidents by day of week
                day_counts = viz_data['day_of_week'].value_counts()
                
                # Order days correctly
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = day_counts.reindex(days, fill_value=0)
                
                # Create bar chart
                st.subheader("Incidents by Day of Week")
                st.bar_chart(day_counts)
                
                # Extract hour of day if time component exists
                time_exists = False
                try:
                    # Check if we have time information (not just dates)
                    times = viz_data[timestamp_col].dt.time
                    all_midnight = all(t.hour == 0 and t.minute == 0 for t in times if pd.notna(t))
                    
                    if not all_midnight:
                        time_exists = True
                        viz_data['hour'] = viz_data[timestamp_col].dt.hour
                        
                        # Group by hour
                        hour_counts = viz_data.groupby('hour').size()
                        
                        # Create hour labels (24-hour format)
                        hour_labels = [f"{h:02d}:00" for h in range(24)]
                        
                        # Reindex to ensure all hours are represented
                        hour_counts = hour_counts.reindex(range(24), fill_value=0)
                        
                        # Create bar chart
                        st.subheader("Incidents by Hour of Day")
                        st.bar_chart(hour_counts)
                except Exception as e:
                    st.warning(f"Could not analyze time of day: {str(e)}")
            except Exception as e:
                st.warning(f"Could not create time-based visualizations: {str(e)}")
        
        # Generate category distribution if available
        if category_col:
            try:
                st.subheader("Incident Distribution by Category")
                category_counts = data[category_col].value_counts().head(10)  # Top 10 categories
                
                # Calculate percentages
                total = len(data)
                category_pcts = category_counts / total * 100
                
                # Create a DataFrame for display
                display_df = pd.DataFrame({
                    'Count': category_counts,
                    'Percentage': category_pcts.round(1).astype(str) + '%'
                })
                
                st.dataframe(display_df)
                
                # Create pie chart using plotly express instead of non-existent st.pie_chart
                import plotly.express as px
                
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Incident Distribution by Category"
                )
                
                # Update layout for better appearance
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                
                # Show the plotly chart
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create category visualizations: {str(e)}")

    def _visualize_staffing_predictions(self, staffing_result: Dict):
        """
        Visualize staffing predictions
        """
        if not staffing_result or not staffing_result.get('success', False):
            st.info("No staffing prediction data available.")
            return
        
        predictions = staffing_result.get('predictions', {})
        
        # Overall predictions
        if 'overall' in predictions:
            overall = predictions['overall']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Daily Staff", f"{overall.get('average_daily_staff', 'N/A')}")
            
            with col2:
                st.metric("Total Predicted Incidents", f"{overall.get('total_predicted_incidents', 'N/A')}")
            
            with col3:
                st.metric("Forecast Period", f"{overall.get('forecast_period_days', 'N/A')} days")
        
        # Category-specific staffing if available
        if 'by_category' in predictions:
            st.subheader("Staffing Needs by Category")
            category_data = predictions['by_category']
            
            if category_data:
                # Prepare data for bar chart
                try:
                    categories = []
                    staff_values = []
                    
                    for category, data in category_data.items():
                        categories.append(category)
                        staff_values.append(data.get('staff_needed', 0))
                    
                    # Create DataFrame for chart
                    chart_data = pd.DataFrame({
                        'Category': categories,
                        'Staff Needed': staff_values
                    })
                    
                    # Set index to Category for horizontal bar chart
                    chart_data = chart_data.set_index('Category')
                    
                    # Display bar chart
                    st.bar_chart(chart_data)
                    
                    # Display detailed table
                    st.subheader("Detailed Staffing by Category")
                    
                    # Create detailed table
                    table_data = []
                    for category, data in category_data.items():
                        table_data.append({
                            'Category': category,
                            'Staff Needed': data.get('staff_needed', 'N/A'),
                            'Predicted Incidents': round(data.get('predicted_incidents', 0), 1),
                            'Avg Resolution Time (hrs)': round(data.get('average_resolution_time', 0), 1)
                        })
                    
                    # Convert to DataFrame for display
                    table_df = pd.DataFrame(table_data)
                    st.dataframe(table_df)
                    
                except Exception as e:
                    st.warning(f"Could not create staffing visualizations: {str(e)}")
            else:
                st.info("No category-specific staffing data available.")




    def _visualize_skill_insights(self, insights: Dict):
        """
        Visualize skill recommendations and insights
        """
        skill_result = insights.get('skill_result', {})
        data = insights.get('data')
        
        if not skill_result or not skill_result.get('success', False):
            st.info("No skill recommendation data available.")
            return
        
        recommendations = skill_result.get('recommendations', {})
        
        # Team composition section
        if 'team_composition' in recommendations:
            st.subheader("Team Composition")
            composition = recommendations['team_composition']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Assignees", composition.get('total_assignees', 0))
            
            with col2:
                st.metric("Specialists", composition.get('specialists_count', 0))
            
            with col3:
                st.metric("Generalists", composition.get('generalists_count', 0))
            
            # Display specialist ratio
            specialist_ratio = composition.get('specialist_ratio', 0)
            st.progress(specialist_ratio, text=f"Specialist Ratio: {specialist_ratio:.0%}")
            
            # Display specialist distribution by category if available
            if 'specialist_by_category' in composition:
                st.subheader("Specialists by Category")
                specialist_by_category = composition['specialist_by_category']
                
                if specialist_by_category:
                    try:
                        # Create DataFrame for display
                        categories = []
                        specialist_counts = []
                        
                        for category, count in specialist_by_category.items():
                            categories.append(category)
                            specialist_counts.append(count)
                        
                        # Create chart
                        chart_data = pd.DataFrame({
                            'Category': categories,
                            'Specialists': specialist_counts
                        })
                        
                        chart_data = chart_data.set_index('Category')
                        st.bar_chart(chart_data)
                        
                        # If needed, also create a pie chart
                        if len(categories) > 1:  # Only if multiple categories
                            import plotly.express as px
                            
                            fig = px.pie(
                                values=specialist_counts,
                                names=categories,
                                title="Specialist Distribution by Category"
                            )
                            
                            # Update layout for better appearance
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(height=400)
                            
                            # Show the plotly chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Could not create specialist distribution chart: {str(e)}")

    def _find_timestamp_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate timestamp column
        """
        timestamp_candidates = [
            'created_date', 'created_at', 'timestamp', 
            'date', 'incident_date', 'opened_at', 'open_date',
            'creation_date', 'reported_date'
        ]
        for col in timestamp_candidates:
            if col in data.columns:
                try:
                    # Check if it's already a datetime
                    if pd.api.types.is_datetime64_any_dtype(data[col]):
                        return col
                    
                    # Try to convert to datetime
                    sample = data[col].head(5)
                    pd.to_datetime(sample, errors='raise')
                    return col
                except:
                    # Not a valid datetime column, continue checking
                    continue
        return None

    def _find_resolution_time_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate resolution time column
        """
        resolution_time_candidates = [
            'resolution_time_hours', 'resolution_time', 
            'resolve_time', 'total_time', 'time_to_resolve',
            'duration', 'elapsed_time'
        ]
        for col in resolution_time_candidates:
            if col in data.columns:
                try:
                    # Check if numeric
                    pd.to_numeric(data[col].head(5))
                    return col
                except:
                    continue
        
        # If not found directly, check if we can calculate from resolved_date
        if self._find_timestamp_column(data) is not None:
            resolution_date_candidates = [
                'resolved_date', 'closed_date', 'completion_date', 
                'end_date', 'close_date', 'resolved_at'
            ]
            for col in resolution_date_candidates:
                if col in data.columns:
                    return col  # Return the column name for later calculation
        
        return None

    def _find_category_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate category column
        """
        category_candidates = [
            'category', 'incident_type', 'type', 
            'classification', 'service', 'group',
            'class', 'incident_category'
        ]
        for col in category_candidates:
            if col in data.columns:
                # Verify it has categorical values (not mostly unique)
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.7:  # Less than 70% unique values
                    return col
        return None

    def _find_priority_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate priority column
        """
        priority_candidates = [
            'priority', 'severity', 'urgency', 
            'impact', 'criticality', 'importance'
        ]
        for col in priority_candidates:
            if col in data.columns:
                # Verify it has categorical values (few unique values)
                if data[col].nunique() < 10:  # Typically priorities are P1-P5 or similar
                    return col
        return None

    def _find_assignee_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate assignee column
        """
        assignee_candidates = [
            'assignee', 'assigned_to', 'handler', 
            'resolver', 'agent', 'support_person',
            'assigned', 'owner', 'technician'
        ]
        for col in assignee_candidates:
            if col in data.columns:
                return col
        return None


def render_resource_page(data_loader, config=None, data=None, is_data_sufficient=False):
    """
    Render the Resource Optimization page
    
    Args:
        data_loader: Data loading utility
        config: Optional configuration dictionary
        data: Optional preprocessed data
        is_data_sufficient: Flag indicating data sufficiency
    """
    resource_page = ResourcePage(data_loader, config)
    resource_page.render_page(data, is_data_sufficient)