import streamlit as st
import pandas as pd
from typing import Dict, List, Any

from analysis.resource_optimizer import ResourceOptimizer
from visualization.chart_generator import ChartGenerator

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
        self.chart_generator = ChartGenerator()

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
                st.error(f"Error loading incident data: {e}")
                return

        # Validate data sufficiency
        if data is None or len(data) < 100:
            st.warning("Insufficient data to generate meaningful resource optimization insights. "
                       "Please upload more incident tickets to enable comprehensive analysis.")
            return

        # Identify columns for analysis
        timestamp_col = self._find_timestamp_column(data)
        resolution_time_col = self._find_resolution_time_column(data)
        category_col = self._find_category_column(data)
        priority_col = self._find_priority_column(data)
        assignee_col = self._find_assignee_column(data)

        # Check if we have minimum required columns
        if not (timestamp_col and resolution_time_col):
            st.warning("Missing critical columns for resource optimization analysis.")
            return

        # Generate resource optimization insights
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
            st.error(f"Error generating resource optimization insights: {e}")

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
        # Analyze workload distribution
        workload_result = self.resource_optimizer.analyze_workload_distribution(
            data, 
            timestamp_col=timestamp_col,
            category_col=category_col,
            priority_col=priority_col,
            assignee_col=assignee_col
        )
        
        # Predict staffing needs
        staffing_result = self.resource_optimizer.predict_staffing_needs(
            data,
            timestamp_col=timestamp_col,
            resolution_time_col=resolution_time_col,
            category_col=category_col,
            priority_col=priority_col
        )
        
        # Get skill recommendations
        skill_result = self.resource_optimizer.get_skill_recommendations(
            data,
            resolution_time_col=resolution_time_col,
            category_col=category_col,
            assignee_col=assignee_col
        )
        
        # Generate comprehensive insights
        optimization_insights = self.resource_optimizer.get_resource_optimization_insights(
            workload_result,
            staffing_result,
            skill_result
        )
        
        return {
            "workload_result": workload_result,
            "staffing_result": staffing_result,
            "skill_result": skill_result,
            "optimization_insights": optimization_insights,
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
        
        # Check if insights were successfully generated
        if not insights.get('optimization_insights'):
            st.info("No resource optimization insights could be generated.")
            return
        
        # Display each insight
        for insight in insights['optimization_insights']:
            # Determine styling based on insight type
            severity = 'info'
            if insight['type'] == 'error':
                severity = 'error'
            elif insight['type'] == 'temporal_peak':
                severity = 'warning'
            
            # Create insight card
            st.markdown(f"""
            <div style="
                border: 1px solid {'red' if severity == 'error' else 'orange' if severity == 'warning' else 'blue'};
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: {'#ffeeee' if severity == 'error' else '#fff3cd' if severity == 'warning' else '#e7f3fe'}
            ">
            <strong>{insight.get('type', 'Insight').replace('_', ' ').title()}:</strong> {insight['message']}
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
            self._visualize_staffing_predictions(insights['staffing_result'])
        
        with tabs[2]:
            self._visualize_skill_insights(insights)

    def _visualize_workload_distribution(self, insights: Dict):
        """
        Visualize workload distribution using chart generator
        """
        data = insights.get('data')
        workload_result = insights.get('workload_result', {})
        
        if not data or not workload_result.get('success', False):
            st.info("No workload distribution data available.")
            return
        
        # Find appropriate columns for visualization
        timestamp_col = self._find_timestamp_column(data)
        category_col = self._find_category_column(data)
        
        # Generate time series chart by category
        if timestamp_col and category_col:
            time_series_chart = self.chart_generator.time_series_chart(
                data, 
                timestamp_col=timestamp_col, 
                category_col=category_col,
                title='Incident Volume by Category Over Time'
            )
            
            if time_series_chart:
                st.subheader("Incident Distribution Over Time")
                st.image(
                    f"data:image/png;base64,{time_series_chart['img_data']}", 
                    use_column_width=True
                )
                
                # Display insights from the chart
                if time_series_chart.get('insights'):
                    for insight in time_series_chart['insights']:
                        st.markdown(f"- {insight['message']}")
        
        # Generate distribution chart 
        if category_col:
            distribution_chart = self.chart_generator.distribution_chart(
                data, 
                category_col=category_col, 
                title='Incident Distribution by Category'
            )
            
            if distribution_chart:
                st.subheader("Category Distribution")
                st.image(
                    f"data:image/png;base64,{distribution_chart['img_data']}", 
                    use_column_width=True
                )
                
                # Display insights from the chart
                if distribution_chart.get('insights'):
                    for insight in distribution_chart['insights']:
                        st.markdown(f"- {insight['message']}")

    def _visualize_staffing_predictions(self, staffing_result: Dict):
        """
        Visualize staffing predictions
        """
        if not staffing_result.get('success', False):
            st.info("No staffing prediction data available.")
            return
        
        predictions = staffing_result.get('predictions', {})
        
        # Overall predictions
        if 'overall' in predictions:
            overall = predictions['overall']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Daily Staff", f"{overall['average_daily_staff']:.1f}")
            
            with col2:
                st.metric("Total Predicted Incidents", f"{overall['total_predicted_incidents']:.0f}")
            
            with col3:
                st.metric("Forecast Period", f"{overall['forecast_period_days']} days")
        
        # Category-specific staffing if available
        if 'by_category' in predictions:
            st.subheader("Staffing Needs by Category")
            category_data = predictions['by_category']
            
            # Prepare data for bar chart
            category_staff = {
                category: data['staff_needed'] 
                for category, data in category_data.items()
            }
            
            st.bar_chart(category_staff)
            
            # Detailed category insights
            for category, data in category_data.items():
                st.markdown(f"""
                **{category}**:
                - Staff Needed: {data['staff_needed']} 
                - Predicted Incidents: {data.get('predicted_incidents', 'N/A')} 
                - Average Resolution Time: {data.get('average_resolution_time', 'N/A')} hours
                """)

    def _visualize_skill_insights(self, insights: Dict):
        """
        Visualize skill recommendations and insights
        """
        skill_result = insights.get('skill_result', {})
        data = insights.get('data')
        
        if not skill_result.get('success', False):
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
        
        # Skill gaps and recommendations
        if 'skill_gaps' in recommendations:
            st.subheader("Skill Development Recommendations")
            
            for gap in recommendations['skill_gaps']:
                if gap.get('issue') != 'no_clear_gaps':
                    st.markdown(f"- {gap.get('recommendation', 'No specific recommendation')}")
        
        # Correlation chart between skills/categories if possible
        assignee_col = self._find_assignee_column(data)
        category_col = self._find_category_column(data)
        
        if assignee_col and category_col:
            st.subheader("Assignee Category Distribution")
            
            correlation_chart = self.chart_generator.correlation_chart(
                data, 
                x_col=assignee_col, 
                y_col=category_col, 
                chart_type='boxplot',
                title='Assignee Specialization by Category'
            )
            
            if correlation_chart:
                st.image(
                    f"data:image/png;base64,{correlation_chart['img_data']}", 
                    use_column_width=True
                )
                
                # Display insights from the chart
                if correlation_chart.get('insights'):
                    for insight in correlation_chart['insights']:
                        st.markdown(f"- {insight['message']}")

    def _find_timestamp_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate timestamp column
        """
        timestamp_candidates = [
            'created_at', 'created_date', 'timestamp', 
            'date', 'incident_date', 'opened_at'
        ]
        for col in timestamp_candidates:
            if col in data.columns and pd.api.types.is_datetime64_any_dtype(data[col]):
                return col
        return None

    def _find_resolution_time_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate resolution time column
        """
        resolution_time_candidates = [
            'resolution_time_hours', 'resolution_time', 
            'resolve_time', 'total_time', 'time_to_resolve'
        ]
        for col in resolution_time_candidates:
            if col in data.columns:
                return col
        return None

    def _find_category_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate category column
        """
        category_candidates = [
            'category', 'incident_type', 'type', 
            'classification', 'service', 'group'
        ]
        for col in category_candidates:
            if col in data.columns:
                return col
        return None

    def _find_priority_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate priority column
        """
        priority_candidates = [
            'priority', 'severity', 'urgency', 
            'impact', 'criticality'
        ]
        for col in priority_candidates:
            if col in data.columns:
                return col
        return None

    def _find_assignee_column(self, data: pd.DataFrame) -> str:
        """
        Find appropriate assignee column
        """
        assignee_candidates = [
            'assignee', 'assigned_to', 'handler', 
            'resolver', 'agent', 'support_person'
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