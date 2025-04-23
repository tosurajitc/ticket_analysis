import streamlit as st
import pandas as pd
from typing import Dict, List, Any

from analysis.resource_optimizer import ResourceOptimizer
from visualization.chart_generator import ChartGenerator

class ResourcePage:
    def __init__(self, data_loader):
        """
        Initialize Resource Optimization Page
        
        Args:
            data_loader: Data loading and preprocessing utility
        """
        self.data_loader = data_loader
        self.resource_optimizer = ResourceOptimizer()
        self.chart_generator = ChartGenerator()

    def render_page(self):
        """
        Render the Resource Optimization page
        """
        st.header("🔍 Resource Optimization Insights")
        
        # Load incident data
        try:
            incident_data = self.data_loader.load_processed_data()
        except Exception as e:
            st.error(f"Error loading incident data: {e}")
            return

        # Validate data sufficiency
        if incident_data is None or len(incident_data) < 100:
            st.warning("Insufficient data to generate meaningful resource optimization insights. "
                       "Please upload more incident tickets to enable comprehensive analysis.")
            return

        # Generate resource optimization insights
        try:
            resource_insights = self._generate_resource_insights(incident_data)
            
            # Display insights
            self._display_resource_insights(resource_insights)
        
        except Exception as e:
            st.error(f"Error generating resource optimization insights: {e}")

    def _generate_resource_insights(self, incident_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate resource optimization opportunities
        
        Args:
            incident_data (pd.DataFrame): Processed incident data
        
        Returns:
            Dict containing resource optimization insights
        """
        # Analyze skill distribution
        skill_analysis = self.resource_optimizer.analyze_skill_distribution(incident_data)
        
        # Identify resource allocation recommendations
        resource_recommendations = self.resource_optimizer.generate_resource_recommendations(
            incident_data, 
            skill_analysis
        )
        
        # Analyze workload distribution
        workload_distribution = self.resource_optimizer.analyze_workload_distribution(incident_data)
        
        return {
            "skill_analysis": skill_analysis,
            "resource_recommendations": resource_recommendations,
            "workload_distribution": workload_distribution
        }

    def _display_resource_insights(self, insights: Dict[str, Any]):
        """
        Display resource optimization insights
        
        Args:
            insights (Dict): Resource optimization insights
        """
        # Skill Distribution Visualization
        st.subheader("Skill Distribution Analysis")
        skill_dist_chart = self.chart_generator.create_skill_distribution_chart(
            insights['skill_analysis']
        )
        st.plotly_chart(skill_dist_chart, use_container_width=True)
        
        # Workload Distribution Visualization
        st.subheader("Workload Distribution")
        workload_chart = self.chart_generator.create_workload_distribution_chart(
            insights['workload_distribution']
        )
        st.plotly_chart(workload_chart, use_container_width=True)
        
        # Resource Recommendations Section
        st.subheader("Resource Optimization Recommendations")
        
        recommendations = insights['resource_recommendations']
        if not recommendations:
            st.info("No specific resource optimization recommendations at this time.")
            return
        
        # Display resource recommendations
        for idx, recommendation in enumerate(recommendations, 1):
            with st.expander(f"Recommendation {idx}: {recommendation['title']}"):
                st.markdown(f"**Description:** {recommendation['description']}")
                st.markdown(f"**Potential Impact:** {recommendation['impact']}")
                st.markdown(f"**Confidence Score:** {recommendation['confidence']}%")
                
                # Visualize recommendation impact
                impact_chart = self.chart_generator.create_resource_impact_chart(recommendation)
                st.plotly_chart(impact_chart, use_container_width=True)

def render_resource_page(data_loader):
    """
    Render the Resource Optimization page
    
    Args:
        data_loader: Data loading utility
    """
    resource_page = ResourcePage(data_loader)
    resource_page.render_page()