import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

def render_automation_page(
    data: Optional[pd.DataFrame] = None, 
    config: Optional[Dict[str, Any]] = None, 
    is_data_sufficient: bool = False
) -> None:
    """
    Render the Automation Opportunities page.
    
    Args:
        data: Processed incident data DataFrame
        config: Application configuration
        is_data_sufficient: Whether the data is sufficient for comprehensive analysis
    """
    # If no data is available, show a message
    if data is None or data.empty:
        st.warning("No data available for automation analysis. Please upload and prepare your incident data.")
        return
    
    st.header("🤖 Automation Opportunities Analysis")
    
    # Check data sufficiency
    if not is_data_sufficient:
        st.warning("Insufficient data for comprehensive automation analysis.")
    
    # Existing logic from the original class method
    # Load incident data
    if data is None or len(data) < 50:
        st.warning("Insufficient data to generate meaningful automation insights. "
                   "Please upload more incident tickets.")
        return

    # Detect automation opportunities
    try:
        # Replace with simplified analysis method
        automation_insights = _generate_automation_insights(data)
        
        # Display insights
        _display_automation_insights(automation_insights)
    
    except Exception as e:
        st.error(f"Error generating automation insights: {e}")

def _generate_automation_insights(incident_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate basic automation opportunities based on incident data.
    
    Args:
        incident_data (pd.DataFrame): Processed incident data
    
    Returns:
        Dict containing automation insights
    """
    # Simplified automation insights generation
    
    # 1. Repetitive Incident Analysis
    repetitive_incidents = incident_data.groupby(['category', 'description']).size().reset_index(name='count')
    repetitive_incidents = repetitive_incidents[repetitive_incidents['count'] > 5]  # Threshold for repetition
    
    # 2. Resolution Time Analysis
    resolution_time_analysis = None
    if 'resolution_time_hours' in incident_data.columns:
        resolution_time_analysis = incident_data.groupby('category')['resolution_time_hours'].agg(['mean', 'count'])
    
    return {
        "repetitive_incidents": repetitive_incidents,
        "resolution_time_analysis": resolution_time_analysis
    }

def _display_automation_insights(insights: Dict[str, Any]):
    """
    Display automation insights.
    
    Args:
        insights (Dict): Automation insights dictionary
    """
    # Repetitive Incidents Section
    st.subheader("Repetitive Incident Patterns")
    if not insights["repetitive_incidents"].empty:
        st.dataframe(insights["repetitive_incidents"])
        st.info("These incident types could be prime candidates for automation.")
    else:
        st.info("No highly repetitive incident patterns detected.")
    
    # Resolution Time Analysis Section
    st.subheader("Resolution Time Analysis")
    if insights["resolution_time_analysis"] is not None:
        st.dataframe(insights["resolution_time_analysis"])
        
        # Identify categories with longest resolution times
        longest_resolution = insights["resolution_time_analysis"].sort_values('mean', ascending=False)
        st.markdown("### Categories with Longest Resolution Times")
        st.dataframe(longest_resolution.head())