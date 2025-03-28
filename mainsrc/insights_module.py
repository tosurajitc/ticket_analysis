# mainsrc/insights_module.py
# Wrapper for insight_generator

import sys
import os
import streamlit as st
import json

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing module
from insight_generator import InsightGenerator

# Import local modules
from .utils import safe_json_dumps

class InsightsModule:
    """
    Module for generating insights from ticket data statistics.
    Wraps around the existing InsightGenerator with better error handling.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the insights module.
        
        Args:
            groq_client: GROQ API client for LLM
        """
        self.insight_generator = InsightGenerator(groq_client)
    
    def generate_insights(self, df, stats):
        """
        Generate insights from statistics.
        
        Args:
            df: Processed DataFrame
            stats: Statistics dictionary
            
        Returns:
            str: Generated insights text
        """
        try:
            # Use safe JSON serialization
            insights = self.insight_generator.generate_insights(df)
            return insights
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            # Generate fallback insights
            return self._generate_fallback_insights(stats)
    
    def _generate_fallback_insights(self, stats):
        """
        Generate basic insights when LLM fails.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            str: Basic insights text
        """
        insights = []
        
        # Add basic ticket volume insight
        if "total_tickets" in stats:
            insights.append(f"• The dataset contains a total of {stats['total_tickets']} tickets.")
        
        # Add priority distribution insight
        if "priority_distribution" in stats:
            priorities = stats["priority_distribution"]
            top_priority = max(priorities.items(), key=lambda x: x[1])
            insights.append(f"• The most common priority level is '{top_priority[0]}' with {top_priority[1]} tickets ({(top_priority[1]/stats['total_tickets'])*100:.1f}% of total).")
        
        # Add resolution time insight
        if "avg_resolution_time_hours" in stats:
            insights.append(f"• The average resolution time is {stats['avg_resolution_time_hours']:.2f} hours.")
        
        # Add top issue insight
        if "top_issues" in stats and stats["top_issues"]:
            top_issue = list(stats["top_issues"].items())[0]
            insights.append(f"• The most common issue is '{top_issue[0]}', appearing in {top_issue[1]} tickets.")
        
        # Add assignment group insight
        if "top_assignment_groups" in stats and stats["top_assignment_groups"]:
            top_group = list(stats["top_assignment_groups"].items())[0]
            insights.append(f"• The team handling the most tickets is '{top_group[0]}' with {top_group[1]} tickets.")
        
        # Add geographic insight
        if "top_regions" in stats and stats["top_regions"]:
            top_region = list(stats["top_regions"].items())[0]
            insights.append(f"• The region with the most tickets is '{top_region[0]}' with {top_region[1]} tickets.")
        
        # Add business hours insight
        if "business_hours_tickets" in stats and "non_business_hours_tickets" in stats:
            bh = stats["business_hours_tickets"]
            nbh = stats["non_business_hours_tickets"]
            total = bh + nbh
            insights.append(f"• {bh} tickets ({(bh/total)*100:.1f}%) were created during business hours, while {nbh} tickets ({(nbh/total)*100:.1f}%) were created outside business hours.")
        
        # Join all insights
        return "\n\n".join(insights)