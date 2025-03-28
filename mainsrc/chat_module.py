# mainsrc/chat_module.py
# Module for chat interface

import sys
import os
import streamlit as st
import json

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from .utils import safe_json_dumps

class ChatModule:
    """
    Module for the chat interface with the ticket data.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the chat module.
        
        Args:
            groq_client: GROQ API client for LLM
        """
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    def process_query(self, query, df, stats, insights):
        """
        Process a natural language query about ticket data.
        
        Args:
            query: User query string
            df: Processed DataFrame
            stats: Statistics dictionary
            insights: Insights text
            
        Returns:
            str: Response to the query
        """
        try:
            # Extract query-specific statistics
            query_stats = self._extract_query_stats(df, query, stats)
            
            # Create prompt for LLM
            prompt = self._create_query_prompt(query, query_stats, insights)
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Provide clear, specific answers to questions about ticket data. Use the provided statistics to inform your responses, and be honest about limitations in the data where relevant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Extract the generated response
            answer = response.choices[0].message.content
            
            return answer
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return f"I'm having trouble processing your query: '{query}'. Please try again or rephrase your question."
    
    def _extract_query_stats(self, df, query, base_stats):
        """
        Extract statistics relevant to a specific query.
        
        Args:
            df: Processed DataFrame
            query: User query string
            base_stats: Base statistics dictionary
            
        Returns:
            dict: Query-relevant statistics
        """
        # Start with a subset of the base stats that are always relevant
        query_stats = {
            "total_tickets": base_stats.get("total_tickets", 0)
        }
        
        # Add other basics if available
        for key in ["priority_distribution", "state_distribution", "top_assignment_groups"]:
            if key in base_stats:
                query_stats[key] = base_stats[key]
        
        # Add statistics based on query keywords
        query_lower = query.lower()
        
        # Geographic/country queries
        if any(word in query_lower for word in ['country', 'region', 'geographic', 'location', 'territory', 'continent', 
                                              'amer', 'apac', 'emea', 'global', 'america', 'europe', 'asia']):
            for key in ["geographic_columns_found", "geographic_distribution", "top_regions", "region_specific_issues"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Priority queries
        if any(word in query_lower for word in ['priority', 'priorities', 'urgent', 'critical', 'high', 'medium', 'low']):
            if "priority_distribution" in base_stats:
                query_stats["priority_distribution"] = base_stats["priority_distribution"]
        
        # Time/resolution queries
        if any(word in query_lower for word in ['time', 'duration', 'resolution', 'resolve', 'solved', 'fast', 'slow', 'sla']):
            for key in ["avg_resolution_time_hours", "median_resolution_time_hours", "resolution_time_quartiles"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Datafix queries
        if any(word in query_lower for word in ['datafix', 'data fix', 'db fix']):
            for key in ["datafix_mentions", "datafix_by_region"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Document failure queries
        if any(word in query_lower for word in ['document', 'doc', 'report', 'fail', 'failure']):
            for key in ["document_failure_count", "document_failures_by_year"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Escalation queries
        if any(word in query_lower for word in ['escalate', 'escalation', 'elevated']):
            for key in ["escalation_mentions", "escalation_by_region"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Issue queries
        if any(word in query_lower for word in ['issue', 'problem', 'common', 'frequent']):
            for key in ["top_issues", "issues_by_region"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        # Automation queries
        if any(word in query_lower for word in ['automate', 'automation', 'bot', 'rpa']):
            for key in ["automation_opportunities_from_descriptions", "automation_opportunities_from_resolutions"]:
                if key in base_stats:
                    query_stats[key] = base_stats[key]
        
        return query_stats
    
    def _create_query_prompt(self, query, stats, insights):
        """
        Create a prompt for the LLM to answer a query.
        
        Args:
            query: User query string
            stats: Query-relevant statistics
            insights: Insights text
            
        Returns:
            str: Prompt for LLM
        """
        # Basic prompt with query and statistics
        prompt = f"""
The user has asked the following question about their ticket data:
"{query}"

Based on the available ticket data statistics:
{safe_json_dumps(stats)}
"""

        # Add insights if available
        if insights:
            prompt += f"""
The following insights have already been identified from this data:
{insights}
"""

        # Add instructions for the response
        prompt += """
Please provide a clear, concise answer to the user's question based on the data provided. 

If the data doesn't contain information needed to fully answer the question, explain what's missing and provide the best possible answer with the available data.

If relevant, suggest further analysis or data collection that would help provide a more complete answer.

If an automation opportunity is relevant to the question, briefly mention it.
"""

        return prompt