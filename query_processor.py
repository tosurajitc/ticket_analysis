
# Module for processing user queries about ticket data

import pandas as pd
import numpy as np
import json
import streamlit as st
import os

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Period):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

class QueryProcessor:
    """
    Class for processing user queries about ticket data.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the QueryProcessor.
        
        Args:
            groq_client: GROQ API client for LLM access
        """
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Get model from env or use default
    
    def process_query(self, query, df, insights=None):
        """
        Process a user query about the ticket data.
        
        Args:
            query (str): User query
            df (pandas.DataFrame): Processed ticket data
            insights (str): Previously generated insights (optional)
            
        Returns:
            str: Response to user query
        """
        # Extract relevant statistics to include with the query
        stats = self._extract_query_relevant_statistics(df, query)
        
        # Create a prompt for the LLM
        prompt = self._create_query_prompt(query, stats, insights)
        
        # Make sure prompt is not None or empty
        if not prompt or prompt.strip() == "":
            prompt = f"The user asked: '{query}'. Please answer this question about ticket data as best as possible with the available information."
        
        # Generate response using LLM
        try:
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
            return "I'm having trouble processing your query. Please try again or rephrase your question."
    
    def _extract_query_relevant_statistics(self, df, query):
        """
        Extract statistics from the data that are relevant to the user's query.
        Performs simple keyword matching to determine what statistics might be relevant.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            query (str): User query
            
        Returns:
            dict: Dictionary of query-relevant statistics
        """
        stats = {}
        query_lower = query.lower()
        
        # Always include basic stats
        stats["total_tickets"] = int(len(df))
        stats["available_columns"] = list(df.columns)
        
        # Check for query keywords and include relevant stats
        
        # Priority related queries
        if any(word in query_lower for word in ["priority", "priorities", "urgent", "critical", "high", "medium", "low"]):
            if 'priority' in df.columns:
                priority_counts = df['priority'].value_counts().to_dict()
                stats["priority_distribution"] = {k: int(v) for k, v in priority_counts.items()}
                
                # If specific priority is mentioned, add more details
                for priority in ["critical", "high", "medium", "low"]:
                    if priority in query_lower and 'priority' in df.columns:
                        priority_tickets = df[df['priority'].str.lower().str.contains(priority, na=False)]
                        if len(priority_tickets) > 0:
                            stats[f"{priority}_priority_count"] = int(len(priority_tickets))
                            
                            if 'resolution_time_hours' in priority_tickets.columns:
                                stats[f"{priority}_avg_resolution_time"] = float(priority_tickets['resolution_time_hours'].mean())
        
        # Time/resolution related queries
        if any(word in query_lower for word in ["time", "duration", "resolution", "resolve", "solved", "fast", "slow", "sla"]):
            if 'resolution_time_hours' in df.columns:
                stats["avg_resolution_time_hours"] = float(df['resolution_time_hours'].mean())
                stats["median_resolution_time_hours"] = float(df['resolution_time_hours'].median())
                stats["min_resolution_time_hours"] = float(df['resolution_time_hours'].min())
                stats["max_resolution_time_hours"] = float(df['resolution_time_hours'].max())
                
                if 'resolution_time_category' in df.columns:
                    resolution_category_counts = df['resolution_time_category'].value_counts().to_dict()
                    stats["resolution_time_categories"] = {k: int(v) for k, v in resolution_category_counts.items()}
        
        # Assignment group related queries
        if any(word in query_lower for word in ["group", "team", "assign", "assignment", "department"]):
            if 'assignment group' in df.columns:
                top_groups = df['assignment group'].value_counts().head(10).to_dict()
                stats["top_assignment_groups"] = {k: int(v) for k, v in top_groups.items()}
                
                # If workload or efficiency is mentioned
                if any(word in query_lower for word in ["workload", "busy", "efficiency", "efficient", "performance"]):
                    group_stats = {}
                    for group in list(top_groups.keys())[:5]:  # Limit to top 5 for efficiency
                        group_tickets = df[df['assignment group'] == group]
                        group_stat = {"ticket_count": int(len(group_tickets))}
                        
                        if 'resolution_time_hours' in group_tickets.columns:
                            group_stat["avg_resolution_time"] = float(group_tickets['resolution_time_hours'].mean())
                            
                        group_stats[group] = group_stat
                        
                    stats["assignment_group_details"] = group_stats
        
        # State/status related queries
        if any(word in query_lower for word in ["state", "status", "open", "closed", "pending", "resolved"]):
            if 'state' in df.columns:
                state_counts = df['state'].value_counts().to_dict()
                stats["state_distribution"] = {k: int(v) for k, v in state_counts.items()}
        
        # Time trends related queries
        if any(word in query_lower for word in ["trend", "volume", "increase", "decrease", "time", "month", "year", "weekly"]):
            if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
                # Monthly ticket counts
                monthly_counts = df.groupby(df['opened'].dt.to_period('M')).size()
                stats["monthly_ticket_counts"] = {str(period): int(count) for period, count in monthly_counts.items()}
                
                # Calculate trend (simple linear comparison of first half vs second half)
                periods = sorted(monthly_counts.index)
                if len(periods) >= 2:
                    mid_point = len(periods) // 2
                    first_half_avg = float(monthly_counts[periods[:mid_point]].mean())
                    second_half_avg = float(monthly_counts[periods[mid_point:]].mean())
                    trend_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
                    stats["ticket_volume_trend_pct"] = float(trend_pct)
        
        # Common issues related queries
        if any(word in query_lower for word in ["issue", "problem", "common", "frequent", "recurring"]):
            issue_columns = [col for col in df.columns if col.startswith('contains_')]
            if issue_columns:
                issue_counts = {}
                for col in issue_columns:
                    issue_name = col.replace('contains_', '')
                    issue_counts[issue_name] = int(df[col].sum())
                
                # Get top issues
                top_issues = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                stats["top_issues"] = top_issues
        
        # Business hours related queries
        if any(word in query_lower for word in ["business hours", "working hours", "after hours", "weekend"]):
            if 'is_business_hours' in df.columns:
                business_hours_count = int(df['is_business_hours'].sum())
                non_business_hours_count = int((~df['is_business_hours']).sum())
                stats["business_hours_tickets"] = business_hours_count
                stats["non_business_hours_tickets"] = non_business_hours_count
                
                # Calculate percentages
                total = business_hours_count + non_business_hours_count
                if total > 0:
                    stats["business_hours_percentage"] = float((business_hours_count / total) * 100)
                    stats["non_business_hours_percentage"] = float((non_business_hours_count / total) * 100)
        
        # Specific country-related queries
        if any(word in query_lower for word in ["country", "region", "location", "geographic"]):
            if any(col.lower() in ["country", "region", "location"] for col in df.columns):
                country_col = next((col for col in df.columns if col.lower() in ["country", "region", "location"]), None)
                if country_col:
                    country_counts = df[country_col].value_counts().head(10).to_dict()
                    stats["country_distribution"] = {k: int(v) for k, v in country_counts.items()}
            else:
                stats["country_info"] = "No country or region information found in the dataset"
                
        # Datafix related queries
        if any(word in query_lower for word in ["datafix", "data fix", "database fix", "db fix"]):
            datafix_info = "No explicit datafix information found in the dataset"
            
            # Look for datafix keywords in descriptions or work notes
            if 'short description' in df.columns:
                datafix_keywords = ['datafix', 'data fix', 'db fix', 'database fix', 'database repair']
                datafix_count = 0
                
                for keyword in datafix_keywords:
                    datafix_count += int(df['short description'].str.lower().str.contains(keyword, na=False).sum())
                
                if datafix_count > 0:
                    stats["datafix_mentions"] = datafix_count
                    datafix_info = f"Found {datafix_count} tickets mentioning datafix in their description"
            
            stats["datafix_info"] = datafix_info
                
        # Escalation related queries
        if any(word in query_lower for word in ["escalate", "escalation", "elevated", "priority increase"]):
            escalation_info = "No explicit escalation information found in the dataset"
            
            # Look for escalation keywords in work notes or state transitions
            if 'work notes' in df.columns:
                escalation_keywords = ['escalate', 'escalation', 'elevated', 'raised to', 'priority increase']
                escalation_count = 0
                
                for keyword in escalation_keywords:
                    escalation_count += int(df['work notes'].str.lower().str.contains(keyword, na=False).sum())
                
                if escalation_count > 0:
                    stats["escalation_mentions"] = escalation_count
                    escalation_info = f"Found {escalation_count} tickets with escalation mentions in work notes"
            
            stats["escalation_info"] = escalation_info
                
        # Document failure related queries
        if any(word in query_lower for word in ["document", "doc", "report", "file", "failed", "failure"]):
            document_info = "No explicit document failure information found in the dataset"
            
            # Look for document failure keywords
            if 'short description' in df.columns:
                doc_keywords = ['document failure', 'report failure', 'failed document', 'document error']
                doc_count = 0
                
                for keyword in doc_keywords:
                    doc_count += int(df['short description'].str.lower().str.contains(keyword, na=False).sum())
                
                if doc_count > 0:
                    stats["document_failure_count"] = doc_count
                    document_info = f"Found {doc_count} tickets related to document failures"
            
            stats["document_failure_info"] = document_info
        
        return stats
    
    def _create_query_prompt(self, query, stats, insights=None):
        """
        Create a prompt for the LLM to answer the user query.
        
        Args:
            query (str): User query
            stats (dict): Dictionary of relevant statistics
            insights (str): Previously generated insights (optional)
            
        Returns:
            str: Prompt for the LLM
        """
        # Basic prompt with query and statistics
        prompt = f"""
The user has asked the following question about their ticket data:
"{query}"

Based on the available ticket data statistics:
{json.dumps(stats, indent=2, cls=NumpyEncoder)}
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