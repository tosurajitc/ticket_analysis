# insight_generator.py
# Module for generating insights from ticket data using LLM

import pandas as pd
import json
import streamlit as st
import os
import numpy as np
from datetime import datetime

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

class InsightGenerator:
    """
    Class for generating insights from ticket data using LLM.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the InsightGenerator.
        
        Args:
            groq_client: GROQ API client for LLM access
        """
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Get model from env or use default
    
    def generate_insights(self, df):
        """
        Generate insights from the processed ticket data.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            str: Markdown formatted insights
        """
        # Extract key statistics to help the LLM with insights
        stats = self._extract_statistics(df)
        
        # Create a prompt for the LLM
        prompt = self._create_insight_prompt(stats)
        
        # Make sure prompt is not None or empty
        if not prompt or prompt.strip() == "":
            prompt = "Please analyze the ticket data statistics and provide insights."
        
        # Generate insights using LLM
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Your task is to provide clear, concise insights based on ticket data statistics. Focus on patterns, trends, and actionable observations. Be direct and highlight key findings without any unnecessary commentary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Extract the generated insights
            insights = response.choices[0].message.content
            
            return insights
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return "Error generating insights. Please check your connection and try again."
    
    def generate_predefined_questions(self, df):
        """
        Generate predefined questions and answers based on ticket data.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            list: List of dictionaries with questions, answers, and automation potential
        """
        # Extract key statistics
        stats = self._extract_statistics(df)
        
        # Create a prompt for the LLM
        prompt = f"""
Based on the following ticket data statistics:
{json.dumps(stats, indent=2, cls=NumpyEncoder)}

Generate exactly 10 insightful questions that would help analyze this ticket data.

Include these 5 specific questions:
1. Which category of incidents consume the greatest number of hours?
2. Which category of incidents require datafix?
3. Which issues are specific to a particular country?
4. Which incidents indicate escalation?
5. How many customer facing documents failed in the last year?

Then generate 5 additional random questions that cover different aspects of the ticket data such as:
- Priority distribution analysis
- Assignment group workload and efficiency
- Ticket volume trends over time
- Business hours vs. non-business hours analysis
- Common issue identification
- Resolution time patterns
- SLA compliance
- Self-service opportunities
- Team performance comparisons

For each question, provide:
1. A clear, data-driven answer based on the statistics provided
2. Whether there's automation potential (Yes/No) and a detailed explanation of what kind of automation would be suitable

For questions where the data doesn't have enough information to provide a complete answer, acknowledge the limitation and suggest what additional data would be needed.

Format your response as a JSON array with objects like:
[
  {{
    "question": "Which category of incidents consume the greatest number of hours?",
    "answer": "Based on the data, Network-related incidents consume the greatest number of hours with an average resolution time of 28.5 hours per ticket. Database issues follow with 24.3 hours on average.",
    "automation_potential": "Yes - Automated diagnostic tools and predefined resolution workflows could be implemented for network issues to reduce resolution time. Machine learning models could predict resolution time and proactively allocate resources to high-consumption categories."
  }}
]
"""
        
        # Make sure prompt is not None or empty
        if not prompt or prompt.strip() == "":
            prompt = "Generate questions and answers about the ticket data statistics."
            
        # Generate questions using LLM
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Generate insightful questions and data-driven answers based on ticket statistics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            # Extract and parse the JSON response
            response_text = response.choices[0].message.content
            
            # Extract JSON if it's embedded in markdown or additional text
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    questions = json.loads(json_str)
                    return questions
                except json.JSONDecodeError:
                    st.warning("Could not parse the generated questions. Using fallback questions.")
                    return self._generate_fallback_questions(stats)
            else:
                st.warning("No valid JSON found in the response. Using fallback questions.")
                return self._generate_fallback_questions(stats)
            
        except Exception as e:
            st.error(f"Error generating predefined questions: {str(e)}")
            return self._generate_fallback_questions(stats)
    
    def _generate_fallback_questions(self, stats):
        """
        Generate fallback questions when LLM fails.
        
        Args:
            stats (dict): Dictionary of key statistics
            
        Returns:
            list: List of dictionaries with questions, answers, and automation potential
        """
        # Create basic fallback questions based on available statistics
        questions = []
        
        # Start with the 5 specific required questions
        specific_questions = [
            {
                "question": "Which category of incidents consume the greatest number of hours?",
                "answer": "This requires detailed resolution time data by category, which may not be fully available in the current dataset.",
                "automation_potential": "Yes - Time tracking automation and category-based analytics could help identify time-consuming incident types."
            },
            {
                "question": "Which category of incidents require datafix?",
                "answer": "This requires specific labeling of datafix incidents in the dataset, which may not be available in the current statistics.",
                "automation_potential": "Yes - Automated classification of tickets that required datafixes based on resolution notes and actions taken."
            },
            {
                "question": "Which issues are specific to a particular country?",
                "answer": "Location or country data appears to be limited in the current dataset. Additional geographic information would be needed for this analysis.",
                "automation_potential": "Yes - Automated geographic tagging and region-specific issue tracking could be implemented."
            },
            {
                "question": "Which incidents indicate escalation?",
                "answer": "Escalation data would need to be extracted from work notes or state transitions, which may require additional analysis of the dataset.",
                "automation_potential": "Yes - Automated escalation detection and workflow management could be implemented to identify and track escalated tickets."
            },
            {
                "question": "How many customer facing documents failed in the last year?",
                "answer": "This specific information about document failures may not be categorized in the current dataset. Additional tagging or categorization would be needed.",
                "automation_potential": "Yes - Automated document monitoring and failure detection systems could be implemented to track customer-facing document issues."
            }
        ]
        questions.extend(specific_questions)
        
        # Add additional general questions to reach 10 total
        if "total_tickets" in stats:
            questions.append({
                "question": "What is the total volume of tickets in the dataset?",
                "answer": f"There are {stats['total_tickets']} tickets in the dataset.",
                "automation_potential": "No - This is a basic statistic that doesn't require automation."
            })
        
        # Question about priority distribution
        if "priority_distribution" in stats:
            priority_str = ", ".join([f"{priority}: {count} tickets" for priority, count in stats["priority_distribution"].items()])
            questions.append({
                "question": "What is the distribution of ticket priorities?",
                "answer": f"The priority distribution is: {priority_str}",
                "automation_potential": "Yes - Automatic ticket prioritization based on text analysis could be implemented."
            })
        
        # Question about resolution time
        if "avg_resolution_time_hours" in stats:
            questions.append({
                "question": "What is the average resolution time for tickets?",
                "answer": f"The average resolution time is {stats['avg_resolution_time_hours']:.2f} hours.",
                "automation_potential": "Yes - Automated SLA monitoring and alerting could be implemented."
            })
        
        # Question about common issues
        if "top_issues" in stats:
            top_issues_str = ", ".join([f"{issue}: {count} tickets" for issue, count in list(stats["top_issues"].items())[:3]])
            questions.append({
                "question": "What are the most common issues in the tickets?",
                "answer": f"The most common issues are: {top_issues_str}",
                "automation_potential": "Yes - Knowledge base articles and automated responses could be created for common issues."
            })
        
        # Question about assignment groups
        if "top_assignment_groups" in stats:
            top_groups_str = ", ".join([f"{group}: {count} tickets" for group, count in list(stats["top_assignment_groups"].items())[:3]])
            questions.append({
                "question": "Which teams handle the most tickets?",
                "answer": f"The teams handling the most tickets are: {top_groups_str}",
                "automation_potential": "Yes - Workload balancing and automatic ticket routing could be implemented."
            })
        
        # If we have more than 10 questions, trim to 10
        return questions[:10]  # Return at most 10 questions
    
    def _extract_statistics(self, df):
        """
        Extract key statistics from the processed ticket data.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            dict: Dictionary of key statistics
        """
        stats = {}
        
        # Convert all NumPy types to standard Python types for JSON serialization
        def convert_to_python_type(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Period):
                return str(obj)
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python_type(item) for item in obj]
            else:
                return obj
        
        # Basic counts
        stats["total_tickets"] = int(len(df))
        
        # Priority distribution if available
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts().to_dict()
            stats["priority_distribution"] = {k: int(v) for k, v in priority_counts.items()}
        
        # State distribution if available
        if 'state' in df.columns:
            state_counts = df['state'].value_counts().to_dict()
            stats["state_distribution"] = {k: int(v) for k, v in state_counts.items()}
        
        # Assignment group distribution if available
        if 'assignment group' in df.columns:
            # Get top 10 assignment groups
            top_groups = df['assignment group'].value_counts().head(10).to_dict()
            stats["top_assignment_groups"] = {k: int(v) for k, v in top_groups.items()}
        
        # Time-based statistics if available
        if 'resolution_time_hours' in df.columns:
            stats["avg_resolution_time_hours"] = float(df['resolution_time_hours'].mean())
            stats["median_resolution_time_hours"] = float(df['resolution_time_hours'].median())
            
            if 'resolution_time_category' in df.columns:
                resolution_category_counts = df['resolution_time_category'].value_counts().to_dict()
                stats["resolution_time_categories"] = {k: int(v) for k, v in resolution_category_counts.items()}
        
        # Time trends if opened column exists and is datetime
        if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
            # Monthly ticket volume
            monthly_counts = df.groupby(df['opened'].dt.to_period('M')).size()
            stats["monthly_ticket_counts"] = {str(period): int(count) for period, count in monthly_counts.items()}
            
            # Day of week distribution
            if 'opened_day_of_week' in df.columns:
                day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                          4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                day_counts = df['opened_day_of_week'].value_counts().sort_index().to_dict()
                stats["day_of_week_distribution"] = {day_map[day]: int(count) for day, count in day_counts.items()}
            
            # Business hours vs. non-business hours
            if 'is_business_hours' in df.columns:
                business_hours_count = int(df['is_business_hours'].sum())
                non_business_hours_count = int((~df['is_business_hours']).sum())
                stats["business_hours_tickets"] = business_hours_count
                stats["non_business_hours_tickets"] = non_business_hours_count
        
        # Common issues if we extracted them
        issue_columns = [col for col in df.columns if col.startswith('contains_')]
        if issue_columns:
            issue_counts = {}
            for col in issue_columns:
                issue_name = col.replace('contains_', '')
                issue_counts[issue_name] = int(df[col].sum())
            
            # Get top issues
            top_issues = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            stats["top_issues"] = top_issues
        
        return stats
    
    def _create_insight_prompt(self, stats):
        """
        Create a prompt for the LLM to generate insights.
        
        Args:
            stats (dict): Dictionary of key statistics
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = f"""
Analyze the following IT service management ticket data statistics and provide clear, actionable insights:
{json.dumps(stats, indent=2)}

Generate 10-15 insights that address:
1. Ticket volume patterns and trends
2. Priority and severity distribution analysis
3. Resolution time performance
4. Assignment group workload and efficiency
5. Common issues and potential systemic problems
6. Service level agreement (SLA) compliance if data available
7. Opportunities for process improvement

Format your insights as bullet points with clear, concise statements. Each insight should be data-driven and highlight something significant about the ticket data."""