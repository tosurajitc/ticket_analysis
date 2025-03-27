# automation_analyzer.py
# Module for analyzing automation opportunities in ticket data

import pandas as pd
import json
import streamlit as st
import os
import numpy as np

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

class AutomationAnalyzer:
    """
    Class for analyzing ticket data to identify automation opportunities.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the AutomationAnalyzer.
        
        Args:
            groq_client: GROQ API client for LLM access
        """
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Get model from env or use default
    
    def analyze(self, df, insights):
        """
        Analyze ticket data to identify automation opportunities.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            insights (str): Generated insights text
            
        Returns:
            list: List of automation opportunity dictionaries
        """
        # Extract statistics to help identify automation opportunities
        stats = self._extract_statistics(df)
        
        # Create a prompt for the LLM
        prompt = self._create_automation_prompt(stats, insights)
        
        # Make sure prompt is not None or empty
        if not prompt or prompt.strip() == "":
            prompt = "Please analyze the ticket data and suggest automation opportunities."
        
        # Generate automation opportunities using LLM
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an IT automation expert specializing in service management and ticket automation. Your task is to analyze ticket data and provide detailed, actionable automation opportunities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            # Extract the generated content
            content = response.choices[0].message.content
            
            # Extract JSON if it's embedded in markdown or additional text
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    opportunities = json.loads(json_str)
                    return opportunities
                except json.JSONDecodeError:
                    st.warning("Could not parse the generated automation opportunities. Using fallback opportunities.")
                    return self._generate_fallback_opportunities(stats)
            else:
                st.warning("No valid JSON found in the response. Using fallback opportunities.")
                return self._generate_fallback_opportunities(stats)
            
        except Exception as e:
            st.error(f"Error generating automation opportunities: {str(e)}")
            return self._generate_fallback_opportunities(stats)
    
    def _extract_statistics(self, df):
        """
        Extract statistics relevant for automation analysis.
        
        Args:
            df (pandas.DataFrame): Processed ticket data
            
        Returns:
            dict: Dictionary of automation-relevant statistics
        """
        stats = {}
        
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
            # Get top assignment groups
            top_groups = df['assignment group'].value_counts().head(10).to_dict()
            stats["top_assignment_groups"] = {k: int(v) for k, v in top_groups.items()}
        
        # Resolution time statistics if available
        if 'resolution_time_hours' in df.columns:
            stats["avg_resolution_time_hours"] = float(df['resolution_time_hours'].mean())
            stats["median_resolution_time_hours"] = float(df['resolution_time_hours'].median())
            
            if 'resolution_time_category' in df.columns:
                resolution_category_counts = df['resolution_time_category'].value_counts().to_dict()
                stats["resolution_time_categories"] = {k: int(v) for k, v in resolution_category_counts.items()}
        
        # Time trends if opened column exists
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
        
        # Check for repetitive assignment patterns
        if 'assignment group' in df.columns and 'short description' in df.columns:
            # This is a simplified analysis - in a real implementation you'd want 
            # to do more sophisticated text clustering and pattern recognition
            
            # Get sample of descriptions for top assignment groups
            assignment_samples = {}
            for group in list(stats.get("top_assignment_groups", {}).keys())[:3]:
                group_tickets = df[df['assignment group'] == group]
                common_terms = []
                
                # Extract common terms from descriptions (very simplified approach)
                if 'short description' in group_tickets.columns:
                    descriptions = group_tickets['short description'].astype(str).str.lower()
                    common_words = {}
                    
                    for desc in descriptions:
                        words = desc.split()
                        for word in words:
                            if len(word) > 3:  # Ignore very short words
                                common_words[word] = common_words.get(word, 0) + 1
                    
                    # Get top words
                    sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
                    top_words = [word for word, count in sorted_words[:10]]
                    common_terms = top_words
                
                assignment_samples[group] = common_terms
            
            stats["assignment_patterns"] = assignment_samples
        
        return stats
    
    def _create_automation_prompt(self, stats, insights):
        """
        Create a prompt for the LLM to identify automation opportunities.
        
        Args:
            stats (dict): Dictionary of ticket statistics
            insights (str): Generated insights text
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = f"""
Based on the following ticket data statistics and insights, identify the top 5 automation opportunities:

STATISTICS:
{json.dumps(stats, indent=2, cls=NumpyEncoder)}

INSIGHTS:
{insights}

For each automation opportunity, provide:
1. A descriptive title
2. Detailed automation scope
3. Justification with data-backed reasoning
4. Type of automation (AI, RPA, scripting, etc.) with specific technology recommendations
5. Detailed implementation plan including steps, timeline, and required resources

Format your response as a JSON array with objects like:
[
  {{
    "title": "Automated Password Reset System",
    "scope": "Implement an automated self-service password reset system that integrates with Active Directory and includes multi-factor authentication...",
    "justification": "Password reset tickets make up 23% of all tickets and have an average resolution time of 1.2 hours. Automating this process could save approximately 450 hours of IT staff time per month...",
    "type": "AI + RPA hybrid solution. Use natural language processing for request detection and RPA for execution of reset procedures in Active Directory...",
    "implementation_plan": "1. Requirements gathering (2 weeks)\\n2. System design and architecture (3 weeks)\\n3. Development of NLP component (4 weeks)\\n4. Development of RPA component (4 weeks)\\n5. Integration and testing (3 weeks)\\n6. User acceptance testing (2 weeks)\\n7. Deployment (1 week)\\n\\nRequired resources: 1 Project Manager, 2 Developers, 1 QA Engineer, 1 Systems Administrator"
  }}
]

Focus on automation opportunities that:
1. Would have the highest impact based on ticket volume and resolution time
2. Address clear patterns in the data
3. Have realistic implementation paths
4. Include a mix of different automation approaches (AI, RPA, scripting, self-service)
5. Consider both technical feasibility and business value
        """
        
        return prompt
    
    def _generate_fallback_opportunities(self, stats):
        """
        Generate fallback automation opportunities when LLM fails.
        
        Args:
            stats (dict): Dictionary of ticket statistics
            
        Returns:
            list: List of automation opportunity dictionaries
        """
        opportunities = []
        
        # Opportunity 1: Password Reset Automation (common in most IT environments)
        if "top_issues" in stats and any(issue in stats["top_issues"] for issue in ["password", "reset", "login", "access"]):
            password_issue_count = sum(stats["top_issues"].get(issue, 0) for issue in ["password", "reset", "login", "access"])
            password_opportunity = {
                "title": "Automated Password Reset System",
                "scope": "Implement a self-service password reset portal that allows users to securely reset their passwords without IT intervention. The system should integrate with Active Directory and include multi-factor authentication for security.",
                "justification": f"Password and access-related issues account for a significant portion of tickets ({password_issue_count} tickets). Automating this process would reduce the workload on IT staff and provide immediate resolution for users.",
                "type": "Self-service portal with RPA integration. Use a web portal for user interaction and RPA to execute the reset procedures in Active Directory.",
                "implementation_plan": "1. Requirements gathering (2 weeks)\n2. System design (2 weeks)\n3. Development of self-service portal (3 weeks)\n4. Integration with Active Directory (2 weeks)\n5. Security testing (1 week)\n6. User acceptance testing (1 week)\n7. Deployment and training (1 week)\n\nRequired resources: 1 Web Developer, 1 Systems Administrator, 1 Security Specialist"
            }
            opportunities.append(password_opportunity)
        
        # Opportunity 2: Ticket Categorization 
        if "total_tickets" in stats:
            categorization_opportunity = {
                "title": "AI-Powered Ticket Categorization and Routing",
                "scope": "Implement an AI system that automatically categorizes incoming tickets based on their description and routes them to the appropriate assignment group. The system should learn from past ticket assignments to improve accuracy over time.",
                "justification": f"With {stats.get('total_tickets', 0)} tickets in the dataset, manual categorization and routing is time-consuming and prone to errors. Automation would reduce response time and ensure tickets reach the right team immediately.",
                "type": "AI/ML solution using Natural Language Processing. Implement a machine learning model trained on historical ticket data to predict the appropriate category and assignment group.",
                "implementation_plan": "1. Data preparation and cleaning (3 weeks)\n2. Model development and training (4 weeks)\n3. Integration with ticketing system (2 weeks)\n4. Testing and validation (2 weeks)\n"
            }