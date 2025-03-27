# Ticket Analysis System - Main Application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
import io
import re
from dotenv import load_dotenv
import groq
import time

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = groq.Client(api_key=groq_api_key)

# Set default model if not specified in .env
if not os.getenv("GROQ_MODEL"):
    os.environ["GROQ_MODEL"] = "llama3-8b-8192"

# Set page configuration
st.set_page_config(
    page_title="Agentic Ticket Analysis System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define application state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'predefined_questions' not in st.session_state:
    st.session_state.predefined_questions = None
if 'automation_opportunities' not in st.session_state:
    st.session_state.automation_opportunities = None
if 'charts' not in st.session_state:
    st.session_state.charts = {}

# Helper functions
def extract_data(uploaded_file, chunk_size=1000):
    """Extract data from uploaded file"""
    try:
        # Get file extension
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Read file based on extension
        if file_extension == 'csv':
            # For large CSV files
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                text_io = io.StringIO(uploaded_file.getvalue().decode('utf-8'))
                chunks = []
                # Read and process chunks
                for chunk in pd.read_csv(text_io, chunksize=chunk_size):
                    chunks.append(chunk)
                # Combine all chunks
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            # For Excel files
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        return df
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        return None

def process_data(df):
    """Process and transform ticket data"""
    try:
        # Make a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Handle missing values
        for col in processed_df.select_dtypes(include=['object']).columns:
            processed_df[col] = processed_df[col].fillna('Unknown')
        
        # Convert date columns to datetime
        date_columns = [col for col in processed_df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'opened', 'closed', 'resolved'])]
        for col in date_columns:
            try:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            except:
                pass
        
        # Feature engineering - only if opened and closed columns exist
        if 'opened' in processed_df.columns and 'closed' in processed_df.columns:
            if pd.api.types.is_datetime64_dtype(processed_df['opened']) and pd.api.types.is_datetime64_dtype(processed_df['closed']):
                # Calculate resolution time in hours
                processed_df['resolution_time_hours'] = (processed_df['closed'] - processed_df['opened']).dt.total_seconds() / 3600
                # Handle negative or NaN resolution times
                processed_df.loc[processed_df['resolution_time_hours'] < 0, 'resolution_time_hours'] = np.nan
        
        # Create day of week if opened column exists and is datetime
        if 'opened' in processed_df.columns and pd.api.types.is_datetime64_dtype(processed_df['opened']):
            processed_df['opened_month'] = processed_df['opened'].dt.month
            processed_df['opened_year'] = processed_df['opened'].dt.year
            processed_df['opened_day_of_week'] = processed_df['opened'].dt.dayofweek
            processed_df['opened_hour'] = processed_df['opened'].dt.hour
            
            # Create day type (weekend/weekday)
            processed_df['is_weekend'] = processed_df['opened_day_of_week'].isin([5, 6])  # 5 = Saturday, 6 = Sunday
            
            # Create business hours flag (assuming 9-5)
            processed_df['is_business_hours'] = ((processed_df['opened_hour'] >= 9) & 
                                                (processed_df['opened_hour'] < 17) & 
                                                ~processed_df['is_weekend'])
        
        # Extract keywords from descriptions
        if 'short description' in processed_df.columns:
            # Convert to string (in case it's not)
            processed_df['short description'] = processed_df['short description'].astype(str)
            
            # Extract common technical terms/issues
            common_issues = [
                'error', 'failed', 'failure', 'broken', 'bug', 'crash', 'issue',
                'password', 'reset', 'access', 'login', 'permission', 'account',
                'slow', 'performance', 'latency', 'timeout', 'hang',
                'install', 'update', 'upgrade', 'patch', 'deploy',
                'network', 'connection', 'wifi', 'internet', 'server',
                'print', 'printer', 'email', 'outlook', 'office'
            ]
            
            # Create issue type columns
            for issue in common_issues:
                col_name = f"contains_{issue}"
                processed_df[col_name] = processed_df['short description'].str.lower().str.contains(issue, regex=False)

        return processed_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return df  # Return original dataframe if processing fails

def extract_statistics(df):
    """Extract key statistics from processed data in a dictionary of primitives"""
    stats = {}
    
    # Basic counts
    stats["total_tickets"] = len(df)
    
    # Priority distribution if available
    if 'priority' in df.columns:
        priority_counts = df['priority'].value_counts()
        stats["priority_distribution"] = {str(k): int(v) for k, v in zip(priority_counts.index, priority_counts.values)}
    
    # State distribution if available
    if 'state' in df.columns:
        state_counts = df['state'].value_counts()
        stats["state_distribution"] = {str(k): int(v) for k, v in zip(state_counts.index, state_counts.values)}
    
    # Assignment group distribution if available
    if 'assignment group' in df.columns:
        # Get top 10 assignment groups
        top_groups = df['assignment group'].value_counts().head(10)
        stats["top_assignment_groups"] = {str(k): int(v) for k, v in zip(top_groups.index, top_groups.values)}
    
    # Time-based statistics if available
    if 'resolution_time_hours' in df.columns:
        stats["avg_resolution_time_hours"] = float(df['resolution_time_hours'].mean())
        stats["median_resolution_time_hours"] = float(df['resolution_time_hours'].median())
    
    # Business hours vs. non-business hours
    if 'is_business_hours' in df.columns:
        stats["business_hours_tickets"] = int(df['is_business_hours'].sum())
        stats["non_business_hours_tickets"] = int((~df['is_business_hours']).sum())
    
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
    
    # Custom checks for specific questions
    
    # Datafix checks
    if 'short description' in df.columns:
        datafix_keywords = ['datafix', 'data fix', 'db fix', 'database fix']
        datafix_count = 0
        for keyword in datafix_keywords:
            datafix_count += df['short description'].str.lower().str.contains(keyword, na=False).sum()
        if datafix_count > 0:
            stats["datafix_mentions"] = int(datafix_count)
    
    # Escalation checks
    if 'work notes' in df.columns:
        escalation_keywords = ['escalate', 'escalation', 'elevated', 'raised to']
        escalation_count = 0
        for keyword in escalation_keywords:
            escalation_count += df['work notes'].str.lower().str.contains(keyword, na=False).sum()
        if escalation_count > 0:
            stats["escalation_mentions"] = int(escalation_count)
    
    # Document failure checks
    if 'short description' in df.columns:
        doc_keywords = ['document failure', 'report failure', 'failed document', 'document error']
        doc_count = 0
        for keyword in doc_keywords:
            doc_count += df['short description'].str.lower().str.contains(keyword, na=False).sum()
        if doc_count > 0:
            stats["document_failure_count"] = int(doc_count)
    
    return stats

def generate_charts(df):
    """Generate charts based on processed data"""
    charts = {}
    
    # Priority distribution chart with improved layout
    if 'priority' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        priority_counts = df['priority'].value_counts()
        
        # Calculate percentages for each slice
        sizes = priority_counts.values
        labels = priority_counts.index
        percentages = [s/sum(sizes)*100 for s in sizes]
        
        # Create autopct function that only shows percentages >= 3%
        def autopct_if_large(pct):
            return f'{pct:.1f}%' if pct >= 3 else ''
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # Remove labels from pie slices
            autopct=autopct_if_large,
            startangle=90,
            colors=plt.cm.Blues(np.linspace(0.4, 0.7, len(sizes)))
        )
        
        # Set properties for percentage text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add title and legend
        ax.set_title('Ticket Distribution by Priority', fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        
        # Add a legend with percentages
        legend_labels = [f"{label} ({size}, {pct:.1f}%)" for label, size, pct in zip(labels, sizes, percentages)]
        ax.legend(wedges, legend_labels, title="Priority", loc="center left", bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        charts['priority_chart'] = fig
    
    # Subcategory distribution chart
    if 'subcategory' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get top subcategories (limit to top 10 for readability)
        subcategory_counts = df['subcategory'].value_counts().head(10)
        
        # Calculate percentages
        sizes = subcategory_counts.values
        labels = subcategory_counts.index
        percentages = [s/sum(sizes)*100 for s in sizes]
        
        # Create autopct function that only shows percentages >= 3%
        def autopct_if_large(pct):
            return f'{pct:.1f}%' if pct >= 3 else ''
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # Remove labels from pie slices
            autopct=autopct_if_large,
            startangle=90,
            colors=plt.cm.Greens(np.linspace(0.4, 0.7, len(sizes)))
        )
        
        # Set properties for percentage text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add title and legend
        ax.set_title('Top 10 Subcategories', fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        
        # Add a legend with percentages
        legend_labels = [f"{label} ({size}, {pct:.1f}%)" for label, size, pct in zip(labels, sizes, percentages)]
        ax.legend(wedges, legend_labels, title="Subcategory", loc="center left", bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        charts['subcategory_chart'] = fig
    
    # State distribution chart with vertical bars (tickets on y-axis)
    if 'state' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        state_counts = df['state'].value_counts()
        
        # Limit to top 10 states if there are too many
        if len(state_counts) > 10:
            state_counts = state_counts.head(10)
            
        # Plot vertical bar chart (tickets on y-axis)
        bars = state_counts.plot(kind='bar', ax=ax, color=plt.cm.Blues(0.6))
        
        # Add data labels above each bar
        for i, value in enumerate(state_counts):
            ax.text(i, value + (state_counts.max() * 0.02), f"{value}", ha='center')
        
        plt.title('Ticket Distribution by State', fontsize=14)
        plt.ylabel('Number of Tickets')
        plt.xlabel('State')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        charts['state_chart'] = fig
    
    # Assignment group chart with vertical bars (tickets on y-axis)
    if 'assignment group' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        group_counts = df['assignment group'].value_counts().head(10)
        
        # Plot vertical bar chart
        bars = group_counts.plot(kind='bar', ax=ax, color=plt.cm.Blues(0.6))
        
        # Add data labels above each bar
        for i, value in enumerate(group_counts):
            ax.text(i, value + (group_counts.max() * 0.02), f"{value}", ha='center')
        
        plt.title('Top 10 Assignment Groups', fontsize=14)
        plt.ylabel('Number of Tickets')
        plt.xlabel('Assignment Group')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        charts['assignment_group_chart'] = fig
    
    # Time series chart if opened column exists and is datetime
    if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by month and count tickets
        monthly_counts = df.resample('M', on='opened').size()
        
        # Plot the line chart
        monthly_counts.plot(marker='o', linestyle='-', ax=ax, color='royalblue')
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()
        
        plt.title('Ticket Volume by Month', fontsize=14)
        plt.xlabel('Month')
        plt.ylabel('Number of Tickets')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        charts['time_series_chart'] = fig
    
    return charts

def generate_insights(stats):
    """Generate insights from statistics using LLM"""
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
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

Format your insights as bullet points with clear, concise statements. Each insight should be data-driven and highlight something significant about the ticket data.
"""
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Your task is to provide clear, concise insights based on ticket data statistics. Focus on patterns, trends, and actionable observations. Be direct and highlight key findings without any unnecessary commentary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        insights = response.choices[0].message.content
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return "Error generating insights. Please check your connection and try again."

def generate_predefined_questions(stats):
    """Generate predefined questions based on statistics using LLM"""
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    prompt = f"""
Based on the following ticket data statistics:
{json.dumps(stats, indent=2)}

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
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
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
                
                # Validate that each question has all required fields
                validated_questions = []
                for q in questions:
                    if 'question' not in q or 'answer' not in q or 'automation_potential' not in q:
                        # Fix any missing fields
                        fixed_question = {
                            'question': q.get('question', 'Missing question'),
                            'answer': q.get('answer', 'No answer available'),
                            'automation_potential': q.get('automation_potential', 'No automation potential information')
                        }
                        validated_questions.append(fixed_question)
                    else:
                        validated_questions.append(q)
                
                return validated_questions[:10]  # Ensure we return at most 10 questions
                
            except json.JSONDecodeError:
                st.warning("Could not parse the generated questions. Using fallback questions.")
                return generate_fallback_questions(stats)
        else:
            st.warning("No valid JSON found in the response. Using fallback questions.")
            return generate_fallback_questions(stats)
        
    except Exception as e:
        st.error(f"Error generating predefined questions: {str(e)}")
        return generate_fallback_questions(stats)

def generate_fallback_questions(stats):
    """Generate fallback questions when LLM fails"""
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
    
    # Add additional general questions based on available statistics
    general_questions = []
    
    if "total_tickets" in stats:
        general_questions.append({
            "question": "What is the total volume of tickets in the dataset?",
            "answer": f"There are {stats['total_tickets']} tickets in the dataset.",
            "automation_potential": "No - This is a basic statistic that doesn't require automation."
        })
    
    if "priority_distribution" in stats:
        priority_str = ", ".join([f"{priority}: {count} tickets" for priority, count in stats["priority_distribution"].items()])
        general_questions.append({
            "question": "What is the distribution of ticket priorities?",
            "answer": f"The priority distribution is: {priority_str}",
            "automation_potential": "Yes - Automatic ticket prioritization based on text analysis could be implemented."
        })
    
    if "avg_resolution_time_hours" in stats:
        general_questions.append({
            "question": "What is the average resolution time for tickets?",
            "answer": f"The average resolution time is {stats['avg_resolution_time_hours']:.2f} hours.",
            "automation_potential": "Yes - Automated SLA monitoring and alerting could be implemented."
        })
    
    if "top_issues" in stats:
        top_issues_str = ", ".join([f"{issue}: {count} tickets" for issue, count in list(stats["top_issues"].items())[:3]])
        general_questions.append({
            "question": "What are the most common issues in the tickets?",
            "answer": f"The most common issues are: {top_issues_str}",
            "automation_potential": "Yes - Knowledge base articles and automated responses could be created for common issues."
        })
    
    if "top_assignment_groups" in stats:
        top_groups_str = ", ".join([f"{group}: {count} tickets" for group, count in list(stats["top_assignment_groups"].items())[:3]])
        general_questions.append({
            "question": "Which teams handle the most tickets?",
            "answer": f"The teams handling the most tickets are: {top_groups_str}",
            "automation_potential": "Yes - Workload balancing and automatic ticket routing could be implemented."
        })
    
    # Add more generic questions to reach 10 total questions
    questions.extend(general_questions[:10 - len(questions)])
    
    return questions[:10]  # Return at most 10 questions

def generate_automation_opportunities(stats, insights):
    """Generate automation opportunities based on statistics using LLM"""
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    prompt = f"""
Based on the following ticket data statistics and insights, identify the top 5 automation opportunities:

STATISTICS:
{json.dumps(stats, indent=2)}

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
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
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
                return generate_fallback_opportunities(stats)
        else:
            st.warning("No valid JSON found in the response. Using fallback opportunities.")
            return generate_fallback_opportunities(stats)
        
    except Exception as e:
        st.error(f"Error generating automation opportunities: {str(e)}")
        return generate_fallback_opportunities(stats)

def generate_fallback_opportunities(stats):
    """Generate fallback automation opportunities when LLM fails"""
    opportunities = []
    
    # Opportunity 1: Password Reset Automation (common in most IT environments)
    if "top_issues" in stats and any(issue in stats["top_issues"] for issue in ["password", "reset", "login", "access"]):
        password_issue_count = sum(stats["top_issues"].get(issue, 0) for issue in ["password", "reset", "login", "access"])
        password_opportunity = {
            "title": "Automated Password Reset System",
            "scope": "Implement a self-service password reset portal that allows users to securely reset their passwords without IT intervention. The system should integrate with Active Directory and include multi-factor authentication for security.",
            "justification": f"Password and access-related issues account for a significant portion of tickets. Automating this process would reduce the workload on IT staff and provide immediate resolution for users.",
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
            "implementation_plan": "1. Data preparation and cleaning (3 weeks)\n2. Model development and training (4 weeks)\n3. Integration with ticketing system (2 weeks)\n4. Testing and validation (2 weeks)\n5. Pilot deployment (2 weeks)\n6. Full deployment and monitoring (1 week)\n\nRequired resources: 1 Data Scientist, 1 ML Engineer, 1 Systems Integrator"
        }
        opportunities.append(categorization_opportunity)
    
    # Opportunity 3: Knowledge Base Article Suggestions
    if "top_issues" in stats:
        kb_opportunity = {
            "title": "Automated Knowledge Base Article Suggestions",
            "scope": "Develop a system that automatically suggests relevant knowledge base articles to agents based on ticket content, and recommends new KB articles to be created for common issues that lack documentation.",
            "justification": "Analysis of ticket data shows recurring issues that could be resolved faster with proper knowledge base articles. This would reduce resolution time and improve consistency in solutions.",
            "type": "AI-powered recommendation system using natural language processing to match ticket text with existing KB articles and identify knowledge gaps.",
            "implementation_plan": "1. KB article inventory and indexing (2 weeks)\n2. Development of text matching algorithm (3 weeks)\n3. Integration with ticketing system (2 weeks)\n4. KB gap analysis functionality (2 weeks)\n5. User interface development (2 weeks)\n6. Testing and refinement (2 weeks)\n7. Deployment and training (1 week)\n\nRequired resources: 1 Knowledge Management Specialist, 1 Developer, 1 UX Designer"
        }
        opportunities.append(kb_opportunity)
    
    # Opportunity 4: SLA Monitoring and Alerting
    if "avg_resolution_time_hours" in stats:
        sla_opportunity = {
            "title": "Proactive SLA Monitoring and Alerting System",
            "scope": "Implement an automated system that monitors ticket SLAs in real-time, sends proactive alerts for tickets at risk of breaching SLA, and provides escalation paths based on ticket priority and age.",
            "justification": f"The average ticket resolution time is {stats.get('avg_resolution_time_hours', 0):.2f} hours, but many tickets likely breach SLA targets. A proactive monitoring system would improve compliance and customer satisfaction.",
            "type": "RPA and Business Rules Engine to monitor tickets and trigger alerts based on configurable rules and thresholds.",
            "implementation_plan": "1. SLA policy definition and mapping (2 weeks)\n2. Alert rules configuration (1 week)\n3. Notification system development (2 weeks)\n4. Dashboard development (2 weeks)\n5. Integration with ticketing system (2 weeks)\n6. Testing across different ticket types (1 week)\n7. Deployment and staff training (1 week)\n\nRequired resources: 1 Business Analyst, 1 Developer, 1 QA Tester"
        }
        opportunities.append(sla_opportunity)
    
    # Opportunity 5: Chatbot for Common Issues
    chatbot_opportunity = {
        "title": "IT Support Chatbot for First-Level Resolution",
        "scope": "Deploy an AI chatbot that can handle common IT issues, guide users through basic troubleshooting steps, and create tickets automatically when it cannot resolve the issue.",
        "justification": "Many common IT issues follow standard troubleshooting patterns that can be automated. A chatbot can provide 24/7 support and immediate responses for these cases.",
        "type": "Conversational AI using natural language understanding and a decision tree-based resolution framework. Integration with existing ticketing system for seamless escalation.",
        "implementation_plan": "1. Define scope and common issues to address (2 weeks)\n2. Design conversation flows (3 weeks)\n3. Build NLU model (4 weeks)\n4. Develop troubleshooting logic (3 weeks)\n5. Integration with ticketing system (2 weeks)\n6. User testing and refinement (3 weeks)\n7. Pilot deployment (2 weeks)\n8. Full deployment and continuous improvement (ongoing)\n\nRequired resources: 1 Conversational AI Specialist, 1 IT Support SME, 1 Systems Integrator, 1 UX Designer"
    }
    opportunities.append(chatbot_opportunity)
    
    return opportunities[:5]  # Return at most 5 opportunities

def process_query(query, df, stats, insights):
    """Process a natural language query about ticket data"""
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    # Extract any additional statistics relevant to the query
    query_stats = extract_query_stats(df, query, stats)
    
    prompt = f"""
The user has asked the following question about their ticket data:
"{query}"

Based on the available ticket data statistics:
{json.dumps(query_stats, indent=2)}

The following insights have already been identified from this data:
{insights}

Please provide a clear, concise answer to the user's question based on the data provided. 

If the data doesn't contain information needed to fully answer the question, explain what's missing and provide the best possible answer with the available data.

If relevant, suggest further analysis or data collection that would help provide a more complete answer.

If an automation opportunity is relevant to the question, briefly mention it.
"""
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Provide clear, specific answers to questions about ticket data. Use the provided statistics to inform your responses, and be honest about limitations in the data where relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I'm having trouble processing your query. Please try again or rephrase your question."

def extract_query_stats(df, query, base_stats):
    """Extract additional statistics relevant to a specific query"""
    query_stats = base_stats.copy()  # Start with the base statistics
    query_lower = query.lower()
    
    # Look for specific aspects mentioned in the query
    
    # Country/region specific issues
    if any(word in query_lower for word in ["country", "region", "geographic", "location"]):
        # Look for any potential country/location fields
        for col in df.columns:
            if any(term in col.lower() for term in ["country", "region", "location", "geography"]):
                counts = df[col].value_counts().head(10)
                query_stats[f"{col}_distribution"] = {str(k): int(v) for k, v in zip(counts.index, counts.values)}
    
    # Datafix related queries
    if any(word in query_lower for word in ["datafix", "data fix", "db fix"]):
        if 'short description' in df.columns:
            datafix_keywords = ['datafix', 'data fix', 'db fix', 'database fix']
            for keyword in datafix_keywords:
                matches = df[df['short description'].str.lower().str.contains(keyword, na=False)]
                if len(matches) > 0:
                    query_stats[f"tickets_mentioning_{keyword}"] = len(matches)
                    
                    # If categories exist, show distribution of datafix issues by category
                    if 'subcategory' in df.columns:
                        category_counts = matches['subcategory'].value_counts().head(5)
                        query_stats[f"{keyword}_by_category"] = {str(k): int(v) for k, v in zip(category_counts.index, category_counts.values)}
    
    # Document failure queries
    if any(word in query_lower for word in ["document", "doc", "report", "failed", "failure"]):
        if 'short description' in df.columns:
            doc_keywords = ['document failure', 'report failure', 'failed document']
            doc_matches = []
            for keyword in doc_keywords:
                matches = df[df['short description'].str.lower().str.contains(keyword, na=False)]
                if len(matches) > 0:
                    doc_matches.append(matches)
                    
            if doc_matches:
                all_matches = pd.concat(doc_matches).drop_duplicates()
                query_stats["document_failure_count"] = len(all_matches)
                
                # Check if we can determine when these failures occurred
                if 'opened' in all_matches.columns and pd.api.types.is_datetime64_dtype(all_matches['opened']):
                    # Count by year
                    year_counts = all_matches.groupby(all_matches['opened'].dt.year).size()
                    query_stats["document_failures_by_year"] = {str(k): int(v) for k, v in zip(year_counts.index, year_counts.values)}
    
    return query_stats

# Application UI
# Application header
st.title("Qualitative Ticket Analysis System")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Ticket Data")
    st.markdown("""
    **Upload meaningful ticket data to generate valuable insights.**
    
    Supported formats:
    - CSV (.csv)
    - Excel (.xls, .xlsx)
    
    The system will automatically process your data, generate insights,
    and suggest automation opportunities.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        with st.spinner("Processing your data... Please wait."):
            # Extract data
            df = extract_data(uploaded_file)
            
            if df is not None:
                # Show sample of the data
                st.write("Data Preview:")
                st.dataframe(df.head(5))
                
                # Store raw data in session state
                st.session_state.data = df
                
                # Process data
                processed_df = process_data(df)
                st.session_state.processed_data = processed_df
                
                # Reset state before adding new data
                st.session_state.insights = None
                st.session_state.predefined_questions = None
                st.session_state.automation_opportunities = None
                st.session_state.charts = {}
                
                # Extract statistics for LLM
                stats = extract_statistics(processed_df)
                
                # Generate insights
                insights = generate_insights(stats)
                st.session_state.insights = insights
                
                # Generate predefined questions with explicit validation
                predefined_questions = generate_predefined_questions(stats)
                # Double-check each question has required fields
                validated_questions = []
                for q in predefined_questions:
                    if isinstance(q, dict):  # Ensure we're dealing with a dictionary
                        validated_q = {
                            'question': q.get('question', 'Question not available'),
                            'answer': q.get('answer', 'Answer not available'),
                            'automation_potential': q.get('automation_potential', 'Automation potential not available')
                        }
                        validated_questions.append(validated_q)
                
                st.session_state.predefined_questions = validated_questions if validated_questions else generate_fallback_questions(stats)
                
                # Generate automation opportunities
                automation_opportunities = generate_automation_opportunities(stats, insights)
                st.session_state.automation_opportunities = automation_opportunities
                
                # Generate charts
                charts = generate_charts(processed_df)
                st.session_state.charts = charts
                
                st.success("Data processed successfully!")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Insights", 
    "Predefined Questions", 
    "Automation Opportunities", 
    "💬 Chat with Your Data"
])

# Tab 1: Data Insights
with tab1:
    st.header("Data Insights")
    
    if st.session_state.data is not None and st.session_state.insights is not None:
        # Display general data statistics
        st.subheader("Dataset Overview")
        st.write(f"Total tickets: {len(st.session_state.data)}")
        
        # Display insights
        st.subheader("Key Insights")
        st.markdown(st.session_state.insights)
        
        # Display charts in an organized way
        st.subheader("Visualizations")
        
        # First row: Pie charts side by side
        if 'priority_chart' in st.session_state.charts or 'subcategory_chart' in st.session_state.charts:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'priority_chart' in st.session_state.charts:
                    st.pyplot(st.session_state.charts['priority_chart'])
            
            with col2:
                if 'subcategory_chart' in st.session_state.charts:
                    st.pyplot(st.session_state.charts['subcategory_chart'])
        
        # Second row: Bar charts side by side
        if 'state_chart' in st.session_state.charts or 'assignment_group_chart' in st.session_state.charts:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'state_chart' in st.session_state.charts:
                    st.pyplot(st.session_state.charts['state_chart'])
            
            with col2:
                if 'assignment_group_chart' in st.session_state.charts:
                    st.pyplot(st.session_state.charts['assignment_group_chart'])
        
        # Third row: Time series chart (full width)
        if 'time_series_chart' in st.session_state.charts:
            st.pyplot(st.session_state.charts['time_series_chart'])
    else:
        st.info("Please upload ticket data to view insights.")

# Tab 2: Predefined Questions
with tab2:
    st.header("Predefined Questions")
    
    if st.session_state.predefined_questions is not None:
        try:
            for i, qa_pair in enumerate(st.session_state.predefined_questions):
                # Ensure minimal display if any field is missing
                question = qa_pair['question'] if 'question' in qa_pair else "Question not available"
                
                with st.expander(f"Q{i+1}: {question}"):
                    answer = qa_pair['answer'] if 'answer' in qa_pair else "Answer not available"
                    st.markdown(f"**Answer:** {answer}")
                    
                    automation = qa_pair['automation_potential'] if 'automation_potential' in qa_pair else "Automation potential not available"
                    st.markdown(f"**Automation Potential:** {automation}")
        except Exception as e:
            st.error(f"Error displaying predefined questions: {str(e)}")
            st.info("There was an issue displaying the predefined questions. Please try uploading the data again.")
    else:
        st.info("Please upload ticket data to view predefined questions and answers.")

# Tab 3: Automation Opportunities
with tab3:
    st.header("Top 5 Automation Opportunities")
    
    if st.session_state.automation_opportunities is not None:
        for i, opportunity in enumerate(st.session_state.automation_opportunities):
            with st.expander(f"Opportunity {i+1}: {opportunity['title']}"):
                st.markdown(f"**Automation Scope:**\n{opportunity['scope']}")
                st.markdown(f"**Justification:**\n{opportunity['justification']}")
                st.markdown(f"**Type of Automation:**\n{opportunity['type']}")
                st.markdown(f"**Implementation Plan:**\n{opportunity['implementation_plan']}")
    else:
        st.info("Please upload ticket data to view automation opportunities.")

# Tab 4: Chat with Your Data
with tab4:
    st.header("Chat with Your Data")
    
    if st.session_state.data is not None:
        # User query input
        user_query = st.text_input("Ask a question about your ticket data:")
        
        if user_query:
            with st.spinner("Processing your query..."):
                stats = extract_statistics(st.session_state.processed_data)
                response = process_query(
                    user_query, 
                    st.session_state.processed_data,
                    stats,
                    st.session_state.insights
                )
                st.markdown(response)
    else:
        st.info("Please upload ticket data to chat with it.")

# Footer
st.markdown("---")
st.markdown("*Agentic Ticket Analysis System - Powered by AI*")