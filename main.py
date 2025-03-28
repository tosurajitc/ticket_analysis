# main.py - Main application entry point
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
import groq
import traceback

# Import modules from mainsrc
from mainsrc.data_module import DataModule
from mainsrc.analysis_module import AnalysisModule
from mainsrc.visualization_module import VisualizationModule
from mainsrc.insights_module import InsightsModule
from mainsrc.questions_module import QuestionsModule
from mainsrc.automation_module import AutomationModule
from mainsrc.chat_module import ChatModule
from mainsrc.utils import NumpyEncoder, safe_json_dumps

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

# Apply custom CSS
st.markdown("""
<style>
    .main-header {text-align: center; color: #1E3A8A;}
    .section-header {color: #1E3A8A; border-bottom: 1px solid #DDDDDD; padding-bottom: 0.5rem;}
    .info-card {background-color: #F0F7FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .metric-card {background-color: #F0FFF4; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #38A169;}
    .warning-card {background-color: #FFFAF0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #DD6B20;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap;}
    .upload-section {border: 2px dashed #CCCCCC; border-radius: 0.5rem; padding: 1rem; text-align: center;}
</style>
""", unsafe_allow_html=True)


## Root cause based Automation opportunities starts
def analyze_root_causes(df):
    """
    Perform root cause analysis on ticket data to identify automation opportunities.
    
    Args:
        df: DataFrame with ticket data
        
    Returns:
        dict: Root causes mapped to ticket clusters and automation potential
    """
    # Get chunk size from environment or use default
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    
    # First, identify the relevant columns
    desc_column = None
    for col in ['short description', 'description', 'short_description']:
        if col in df.columns:
            desc_column = col
            break
    
    notes_column = None
    for col in ['close notes', 'closed notes', 'resolution notes', 'work notes', 'close_notes']:
        if col in df.columns:
            notes_column = col
            break
    
    category_columns = []
    for col in df.columns:
        if any(term in col.lower() for term in ['category', 'subcategory', 'type', 'group']):
            category_columns.append(col)
    
    # Initialize root cause clusters
    root_causes = {}
    
    # Process data in chunks if it's large
    if len(df) > chunk_size:
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_causes = _identify_root_causes_in_chunk(
                chunk_df, 
                desc_column, 
                notes_column, 
                category_columns
            )
            
            # Merge chunk results
            for cause_key, cause_data in chunk_causes.items():
                if cause_key in root_causes:
                    # Update existing cause
                    root_causes[cause_key]['count'] += cause_data['count']
                    root_causes[cause_key]['tickets'].extend(cause_data['tickets'][:3])
                    # Update resolution patterns
                    for resolution in cause_data['resolutions']:
                        if resolution not in root_causes[cause_key]['resolutions']:
                            root_causes[cause_key]['resolutions'].append(resolution)
                else:
                    # Add new cause
                    root_causes[cause_key] = cause_data
    else:
        # For small datasets, process directly
        root_causes = _identify_root_causes_in_chunk(
            df, 
            desc_column, 
            notes_column, 
            category_columns
        )
    
    # Sort by count and keep top causes
    sorted_causes = {k: v for k, v in sorted(
        root_causes.items(), 
        key=lambda item: item[1]['count'], 
        reverse=True
    )[:10]}  # Top 10 causes
    
    # For each root cause, identify automation potential
    for cause_key, cause_data in sorted_causes.items():
        cause_data['automation_potential'] = _assess_automation_potential(
            cause_key,
            cause_data
        )
    
    return sorted_causes

def _identify_root_causes_in_chunk(df, desc_column, notes_column, category_columns):
    """
    Identify root causes in a chunk of data.
    
    Args:
        df: DataFrame chunk
        desc_column: Column with descriptions
        notes_column: Column with resolution notes
        category_columns: List of category-related columns
        
    Returns:
        dict: Root causes identified in this chunk
    """
    root_causes = {}
    
    # Group by categories if available
    if category_columns:
        for col in category_columns:
            if col in df.columns:
                # Group by this category
                for category, group in df.groupby(col):
                    if pd.isna(category) or str(category).strip() == '':
                        continue
                        
                    cause_key = f"{col}: {category}"
                    
                    # Initialize if new
                    if cause_key not in root_causes:
                        root_causes[cause_key] = {
                            'category': col,
                            'value': category,
                            'count': 0,
                            'tickets': [],
                            'resolutions': []
                        }
                    
                    # Update count
                    root_causes[cause_key]['count'] += len(group)
                    
                    # Add sample tickets
                    for _, ticket in group.head(3).iterrows():
                        ticket_id = ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                        
                        # Get description
                        ticket_desc = ''
                        if desc_column and desc_column in ticket:
                            ticket_desc = str(ticket[desc_column])
                            if len(ticket_desc) > 100:
                                ticket_desc = ticket_desc[:97] + "..."
                        
                        # Get resolution if available
                        resolution = ''
                        if notes_column and notes_column in ticket and not pd.isna(ticket[notes_column]):
                            resolution = str(ticket[notes_column])
                            if len(resolution) > 150:
                                resolution = resolution[:147] + "..."
                            
                            # Add to resolutions if not already there
                            if resolution and resolution not in root_causes[cause_key]['resolutions']:
                                root_causes[cause_key]['resolutions'].append(resolution)
                        
                        # Add ticket to samples
                        root_causes[cause_key]['tickets'].append({
                            'id': ticket_id,
                            'description': ticket_desc
                        })
    
    # Text-based clustering for descriptions
    if desc_column:
        # Get descriptions
        descriptions = df[desc_column].fillna('').astype(str)
        
        # Simple pattern matching for common issues
        patterns = [
            ('password reset', ['password reset', 'reset password', 'forgot password']),
            ('access request', ['access request', 'request access', 'need access']),
            ('system unavailable', ['system down', 'cannot access', 'not available']),
            ('login issue', ['cannot login', 'login failed', 'unable to log in']),
            ('error message', ['error message', 'getting error', 'shows error']),
            ('software installation', ['install software', 'need application', 'deploy application']),
            ('hardware issue', ['hardware problem', 'device not working', 'broken']),
            ('data issue', ['incorrect data', 'missing data', 'data problem']),
            ('performance issue', ['slow', 'performance', 'taking too long'])
        ]
        
        # Check for patterns
        for pattern_name, keywords in patterns:
            cause_key = f"Issue: {pattern_name}"
            
            # Initialize if new
            if cause_key not in root_causes:
                root_causes[cause_key] = {
                    'category': 'Issue Type',
                    'value': pattern_name,
                    'count': 0,
                    'tickets': [],
                    'resolutions': []
                }
            
            # Find matching tickets
            for keyword in keywords:
                matches = descriptions.str.lower().str.contains(keyword, regex=False)
                matching_tickets = df[matches]
                
                # Update count
                root_causes[cause_key]['count'] += len(matching_tickets)
                
                # Add sample tickets
                for _, ticket in matching_tickets.head(3).iterrows():
                    ticket_id = ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                    
                    # Check if we already have this ticket ID
                    ticket_ids = [t['id'] for t in root_causes[cause_key]['tickets']]
                    if ticket_id in ticket_ids:
                        continue
                    
                    # Get description
                    ticket_desc = str(ticket[desc_column])
                    if len(ticket_desc) > 100:
                        ticket_desc = ticket_desc[:97] + "..."
                    
                    # Get resolution if available
                    if notes_column and notes_column in ticket and not pd.isna(ticket[notes_column]):
                        resolution = str(ticket[notes_column])
                        if len(resolution) > 150:
                            resolution = resolution[:147] + "..."
                        
                        # Add to resolutions if not already there
                        if resolution and resolution not in root_causes[cause_key]['resolutions']:
                            root_causes[cause_key]['resolutions'].append(resolution)
                    
                    # Add ticket to samples
                    if len(root_causes[cause_key]['tickets']) < 5:  # Limit to 5 examples
                        root_causes[cause_key]['tickets'].append({
                            'id': ticket_id,
                            'description': ticket_desc
                        })
    
    return root_causes

def _assess_automation_potential(cause_key, cause_data):
    """
    Assess automation potential for a root cause.
    
    Args:
        cause_key: Key for the root cause
        cause_data: Data about the root cause
        
    Returns:
        dict: Automation potential details
    """
    # Check count - higher count means better automation potential
    count = cause_data['count']
    
    # Check resolution patterns - consistent resolutions are more automatable
    resolutions = cause_data['resolutions']
    has_consistent_resolution = len(resolutions) > 0 and len(resolutions) < count/2
    
    # Initial assessment
    if count >= 50 and has_consistent_resolution:
        automation_level = "High"
    elif count >= 20 or has_consistent_resolution:
        automation_level = "Medium"
    else:
        automation_level = "Low"
    
    # Generate automation type based on the cause
    automation_type = "Unknown"
    implementation_steps = []
    
    # Recognize specific patterns in cause_key
    if 'password' in cause_key.lower():
        automation_type = "Self-service password reset portal"
        implementation_steps = [
            "Implement identity verification",
            "Create self-service portal",
            "Integrate with Active Directory",
            "Add security controls and auditing",
            "Create user documentation"
        ]
    elif 'access' in cause_key.lower():
        automation_type = "Access request and provisioning workflow"
        implementation_steps = [
            "Map access types and approvers",
            "Create request workflow",
            "Build approval routing",
            "Automate provisioning where possible",
            "Add compliance reporting"
        ]
    elif 'install' in cause_key.lower() or 'software' in cause_key.lower():
        automation_type = "Software deployment automation"
        implementation_steps = [
            "Create software package repository",
            "Build deployment automation",
            "Create self-service portal",
            "Implement approval workflows",
            "Add reporting and tracking"
        ]
    elif 'system' in cause_key.lower() and ('down' in cause_key.lower() or 'unavailable' in cause_key.lower()):
        automation_type = "System monitoring and auto-recovery"
        implementation_steps = [
            "Implement monitoring for key systems",
            "Create automated health checks",
            "Build auto-recovery scripts",
            "Set up alerting and escalation",
            "Create runbooks for manual intervention"
        ]
    elif 'data' in cause_key.lower():
        automation_type = "Data validation and correction workflows"
        implementation_steps = [
            "Create data validation rules",
            "Build validation automation",
            "Implement correction workflows",
            "Add monitoring and reporting",
            "Create exception handling process"
        ]
    elif 'login' in cause_key.lower():
        automation_type = "Login troubleshooting tool"
        implementation_steps = [
            "Create diagnostic workflow",
            "Build self-service portal",
            "Implement common fixes",
            "Add reporting and tracking",
            "Create escalation process for complex issues"
        ]
    else:
        # Generic workflow for other types
        automation_type = "Process automation workflow"
        implementation_steps = [
            "Document current process",
            "Identify automation triggers",
            "Create workflow automation",
            "Implement validation and exception handling",
            "Add reporting and monitoring"
        ]
    
    # Calculate impact score
    if automation_level == "High":
        impact_score = min(80 + (count // 20), 100)  # Higher count, higher score, max 100
    elif automation_level == "Medium":
        impact_score = min(50 + (count // 30), 79)   # Medium range 50-79
    else:
        impact_score = min(30 + (count // 50), 49)   # Low range 30-49
    
    # Create automation potential object
    automation_potential = {
        "level": automation_level,
        "score": impact_score,
        "type": automation_type,
        "implementation_steps": implementation_steps
    }
    
    return automation_potential

def generate_automation_opportunity_from_root_cause(cause_key, cause_data):
    """
    Generate a detailed automation opportunity from a root cause.
    
    Args:
        cause_key: Key for the root cause
        cause_data: Data about the root cause
        
    Returns:
        dict: Automation opportunity
    """
    automation = cause_data['automation_potential']
    
    # Create ticket examples text
    example_texts = []
    for ticket in cause_data['tickets']:
        example_texts.append(f"{ticket['id']}: {ticket['description']}")
    
    # Create resolution text
    resolution_text = ""
    if cause_data['resolutions']:
        resolution_text = "Based on the resolution patterns:\n\n"
        for i, resolution in enumerate(cause_data['resolutions'][:3]):  # Top 3 resolutions
            resolution_text += f"- Resolution example {i+1}: \"{resolution}\"\n"
    
    # Generate opportunity
    opportunity = {
        "title": f"Automate {cause_data['value']} Resolution",
        "scope": f"Create an automated solution for {cause_data['value']} tickets",
        "justification": f"Analysis identified {cause_data['count']} tickets related to '{cause_data['value']}'. {resolution_text}",
        "type": automation['type'],
        "implementation_plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(automation['implementation_steps'])]),
        "impact_score": automation['score'],
        "examples": example_texts[:3]  # Limit to 3 examples
    }
    
    return opportunity

def get_data_driven_automation_opportunities(processed_data, stats):
    """
    Generate completely data-driven automation opportunities based on root cause analysis.
    
    Args:
        processed_data: DataFrame with processed ticket data
        stats: Statistics dictionary
        
    Returns:
        list: Data-driven automation opportunities
    """
    try:
        # Perform root cause analysis
        root_causes = analyze_root_causes(processed_data)
        
        # Generate opportunities from root causes
        opportunities = []
        
        for cause_key, cause_data in root_causes.items():
            # Only create opportunities for causes with automation potential
            if cause_data['automation_potential']['level'] != "Low":
                opportunity = generate_automation_opportunity_from_root_cause(
                    cause_key, cause_data
                )
                opportunities.append(opportunity)
        
        # Sort by impact score
        opportunities.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Enhance top opportunities with LLM if possible
        try:
            top_opps = opportunities[:5]
            if top_opps:
                for i, opp in enumerate(top_opps):
                    opp = enhance_single_opportunity(opp, stats)
                    opportunities[i] = opp
        except Exception as e:
            st.warning(f"Could not enhance opportunities with LLM: {str(e)}")
        
        return opportunities[:5]  # Return top 5
        
    except Exception as e:
        st.error(f"Error in root cause analysis: {str(e)}")
        return []

def enhance_single_opportunity(opportunity, stats):
    """
    Enhance a single opportunity with LLM.
    
    Args:
        opportunity: Opportunity dictionary
        stats: Statistics dictionary
        
    Returns:
        dict: Enhanced opportunity
    """
    try:
        # Create a prompt for just this opportunity
        prompt = f"""
You are an expert IT automation consultant. Enhance this automation opportunity with detailed, data-driven improvements.

Original opportunity:
{json.dumps(opportunity, indent=2)}

Total tickets in dataset: {stats.get('total_tickets', 'unknown')}
Average resolution time: {stats.get('avg_resolution_time_hours', 'unknown')} hours

Enhance this opportunity by:
1. Adding specific technical details to the implementation plan
2. Adding data-driven justification with potential time and cost savings
3. Adding an ROI estimate section with timeframe and projected savings

Keep the original structure but enhance the content. Do NOT invent new data that doesn't exist in the original ticket information.

FORMAT YOUR RESPONSE AS JSON with the same structure as the input but with enhanced fields and a new 'roi_estimate' field.
"""

        # Generate response with controlled token count
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            messages=[
                {"role": "system", "content": "You are an expert IT automation consultant who provides detailed, technical recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000  # Limit output tokens
        )
        
        # Extract the generated response
        response_text = response.choices[0].message.content
        
        # Extract JSON
        import re
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}', response_text)
        
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            try:
                enhanced_opp = json.loads(json_str)
                return enhanced_opp
            except json.JSONDecodeError:
                return opportunity
        else:
            return opportunity
            
    except Exception as e:
        st.warning(f"Could not enhance opportunity: {str(e)}")
        return opportunity

## Root cause based automation opportunity ends




# Modify these functions to match what the modules expect

# Add this function to your helper functions
def analyze_custom_scenario(scenario_text, processed_data, stats):
    """
    Analyze a custom automation scenario described by the user.
    
    Args:
        scenario_text: User's description of the scenario
        processed_data: DataFrame with processed ticket data
        stats: Statistics dictionary
        
    Returns:
        dict: Analysis results
    """
    try:
        # Take a small sample of the DataFrame
        sample_size = min(20, len(processed_data))
        data_sample = processed_data.sample(sample_size, random_state=42)
        
        # Convert stats to JSON-safe format
        json_stats = prepare_stats_for_json(stats)
        
        # Create prompt for LLM
        prompt = f"""
You are an expert IT automation consultant analyzing ticket data for automation opportunities.

The user has described this custom automation scenario they're interested in:
"{scenario_text}"

Based on the ticket data sample:
{json_stats}

Provide a detailed analysis with the following information:
1. Feasibility: How feasible is this automation based on the available data? (High, Medium, Low)
2. Potential Impact: What impact would this automation have on efficiency, cost, and user experience?
3. Required Technology: What technologies would be needed to implement this automation?
4. Implementation Approach: A step-by-step approach to implementing this automation.
5. Data Requirements: What additional data might be needed to fully implement this solution?
6. Challenges: What potential challenges might arise during implementation?
7. ROI Timeline: When could the organization expect to see returns on this investment?

FORMAT YOUR RESPONSE AS JSON:
{{
  "feasibility": "High/Medium/Low",
  "impact": "Description of impact...",
  "required_technology": ["Tech 1", "Tech 2"],
  "implementation_approach": "Step-by-step implementation...",
  "data_requirements": "Additional data needed...",
  "challenges": ["Challenge 1", "Challenge 2"],
  "roi_timeline": "Timeline description..."
}}
"""

        # Generate response
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            messages=[
                {"role": "system", "content": "You are an expert IT automation consultant analyzing ticket data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        # Extract the generated response
        response_text = response.choices[0].message.content
        
        # Extract JSON if it's embedded in markdown or additional text
        import re
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}', response_text)
        
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            try:
                analysis = json.loads(json_str)
                return analysis
            except json.JSONDecodeError:
                # Fallback to returning the whole text if JSON parsing fails
                return {
                    "feasibility": "Unknown",
                    "impact": "Could not analyze impact",
                    "required_technology": [],
                    "implementation_approach": response_text,
                    "data_requirements": "Unknown",
                    "challenges": ["JSON parsing failed"],
                    "roi_timeline": "Unknown"
                }
        else:
            # Return the whole text if no JSON format is found
            return {
                "feasibility": "Unknown",
                "impact": "Analysis generated in non-standard format",
                "required_technology": [],
                "implementation_approach": response_text,
                "data_requirements": "Unknown",
                "challenges": ["Non-standard response format"],
                "roi_timeline": "Unknown"
            }
            
    except Exception as e:
        st.error(f"Error analyzing custom scenario: {str(e)}")
        return {
            "feasibility": "Error",
            "impact": "Could not analyze due to error",
            "required_technology": [],
            "implementation_approach": f"Error occurred: {str(e)}",
            "data_requirements": "Unknown",
            "challenges": ["Error during analysis"],
            "roi_timeline": "Unknown"
        }


# For insights module
def get_insights(processed_data, stats):
    """
    Generate insights from the processed data and stats.
    Adapts the data format to what the insights module expects.
    
    Args:
        processed_data: DataFrame with processed ticket data
        stats: Statistics dictionary 
        
    Returns:
        dict: Generated insights
    """
    try:
        # Take a small sample of the DataFrame
        sample_size = min(30, len(processed_data))
        data_sample = processed_data.sample(sample_size, random_state=42)
        
        # Convert stats to JSON-safe format
        json_stats = prepare_stats_for_json(stats)
        
        # Call the insights module with the DataFrame directly
        insights = insights_module.generate_insights(data_sample, json_stats)
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return {
            "key_findings": [
                {"title": "Error generating insights", "description": str(e)}
            ]
        }


def get_automation_opportunities(processed_data, stats):
    """
    Generate more specific automation opportunities from data analysis.
    Uses chunking for processing large datasets.
    
    Args:
        processed_data: DataFrame with processed ticket data
        stats: Statistics dictionary
        
    Returns:
        list: Generated automation opportunities with impact scores
    """
    try:
        # Analyze specific columns for automation patterns
        automation_patterns = analyze_ticket_content(processed_data)
        
        # Generate opportunities based on patterns
        opportunities = []
        
        # Process each identified pattern
        for pattern, details in automation_patterns.items():
            if details['count'] > 0:
                # Calculate an impact score based on volume and time saved
                impact_score = calculate_impact_score(
                    count=details['count'],
                    total_tickets=len(processed_data),
                    avg_time=stats.get('avg_resolution_time_hours', 1.0)
                )
                
                # Create an opportunity with the impact score
                opp = {
                    "title": details['title'],
                    "scope": details['scope'],
                    "justification": f"{details['justification']} This affects {details['count']} tickets ({(details['count']/len(processed_data)*100):.1f}% of total).",
                    "type": details['type'],
                    "implementation_plan": details['implementation'],
                    "impact_score": impact_score,
                    "examples": details.get('examples', [])[:3]  # Limit to 3 examples
                }
                
                opportunities.append(opp)
        
        # Sort by impact score (highest first)
        opportunities.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Use LLM to enhance top opportunities if there are any
        if opportunities:
            # Limit to top 5 opportunities to reduce token count
            top_opps = opportunities[:5]
            enhanced_opportunities = enhance_opportunities_with_llm(top_opps, stats)
            return enhanced_opportunities
        else:
            # Fallback to simpler opportunities if no patterns found
            return generate_fallback_opportunities(stats)
            
    except Exception as e:
        st.error(f"Error in automation analysis: {str(e)}")
        return generate_fallback_opportunities(stats)



def generate_fallback_opportunities(stats):
    """
    Generate fallback opportunities when pattern analysis fails.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        list: List of basic opportunities
    """
    # Create simple opportunities based on common patterns
    fallback_opportunities = [
        {
            "title": "Self-Service Password Reset",
            "scope": "Implement a self-service password reset solution",
            "justification": "Password reset requests are common in most organizations and follow a standard process that can be automated.",
            "type": "Self-service portal with identity verification",
            "implementation_plan": "1. Evaluate and select a self-service solution\n2. Configure identity verification\n3. Integrate with existing systems\n4. Test and validate\n5. Deploy and train users",
            "impact_score": 75,
            "examples": []
        },
        {
            "title": "IT Service Chatbot",
            "scope": "Deploy an AI chatbot for common IT requests and issues",
            "justification": "Many common IT requests follow standard patterns that can be handled by a chatbot, freeing up analysts for more complex tasks.",
            "type": "AI-powered chatbot with knowledge base integration",
            "implementation_plan": "1. Select a chatbot platform\n2. Build knowledge articles\n3. Configure conversation flows\n4. Train the AI model\n5. Deploy and improve iteratively",
            "impact_score": 65,
            "examples": []
        },
        {
            "title": "Knowledge Base Enhancement",
            "scope": "Create a searchable knowledge base from ticket resolutions",
            "justification": "Many tickets are resolved with similar solutions that could be documented and reused.",
            "type": "Knowledge management system with automated suggestions",
            "implementation_plan": "1. Select a knowledge management platform\n2. Extract solutions from past tickets\n3. Organize and categorize content\n4. Implement smart search\n5. Create process for ongoing maintenance",
            "impact_score": 55,
            "examples": []
        }
    ]
    
    return fallback_opportunities


def analyze_ticket_content(df):
    """
    Analyze ticket content to find automation patterns.
    Uses chunking for large datasets.
    
    Args:
        df: DataFrame with ticket data
        
    Returns:
        dict: Mapping of patterns to details
    """
    # Get chunk size from environment or use default
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    
    patterns = {
        "password_reset": {
            "title": "Password Reset Automation",
            "scope": "Implement a self-service password reset system with secure verification",
            "justification": "Password reset tickets are frequent and follow a standard resolution process.",
            "type": "Self-service portal with automated backend integration",
            "implementation": "1. Implement identity verification\n2. Create self-service portal\n3. Integrate with Active Directory\n4. Add audit logging\n5. Create user documentation",
            "keywords": ["password reset", "reset password", "forgot password", "pw reset", "changed password"],
            "count": 0,
            "examples": []
        },
        # More patterns as before...
    }
    
    # Check for description column
    desc_column = None
    for col in ['short description', 'description', 'short_description']:
        if col in df.columns:
            desc_column = col
            break
    
    # Check for resolution notes column
    notes_column = None
    for col in ['close notes', 'closed notes', 'resolution notes', 'work notes', 'close_notes']:
        if col in df.columns:
            notes_column = col
            break
    
    # Process data in chunks if it's large
    if len(df) > chunk_size:
        # Create empty result pattern to collect results from each chunk
        result_patterns = {k: v.copy() for k, v in patterns.items()}
        
        # Process data in chunks
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            
            # Analyze this chunk
            chunk_patterns = _analyze_chunk(chunk_df, patterns.copy(), desc_column, notes_column)
            
            # Merge results
            for pattern_key, pattern_data in chunk_patterns.items():
                result_patterns[pattern_key]['count'] += pattern_data['count']
                # Limit to top examples only
                result_patterns[pattern_key]['examples'].extend(pattern_data['examples'])
                if len(result_patterns[pattern_key]['examples']) > 5:
                    result_patterns[pattern_key]['examples'] = result_patterns[pattern_key]['examples'][:5]
        
        return result_patterns
    else:
        # For small datasets, process directly
        return _analyze_chunk(df, patterns, desc_column, notes_column)
    
def _analyze_chunk(df, patterns, desc_column, notes_column):
    """
    Analyze a chunk of ticket data.
    
    Args:
        df: DataFrame chunk
        patterns: Pattern dictionary to fill
        desc_column: Column name for ticket description
        notes_column: Column name for resolution notes
        
    Returns:
        dict: Updated patterns dictionary
    """
    # If we have description column, analyze for patterns
    if desc_column:
        # Convert to lowercase for better matching
        desc_series = df[desc_column].fillna('').astype(str).str.lower()
        
        # For each pattern, count occurrences and get example tickets
        for pattern_key, pattern_data in patterns.items():
            keywords = pattern_data['keywords']
            
            # Count tickets matching any keyword
            for keyword in keywords:
                matches = desc_series.str.contains(keyword, regex=False)
                pattern_data['count'] += matches.sum()
                
                # Get examples of matching tickets
                if matches.sum() > 0:
                    example_tickets = df[matches].head(2)
                    for _, ticket in example_tickets.iterrows():
                        ticket_id = ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                        ticket_desc = ticket.get(desc_column, '')
                        # Truncate long descriptions
                        if len(ticket_desc) > 100:
                            ticket_desc = ticket_desc[:97] + "..."
                        pattern_data['examples'].append(f"{ticket_id}: {ticket_desc}")
    
    # If we have resolution notes, analyze for additional patterns
    if notes_column:
        notes_series = df[notes_column].fillna('').astype(str).str.lower()
        
        # Look for resolution patterns
        for pattern_key, pattern_data in patterns.items():
            # If this pattern already has many examples, skip additional analysis
            if len(pattern_data['examples']) >= 3:
                continue
                
            # Look for additional examples in resolution notes
            for keyword in pattern_data['keywords']:
                matches = notes_series.str.contains(keyword, regex=False)
                
                # Get examples if we found matches
                if matches.sum() > 0:
                    example_tickets = df[matches].head(2)
                    for _, ticket in example_tickets.iterrows():
                        ticket_id = ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                        ticket_notes = ticket.get(notes_column, '')
                        if ticket_notes:
                            # Trim long notes
                            if len(ticket_notes) > 100:
                                ticket_notes = ticket_notes[:97] + "..."
                            pattern_data['examples'].append(f"{ticket_id}: {ticket_notes}")
    
    return patterns


def calculate_impact_score(count, total_tickets, avg_time=1.0):
    """
    Calculate an impact score for an automation opportunity.
    
    Args:
        count: Number of tickets affected
        total_tickets: Total number of tickets
        avg_time: Average time spent on these tickets (hours)
        
    Returns:
        float: Impact score (0-100)
    """
    # Calculate percentage of total tickets
    percentage = (count / total_tickets) * 100
    
    # Calculate time impact (higher time = higher impact)
    time_factor = min(avg_time / 2.0, 1.0)  # Cap at 1.0
    
    # Calculate base score (0-100)
    # More weight on volume (70%) than time (30%)
    base_score = (percentage * 0.7) + (time_factor * 30)
    
    # Apply diminishing returns for very low volume
    if percentage < 1:
        base_score = base_score * (percentage / 1.0)
    
    # Cap at 100
    return min(base_score, 100)



def enhance_opportunities_with_llm(opportunities, stats):
    """
    Enhance opportunities with LLM, using chunking to avoid token limits.
    
    Args:
        opportunities: List of opportunity dictionaries
        stats: Statistics dictionary
        
    Returns:
        list: Enhanced opportunities
    """
    try:
        # Get chunk size from environment or use default
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        
        # Only keep essential stats to reduce token count
        essential_stats = {
            "total_tickets": stats.get("total_tickets", 0),
            "avg_resolution_time_hours": stats.get("avg_resolution_time_hours", 0)
        }
        
        # Process opportunities in smaller chunks to avoid token limit
        enhanced_opportunities = []
        
        # Process 2 opportunities at a time to stay well under token limits
        for i in range(0, len(opportunities), 2):
            chunk = opportunities[i:i+2]
            
            # Create a prompt for the LLM with just this chunk
            prompt = f"""
You are an expert IT automation consultant. Enhance the following automation opportunities based on the statistics provided.

Ticket Statistics Overview:
- Total Tickets: {essential_stats.get('total_tickets', 0)}
- Avg Resolution Time: {essential_stats.get('avg_resolution_time_hours', 0)} hours

For each opportunity, keep the existing structure but enhance:
1. The justification with more data-driven reasoning
2. The implementation plan with more specific technical steps
3. Add concrete ROI estimates

Opportunities to enhance (enhance ONLY these specific opportunities):
{json.dumps(chunk, indent=2)}

FORMAT YOUR RESPONSE AS JSON with the same structure as the input, but with enhanced fields.
"""

            try:
                # Generate response with controlled token count
                response = groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                    messages=[
                        {"role": "system", "content": "You are an expert IT automation consultant specializing in ITSM."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500  # Limit output tokens
                )
                
                # Extract the generated response
                response_text = response.choices[0].message.content
                
                # Extract JSON
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}|\[[\s\S]*\]', response_text)
                
                if json_match:
                    json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                    try:
                        chunk_enhanced = json.loads(json_str)
                        # Handle if response is a dict instead of a list
                        if isinstance(chunk_enhanced, dict):
                            chunk_enhanced = [chunk_enhanced]
                        enhanced_opportunities.extend(chunk_enhanced)
                    except json.JSONDecodeError:
                        # If parsing fails, keep original chunk
                        enhanced_opportunities.extend(chunk)
                else:
                    # Keep original if no JSON format is found
                    enhanced_opportunities.extend(chunk)
                    
            except Exception as e:
                st.warning(f"Error enhancing chunk {i//2+1}: {str(e)}. Using original data for this chunk.")
                # Add original opportunities from this chunk
                enhanced_opportunities.extend(chunk)
                
            # Add a slight delay between API calls
            import time
            time.sleep(0.5)
        
        return enhanced_opportunities
            
    except Exception as e:
        st.error(f"Error in enhancement process: {str(e)}")
        # Return original opportunities
        return opportunities



# Helper functions for JSON serialization
def prepare_dataframe_for_json(df, max_rows=100):
    """
    Prepare a DataFrame for JSON serialization by:
    1. Converting to a smaller sample
    2. Converting timestamps and other problematic types to strings
    3. Handling NaN values
    
    Args:
        df: DataFrame to prepare or dictionary
        max_rows: Maximum number of rows to include
        
    Returns:
        dict: JSON-safe dictionary
    """
    # If already a dictionary, return it (assuming it's already prepared)
    if isinstance(df, dict):
        return df
    
    # If None or empty, return empty dict
    if df is None or len(df) == 0:
        return {}
    
    # Take a sample if dataframe is large
    if len(df) > max_rows:
        sample_df = df.sample(max_rows, random_state=42)
    else:
        sample_df = df.copy()
    
    # Convert to dictionary with preprocessing
    result = {}
    
    for col in sample_df.columns:
        column_data = sample_df[col].tolist()
        
        # Convert each value to a serializable format
        for i, val in enumerate(column_data):
            # Handle timestamps
            if pd.api.types.is_datetime64_dtype(sample_df[col].dtype):
                column_data[i] = str(val) if pd.notna(val) else None
            # Handle other objects that might not be serializable
            elif isinstance(val, (pd.Timestamp, pd.Period)):
                column_data[i] = str(val)
            # Handle NaN/None
            elif pd.isna(val):
                column_data[i] = None
            # Handle numpy types
            elif isinstance(val, (np.integer, np.int64)):
                column_data[i] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                column_data[i] = float(val)
        
        result[col] = column_data
    
    return result

def prepare_stats_for_json(stats):
    """
    Prepare statistics dictionary for JSON serialization.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        dict: JSON-safe dictionary
    """
    if not stats:
        return {}
    
    # Create a new dict to avoid modifying the original
    safe_stats = {}
    
    for key, value in stats.items():
        # Handle different types of values
        if isinstance(value, dict):
            safe_stats[key] = prepare_stats_for_json(value)
        elif isinstance(value, (list, tuple)):
            safe_stats[key] = [prepare_value_for_json(item) for item in value]
        else:
            safe_stats[key] = prepare_value_for_json(value)
    
    return safe_stats

def prepare_value_for_json(value):
    """
    Convert a single value to a JSON-safe format.
    
    Args:
        value: Value to convert
        
    Returns:
        JSON-safe value
    """
    if pd.isna(value):
        return None
    elif isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    elif isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: prepare_value_for_json(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [prepare_value_for_json(item) for item in value]
    return value

def handle_error(e, message="An error occurred"):
    """
    Handle exceptions gracefully.
    
    Args:
        e: Exception object
        message: Error message to display
    """
    st.error(f"{message}: {str(e)}")
    st.code(traceback.format_exc())

# Initialize modules
data_module = DataModule()
analysis_module = AnalysisModule()
visualization_module = VisualizationModule()
insights_module = InsightsModule(groq_client)
questions_module = QuestionsModule(groq_client)
automation_module = AutomationModule(groq_client)
chat_module = ChatModule(groq_client)

# Define application state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'charts' not in st.session_state:
    st.session_state.charts = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'predefined_questions' not in st.session_state:
    st.session_state.predefined_questions = None
if 'automation_opportunities' not in st.session_state:
    st.session_state.automation_opportunities = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'tab' not in st.session_state:
    st.session_state.tab = 0
if 'errors' not in st.session_state:
    st.session_state.errors = []

# Main application header
st.markdown("<h1 class='main-header'>Qualitative Ticket Analysis System</h1>", unsafe_allow_html=True)
st.markdown("Analyze and extract insights from IT support ticket data using LLM-powered intelligence.")

# Show errors if any
if st.session_state.errors:
    with st.expander("View Error Log", expanded=False):
        for i, error in enumerate(st.session_state.errors):
            st.error(f"Error {i+1}: {error}")
        if st.button("Clear Error Log"):
            st.session_state.errors = []
            st.experimental_rerun()


# Sidebar for app navigation and controls
with st.sidebar:
    st.title("Navigation")
    
    # Upload section in sidebar
    st.header("Data Upload")
    
    with st.expander("Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing data..."):
                    # Extract data
                    raw_data = data_module.extract_data(uploaded_file)
                    
                    if raw_data is not None and data_module.validate_data(raw_data):
                        # Store raw data
                        st.session_state.data = raw_data
                        
                        # Process data
                        st.session_state.processed_data = data_module.process_data(raw_data)
                        
                        # Generate statistics
                        st.session_state.stats = analysis_module.extract_statistics(st.session_state.processed_data)
                        
                        # Generate charts
                        st.session_state.charts = visualization_module.generate_charts(st.session_state.processed_data)
                        
                        st.success(f"✅ Loaded {len(raw_data)} records!")
            except Exception as e:
                handle_error(e, "Error processing uploaded file")
                st.session_state.errors.append(f"File upload error: {str(e)}")
    
    # Sample data button
    if st.button("Load Sample Data"):
        with st.spinner("Loading sample data..."):
            # TODO: Add sample data logic or placeholder for now
            st.info("Sample data functionality will be implemented in a future version.")
    
    # Model selection
    st.header("LLM Model")
    selected_model = st.selectbox(
        "Select model:",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    os.environ["GROQ_MODEL"] = selected_model
    
    # Data stats if data is loaded
    if st.session_state.data is not None:
        st.header("Data Summary")
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown(f"**Data Loaded:** {len(st.session_state.data)} records")
        
        if st.session_state.stats:
            if 'priority_distribution' in st.session_state.stats:
                priorities = st.session_state.stats['priority_distribution']
                if priorities:
                    top_priority = max(priorities.items(), key=lambda x: x[1])[0]
                    st.markdown(f"**Top Priority:** {top_priority}")
            
            if 'avg_resolution_time_hours' in st.session_state.stats:
                st.markdown(f"**Avg Resolution:** {st.session_state.stats['avg_resolution_time_hours']:.1f} hours")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Clear Data"):
            # Reset session state
            st.session_state.data = None
            st.session_state.processed_data = None
            st.session_state.stats = None
            st.session_state.charts = None
            st.session_state.insights = None
            st.session_state.predefined_questions = None
            st.session_state.automation_opportunities = None
            st.session_state.chat_history = []
            st.session_state.tab = 0
            st.experimental_rerun()
    
    # About section
    st.markdown("---")
    st.markdown("""
    ### About
    This tool analyzes IT support tickets to:
    - Extract key statistics
    - Visualize ticket patterns
    - Generate LLM-powered insights
    - Identify automation opportunities
    - Provide natural language chat
    """)

# Main tabs
tabs = st.tabs([
    "Upload", 
    "Data Analysis", 
    "Data Visualization", 
    "Ticket Insights",
    "Qualitative Questions",
    "Automation Suggestion",
    "💬 Chat"
])

# Tab 1: Upload Data
# Tab 1: Data Preview 
with tabs[0]:
    st.markdown("<h2 class='section-header'>Data Preview</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload a file using the upload control in the sidebar.")
        
        # Show placeholder card with instructions
        st.markdown("""
        <div style="padding: 20px; border-radius: 5px; border: 1px dashed #ccc; text-align: center; margin-top: 20px;">
            <h3>Getting Started</h3>
            <p>Upload your ticket data CSV or Excel file using the sidebar to begin analysis.</p>
            <p>Once loaded, you'll see a preview of your data here and can explore analysis, visualizations, and insights in the other tabs.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Show data overview
        st.subheader("Data Overview")
        
        # Create metrics row for quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            num_cols = len(st.session_state.data.columns)
            st.metric("Columns", num_cols)
        with col3:
            # Count numeric columns
            numeric_cols = sum(pd.api.types.is_numeric_dtype(st.session_state.data[col].dtype) 
                              for col in st.session_state.data.columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            # Count date columns
            date_cols = sum(pd.api.types.is_datetime64_dtype(st.session_state.data[col].dtype) 
                           for col in st.session_state.data.columns)
            st.metric("Date Columns", date_cols)
        
        # Preview the data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Column info
        st.subheader("Column Information")
        
        # Tabs for different column info views
        col_info_tabs = st.tabs(["List View", "Data Types", "Missing Values"])
        
        with col_info_tabs[0]:
            # Simple column list
            st.markdown("**Column Names:**")
            st.write(", ".join(st.session_state.data.columns.tolist()))
        
        with col_info_tabs[1]:
            # Data types table
            dtypes_df = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Data Type': [str(st.session_state.data[col].dtype) for col in st.session_state.data.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)
        
        with col_info_tabs[2]:
            # Missing values information
            missing_df = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Missing Values': [st.session_state.data[col].isna().sum() for col in st.session_state.data.columns],
                'Missing %': [(st.session_state.data[col].isna().sum() / len(st.session_state.data)) * 100 
                             for col in st.session_state.data.columns]
            })
            missing_df['Missing %'] = missing_df['Missing %'].round(2)
            st.dataframe(missing_df, use_container_width=True)
        
        # Data export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Data as CSV"):
                csv_data = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="ticket_data_export.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Column Info"):
                # Create a comprehensive column info dataframe
                col_info = pd.DataFrame({
                    'Column': st.session_state.data.columns,
                    'Data Type': [str(st.session_state.data[col].dtype) for col in st.session_state.data.columns],
                    'Missing Values': [st.session_state.data[col].isna().sum() for col in st.session_state.data.columns],
                    'Missing %': [(st.session_state.data[col].isna().sum() / len(st.session_state.data)) * 100 
                                 for col in st.session_state.data.columns],
                    'Unique Values': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
                })
                col_info['Missing %'] = col_info['Missing %'].round(2)
                
                csv_info = col_info.to_csv(index=False)
                st.download_button(
                    label="Download Column Info",
                    data=csv_info,
                    file_name="column_info.csv",
                    mime="text/csv"
                )

# Tab 2: Analysis
with tabs[1]:
    st.markdown("<h2 class='section-header'>Data Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to view analysis.")
    else:
        try:
            analysis_tabs = st.tabs(["Overview", "Statistics", "Data Explorer"])
            
            with analysis_tabs[0]:
                # Basic statistics overview
                st.subheader("Overview Statistics")
                
                # Display key metrics in multiple columns
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Total Tickets", len(st.session_state.data))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metrics_cols[1]:
                    avg_resolution = st.session_state.stats.get('avg_resolution_time_hours', 0)
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Avg Resolution Time", f"{avg_resolution:.1f} hours")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metrics_cols[2]:
                    if 'state_distribution' in st.session_state.stats:
                        open_tickets = sum([
                            st.session_state.stats['state_distribution'].get(state, 0) 
                            for state in ['Open', 'In Progress', 'Assigned', 'Pending']
                        ])
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Open Tickets", open_tickets)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Open Tickets", "N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with metrics_cols[3]:
                    if 'priority_distribution' in st.session_state.stats:
                        high_priority = sum([
                            st.session_state.stats['priority_distribution'].get(priority, 0) 
                            for priority in ['Critical', 'High', '1', '2']
                        ])
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("High Priority", high_priority)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("High Priority", "N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Distribution charts
                st.subheader("Key Distributions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Priority distribution
                    if 'priority_distribution' in st.session_state.stats:
                        st.subheader("Priority Distribution")
                        priority_df = pd.DataFrame(
                            st.session_state.stats['priority_distribution'].items(), 
                            columns=['Priority', 'Count']
                        )
                        st.bar_chart(priority_df.set_index('Priority'))
                
                with col2:
                    # State distribution
                    if 'state_distribution' in st.session_state.stats:
                        st.subheader("State Distribution")
                        state_df = pd.DataFrame(
                            st.session_state.stats['state_distribution'].items(), 
                            columns=['State', 'Count']
                        )
                        st.bar_chart(state_df.set_index('State'))
            
            with analysis_tabs[1]:
                # Detailed statistics
                st.subheader("Detailed Statistics")
                
                # Geography data
                if 'geographic_distribution' in st.session_state.stats:
                    st.markdown("#### Geographic Analysis")
                    geo_data = st.session_state.stats['geographic_distribution']
                    for col, regions in geo_data.items():
                        st.markdown(f"**{col}**")
                        geo_df = pd.DataFrame(regions.items(), columns=['Region', 'Count'])
                        st.bar_chart(geo_df.set_index('Region'))
                
                # Time-based statistics
                if 'avg_resolution_time_hours' in st.session_state.stats:
                    st.markdown("#### Resolution Time Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                        st.markdown("**Resolution Time Statistics:**")
                        st.markdown(f"- Average: {st.session_state.stats['avg_resolution_time_hours']:.2f} hours")
                        
                        if 'median_resolution_time_hours' in st.session_state.stats:
                            st.markdown(f"- Median: {st.session_state.stats['median_resolution_time_hours']:.2f} hours")
                        
                        if 'resolution_time_quartiles' in st.session_state.stats:
                            q = st.session_state.stats['resolution_time_quartiles']
                            st.markdown(f"- 25% of tickets: ≤ {q['25%']:.2f} hours")
                            st.markdown(f"- 75% of tickets: ≤ {q['75%']:.2f} hours")
                            st.markdown(f"- 90% of tickets: ≤ {q['90%']:.2f} hours")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        if 'business_hours_tickets' in st.session_state.stats and 'non_business_hours_tickets' in st.session_state.stats:
                            bh = st.session_state.stats['business_hours_tickets']
                            nbh = st.session_state.stats['non_business_hours_tickets']
                            total = bh + nbh
                            
                            # Create data for pie chart
                            labels = ['Business Hours', 'Non-Business Hours']
                            values = [bh, nbh]
                            
                            # Create pie chart with plotly
                            import plotly.graph_objects as go
                            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                            st.plotly_chart(fig, use_container_width=True)
                
                # Top issues
                if 'top_issues' in st.session_state.stats:
                    st.markdown("#### Top Issues")
                    issues_df = pd.DataFrame(
                        st.session_state.stats['top_issues'].items(), 
                        columns=['Issue', 'Count']
                    )
                    st.bar_chart(issues_df.set_index('Issue'))
                
                # Assignment groups
                if 'top_assignment_groups' in st.session_state.stats:
                    st.markdown("#### Top Assignment Groups")
                    groups_df = pd.DataFrame(
                        st.session_state.stats['top_assignment_groups'].items(), 
                        columns=['Group', 'Count']
                    )
                    st.bar_chart(groups_df.set_index('Group'))
            
            with analysis_tabs[2]:
                # Data explorer with filters
                st.subheader("Data Explorer")
                
                # Column selector
                selected_columns = st.multiselect(
                    "Select columns to display:",
                    options=st.session_state.data.columns.tolist(),
                    default=st.session_state.data.columns[:5].tolist()
                )
                
                # Filtering options
                st.markdown("#### Filter Data")
                filter_col1, filter_col2 = st.columns(2)
                
                # We'll allow filtering by one column for simplicity
                with filter_col1:
                    filter_column = st.selectbox(
                        "Filter by column:", 
                        options=["None"] + st.session_state.data.columns.tolist()
                    )
                
                filter_value = None
                if filter_column != "None":
                    with filter_col2:
                        unique_values = st.session_state.data[filter_column].dropna().unique()
                        if len(unique_values) <= 50:  # Only show dropdown for reasonable number of values
                            filter_value = st.selectbox(
                                f"Select {filter_column} value:",
                                options=["All"] + sorted([str(x) for x in unique_values])
                            )
                        else:
                            filter_value = st.text_input(f"Enter {filter_column} value to filter:")
                
                # Apply filters and display data
                filtered_data = st.session_state.data
                
                if filter_column != "None" and filter_value and filter_value != "All":
                    filtered_data = filtered_data[filtered_data[filter_column].astype(str) == filter_value]
                
                if selected_columns:
                    display_data = filtered_data[selected_columns]
                    st.dataframe(display_data, use_container_width=True)
                    
                    st.markdown(f"Showing {len(display_data)} records")
                    
                    # Download button for filtered data
                    csv_data = display_data.to_csv(index=False)
                    st.download_button(
                        label="Download filtered data as CSV",
                        data=csv_data,
                        file_name="filtered_ticket_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Please select at least one column to display data.")
        except Exception as e:
            handle_error(e, "Error in analysis tab")
            st.session_state.errors.append(f"Analysis error: {str(e)}")

# Tab 3: Visualization
with tabs[2]:
    st.markdown("<h2 class='section-header'>Data Visualization</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to view visualizations.")
    else:
        try:
            if st.session_state.charts is None:
                with st.spinner("Generating charts..."):
                    st.session_state.charts = visualization_module.generate_charts(st.session_state.processed_data)
            
            if not st.session_state.charts:
                st.warning("Could not generate charts from the data. Please check the data format.")
            else:
                visualization_module.display_charts(st.session_state.charts)
                
                st.markdown("---")
                
                # Option to export charts
                st.subheader("Export Visualizations")
                st.info("Export functionality will be implemented in a future version.")
                
                # Placeholder for future export feature
                # export_format = st.selectbox("Export format", ["PNG", "PDF", "HTML"])
                # if st.button("Export All Charts"):
                #     with st.spinner("Exporting charts..."):
                #         # Export logic would go here
                #         st.success(f"Charts exported successfully in {export_format} format!")
        except Exception as e:
            handle_error(e, "Error in visualization tab")
            st.session_state.errors.append(f"Visualization error: {str(e)}")


# Tab 4: Insights
with tabs[3]:
    st.markdown("<h2 class='section-header'>AI-Generated Insights</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to generate insights.")
    else:
        # Generate insights if not already done
        if st.session_state.insights is None or st.button("Regenerate Insights"):
            with st.spinner("Generating insights with LLM..."):
                # Use the get_insights helper function to generate insights
                st.session_state.insights = get_insights(
                    st.session_state.processed_data, 
                    st.session_state.stats
                )
        
        # Display insights
        if isinstance(st.session_state.insights, str):
            # If insights is a string (fallback mode), display directly
            st.markdown(st.session_state.insights)
        elif isinstance(st.session_state.insights, dict):
            # If insights is a structured dictionary
            
            # Key findings
            if 'key_findings' in st.session_state.insights:
                st.subheader("🔑 Key Findings")
                for i, finding in enumerate(st.session_state.insights['key_findings']):
                    with st.expander(f"Finding {i+1}: {finding.get('title', 'Insight')}", expanded=False):
                        st.markdown(finding.get('description', 'No details available'))
            
            # Recommendations
            if 'recommendations' in st.session_state.insights:
                st.subheader("Recommendations")
                for i, rec in enumerate(st.session_state.insights['recommendations']):
                    with st.expander(f"Recommendation {i+1}: {rec.get('title', 'Recommendation')}", expanded=False):
                        st.markdown(rec.get('description', 'No details available'))
                        
                        # Impact and effort if available
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Impact:** {rec.get('impact', 'Not specified')}")
                        with col2:
                            st.markdown(f"**Effort:** {rec.get('effort', 'Not specified')}")
            
            # Patterns
            if 'patterns' in st.session_state.insights:
                st.subheader("🔄 Patterns")
                for i, pattern in enumerate(st.session_state.insights['patterns']):
                    with st.expander(f"Pattern {i+1}: {pattern.get('title', 'Pattern')}", expanded=False):
                        st.markdown(pattern.get('description', 'No details available'))
        else:
            st.error("Insights data is in an unexpected format. Please regenerate insights.")


# Tab 5: Questions
with tabs[4]:
    st.markdown("<h2 class='section-header'>Predefined Questions</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to use the questions module.")
    else:
        try:
            # Generate predefined questions if not already done
            if st.session_state.predefined_questions is None or st.button("Generate Questions"):
                with st.spinner("Generating predefined questions with LLM..."):
                    # Much more aggressive sampling to avoid token limit issues
                    # Reduce the sample size dramatically for the LLM request
                    sample_size = min(20, len(st.session_state.processed_data))
                    
                    # Get a small representative sample
                    df_sample = st.session_state.processed_data.sample(sample_size, random_state=42)
                    
                    # Convert to JSON-safe format with very limited fields
                    # Only include the most important columns to reduce token count
                    important_columns = []
                    
                    # Try to identify important columns based on common ticket field names
                    potential_columns = [
                        'number', 'id', 'priority', 'category', 'subcategory', 'state', 'status',
                        'short description', 'description', 'assignment group', 'assigned to',
                        'opened', 'closed', 'resolution time', 'sla'
                    ]
                    
                    # Only include columns that actually exist in the dataframe
                    for col in potential_columns:
                        if col in df_sample.columns:
                            important_columns.append(col)
                    
                    # If we found less than 5 columns, add more until we reach 5 or run out
                    if len(important_columns) < 5:
                        remaining_cols = [col for col in df_sample.columns if col not in important_columns]
                        important_columns.extend(remaining_cols[:5 - len(important_columns)])
                    
                    # Create a smaller dataframe with just the important columns
                    if important_columns:
                        df_sample = df_sample[important_columns]
                    
                    # Convert to minimal JSON representation
                    json_data = prepare_dataframe_for_json(df_sample, max_rows=20)
                    
                    # Create a simplified version of the stats
                    # Only include the most critical statistics
                    simplified_stats = {}
                    key_stat_fields = [
                        'total_tickets', 'avg_resolution_time_hours', 'priority_distribution',
                        'state_distribution', 'top_issues'
                    ]
                    
                    for field in key_stat_fields:
                        if field in st.session_state.stats:
                            simplified_stats[field] = st.session_state.stats[field]
                    
                    json_stats = prepare_stats_for_json(simplified_stats)
                    
                    # Generate predefined questions with the reduced data
                    st.session_state.predefined_questions = questions_module.generate_predefined_questions(json_data, json_stats)
            
            # Display questions with answers automatically shown
            if st.session_state.predefined_questions:
                st.markdown("### Questions and Answers Based on Your Ticket Data")
                
                # Display each question and answer
                for i, qa_pair in enumerate(st.session_state.predefined_questions):
                    question = qa_pair.get('question', 'Question not available')
                    answer = qa_pair.get('answer', 'No answer available')
                    automation = qa_pair.get('automation_potential', 'No automation potential information available')
                    
                    # Create expandable section with the question as header
                    with st.expander(f"**Q{i+1}: {question}**", expanded=False):
                        st.markdown("#### Answer:")
                        st.markdown(answer)
                        
                        st.markdown("#### Automation Potential:")
                        st.markdown(automation)
            else:
                st.warning("Failed to generate predefined questions. Please try again or check your data.")
        except Exception as e:
            handle_error(e, "Error generating predefined questions")
            st.session_state.errors.append(f"Questions error: {str(e)}")

# Tab 6: Automation
with tabs[5]:
    st.markdown("<h2 class='section-header'>Automation Opportunities</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to identify automation opportunities.")
    else:
        # Generate automation opportunities if not already done
        if st.session_state.automation_opportunities is None or st.button("Identify Automation Opportunities"):
            with st.spinner("Analyzing ticket data for root causes and automation opportunities..."):
                # Use our root cause analysis function to generate data-driven opportunities
                st.session_state.automation_opportunities = get_data_driven_automation_opportunities(
                    st.session_state.processed_data,
                    st.session_state.stats
                )
        
        # Add a placeholder if no opportunities are found
        if not st.session_state.automation_opportunities:
            st.warning("No automation opportunities identified. Try regenerating or check your data.")
        
        # Display automation opportunities
        elif isinstance(st.session_state.automation_opportunities, list):
            # Count valid opportunities (with titles)
            valid_opps = [opp for opp in st.session_state.automation_opportunities if opp.get('title')]
            
            if valid_opps:
                st.subheader(f"Found {len(valid_opps)} Root Cause-Based Automation Opportunities")
                
                # Add an overall automation heat map
                st.markdown("### Automation Impact Assessment")
                
                # Create a heat map of opportunities
                import plotly.graph_objects as go
                
                # Get data for the heat map
                titles = [opp.get('title', 'Unknown') for opp in valid_opps]
                impact_scores = [opp.get('impact_score', 0) for opp in valid_opps]
                
                # Create figure
                fig = go.Figure(data=[go.Bar(
                    x=impact_scores,
                    y=titles,
                    orientation='h',
                    marker=dict(
                        color=impact_scores,
                        colorscale='Viridis',
                        colorbar=dict(title="Impact Score")
                    )
                )])
                
                # Update layout
                fig.update_layout(
                    title="Automation Opportunities by Impact Score",
                    xaxis_title="Impact Score (0-100)",
                    yaxis_title="Automation Opportunity",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display each opportunity
                for i, opp in enumerate(valid_opps):
                    # Calculate impact level for color coding
                    impact_score = opp.get('impact_score', 0)
                    if impact_score >= 70:
                        impact_color = "#38A169"  # Green
                        impact_level = "High Impact"
                    elif impact_score >= 40:
                        impact_color = "#DD6B20"  # Orange
                        impact_level = "Medium Impact"
                    else:
                        impact_color = "#718096"  # Gray
                        impact_level = "Lower Impact"
                    
                    # Create a colored header based on impact score
                    header_html = f"""
                    <div style="display: flex; align-items: center; margin-bottom: 0;">
                        <div style="background-color: {impact_color}; color: white; padding: 2px 8px; 
                              border-radius: 12px; font-size: 12px; margin-right: 10px;">
                            {impact_level}
                        </div>
                        <div style="font-weight: bold; font-size: 16px;">
                            {opp.get('title', 'Automation Opportunity')}
                        </div>
                    </div>
                    """
                    
                    # Display the opportunity with expandable details
                    with st.expander(f"Opportunity {i+1}", expanded=False):
                        # Display the custom header
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # Show impact score
                        st.progress(min(impact_score/100, 1.0))
                        st.markdown(f"**Impact Score:** {impact_score:.1f}/100")
                        
                        st.markdown(f"**Scope:** {opp.get('scope', 'No scope information')}")
                        
                        st.markdown("**Justification:**")
                        st.markdown(opp.get('justification', 'No justification provided'))
                        
                        # Show example tickets if available
                        examples = opp.get('examples', [])
                        if examples:
                            st.markdown("**Example Tickets:**")
                            for example in examples:
                                st.markdown(f"- `{example}`")
                        
                        # Display in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Type:**")
                            st.markdown(opp.get('type', 'No type information'))
                            
                            # Display ROI estimate if available
                            if 'roi_estimate' in opp:
                                st.markdown("**ROI Estimate:**")
                                st.markdown(opp.get('roi_estimate', 'No ROI estimate available'))
                        
                        with col2:
                            st.markdown("**Implementation Plan:**")
                            st.markdown(opp.get('implementation_plan', 'No implementation plan provided'))
                
                # Add a download button for the opportunities
                st.download_button(
                    "Download Automation Opportunities as CSV",
                    data=pd.DataFrame(valid_opps).to_csv(index=False),
                    file_name="automation_opportunities.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid automation opportunities were identified.")
        
        # Custom automation scenario analysis
        st.markdown("---")
        st.subheader("Custom Automation Scenario")
        
        custom_scenario = st.text_area(
            "Describe a specific automation scenario you're interested in:",
            placeholder="Example: We'd like to automate password reset requests to reduce the workload on our service desk.",
            height=100
        )
        
        if st.button("Analyze Custom Scenario") and custom_scenario:
            with st.spinner("Analyzing your custom scenario..."):
                # Use the analyze_custom_scenario helper function
                analysis = analyze_custom_scenario(
                    custom_scenario,
                    st.session_state.processed_data,
                    st.session_state.stats
                )
            
            # Display the analysis in a structured way
            st.markdown("### Scenario Analysis")
            
            # Create a colored card based on feasibility
            feasibility = analysis.get("feasibility", "Unknown")
            feasibility_color = {
                "High": "green",
                "Medium": "orange",
                "Low": "red"
            }.get(feasibility, "gray")
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 5px; background-color: {feasibility_color}25; 
                 border-left: 5px solid {feasibility_color}; margin-bottom: 15px;">
                <h4 style="margin-top: 0; color: {feasibility_color};">Feasibility: {feasibility}</h4>
                <p>{analysis.get("impact", "No impact analysis available")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display main sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Implementation Approach")
                st.markdown(analysis.get("implementation_approach", "No implementation approach provided"))
                
                st.markdown("#### ROI Timeline")
                st.markdown(analysis.get("roi_timeline", "No ROI timeline provided"))
            
            with col2:
                st.markdown("#### Required Technology")
                technologies = analysis.get("required_technology", [])
                if isinstance(technologies, list) and technologies:
                    for tech in technologies:
                        st.markdown(f"- {tech}")
                else:
                    st.markdown("No specific technologies identified")
                
                st.markdown("#### Challenges")
                challenges = analysis.get("challenges", [])
                if isinstance(challenges, list) and challenges:
                    for challenge in challenges:
                        st.markdown(f"- {challenge}")
                else:
                    st.markdown("No specific challenges identified")
            
            st.markdown("#### Additional Data Requirements")
            st.markdown(analysis.get("data_requirements", "No additional data requirements specified"))

# Tab 7: Chat
with tabs[6]:
    st.markdown("<h2 class='section-header'>Chat with Your Ticket Data</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to chat with it.")
    else:
        try:
            st.markdown("Ask questions about your ticket data in natural language.")
            
            # Initialize chat history if empty
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
            
            # Add a default message if chat is empty
            if not st.session_state.chat_history:
                with st.chat_message("assistant"):
                    st.markdown("Hello! I'm your ticket data assistant. Ask me anything about your ticket data, such as:")
                    st.markdown("""
                    - What are the most common ticket categories?
                    - What's the average resolution time for high priority tickets?
                    - Which team handles the most tickets?
                    - Are there any patterns in ticket creation time?
                    - What automation opportunities do you see in this data?
                    """)
            
            # Chat input
            user_query = st.chat_input("Ask something about your ticket data...")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Process query and get response
                with st.spinner("Processing your question..."):
                    # Convert data to JSON-safe format
                    json_data = prepare_dataframe_for_json(st.session_state.processed_data, max_rows=100)
                    json_stats = prepare_stats_for_json(st.session_state.stats)
                    
                    # Get insights as text if available, otherwise empty string
                    insights_text = ""
                    if st.session_state.insights:
                        if isinstance(st.session_state.insights, str):
                            insights_text = st.session_state.insights
                        elif isinstance(st.session_state.insights, dict):
                            # Flatten the insights dict to text
                            for section in ['key_findings', 'recommendations', 'patterns']:
                                if section in st.session_state.insights:
                                    insights_text += f"\n\n{section.upper()}:\n"
                                    for item in st.session_state.insights[section]:
                                        insights_text += f"\n- {item.get('title', '')}: {item.get('description', '')}"
                    
                    try:
                        response = chat_module.process_query(
                            user_query,
                            json_data,
                            json_stats,
                            insights_text
                        )
                    except Exception as e:
                        response = f"I'm sorry, I had trouble processing your question: {str(e)}. Could you please rephrase or try a different question?"
                        st.session_state.errors.append(f"Chat query error: {str(e)}")
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Rerun the app to show the updated chat
                st.experimental_rerun()
        
        except Exception as e:
            handle_error(e, "Error in chat module")
            st.session_state.errors.append(f"Chat error: {str(e)}")            

    