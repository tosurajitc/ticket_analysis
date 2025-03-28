# main.py - Main application entry point
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
import groq
import traceback
import re
from collections import Counter

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


################################## Root cause based Automation opportunities starts #########################################
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

# Update the get_data_driven_automation_opportunities function to use the varied impact scores

def generate_automation_opportunity_from_root_cause(cause_key, cause_data):
    """
    Generate a detailed automation opportunity from a root cause with varied impact scores.
    
    Args:
        cause_key: Key for the root cause
        cause_data: Data about the root cause
        
    Returns:
        dict: Automation opportunity
    """
    automation = cause_data['automation_potential']
    
    # Extract opportunity type from the cause_key
    opportunity_type = "general"
    
    # Try to identify type from key or value
    if "password" in cause_key.lower():
        opportunity_type = "password_reset"
    elif "access" in cause_key.lower():
        opportunity_type = "access_request"
    elif "software" in cause_key.lower() or "install" in cause_key.lower():
        opportunity_type = "software_installation"
    elif "account" in cause_key.lower() and "lock" in cause_key.lower():
        opportunity_type = "account_lockout"
    elif "vpn" in cause_key.lower():
        opportunity_type = "vpn_issues"
    elif "onboard" in cause_key.lower() or "hire" in cause_key.lower():
        opportunity_type = "onboarding"
    elif "print" in cause_key.lower():
        opportunity_type = "printer_issues"
    elif "email" in cause_key.lower() or "outlook" in cause_key.lower():
        opportunity_type = "email_issues"
    elif "data" in cause_key.lower() and "fix" in cause_key.lower():
        opportunity_type = "datafix"
    elif ("system" in cause_key.lower() and "down" in cause_key.lower()) or "unavailable" in cause_key.lower():
        opportunity_type = "system_unavailable"
    elif "login" in cause_key.lower():
        opportunity_type = "login_issue"
    elif "error" in cause_key.lower():
        opportunity_type = "error_message"
    elif "hardware" in cause_key.lower():
        opportunity_type = "hardware_issue"
    elif "slow" in cause_key.lower() or "performance" in cause_key.lower():
        opportunity_type = "performance_issue"
    
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
    
    # Calculate varied impact score
    impact_score = calculate_varied_impact_score(
        opportunity_type,
        cause_data['count'],
        cause_data.get('total_tickets', 1000),  # Default estimate if not available
        automation.get('avg_time', 1.0),        # Use average time if available
        automation.get('level', "Medium")       # Use automation level as complexity
    )
    
    # Generate opportunity
    opportunity = {
        "title": f"Automate {cause_data['value']} Resolution",
        "scope": f"Create an automated solution for {cause_data['value']} tickets",
        "justification": f"Analysis identified {cause_data['count']} tickets related to '{cause_data['value']}'. {resolution_text}",
        "type": automation['type'],
        "implementation_plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(automation['implementation_steps'])]),
        "impact_score": impact_score,
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

############################################## Root cause based automation opportunity ends  #########################################


def deep_analyze_ticket_content(df, focus_area=None):
    """
    Perform deep analysis of ticket content to identify specific issues, patterns and automation possibilities.
    
    Args:
        df: DataFrame with ticket data
        focus_area: Optional specific area to focus analysis on (e.g., "errors", "access", etc.)
        
    Returns:
        dict: Detailed patterns with specific examples and automation suggestions
    """
    # Get description and resolution columns
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
    
    # Initialize results
    patterns = {
        "specific_issues": [],
        "error_patterns": {},
        "resolution_patterns": {},
        "system_patterns": {},
        "examples": {}
    }
    
    # If no description column, can't do much analysis
    if not desc_column:
        return patterns
    
    # 1. First pass - extract context around key terms
    if focus_area == "errors" or focus_area is None:
        # Look for specific errors
        error_contexts = _extract_error_contexts(df, desc_column, notes_column)
        patterns["error_patterns"] = error_contexts
    
    # 2. Extract systems and applications mentioned
    system_patterns = _extract_system_mentions(df, desc_column)
    patterns["system_patterns"] = system_patterns
    
    # 3. Extract resolution types when we have notes
    if notes_column:
        resolution_patterns = _extract_resolution_patterns(df, notes_column)
        patterns["resolution_patterns"] = resolution_patterns
    
    # 4. Use n-gram analysis to find specific issue patterns
    ngram_patterns = _extract_ngram_patterns(df, desc_column, 2, 3)  # Bigrams and trigrams
    patterns["specific_issues"] = ngram_patterns
    
    # 5. Connect issues to resolutions when possible
    if notes_column:
        # For each specific error type, find common resolutions
        _connect_issues_to_resolutions(patterns, df, desc_column, notes_column)
    
    return patterns


def _connect_issues_to_resolutions(patterns, df, desc_column, notes_column):
    """
    Connect issue patterns to resolution patterns.
    
    Args:
        patterns: Dictionary containing pattern data
        df: DataFrame with ticket data
        desc_column: Column name containing description text
        notes_column: Column name containing resolution notes
        
    Returns:
        None (updates patterns dict in place)
    """
    # For each error pattern, try to find common resolutions
    for error_term, error_data in patterns["error_patterns"].items():
        # Get rows with this error
        matching_rows = df[df[desc_column].str.contains(error_term, case=False, na=False)]
        
        if len(matching_rows) < 5:
            continue
        
        # Look for common words in resolution notes
        notes = matching_rows[notes_column].dropna().astype(str)
        
        # Combine all notes
        all_notes = ' '.join(notes)
        
        # Count word frequencies
        from collections import Counter
        import re
        
        # Clean and split into words
        clean_notes = re.sub(r'[^\w\s]', ' ', all_notes.lower())
        words = clean_notes.split()
        
        # Count
        word_counts = Counter(words)
        
        # Filter for action words
        action_words = ['restart', 'reset', 'reinstall', 'update', 'install', 'configure', 
                        'grant', 'remove', 'add', 'change', 'fix', 'resolve', 'clear', 'delete',
                        'replace', 'create', 'verify', 'contact', 'escalate', 'assist', 'enable',
                        'disable', 'unlock', 'restore', 'recover', 'rebuild', 'recommend', 'approve']
        
        common_actions = []
        for action in action_words:
            # Check for the action word and variations
            count = 0
            for word in word_counts:
                if action in word:  # Partial match to catch variations
                    count += word_counts[word]
            
            if count >= 2:
                common_actions.append({
                    "action": action,
                    "count": count
                })
        
        # Sort by frequency
        common_actions.sort(key=lambda x: x["count"], reverse=True)
        
        # Add to error data
        if common_actions:
            error_data["common_resolutions"] = common_actions[:5]  # Top 5 resolutions



def _extract_error_contexts(df, desc_column, notes_column=None):
    """
    Extract specific contexts around error mentions.
    
    Args:
        df: DataFrame with ticket data
        desc_column: Column name containing description text
        notes_column: Optional column name containing resolution notes
        
    Returns:
        dict: Error patterns with context information
    """
    error_patterns = {}
    
    # Define a list of error-related terms to look for
    error_terms = ['error', 'fail', 'exception', 'crash', 'broken', 'issue', 'problem']
    
    # Look for context around these terms
    for term in error_terms:
        # Find rows with this term
        matching_rows = df[df[desc_column].str.contains(term, case=False, na=False)]
        
        if len(matching_rows) < 5:  # Skip terms with too few matches
            continue
            
        # Extract phrases containing the error term
        contexts = []
        for _, row in matching_rows.head(50).iterrows():  # Limit to 50 examples for efficiency
            desc = str(row[desc_column])
            
            # Extract words before and after the term
            import re
            matches = re.finditer(r'\b' + re.escape(term) + r'\b', desc.lower())
            
            for match in matches:
                start_pos = max(0, match.start() - 30)
                end_pos = min(len(desc), match.end() + 30)
                
                # Get the surrounding context
                context = desc[start_pos:end_pos].strip()
                contexts.append(context)
        
        if contexts:
            # Analyze contexts to find patterns
            _analyze_context_patterns(contexts, term, error_patterns)
    
    return error_patterns

def _analyze_context_patterns(contexts, term, patterns_dict):
    """
    Analyze context strings to find common patterns around a term.
    
    Args:
        contexts: List of context strings
        term: The error term being analyzed
        patterns_dict: Dictionary to store the patterns
        
    Returns:
        None (updates patterns_dict in place)
    """
    # We'll look for common words immediately preceding or following the term
    preceding_words = {}
    following_words = {}
    
    for context in contexts:
        context_lower = context.lower()
        term_lower = term.lower()
        
        try:
            term_index = context_lower.index(term_lower)
            
            # Get preceding words
            preceding_text = context_lower[:term_index].strip()
            preceding_words_list = preceding_text.split()[-3:]  # Get up to 3 preceding words
            
            for word in preceding_words_list:
                # Clean the word
                word = word.strip(',.();:"\'')
                if word and len(word) > 2:  # Skip very short words
                    preceding_words[word] = preceding_words.get(word, 0) + 1
            
            # Get following words
            following_text = context_lower[term_index + len(term_lower):].strip()
            following_words_list = following_text.split()[:3]  # Get up to 3 following words
            
            for word in following_words_list:
                # Clean the word
                word = word.strip(',.();:"\'')
                if word and len(word) > 2:  # Skip very short words
                    following_words[word] = following_words.get(word, 0) + 1
                    
        except ValueError:
            # Term not found in context (shouldn't happen but just in case)
            pass
    
    # Find the most common preceding and following words
    common_preceding = sorted(preceding_words.items(), key=lambda x: x[1], reverse=True)[:5]
    common_following = sorted(following_words.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Look for specific combinations
    common_pairs = []
    
    for p_word, p_count in common_preceding:
        for f_word, f_count in common_following:
            # Look for instances where both words appear in context
            pair_count = 0
            for context in contexts:
                if p_word in context.lower() and f_word in context.lower():
                    pair_count += 1
            
            if pair_count >= 2:  # At least 2 occurrences
                common_pairs.append({
                    "pattern": f"{p_word} {term} {f_word}",
                    "count": pair_count,
                    "examples": [c for c in contexts[:5] if p_word in c.lower() and f_word in c.lower()]
                })
    
    # Add to patterns dictionary
    patterns_dict[term] = {
        "count": len(contexts),
        "preceding_terms": [{"term": term, "count": count} for term, count in common_preceding],
        "following_terms": [{"term": term, "count": count} for term, count in common_following],
        "common_patterns": common_pairs,
        "examples": contexts[:5]  # Top 5 examples
    }


def _extract_system_mentions(df, desc_column):
    """
    Extract mentions of systems and applications from ticket descriptions.
    
    Args:
        df: DataFrame with ticket data
        desc_column: Column name containing description text
        
    Returns:
        dict: System patterns with frequency and examples
    """
    system_patterns = {}
    
    # Common system and application terms to look for
    system_terms = [
        'SAP', 'Oracle', 'Salesforce', 'ServiceNow', 'Excel', 'Outlook', 'SharePoint', 'OneDrive',
        'Teams', 'Windows', 'Office', 'Exchange', 'Active Directory', 'SQL', 'Database', 'ERP',
        'CRM', 'VPN', 'Network', 'Server', 'Desktop', 'Laptop', 'Mobile', 'iOS', 'Android',
        'Chrome', 'Firefox', 'Safari', 'Edge', 'Internet Explorer', 'Word', 'PowerPoint',
        'Access', 'Power BI', 'Tableau', 'Jira', 'Confluence', 'Azure', 'AWS', 'Google Cloud'
    ]
    
    # Add some potential custom systems detection
    descriptions = df[desc_column].dropna().astype(str).str.lower()
    
    # Look for capitalized terms that might be system names
    import re
    potential_systems = set()
    
    for desc in descriptions.sample(min(500, len(descriptions))):  # Sample for efficiency
        # Find capitalized terms that might be system names
        caps = re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', desc)
        potential_systems.update(caps)
    
    # Combine with known system terms
    all_system_terms = system_terms + list(potential_systems)
    
    # Find occurrences of system terms
    for system in all_system_terms:
        # Skip very common words that might be mistaken for systems
        if system.lower() in ['the', 'and', 'this', 'that', 'with', 'from', 'have', 'issue']:
            continue
            
        # Count occurrences
        count = df[desc_column].str.contains(r'\b' + re.escape(system) + r'\b', case=False, regex=True, na=False).sum()
        
        if count >= 3:  # At least 3 mentions
            # Get examples
            examples = df[df[desc_column].str.contains(r'\b' + re.escape(system) + r'\b', case=False, regex=True, na=False)].head(3)
            example_texts = examples[desc_column].astype(str).tolist()
            
            system_patterns[system] = {
                "count": int(count),
                "examples": example_texts
            }
    
    return system_patterns

def _extract_resolution_patterns(df, notes_column):
    """
    Extract patterns from resolution notes.
    
    Args:
        df: DataFrame with ticket data
        notes_column: Column name containing resolution notes
        
    Returns:
        dict: Resolution patterns with frequency and examples
    """
    resolution_patterns = {}
    
    # Look for common resolution actions
    action_terms = [
        'restarted', 'reset', 'reinstalled', 'updated', 'installed', 'configured', 'modified',
        'granted', 'removed', 'added', 'changed', 'fixed', 'resolved', 'cleared', 'deleted',
        'replaced', 'created', 'verified', 'contacted', 'escalated', 'assisted', 'enabled',
        'disabled', 'unlocked', 'restored', 'recovered', 'rebuilt', 'recommended', 'approved'
    ]
    
    for action in action_terms:
        # Count occurrences
        notes = df[notes_column].dropna().astype(str)
        count = notes.str.contains(r'\b' + action + r'\b', case=False, regex=True).sum()
        
        if count >= 5:  # At least 5 mentions
            # Get examples
            matching_rows = df[notes.str.contains(r'\b' + action + r'\b', case=False, regex=True)].head(3)
            example_texts = matching_rows[notes_column].astype(str).tolist()
            
            resolution_patterns[action] = {
                "count": int(count),
                "examples": example_texts
            }
    
    return resolution_patterns


def _extract_ngram_patterns(df, desc_column, min_n=2, max_n=3):
    """
    Extract n-gram patterns from descriptions.
    
    Args:
        df: DataFrame with ticket data
        desc_column: Column name containing description text
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        
    Returns:
        list: List of n-gram patterns with frequency and examples
    """
    from collections import Counter
    import re
    
    # Get clean descriptions
    descriptions = df[desc_column].dropna().astype(str)
    cleaned_descs = [re.sub(r'[^\w\s]', '', desc.lower()) for desc in descriptions]
    
    # Extract n-grams
    all_ngrams = []
    
    for desc in cleaned_descs:
        words = desc.split()
        
        for n in range(min_n, max_n + 1):
            if len(words) >= n:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)
    
    # Count occurrences
    ngram_counts = Counter(all_ngrams)
    
    # Filter for meaningful patterns
    # Exclude very common phrases that aren't issue-specific
    exclude_phrases = ['i have a', 'i need a', 'i need to', 'can you please', 'please help', 
                       'i am having', 'we have a', 'there is a', 'this is a', 'how to']
    
    filtered_ngrams = [(ngram, count) for ngram, count in ngram_counts.most_common(100) 
                       if count >= 3 and not any(excl in ngram for excl in exclude_phrases)]
    
    # Create pattern objects
    ngram_patterns = []
    
    for ngram, count in filtered_ngrams:
        # Find examples
        examples = []
        for desc in descriptions.head(100):  # Limit search to first 100 for efficiency
            if ngram in desc.lower():
                examples.append(desc)
                if len(examples) >= 3:  # Limit to 3 examples
                    break
        
        if examples:  # Only add if we found examples
            ngram_patterns.append({
                "pattern": ngram,
                "count": count,
                "examples": examples
            })
    
    return ngram_patterns


def determine_specific_automation_type(opportunity_type, pattern_data):
    """
    Determine specific, differentiated automation types based on the opportunity.
    
    Args:
        opportunity_type: Type of opportunity (password, access, etc.)
        pattern_data: Data about the pattern
        
    Returns:
        str: Specific automation type
    """
    # Extract data attributes
    count = pattern_data.get("count", 0)
    complexity = pattern_data.get("complexity", "Medium")
    has_examples = len(pattern_data.get("examples", [])) > 0
    common_resolutions = pattern_data.get("common_resolutions", [])
    
    # Specific automation types by opportunity category
    if opportunity_type == "password_reset":
        if count > 50:
            return "Self-service portal with identity verification and Active Directory integration"
        else:
            return "Chatbot-assisted password reset with security verification"
    
    elif opportunity_type == "access_request":
        if "approval" in str(common_resolutions).lower():
            return "Multi-stage approval workflow with identity governance integration"
        else:
            return "Role-based access provisioning system with compliance controls"
    
    elif opportunity_type == "software_installation":
        if count > 100:
            return "Software distribution platform with self-service catalog"
        else:
            return "Automated package deployment with version control"
    
    elif opportunity_type == "account_lockout":
        return "Self-service account unlock with multi-factor authentication"
    
    elif opportunity_type == "vpn_issues":
        if "configuration" in str(common_resolutions).lower():
            return "VPN configuration assistant with automated troubleshooting"
        else:
            return "Remote connectivity diagnostic and remediation system"
    
    elif opportunity_type == "onboarding":
        return "End-to-end onboarding orchestration with cross-system provisioning"
    
    elif opportunity_type == "printer_issues":
        return "Print system diagnostic and repair automation"
    
    elif opportunity_type == "email_issues":
        if "quota" in str(pattern_data).lower():
            return "Mailbox quota management with proactive notifications"
        else:
            return "Email connectivity triage and restoration system"
    
    elif opportunity_type == "datafix":
        return "Data validation and correction framework with exception handling"
    
    elif opportunity_type == "system_unavailable":
        if "restart" in str(common_resolutions).lower():
            return "Automated service monitoring with self-healing capabilities"
        else:
            return "System availability manager with failover orchestration"
    
    elif opportunity_type == "login_issue":
        return "Authentication troubleshooter with credential verification"
    
    elif opportunity_type == "error_message":
        return "Error pattern recognition and automated resolution system"
    
    elif opportunity_type == "hardware_issue":
        return "Hardware diagnostic automation with predictive maintenance"
    
    elif opportunity_type == "performance_issue":
        return "Performance analysis and optimization framework"
    
    # For more specific categorization based on resolution patterns
    if common_resolutions:
        top_resolution = common_resolutions[0]['action'] if isinstance(common_resolutions, list) and len(common_resolutions) > 0 else ""
        
        if top_resolution == "restart":
            return "Service monitoring and automated restart system"
        elif top_resolution == "reset":
            return "Configuration reset automation with state preservation"
        elif top_resolution == "install" or top_resolution == "update":
            return "Software deployment and patching orchestration"
        elif top_resolution == "configure":
            return "Configuration management automation with validation"
        elif top_resolution == "grant" or top_resolution == "add":
            return "Permission management system with governance controls"
        elif top_resolution == "clear" or top_resolution == "delete":
            return "Automated cleanup system with safety verification"
    
    # Default with complexity differentiation
    if complexity == "Low":
        return "Lightweight process automation with minimal integration"
    elif complexity == "High":
        return "Complex orchestration platform with cross-system integration"
    else:
        return "Standardized workflow automation with self-service interface"



# Update the recommendation functions to use specific automation types

def generate_automation_recommendation(pattern_type, pattern_data):
    """
    Generate specific automation recommendations for a pattern with varied impact scores and specific types.
    
    Args:
        pattern_type: Type of pattern ("error", "system", etc.)
        pattern_data: Data about the pattern
        
    Returns:
        dict: Specific automation recommendation
    """
    # Base recommendation structure
    recommendation = {
        "approach": "",
        "technologies": [],
        "implementation_steps": [],
        "benefits": [],
        "complexity": "Medium"
    }
    
    # Opportunity type for impact score calculation
    opportunity_type = "general"
    
    # Determine automation approach based on pattern type
    if pattern_type == "error":
        count = pattern_data.get("count", 0)
        term = pattern_data.get("term", "error")
        
        if "password" in term or any("password" in p.get("pattern", "") for p in pattern_data.get("common_patterns", [])):
            # Password-related errors
            opportunity_type = "password_reset"
            recommendation["approach"] = "Self-service password reset automation"
            recommendation["technologies"] = ["Microsoft Identity Manager", "ForgeRock", "Okta", "UiPath RPA"]
            recommendation["implementation_steps"] = [
                "Deploy self-service password reset portal",
                "Integrate with Active Directory/IAM system",
                "Implement secure verification methods",
                "Configure end-user notifications",
                "Create usage analytics dashboard"
            ]
            recommendation["benefits"] = [
                f"Eliminate {count} password-related tickets annually",
                "Immediate resolution for users 24/7",
                "Reduce helpdesk workload by 15-20%",
                "Improve security through standardized processes"
            ]
            recommendation["complexity"] = "Low"  # Password resets are relatively easy to automate
        
        elif "access" in term or any("access" in p.get("pattern", "") for p in pattern_data.get("common_patterns", [])):
            # Access-related errors
            opportunity_type = "access_request"
            recommendation["approach"] = "Access request and provisioning workflow automation"
            recommendation["technologies"] = ["ServiceNow", "SailPoint", "CyberArk", "Microsoft Power Automate", "Automation Anywhere"]
            recommendation["implementation_steps"] = [
                "Map access requirements and approval workflows",
                "Create self-service request catalog",
                "Build approval routing logic",
                "Integrate with IAM/directory systems",
                "Implement audit and compliance reporting"
            ]
            recommendation["benefits"] = [
                f"Streamline resolution for {count} access-related tickets annually",
                "Reduce manual provisioning errors",
                "Accelerate access delivery by 70%",
                "Improve security governance and compliance"
            ]
            recommendation["complexity"] = "Medium"
        
        elif any(term in ["system", "application", "app", "server", "service"] for term in pattern_data.get("preceding_terms", [])):
            # System/application errors
            opportunity_type = "system_unavailable"
            recommendation["approach"] = "Automated system monitoring and remediation"
            recommendation["technologies"] = ["SolarWinds", "Nagios", "Splunk", "Microsoft SCOM", "New Relic", "Dynatrace", "Jenkins"]
            recommendation["implementation_steps"] = [
                "Deploy monitoring for critical systems",
                "Create automated health checks",
                "Implement self-healing scripts for common issues",
                "Build incident creation for failed self-healing",
                "Establish performance baselines and trending"
            ]
            recommendation["benefits"] = [
                f"Proactively address {count} system error tickets annually",
                "Reduce service interruptions through early detection",
                "Decrease MTTR by 60% through automated remediation",
                "Improve system reliability and performance"
            ]
            recommendation["complexity"] = "High"  # System monitoring can be complex
        
        else:
            # Generic error handling
            opportunity_type = "error_message"
            recommendation["approach"] = "Intelligent error detection and resolution system"
            recommendation["technologies"] = ["ServiceNow", "Microsoft Power Automate", "UiPath", "Python scripts", "Ansible"]
            recommendation["implementation_steps"] = [
                "Catalog error types and resolution patterns",
                "Develop automated diagnostics",
                "Create resolution playbooks for common errors",
                "Build knowledge base integration",
                "Implement success rate tracking"
            ]
            recommendation["benefits"] = [
                f"Accelerate resolution for {count} error tickets annually",
                "Standardize troubleshooting and resolution processes",
                "Reduce mean time to resolution by 40-50%",
                "Decrease ticket reopens through consistent solutions"
            ]
            recommendation["complexity"] = "Medium"
    
    # Similar logic for system and resolution patterns (not showing all for brevity)
    # ...
    
    # Calculate the varied impact score
    impact_score = calculate_varied_impact_score(
        opportunity_type, 
        pattern_data.get("count", 0), 
        1000,  # Default total tickets if not available
        avg_time=1.5,  # Default average time
        complexity=recommendation["complexity"]
    )
    
    # Get specific automation type instead of generic "Process automation workflow"
    specific_type = determine_specific_automation_type(opportunity_type, pattern_data)
    recommendation["type"] = specific_type
    
    recommendation["roi"] = "High" if impact_score >= 70 else "Medium" if impact_score >= 40 else "Low"
    recommendation["impact_score"] = impact_score
    
    return recommendation

def generate_automation_opportunity_from_root_cause(cause_key, cause_data):
    """
    Generate a detailed automation opportunity from a root cause with varied impact scores and specific types.
    
    Args:
        cause_key: Key for the root cause
        cause_data: Data about the root cause
        
    Returns:
        dict: Automation opportunity
    """
    automation = cause_data['automation_potential']
    
    # Extract opportunity type from the cause_key
    opportunity_type = "general"
    
    # Try to identify type from key or value
    if "password" in cause_key.lower():
        opportunity_type = "password_reset"
    elif "access" in cause_key.lower():
        opportunity_type = "access_request"
    elif "software" in cause_key.lower() or "install" in cause_key.lower():
        opportunity_type = "software_installation"
    elif "account" in cause_key.lower() and "lock" in cause_key.lower():
        opportunity_type = "account_lockout"
    elif "vpn" in cause_key.lower():
        opportunity_type = "vpn_issues"
    elif "onboard" in cause_key.lower() or "hire" in cause_key.lower():
        opportunity_type = "onboarding"
    elif "print" in cause_key.lower():
        opportunity_type = "printer_issues"
    elif "email" in cause_key.lower() or "outlook" in cause_key.lower():
        opportunity_type = "email_issues"
    elif "data" in cause_key.lower() and "fix" in cause_key.lower():
        opportunity_type = "datafix"
    elif ("system" in cause_key.lower() and "down" in cause_key.lower()) or "unavailable" in cause_key.lower():
        opportunity_type = "system_unavailable"
    elif "login" in cause_key.lower():
        opportunity_type = "login_issue"
    elif "error" in cause_key.lower():
        opportunity_type = "error_message"
    elif "hardware" in cause_key.lower():
        opportunity_type = "hardware_issue"
    elif "slow" in cause_key.lower() or "performance" in cause_key.lower():
        opportunity_type = "performance_issue"
    
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
    
    # Calculate varied impact score
    impact_score = calculate_varied_impact_score(
        opportunity_type,
        cause_data['count'],
        cause_data.get('total_tickets', 1000),  # Default estimate if not available
        automation.get('avg_time', 1.0),        # Use average time if available
        automation.get('level', "Medium")       # Use automation level as complexity
    )
    
    # Get specific automation type
    specific_type = determine_specific_automation_type(opportunity_type, cause_data)
    
    # Generate opportunity
    opportunity = {
        "title": f"Automate {cause_data['value']} Resolution",
        "scope": f"Create an automated solution for {cause_data['value']} tickets",
        "justification": f"Analysis identified {cause_data['count']} tickets related to '{cause_data['value']}'. {resolution_text}",
        "type": specific_type,  # Use the specific type instead of generic
        "implementation_plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(automation['implementation_steps'])]),
        "impact_score": impact_score,
        "examples": example_texts[:3]  # Limit to 3 examples
    }
    
    return opportunity





def extract_time_metrics(df, key_concepts, relevant_columns):
    """
    Extract time-related metrics relevant to the question.
    
    Args:
        df: DataFrame with ticket data
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        
    Returns:
        dict: Time metrics
    """
    metrics = {
        "resolution_times": {},
        "time_trends": {},
        "sla_metrics": {}
    }
    
    # Look for resolution time column
    resolution_time_col = None
    for col in df.columns:
        if 'resolution_time' in col.lower() or 'resolve_time' in col.lower():
            resolution_time_col = col
            break
    
    # Look for opened/closed date columns
    opened_col = None
    closed_col = None
    for col in df.columns:
        if 'open' in col.lower() or 'created' in col.lower():
            opened_col = col
        elif 'close' in col.lower() or 'resolved' in col.lower():
            closed_col = col
    
    # Calculate resolution time metrics if available
    if resolution_time_col and resolution_time_col in df.columns:
        # Overall metrics
        resolution_times = df[resolution_time_col].dropna()
        
        if not resolution_times.empty:
            metrics["resolution_times"]["overall"] = {
                "mean": float(resolution_times.mean()),
                "median": float(resolution_times.median()),
                "min": float(resolution_times.min()),
                "max": float(resolution_times.max()),
                "p90": float(resolution_times.quantile(0.9))
            }
        
        # Breakdown by relevant categories
        for col in relevant_columns:
            if col in df.columns and col not in [resolution_time_col, opened_col, closed_col]:
                # Group by this column and get resolution time stats
                grouped = df.groupby(col)[resolution_time_col].agg(['mean', 'median', 'min', 'max'])
                if not grouped.empty:
                    # Get top 3 values with highest mean resolution time
                    top_values = grouped.sort_values('mean', ascending=False).head(3)
                    
                    breakdown = {}
                    for value in top_values.index:
                        breakdown[str(value)] = {
                            "mean": float(top_values.loc[value, 'mean']),
                            "median": float(top_values.loc[value, 'median'])
                        }
                    
                    metrics["resolution_times"][col] = breakdown
    
    # Calculate time trends if date columns are available
    if opened_col and opened_col in df.columns:
        # Ensure column is datetime
        if pd.api.types.is_datetime64_dtype(df[opened_col]):
            # Group by month and count
            try:
                monthly_counts = df.groupby(df[opened_col].dt.to_period('M')).size()
                
                trends = {}
                for period, count in monthly_counts.items():
                    trends[str(period)] = int(count)
                
                metrics["time_trends"]["monthly_volume"] = trends
            except:
                # If datetime operations fail, skip
                pass
    
    # Calculate SLA metrics if relevant columns exist
    sla_col = None
    for col in df.columns:
        if 'sla' in col.lower() or 'target' in col.lower() or 'breach' in col.lower():
            sla_col = col
            break
    
    if sla_col and sla_col in df.columns:
        # Count SLA breaches/compliance
        value_counts = df[sla_col].value_counts()
        
        sla_breakdown = {}
        for value, count in value_counts.items():
            sla_breakdown[str(value)] = int(count)
        
        metrics["sla_metrics"]["breakdown"] = sla_breakdown
    
    return metrics

def extract_category_metrics(df, key_concepts, relevant_columns):
    """
    Extract category-related metrics relevant to the question.
    
    Args:
        df: DataFrame with ticket data
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        
    Returns:
        dict: Category metrics
    """
    metrics = {
        "category_counts": {},
        "subcategory_counts": {},
        "cross_tabulations": {}
    }
    
    # Find category and subcategory columns
    category_cols = []
    for col in df.columns:
        if 'category' in col.lower() or 'type' in col.lower() or 'group' in col.lower():
            category_cols.append(col)
    
    # Get counts for each category column
    for col in category_cols:
        value_counts = df[col].value_counts()
        
        # Store top values
        top_counts = {}
        for value, count in value_counts.head(10).items():
            top_counts[str(value)] = int(count)
        
        # Determine if this is a main category or subcategory
        if 'sub' in col.lower() or len(category_cols) > 1 and col != category_cols[0]:
            metrics["subcategory_counts"][col] = top_counts
        else:
            metrics["category_counts"][col] = top_counts
    
    # Create cross-tabulations between relevant columns
    if len(category_cols) >= 2:
        for i, col1 in enumerate(category_cols[:3]):  # Limit to first 3 to avoid too many combinations
            for col2 in category_cols[i+1:]:
                # Create a cross-tabulation
                cross_tab = pd.crosstab(df[col1], df[col2])
                
                # Convert to a nested dictionary
                cross_dict = {}
                for idx, row in cross_tab.iterrows():
                    cross_dict[str(idx)] = {str(col): int(val) for col, val in row.items()}
                
                metrics["cross_tabulations"][f"{col1}_vs_{col2}"] = cross_dict
    
    # Focus on key concepts if available
    if key_concepts:
        for concept in key_concepts:
            concept_lower = concept.lower()
            
            # Look for categories matching this concept
            for col in category_cols:
                # Filter for values containing the concept
                if df[col].dtype == 'object':  # Only for string columns
                    matching_values = df[df[col].str.lower().str.contains(concept_lower, na=False)]
                    
                    if not matching_values.empty:
                        concept_metrics = {}
                        
                        # Count by category
                        concept_counts = matching_values[col].value_counts()
                        
                        # Store counts
                        concept_counts_dict = {}
                        for value, count in concept_counts.items():
                            concept_counts_dict[str(value)] = int(count)
                        
                        concept_metrics["counts"] = concept_counts_dict
                        
                        # If we have resolution time, get that too
                        resolution_time_col = next((c for c in df.columns if 'resolution_time' in c.lower()), None)
                        if resolution_time_col:
                            concept_metrics["avg_resolution_time"] = float(matching_values[resolution_time_col].mean())
                        
                        metrics[f"{concept}_specific"] = concept_metrics
    
    return metrics

def extract_error_metrics(df, key_concepts, relevant_columns, desc_column):
    """
    Extract error-related metrics from descriptions.
    
    Args:
        df: DataFrame with ticket data
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        desc_column: Description column name
        
    Returns:
        dict: Error metrics
    """
    metrics = {
        "error_counts": {},
        "error_categories": {},
        "resolution_patterns": {},
        "system_specific_errors": {}
    }
    
    if not desc_column or desc_column not in df.columns:
        return metrics  # No description column to analyze
    
    # Define error keywords to look for
    error_keywords = [
        "error", "fail", "exception", "crash", "issue", "problem", 
        "broken", "down", "timeout", "unavailable", "unable",
        "invalid", "incorrect", "not working", "bug"
    ]
    
    # Count occurrences of each error keyword
    error_counts = {}
    for keyword in error_keywords:
        if df[desc_column].dtype == 'object':  # Only for string columns
            count = df[desc_column].str.lower().str.contains(r'\b' + keyword + r'\b', regex=True, na=False).sum()
            if count > 0:
                error_counts[keyword] = int(count)
    
    metrics["error_counts"] = error_counts
    
    # Look for specific error patterns
    error_df = df[df[desc_column].str.lower().str.contains('|'.join(error_keywords), regex=True, na=False)]
    
    # If we have a category column, categorize errors
    category_col = next((col for col in relevant_columns if 'category' in col.lower()), None)
    if category_col and category_col in df.columns:
        category_counts = error_df[category_col].value_counts()
        
        category_breakdown = {}
        for category, count in category_counts.items():
            category_breakdown[str(category)] = int(count)
        
        metrics["error_categories"]["by_category"] = category_breakdown
    
    # Extract system-specific errors if key concepts include system names
    systems = ["database", "server", "network", "application", "system", "software", "hardware", "desktop", "laptop", "mobile"]
    systems.extend([concept for concept in key_concepts if concept.lower() in ["sap", "oracle", "sql", "windows", "office", "excel"]])
    
    for system in systems:
        system_errors = error_df[error_df[desc_column].str.lower().str.contains(system, na=False)]
        
        if not system_errors.empty:
            system_metrics = {
                "count": len(system_errors),
                "percentage": len(system_errors) / len(error_df) * 100 if len(error_df) > 0 else 0
            }
            
            # Get top error types for this system
            for keyword in error_keywords:
                keyword_count = system_errors[desc_column].str.lower().str.contains(r'\b' + keyword + r'\b', regex=True, na=False).sum()
                if keyword_count > 0:
                    system_metrics[f"{keyword}_count"] = int(keyword_count)
            
            metrics["system_specific_errors"][system] = system_metrics
    
    # Find resolution patterns for errors if we have resolution notes
    notes_column = next((col for col in df.columns if 'notes' in col.lower() or 'resolution' in col.lower()), None)
    if notes_column and notes_column in df.columns:
        # Look for common resolution actions
        resolution_actions = ["restart", "reset", "update", "install", "configure", "replace", 
                             "fix", "change", "modify", "add", "remove", "clear", "check"]
        
        resolution_counts = {}
        for action in resolution_actions:
            if error_df[notes_column].dtype == 'object':  # Only for string columns
                count = error_df[notes_column].str.lower().str.contains(r'\b' + action + r'\b', regex=True, na=False).sum()
                if count > 0:
                    resolution_counts[action] = int(count)
        
        metrics["resolution_patterns"]["action_counts"] = resolution_counts
    
    return metrics

def extract_region_metrics(df, key_concepts, relevant_columns):
    """
    Extract region-related metrics from the data.
    
    Args:
        df: DataFrame with ticket data
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        
    Returns:
        dict: Region metrics
    """
    metrics = {
        "region_counts": {},
        "country_counts": {},
        "region_specific_issues": {}
    }
    
    # Find geographic columns
    geo_columns = []
    for col in df.columns:
        if any(term in col.lower() for term in ['region', 'country', 'location', 'geo']):
            geo_columns.append(col)
    
    # Also check subcategory columns which often contain region info
    for col in df.columns:
        if 'subcategory' in col.lower() and col not in geo_columns:
            # Check for region-like values
            if df[col].dtype == 'object':  # Only for string columns
                region_terms = ['amer', 'emea', 'apac', 'us', 'uk', 'eu', 'asia', 'america', 'europe']
                has_regions = any(df[col].str.lower().str.contains('|'.join(region_terms), regex=True, na=False))
                if has_regions:
                    geo_columns.append(col)
    
    # Get counts for each geographic column
    for col in geo_columns:
        value_counts = df[col].value_counts()
        
        # Determine if this is a region or country column
        if 'region' in col.lower() or any(df[col].astype(str).str.lower().str.contains('|'.join(['amer', 'emea', 'apac']), regex=True).any()):
            # Region column
            region_counts = {}
            for region, count in value_counts.items():
                region_counts[str(region)] = int(count)
            
            metrics["region_counts"][col] = region_counts
        else:
            # Country column
            country_counts = {}
            for country, count in value_counts.head(10).items():  # Top 10 countries
                country_counts[str(country)] = int(count)
            
            metrics["country_counts"][col] = country_counts
    
    # Find region-specific issues
    if geo_columns and 'category' in df.columns:
        for geo_col in geo_columns:
            # Group by region and category
            region_issues = {}
            
            for region, region_df in df.groupby(geo_col):
                # Get category breakdown
                cat_counts = region_df['category'].value_counts()
                
                # Store top categories
                top_cats = {}
                for cat, count in cat_counts.head(3).items():
                    top_cats[str(cat)] = int(count)
                
                # Store in metrics if we have results
                if top_cats:
                    region_issues[str(region)] = top_cats
            
            if region_issues:
                metrics["region_specific_issues"][geo_col] = region_issues
    
    # If key concepts contain specific regions, analyze those
    if key_concepts:
        for concept in key_concepts:
            concept_lower = concept.lower()
            
            # Check if this concept is a region
            if concept_lower in ['amer', 'emea', 'apac', 'americas', 'europe', 'asia', 'global']:
                # Find data for this region
                region_data = None
                
                for geo_col in geo_columns:
                    if df[geo_col].dtype == 'object':  # Only for string columns
                        matching = df[df[geo_col].str.lower().str.contains(concept_lower, na=False)]
                        
                        if not matching.empty:
                            region_data = matching
                            break
                
                if region_data is not None:
                    # Calculate metrics for this specific region
                    region_specific = {
                        "count": len(region_data),
                        "percentage": len(region_data) / len(df) * 100
                    }
                    
                    # Get top categories
                    if 'category' in df.columns:
                        cat_counts = region_data['category'].value_counts()
                        top_cats = {}
                        for cat, count in cat_counts.head(3).items():
                            top_cats[str(cat)] = int(count)
                        
                        region_specific["top_categories"] = top_cats
                    
                    # Get average resolution time
                    resolution_time_col = next((c for c in df.columns if 'resolution_time' in c.lower()), None)
                    if resolution_time_col:
                        region_specific["avg_resolution_time"] = float(region_data[resolution_time_col].mean())
                    
                    metrics[f"{concept}_specific"] = region_specific
    
    return metrics

def segment_data_for_question(df, question_type, key_concepts, relevant_columns):
    """
    Segment the data based on the question focus.
    
    Args:
        df: DataFrame with ticket data
        question_type: Type of question being asked
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        
    Returns:
        list: Segmented data groups
    """
    segments = []
    
    # Different segmentation strategies based on question type
    if question_type == "VOLUME":
        # Segment by top categories
        for col in relevant_columns:
            if col in df.columns:
                top_values = df[col].value_counts().head(3).index.tolist()
                
                for value in top_values:
                    segment = {
                        "name": f"{col}: {value}",
                        "filter": f"{col} == '{value}'",
                        "count": int(df[df[col] == value].shape[0]),
                        "percentage": float(df[df[col] == value].shape[0] / len(df) * 100)
                    }
                    segments.append(segment)
    
    elif question_type == "TIME":
        # Segment by resolution time quartiles
        resolution_time_col = next((c for c in df.columns if 'resolution_time' in c.lower()), None)
        if resolution_time_col and resolution_time_col in df.columns:
            # Calculate quartiles
            resolution_times = df[resolution_time_col].dropna()
            if not resolution_times.empty:
                quartiles = [
                    resolution_times.quantile(0.25),
                    resolution_times.quantile(0.5),
                    resolution_times.quantile(0.75)
                ]
                
                # Create segments for each quartile
                segments.append({
                    "name": "Fast resolution (Q1)",
                    "filter": f"{resolution_time_col} <= {quartiles[0]}",
                    "count": int(df[df[resolution_time_col] <= quartiles[0]].shape[0]),
                    "percentage": float(df[df[resolution_time_col] <= quartiles[0]].shape[0] / len(df) * 100)
                })
                
                segments.append({
                    "name": "Medium resolution (Q2)",
                    "filter": f"{quartiles[0]} < {resolution_time_col} <= {quartiles[1]}",
                    "count": int(df[(df[resolution_time_col] > quartiles[0]) & (df[resolution_time_col] <= quartiles[1])].shape[0]),
                    "percentage": float(df[(df[resolution_time_col] > quartiles[0]) & (df[resolution_time_col] <= quartiles[1])].shape[0] / len(df) * 100)
                })
                
                segments.append({
                    "name": "Slow resolution (Q3)",
                    "filter": f"{quartiles[1]} < {resolution_time_col} <= {quartiles[2]}",
                    "count": int(df[(df[resolution_time_col] > quartiles[1]) & (df[resolution_time_col] <= quartiles[2])].shape[0]),
                    "percentage": float(df[(df[resolution_time_col] > quartiles[1]) & (df[resolution_time_col] <= quartiles[2])].shape[0] / len(df) * 100)
                })
                
                segments.append({
                    "name": "Very slow resolution (Q4)",
                    "filter": f"{resolution_time_col} > {quartiles[2]}",
                    "count": int(df[df[resolution_time_col] > quartiles[2]].shape[0]),
                    "percentage": float(df[df[resolution_time_col] > quartiles[2]].shape[0] / len(df) * 100)
                })
    
    elif question_type == "CATEGORY":
        # Segment by main categories and their top subcategories
        category_col = next((col for col in df.columns if 'category' in col.lower() and 'sub' not in col.lower()), None)
        subcategory_col = next((col for col in df.columns if 'subcategory' in col.lower()), None)
        
        if category_col and category_col in df.columns:
            top_categories = df[category_col].value_counts().head(3).index.tolist()
            
            for category in top_categories:
                segment = {
                    "name": f"Category: {category}",
                    "filter": f"{category_col} == '{category}'",
                    "count": int(df[df[category_col] == category].shape[0]),
                    "percentage": float(df[df[category_col] == category].shape[0] / len(df) * 100)
                }
                
                # Add subcategories if available
                if subcategory_col and subcategory_col in df.columns:
                    category_df = df[df[category_col] == category]
                    top_subcats = category_df[subcategory_col].value_counts().head(2).index.tolist()
                    
                    sub_segments = []
                    for subcat in top_subcats:
                        sub_segment = {
                            "name": f"Subcategory: {subcat}",
                            "filter": f"{subcategory_col} == '{subcat}'",
                            "count": int(category_df[category_df[subcategory_col] == subcat].shape[0]),
                            "percentage": float(category_df[category_df[subcategory_col] == subcat].shape[0] / len(category_df) * 100)
                        }
                        sub_segments.append(sub_segment)
                    
                    segment["subcategories"] = sub_segments
                
                segments.append(segment)
    
    elif question_type == "ERROR":
        # Segment by error types
        desc_column = next((col for col in df.columns if 'description' in col.lower()), None)
        
        if desc_column and desc_column in df.columns:
            # Define error keywords
            error_keywords = ["error", "fail", "exception", "crash", "issue", "problem"]
            
            for keyword in error_keywords:
                if df[desc_column].dtype == 'object':  # Only for string columns
                    matching = df[df[desc_column].str.lower().str.contains(r'\b' + keyword + r'\b', regex=True, na=False)]
                    
                    if not matching.empty:
                        segment = {
                            "name": f"{keyword.capitalize()} tickets",
                            "filter": f"{desc_column}.str.contains('{keyword}')",
                            "count": len(matching),
                            "percentage": float(len(matching) / len(df) * 100)
                        }
                        segments.append(segment)
            
            # Also segment by system/component if mentioned in key concepts
            systems = ["database", "server", "network", "application", "system", "software", "hardware"]
            systems.extend([concept for concept in key_concepts if concept.lower() in ["sap", "oracle", "sql", "windows", "office", "excel"]])
            
            for system in systems:
                matching = df[df[desc_column].str.lower().str.contains(system, na=False)]
                
                if not matching.empty:
                    error_matching = matching[matching[desc_column].str.lower().str.contains('|'.join(error_keywords), regex=True, na=False)]
                    
                    if not error_matching.empty:
                        segment = {
                            "name": f"{system.capitalize()} errors",
                            "filter": f"{desc_column}.str.contains('{system}') & {desc_column}.str.contains('|'.join({error_keywords}))",
                            "count": len(error_matching),
                            "percentage": float(len(error_matching) / len(df) * 100)
                        }
                        segments.append(segment)
    
    elif question_type == "REGION":
        # Segment by regions
        geo_columns = [col for col in df.columns if any(term in col.lower() for term in ['region', 'country', 'location', 'geo'])]
        
        # Also check subcategory columns for regions
        for col in df.columns:
            if 'subcategory' in col.lower() and col not in geo_columns:
                if df[col].dtype == 'object':  # Only for string columns
                    region_terms = ['amer', 'emea', 'apac', 'us', 'uk', 'eu', 'asia', 'america', 'europe']
                    has_regions = any(df[col].str.lower().str.contains('|'.join(region_terms), regex=True, na=False))
                    if has_regions:
                        geo_columns.append(col)
        
        for geo_col in geo_columns:
            top_regions = df[geo_col].value_counts().head(5).index.tolist()
            
            for region in top_regions:
                segment = {
                    "name": f"{geo_col}: {region}",
                    "filter": f"{geo_col} == '{region}'",
                    "count": int(df[df[geo_col] == region].shape[0]),
                    "percentage": float(df[df[geo_col] == region].shape[0] / len(df) * 100)
                }
                segments.append(segment)
    
    # If we have key concepts, also segment by them
    if key_concepts:
        desc_column = next((col for col in df.columns if 'description' in col.lower()), None)
        
        if desc_column and desc_column in df.columns:
            for concept in key_concepts:
                if df[desc_column].dtype == 'object':  # Only for string columns
                    matching = df[df[desc_column].str.lower().str.contains(concept.lower(), na=False)]
                    
                    if not matching.empty:
                        segment = {
                            "name": f"{concept.capitalize()} mentioned",
                            "filter": f"{desc_column}.str.contains('{concept}')",
                            "count": len(matching),
                            "percentage": float(len(matching) / len(df) * 100)
                        }
                        segments.append(segment)
    
    return segments



def classify_question(question):
    """
    Classify the question type and extract key concepts.
    
    Args:
        question: The question text
        
    Returns:
        tuple: Question type and key concepts
    """
    question_lower = question.lower()
    
    # Define question type patterns
    patterns = {
        "VOLUME": ["how many", "count", "volume", "number of", "most common"],
        "TIME": ["how long", "time", "duration", "average time", "resolution time"],
        "CATEGORY": ["which category", "what type", "categories", "types of"],
        "ERROR": ["error", "issue", "problem", "fail", "bug"],
        "REGION": ["country", "region", "location", "geographic", "where"]
    }
    
    # Identify question type
    question_type = "GENERAL"
    for qtype, keywords in patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            question_type = qtype
            break
    
    # Extract key concepts based on question type
    key_concepts = []
    
    # Look for specific entities based on question type
    if question_type == "VOLUME":
        # Look for categories, issues, etc.
        for category in ["password", "access", "software", "hardware", "network", "data", "email"]:
            if category in question_lower:
                key_concepts.append(category)
    
    elif question_type == "TIME":
        # Look for resolution mentions, categories
        for term in ["resolution", "response", "fix", "solve"]:
            if term in question_lower:
                key_concepts.append(term)
    
    elif question_type == "CATEGORY":
        # Look for specific categories or issues
        for category in ["incident", "request", "problem", "change", "high priority", "critical"]:
            if category in question_lower:
                key_concepts.append(category)
    
    elif question_type == "ERROR":
        # Look for specific error types
        for error in ["system", "application", "login", "connection", "data", "security"]:
            if error in question_lower:
                key_concepts.append(error)
    
    elif question_type == "REGION":
        # Look for specific regions
        for region in ["north", "south", "east", "west", "america", "europe", "asia", "global"]:
            if region in question_lower:
                key_concepts.append(region)
    
    # If no specific concepts found, extract nouns as key concepts
    if not key_concepts:
        import re
        # Simple noun extraction - looking for capitalized words or words following "the", "a", "an"
        nouns = re.findall(r'\b([A-Z][a-z]+|\b(?:the|a|an)\s+(\w+))', question)
        key_concepts = [n[0] if isinstance(n, tuple) else n for n in nouns]
    
    return question_type, key_concepts

def identify_relevant_columns(df, question_type, key_concepts):
    """
    Identify columns in the dataframe that are relevant to the question.
    
    Args:
        df: DataFrame with ticket data
        question_type: Type of question being asked
        key_concepts: Key concepts from the question
        
    Returns:
        list: Relevant column names
    """
    relevant_columns = []
    
    # Map question types to potential column patterns
    column_patterns = {
        "VOLUME": ["category", "type", "group", "priority"],
        "TIME": ["time", "duration", "date", "created", "resolved", "closed", "open"],
        "CATEGORY": ["category", "subcategory", "type", "group", "classification"],
        "ERROR": ["description", "short_description", "notes", "work_notes", "close_notes"],
        "REGION": ["region", "country", "location", "site", "geo", "subcategory"]
    }
    
    # Get potential column patterns for this question type
    patterns = column_patterns.get(question_type, [])
    
    # Find columns matching the patterns
    for col in df.columns:
        col_lower = col.lower()
        # Check if column matches any pattern
        if any(pattern in col_lower for pattern in patterns):
            relevant_columns.append(col)
    
    # If key concepts are available, find more specific columns
    if key_concepts:
        for concept in key_concepts:
            concept_lower = concept.lower()
            for col in df.columns:
                col_lower = col.lower()
                if concept_lower in col_lower and col not in relevant_columns:
                    relevant_columns.append(col)
    
    # Always include essential columns if they exist
    essential_columns = ["priority", "category", "state", "opened", "closed"]
    for col in essential_columns:
        if col in df.columns and col not in relevant_columns:
            relevant_columns.append(col)
    
    return relevant_columns

def extract_volume_metrics(df, key_concepts, relevant_columns):
    """
    Extract volume-related metrics relevant to the question.
    
    Args:
        df: DataFrame with ticket data
        key_concepts: Key concepts from the question
        relevant_columns: Relevant columns for analysis
        
    Returns:
        dict: Volume metrics
    """
    metrics = {
        "total_count": len(df),
        "breakdowns": {},
        "top_values": {}
    }
    
    # Check if we should focus on specific concepts
    focused_columns = []
    if key_concepts:
        # Find columns related to key concepts
        for concept in key_concepts:
            concept_lower = concept.lower()
            for col in relevant_columns:
                if concept_lower in col.lower():
                    focused_columns.append(col)
    
    # If no focused columns found, use all relevant columns
    if not focused_columns:
        focused_columns = relevant_columns
    
    # Get breakdowns for each focused column
    for col in focused_columns:
        if col in df.columns:
            # Get value counts
            value_counts = df[col].value_counts()
            
            # Store breakdown (for top 5 values)
            breakdown = {}
            for value, count in value_counts.head(5).items():
                breakdown[str(value)] = int(count)
            
            metrics["breakdowns"][col] = breakdown
            
            # Store top value
            if not value_counts.empty:
                top_value = value_counts.index[0]
                metrics["top_values"][col] = {
                    "value": str(top_value),
                    "count": int(value_counts.iloc[0]),
                    "percentage": float(value_counts.iloc[0] / len(df) * 100)
                }
    
    return metrics




def calculate_varied_impact_score(opportunity_type, count, total_tickets, avg_time=1.0, complexity="Medium"):
    """
    Calculate a varied impact score based on multiple factors.
    
    Args:
        opportunity_type: Type of opportunity (password, access, etc.)
        count: Number of tickets affected
        total_tickets: Total number of tickets
        avg_time: Average resolution time in hours
        complexity: Implementation complexity
        
    Returns:
        float: Impact score (0-100)
    """
    # Base factors
    percentage = (count / total_tickets) * 100 if total_tickets > 0 else 0
    time_factor = min(avg_time / 2.0, 1.0)  # Cap at 1.0
    
    # Base score calculation (volume and time impact)
    base_score = (percentage * 0.6) + (time_factor * 30)
    
    # Opportunity type multipliers - some issues have higher business impact
    type_multipliers = {
        "password_reset": 1.1,     # High volume, low complexity
        "access_request": 1.3,     # Security impact, compliance requirements
        "software_installation": 0.9,  # Common but less critical
        "account_lockout": 1.2,    # High urgency
        "data_export": 1.0,        # Medium impact
        "vpn_issues": 1.25,        # Remote work impact
        "onboarding": 1.35,        # Business process impact
        "printer_issues": 0.8,     # Lower business impact
        "password_expiry": 1.05,   # Predictable issue
        "email_issues": 1.15,      # Productivity impact
        "datafix": 1.2,            # Data integrity impact
        "system_unavailable": 1.4, # High business impact
        "login_issue": 1.1,        # Productivity impact
        "error_message": 0.95,     # Generic issue
        "hardware_issue": 1.05,    # Physical dependency
        "performance_issue": 1.15  # Productivity impact
    }
    
    # Default multiplier for unmapped types
    type_multiplier = type_multipliers.get(opportunity_type, 1.0)
    
    # Complexity adjustment
    complexity_factors = {
        "Low": 1.1,     # Easier to implement = higher impact score
        "Medium": 1.0,  # Neutral
        "High": 0.9     # Harder to implement = lower impact score
    }
    complexity_factor = complexity_factors.get(complexity, 1.0)
    
    # Additional factors - random variation to avoid identical scores
    import random
    variation = random.uniform(0.95, 1.05)
    
    # Calculate final score
    final_score = base_score * type_multiplier * complexity_factor * variation
    
    # Cap at 100
    final_score = min(final_score, 100)
    
    # Ensure minimum score of 30 if there are any tickets
    if count > 0:
        final_score = max(final_score, 30)
    
    return round(final_score, 1)  # Round to 1 decimal place



def enhance_common_issues_answer(processed_data, question, answer):
    """
    Enhance the common issues answer with deep data analysis and specific automation recommendations.
    
    Args:
        processed_data: DataFrame with processed ticket data
        question: The question being asked
        answer: Current answer
        
    Returns:
        dict: Enhanced answer with specific examples and recommendations
    """
    # Perform deep analysis
    deep_patterns = deep_analyze_ticket_content(processed_data, "errors")
    
    # Extract top error patterns
    error_patterns = deep_patterns.get("error_patterns", {})
    system_patterns = deep_patterns.get("system_patterns", {})
    specific_issues = deep_patterns.get("specific_issues", [])
    
    # Sort error patterns by count
    sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1].get("count", 0), reverse=True)
    
    # Sort system patterns by count
    sorted_systems = sorted(system_patterns.items(), key=lambda x: x[1].get("count", 0), reverse=True)
    
    # Sort specific issues by count
    sorted_issues = sorted(specific_issues, key=lambda x: x.get("count", 0), reverse=True)
    
    # Create enhanced answer
    enhanced_answer = answer + "\n\n**Detailed Analysis:**\n\n"
    
    # Add specific error patterns
    if sorted_errors:
        enhanced_answer += "**Specific Error Patterns:**\n\n"
        for term, data in sorted_errors[:5]:  # Top 5 error patterns
            enhanced_answer += f"**{term.upper()} issues ({data.get('count', 0)} tickets)**\n\n"
            
            # Add common patterns if available
            common_patterns = data.get("common_patterns", [])
            if common_patterns:
                enhanced_answer += "Common contexts:\n"
                for pattern in common_patterns[:3]:  # Top 3 patterns
                    enhanced_answer += f"- \"{pattern.get('pattern', '')}\" ({pattern.get('count', 0)} occurrences)\n"
            
            # Add example
            if "examples" in data and data["examples"]:
                enhanced_answer += f"\nExample: \"{data['examples'][0]}\"\n\n"
            
            # Add common resolutions if available
            if "common_resolutions" in data and data["common_resolutions"]:
                enhanced_answer += "Common resolutions:\n"
                for res in data["common_resolutions"][:3]:  # Top 3 resolutions
                    enhanced_answer += f"- {res.get('action', '')} ({res.get('count', 0)} instances)\n"
            
            enhanced_answer += "\n"
    
    # Add specific systems involved
    if sorted_systems:
        enhanced_answer += "**Top Systems Mentioned:**\n\n"
        for system, data in sorted_systems[:5]:  # Top 5 systems
            enhanced_answer += f"- **{system}**: {data.get('count', 0)} tickets\n"
            if "examples" in data and data["examples"]:
                enhanced_answer += f"  Example: \"{data['examples'][0]}\"\n"
        
        enhanced_answer += "\n"
    
    # Add specific issue n-grams
    if sorted_issues:
        enhanced_answer += "**Specific Issue Phrases:**\n\n"
        for issue in sorted_issues[:5]:  # Top 5 specific issues
            enhanced_answer += f"- \"{issue.get('pattern', '')}\" ({issue.get('count', 0)} occurrences)\n"
            if "examples" in issue and issue["examples"]:
                enhanced_answer += f"  Example: \"{issue['examples'][0]}\"\n"
        
        enhanced_answer += "\n"
    
    # Generate automation recommendations
    automation_potential = "Based on detailed analysis of the ticket data, here are specific automation opportunities:\n\n"
    
    recommendations = []
    
    # Add recommendations for top error patterns
    for term, data in sorted_errors[:3]:  # Top 3 error patterns
        if data.get("count", 0) >= 10:  # Only for significant patterns
            recommendation = generate_automation_recommendation("error", data)
            
            # Create recommendation text
            rec_text = f"**{recommendation['approach']}**\n\n"
            rec_text += f"*For {term.upper()} issues ({data.get('count', 0)} tickets)*\n\n"
            
            rec_text += "**Technologies:**\n"
            for tech in recommendation["technologies"]:
                rec_text += f"- {tech}\n"
            
            rec_text += "\n**Implementation Steps:**\n"
            for step in recommendation["implementation_steps"]:
                rec_text += f"- {step}\n"
            
            rec_text += "\n**Benefits:**\n"
            for benefit in recommendation["benefits"]:
                rec_text += f"- {benefit}\n"
            
            rec_text += f"\n**Complexity:** {recommendation['complexity']} | **ROI:** {recommendation['roi']}\n\n"
            
            recommendations.append(rec_text)
    
    # Add recommendations for top systems
    for system, data in sorted_systems[:2]:  # Top 2 systems
        if data.get("count", 0) >= 10:  # Only for significant patterns
            system_data = {
                "system": system,
                "count": data.get("count", 0)
            }
            recommendation = generate_automation_recommendation("system", system_data)
            
            # Create recommendation text
            rec_text = f"**{recommendation['approach']}**\n\n"
            rec_text += f"*For {system} issues ({data.get('count', 0)} tickets)*\n\n"
            
            rec_text += "**Technologies:**\n"
            for tech in recommendation["technologies"]:
                rec_text += f"- {tech}\n"
            
            rec_text += "\n**Implementation Steps:**\n"
            for step in recommendation["implementation_steps"]:
                rec_text += f"- {step}\n"
            
            rec_text += "\n**Benefits:**\n"
            for benefit in recommendation["benefits"]:
                rec_text += f"- {benefit}\n"
            
            rec_text += f"\n**Complexity:** {recommendation['complexity']} | **ROI:** {recommendation['roi']}\n\n"
            
            recommendations.append(rec_text)
            
    # Add recommendations from specific issue patterns if any are substantial
    for issue in sorted_issues[:1]:  # Top specific issue
        if issue.get("count", 0) >= 15:  # Only for significant patterns
            pattern = issue.get("pattern", "")
            
            # Determine most likely type based on pattern
            issue_data = {
                "term": pattern,
                "count": issue.get("count", 0),
                "common_patterns": []
            }
            
            recommendation = generate_automation_recommendation("error", issue_data)
            
            # Create recommendation text
            rec_text = f"**{recommendation['approach']}**\n\n"
            rec_text += f"*For \"{pattern}\" issues ({issue.get('count', 0)} tickets)*\n\n"
            
            rec_text += "**Technologies:**\n"
            for tech in recommendation["technologies"]:
                rec_text += f"- {tech}\n"
            
            rec_text += "\n**Implementation Steps:**\n"
            for step in recommendation["implementation_steps"]:
                rec_text += f"- {step}\n"
            
            rec_text += "\n**Benefits:**\n"
            for benefit in recommendation["benefits"]:
                rec_text += f"- {benefit}\n"
            
            rec_text += f"\n**Complexity:** {recommendation['complexity']} | **ROI:** {recommendation['roi']}\n\n"
            
            recommendations.append(rec_text)
            
    # Combine all recommendations
    if recommendations:
        automation_potential += "\n\n".join(recommendations)
    else:
        automation_potential += "No clear automation opportunities were identified from the available data. Consider gathering more detailed information about error types and resolution methods."
    
    # Create final enhanced answer
    enhanced = {
        "answer": enhanced_answer,
        "automation_potential": automation_potential,
        "examples": []  # Examples are already included in the answer
    }
    
    return enhanced


def enhance_question_answer(processed_data, question, answer):
    """
    Enhance a question answer with specific data examples and automation potential.
    Routes to appropriate analysis function based on question type.
    
    Args:
        processed_data: DataFrame with processed ticket data
        question: The question being asked
        answer: Current answer
        
    Returns:
        dict: Enhanced answer with specific automation potential
    """
    # Check if this is the common issues question
    if 'common issues' in question.lower():
        # Use special deep analysis for common issues
        return enhance_common_issues_answer(processed_data, question, answer)
    
    # For other questions, use the standard approach
    context = analyze_data_for_question_context(processed_data, question)
    
    # Generate specific automation potential
    specific_automation = generate_specific_automation_potential(question, context["patterns"])
    
    # Create enhanced answer
    enhanced = {
        "answer": answer,
        "automation_potential": specific_automation,
        "examples": context["examples"][:3]  # Limit to 3 examples
    }
    
    return enhanced




####################################################### Data based Question answer ###################################

def analyze_data_for_question_context(processed_data, question):
    """
    Extract specific data examples relevant to a question to provide context.
    
    Args:
        processed_data: DataFrame with processed ticket data
        question: The specific question being asked
        
    Returns:
        dict: Context with examples and patterns
    """
    context = {
        "examples": [],
        "patterns": {}
    }
    
    # Identify key terms in the question
    question_lower = question.lower()
    
    # Find description column
    desc_column = None
    for col in ['short description', 'description', 'short_description']:
        if col in processed_data.columns:
            desc_column = col
            break
    
    # Find notes column
    notes_column = None
    for col in ['close notes', 'closed notes', 'resolution notes', 'work notes', 'close_notes']:
        if col in processed_data.columns:
            notes_column = col
            break
    
    # Find matching tickets based on the question
    matching_tickets = []
    
    # If asking about categories that consume the most hours
    if 'consume' in question_lower and 'hours' in question_lower:
        # Check if we have necessary columns
        if 'category' in processed_data.columns and 'resolution_time_hours' in processed_data.columns:
            # Group by category and get average resolution time
            category_time = processed_data.groupby('category')['resolution_time_hours'].agg(['mean', 'count'])
            # Sort by mean time
            category_time = category_time.sort_values('mean', ascending=False)
            
            # Get top categories by time
            for category, data in category_time.head(3).iterrows():
                # Find example tickets from this category
                cat_tickets = processed_data[processed_data['category'] == category].nlargest(2, 'resolution_time_hours')
                
                for _, ticket in cat_tickets.iterrows():
                    ticket_dict = {
                        "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID',
                        "category": category,
                        "resolution_time": f"{ticket['resolution_time_hours']:.2f} hours"
                    }
                    
                    if desc_column:
                        ticket_dict["description"] = str(ticket[desc_column])[:100]
                    
                    if notes_column and notes_column in ticket:
                        ticket_dict["resolution"] = str(ticket[notes_column])[:150]
                    
                    matching_tickets.append(ticket_dict)
                
                # Add pattern
                context["patterns"][f"category_{category}"] = {
                    "category": category,
                    "avg_time": f"{data['mean']:.2f} hours",
                    "count": int(data['count']),
                    "automation_potential": "High" if data['mean'] > 4 and data['count'] > 10 else "Medium"
                }
    
    # If asking about datafix
    elif 'datafix' in question_lower:
        if desc_column:
            # Find tickets mentioning datafix
            datafix_keywords = ['datafix', 'data fix', 'fix data', 'database correction']
            
            for keyword in datafix_keywords:
                matches = processed_data[desc_column].str.contains(keyword, case=False, na=False)
                if matches.sum() > 0:
                    datafix_tickets = processed_data[matches].head(3)
                    
                    for _, ticket in datafix_tickets.iterrows():
                        ticket_dict = {
                            "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                        }
                        
                        if desc_column:
                            ticket_dict["description"] = str(ticket[desc_column])[:100]
                        
                        if notes_column and notes_column in ticket:
                            ticket_dict["resolution"] = str(ticket[notes_column])[:150]
                        
                        if 'category' in processed_data.columns:
                            ticket_dict["category"] = ticket['category']
                        
                        matching_tickets.append(ticket_dict)
                    
                    # Add pattern
                    if 'category' in processed_data.columns:
                        # Group by category
                        datafix_by_category = processed_data[matches].groupby('category').size()
                        for category, count in datafix_by_category.items():
                            context["patterns"][f"datafix_{category}"] = {
                                "category": category,
                                "keyword": keyword,
                                "count": int(count),
                                "automation_potential": "High" if count > 5 else "Medium"
                            }
                    else:
                        context["patterns"]["datafix_general"] = {
                            "keyword": keyword,
                            "count": int(matches.sum()),
                            "automation_potential": "High" if matches.sum() > 10 else "Medium"
                        }
    
    # If asking about country-specific issues
    elif 'country' in question_lower or 'region' in question_lower:
        # Look for country/region columns
        geo_columns = []
        
        for col in processed_data.columns:
            if any(term in col.lower() for term in ['country', 'region', 'location', 'geo']):
                geo_columns.append(col)
        
        # Also check subcategory columns which often contain region info
        for col in processed_data.columns:
            if 'subcategory' in col.lower() and col not in geo_columns:
                geo_columns.append(col)
        
        # For each geo column, find patterns
        for geo_col in geo_columns:
            # Check for values that look like regions
            region_keywords = ['amer', 'emea', 'apac', 'us', 'uk', 'europe', 'asia', 'australia']
            
            # Get values and counts
            value_counts = processed_data[geo_col].value_counts()
            
            for region, count in value_counts.items():
                if count < 3:  # Skip regions with very few tickets
                    continue
                    
                region_str = str(region).lower()
                if any(keyword in region_str for keyword in region_keywords):
                    # Find example tickets from this region
                    region_tickets = processed_data[processed_data[geo_col] == region].head(2)
                    
                    for _, ticket in region_tickets.iterrows():
                        ticket_dict = {
                            "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID',
                            "region": region,
                            "region_column": geo_col
                        }
                        
                        if desc_column:
                            ticket_dict["description"] = str(ticket[desc_column])[:100]
                        
                        if 'category' in processed_data.columns:
                            ticket_dict["category"] = ticket['category']
                        
                        matching_tickets.append(ticket_dict)
                    
                    # Find common issues in this region
                    if 'category' in processed_data.columns:
                        region_df = processed_data[processed_data[geo_col] == region]
                        category_counts = region_df['category'].value_counts()
                        
                        for category, cat_count in category_counts.head(2).items():
                            context["patterns"][f"region_{region}_{category}"] = {
                                "region": region,
                                "region_column": geo_col,
                                "category": category,
                                "count": int(cat_count),
                                "total_region_tickets": int(count),
                                "automation_potential": "High" if cat_count > 5 else "Medium"
                            }
    
    # If asking about escalation
    elif 'escalation' in question_lower or 'escalate' in question_lower:
        # Find tickets mentioning escalation
        escalation_keywords = ['escalate', 'escalation', 'elevate', 'priority increase']
        
        # Check both description and notes
        for keyword in escalation_keywords:
            # Check description
            if desc_column:
                desc_matches = processed_data[desc_column].str.contains(keyword, case=False, na=False)
                
                if desc_matches.sum() > 0:
                    for _, ticket in processed_data[desc_matches].head(2).iterrows():
                        ticket_dict = {
                            "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID',
                            "source": "description"
                        }
                        
                        if desc_column:
                            ticket_dict["description"] = str(ticket[desc_column])[:100]
                        
                        if 'category' in processed_data.columns:
                            ticket_dict["category"] = ticket['category']
                        
                        matching_tickets.append(ticket_dict)
            
            # Check notes
            if notes_column:
                notes_matches = processed_data[notes_column].str.contains(keyword, case=False, na=False)
                
                if notes_matches.sum() > 0:
                    for _, ticket in processed_data[notes_matches].head(2).iterrows():
                        ticket_dict = {
                            "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID',
                            "source": "notes"
                        }
                        
                        if desc_column:
                            ticket_dict["description"] = str(ticket[desc_column])[:100]
                        
                        if notes_column:
                            ticket_dict["notes"] = str(ticket[notes_column])[:150]
                        
                        if 'category' in processed_data.columns:
                            ticket_dict["category"] = ticket['category']
                        
                        matching_tickets.append(ticket_dict)
                
                # Analyze escalation patterns by category
                if notes_matches.sum() > 0 and 'category' in processed_data.columns:
                    category_counts = processed_data[notes_matches]['category'].value_counts()
                    
                    for category, count in category_counts.head(3).items():
                        context["patterns"][f"escalation_{category}"] = {
                            "category": category,
                            "count": int(count),
                            "keyword": keyword,
                            "automation_potential": "High" if count > 5 else "Medium"
                        }
    
    # If asking about document failures
    elif 'document' in question_lower and ('fail' in question_lower or 'failure' in question_lower):
        # Look for document failure keywords
        doc_keywords = ['document failure', 'doc fail', 'report failure', 'failed document', 'failed report']
        
        if desc_column:
            for keyword in doc_keywords:
                matches = processed_data[desc_column].str.contains(keyword, case=False, na=False)
                
                if matches.sum() > 0:
                    for _, ticket in processed_data[matches].head(3).iterrows():
                        ticket_dict = {
                            "id": ticket.get('number', '') or ticket.get('id', '') or 'TICKET-ID'
                        }
                        
                        if desc_column:
                            ticket_dict["description"] = str(ticket[desc_column])[:100]
                        
                        if 'category' in processed_data.columns:
                            ticket_dict["category"] = ticket['category']
                        
                        if 'opened' in processed_data.columns:
                            ticket_dict["date"] = str(ticket['opened'])
                        
                        matching_tickets.append(ticket_dict)
                    
                    # Check if we can group by year
                    if 'opened' in processed_data.columns and hasattr(processed_data['opened'], 'dt'):
                        # Group by year
                        matches_df = processed_data[matches]
                        if hasattr(matches_df['opened'], 'dt'):
                            year_counts = matches_df['opened'].dt.year.value_counts()
                            
                            for year, count in year_counts.items():
                                context["patterns"][f"doc_failure_{year}"] = {
                                    "year": int(year),
                                    "count": int(count),
                                    "keyword": keyword,
                                    "automation_potential": "High" if count > 5 else "Medium"
                                }
    
    # Add matching tickets to context
    context["examples"] = matching_tickets
    
    return context

def generate_specific_automation_potential(question, patterns):
    """
    Generate specific automation potential based on identified patterns.
    
    Args:
        question: The question being asked
        patterns: Patterns identified in the data
        
    Returns:
        str: Specific automation potential
    """
    question_lower = question.lower()
    
    if not patterns:
        return "Unable to determine specific automation potential due to limited data patterns."
    
    # Start with an empty potential
    automation_potential = "Based on analysis of the ticket data:\n\n"
    
    # For category time consumption
    if 'consume' in question_lower and 'hours' in question_lower:
        high_potential_categories = []
        medium_potential_categories = []
        
        for pattern_key, pattern in patterns.items():
            if pattern_key.startswith('category_'):
                if pattern['automation_potential'] == "High":
                    high_potential_categories.append({
                        "name": pattern['category'],
                        "time": pattern['avg_time'],
                        "count": pattern['count']
                    })
                else:
                    medium_potential_categories.append({
                        "name": pattern['category'],
                        "time": pattern['avg_time'],
                        "count": pattern['count']
                    })
        
        if high_potential_categories:
            automation_potential += "**High Automation Potential:**\n"
            for cat in high_potential_categories:
                automation_potential += f"- {cat['name']} incidents ({cat['time']} avg. resolution time, {cat['count']} tickets): "
                automation_potential += f"Implement standardized workflow automation with pre-defined diagnostic steps and resolution paths. "
                automation_potential += f"This could reduce resolution time by 40-60% through eliminating manual diagnosis steps.\n"
        
        if medium_potential_categories:
            automation_potential += "\n**Medium Automation Potential:**\n"
            for cat in medium_potential_categories:
                automation_potential += f"- {cat['name']} incidents: Partial automation through guided troubleshooting tools and knowledge base integration.\n"
        
        if not high_potential_categories and not medium_potential_categories:
            automation_potential += "No clear automation potential identified based on resolution time analysis."
    
    # For datafix questions
    elif 'datafix' in question_lower:
        datafix_patterns = {}
        
        for pattern_key, pattern in patterns.items():
            if pattern_key.startswith('datafix_'):
                if 'category' in pattern:
                    cat = pattern['category']
                    count = pattern['count']
                    datafix_patterns[cat] = count
        
        if datafix_patterns:
            automation_potential += "**Datafix Automation Opportunities:**\n"
            for category, count in sorted(datafix_patterns.items(), key=lambda x: x[1], reverse=True):
                automation_potential += f"- {category} datafixes ({count} tickets): "
                if count > 10:
                    automation_potential += f"Develop data validation rules and automated correction workflows specific to {category} data issues. "
                    automation_potential += f"Implement proactive data quality monitoring to catch issues before they require tickets.\n"
                else:
                    automation_potential += f"Create standard data correction procedures with partial automation for common {category} data issues.\n"
        else:
            if 'datafix_general' in patterns:
                count = patterns['datafix_general']['count']
                automation_potential += f"**General Datafix Automation ({count} tickets):**\n"
                automation_potential += "Implement a data validation framework with automated correction capabilities for common data issues. "
                automation_potential += "Include self-service options for users to request and track specific datafixes."
            else:
                automation_potential += "No clear datafix automation patterns identified in the data."
    
    # For country/region specific issues
    elif 'country' in question_lower or 'region' in question_lower:
        region_patterns = {}
        
        for pattern_key, pattern in patterns.items():
            if pattern_key.startswith('region_'):
                region = pattern['region']
                if region not in region_patterns:
                    region_patterns[region] = []
                
                region_patterns[region].append({
                    "category": pattern['category'],
                    "count": pattern['count'],
                    "total": pattern['total_region_tickets']
                })
        
        if region_patterns:
            automation_potential += "**Region-Specific Automation Opportunities:**\n"
            for region, issues in region_patterns.items():
                automation_potential += f"- {region} region:\n"
                for issue in issues:
                    category = issue['category']
                    count = issue['count']
                    percentage = (count / issue['total']) * 100 if issue['total'] > 0 else 0
                    
                    automation_potential += f"  * {category} ({count} tickets, {percentage:.1f}% of region's tickets): "
                    automation_potential += f"Develop region-specific automated solutions for {category} issues, "
                    automation_potential += f"taking into account local systems and processes unique to {region}.\n"
        else:
            automation_potential += "No clear region-specific automation patterns identified in the data."
    
    # For escalation
    elif 'escalation' in question_lower or 'escalate' in question_lower:
        escalation_patterns = {}
        
        for pattern_key, pattern in patterns.items():
            if pattern_key.startswith('escalation_'):
                category = pattern['category']
                count = pattern['count']
                escalation_patterns[category] = count
        
        if escalation_patterns:
            automation_potential += "**Escalation Process Automation Opportunities:**\n"
            for category, count in sorted(escalation_patterns.items(), key=lambda x: x[1], reverse=True):
                automation_potential += f"- {category} escalations ({count} tickets): "
                automation_potential += f"Implement automated escalation triggers and workflows for {category} tickets "
                automation_potential += f"with predefined criteria and notification paths. Include proactive monitoring "
                automation_potential += f"to identify potential escalation scenarios before they occur.\n"
        else:
            automation_potential += "No clear escalation patterns identified for automation in the data."
    
    # For document failures
    elif 'document' in question_lower and ('fail' in question_lower or 'failure' in question_lower):
        doc_failure_patterns = {}
        
        for pattern_key, pattern in patterns.items():
            if pattern_key.startswith('doc_failure_'):
                year = pattern['year']
                count = pattern['count']
                doc_failure_patterns[year] = count
        
        if doc_failure_patterns:
            automation_potential += "**Document Failure Automation Opportunities:**\n"
            total_failures = sum(doc_failure_patterns.values())
            automation_potential += f"Based on {total_failures} document failure tickets:\n"
            automation_potential += "- Implement document generation monitoring system with automated error detection\n"
            automation_potential += "- Create automated retry mechanisms for failed document generation\n"
            automation_potential += "- Develop proactive notification system for document failures\n"
            automation_potential += "- Build self-healing capabilities for common document failure scenarios\n"
            
            # Mention year distribution
            if len(doc_failure_patterns) > 1:
                automation_potential += "\nYear-over-year trend: "
                sorted_years = sorted(doc_failure_patterns.items())
                trend = []
                for year, count in sorted_years:
                    trend.append(f"{year}: {count}")
                automation_potential += ", ".join(trend)
                
                # Check if increasing or decreasing
                if len(sorted_years) >= 2:
                    first_year = sorted_years[0][1]
                    last_year = sorted_years[-1][1]
                    if last_year > first_year:
                        automation_potential += "\nWith an increasing trend, automation would provide growing ROI."
                    else:
                        automation_potential += "\nWith a decreasing trend, existing measures may be working but could be enhanced with automation."
        else:
            automation_potential += "No clear document failure patterns identified for automation in the data."
    
    # Generic fallback
    else:
        automation_potential += "The data does not show clear patterns for specific automation related to this question. Consider broader analysis of ticket categories and resolution patterns."
    
    return automation_potential



###################################################### data based Question Answer ends ##########################################  




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
            analysis_tabs = st.tabs(["Overview", "Statistics"])
            
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
            
        except Exception as e:
            handle_error(e, "Error in analysis tab")
            st.session_state.errors.append(f"Analysis error: {str(e)}")


# Tab 4: Insights
with tabs[2]:
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
with tabs[3]:
    st.markdown("<h2 class='section-header'>Predefined Questions</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.info("Please upload data first to use the questions module.")
    else:
        try:
            # Generate predefined questions if not already done
            if st.session_state.predefined_questions is None or st.button("Generate Questions"):
                with st.spinner("Generating predefined questions with LLM..."):
                    # Sample the data to reduce size
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
                st.markdown("### Ticket Data Analysis Questions")
                
                # Display each question with enhanced answers
                for i, qa_pair in enumerate(st.session_state.predefined_questions):
                    question = qa_pair.get('question', 'Question not available')
                    answer = qa_pair.get('answer', 'No answer available')
                    automation = qa_pair.get('automation_potential', 'No automation potential information available')
                    
                    # Enhance the answer with specific data-driven automation potential
                    enhanced = enhance_question_answer(
                        st.session_state.processed_data,
                        question,
                        answer
                    )
                    
                    # Create expandable section with the question as header
                    with st.expander(f"**Q{i+1}: {question}**", expanded=False):
                        st.markdown("#### Answer:")
                        st.markdown(enhanced["answer"])
                        
                        # Show real examples from the data
                        if enhanced["examples"]:
                            st.markdown("#### Data Examples:")
                            for ex in enhanced["examples"]:
                                ex_text = f"**Ticket {ex.get('id', 'ID')}**"
                                
                                if 'category' in ex:
                                    ex_text += f" (Category: {ex['category']})"
                                
                                if 'description' in ex:
                                    ex_text += f"\n\nDescription: {ex['description']}"
                                
                                if 'resolution' in ex:
                                    ex_text += f"\n\nResolution: {ex['resolution']}"
                                
                                if 'resolution_time' in ex:
                                    ex_text += f"\n\nResolution Time: {ex['resolution_time']}"
                                
                                st.markdown(ex_text)
                                st.markdown("---")
                        
                        st.markdown("#### Automation Potential:")
                        st.markdown(enhanced["automation_potential"])
            else:
                st.warning("Failed to generate predefined questions. Please try again or check your data.")
        except Exception as e:
            handle_error(e, "Error generating questions")
            st.session_state.errors.append(f"Questions error: {str(e)}")

# Tab 6: Automation
with tabs[4]:
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
with tabs[5]:
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

    