import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import time
import threading
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Import custom JSON utilities
from utils.json_utils import make_json_serializable, safe_json_dumps, safe_json_loads

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")  # Default if not specified

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL_NAME,
    temperature=0.2,
    max_tokens=8192
)

# Page setup
st.set_page_config(
    page_title="Ticket Analysis System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'ticket_data' not in st.session_state:
    st.session_state.ticket_data = None
if 'chunked_data' not in st.session_state:
    st.session_state.chunked_data = []
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'automation_suggestions' not in st.session_state:
    st.session_state.automation_suggestions = None
if 'qualitative_answers' not in st.session_state:
    st.session_state.qualitative_answers = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Import agent modules
from agents.data_agent import DataProcessingAgent
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.automation_agent import AutomationRecommendationAgent
from agents.qa_agent import QualitativeAnswerAgent
from agents.chat_agent import ChatAgent
from agents.supervisor_agent import SupervisorAgent

# Initialize agents
def init_agents():
    data_agent = DataProcessingAgent(llm)
    analysis_agent = AnalysisAgent(llm)
    visualization_agent = VisualizationAgent(llm)
    automation_agent = AutomationRecommendationAgent(llm)
    qa_agent = QualitativeAnswerAgent(llm)
    chat_agent = ChatAgent(llm)
    
    # Initialize supervisor with all agent references
    supervisor = SupervisorAgent(
        llm,
        data_agent,
        analysis_agent,
        visualization_agent,
        automation_agent,
        qa_agent,
        chat_agent
    )
    
    return supervisor

# Create sidebar
st.sidebar.title("Ticket Analysis System")
st.sidebar.write("Upload your ticket data and get AI-powered insights")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload ticket data (.csv or .xlsx)", type=["csv", "xlsx"])

# Optional column names input
st.sidebar.subheader("Optional: Specify important columns")
column_hints = st.sidebar.text_area(
    "Enter important column names (one per line):",
    placeholder="E.g.:\nticket_id\ncategory\npriority\nresolution_time\ndescription"
)

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Processing your data..."):
        try:
            # Initialize session state if needed
            if 'ticket_data' not in st.session_state or st.session_state.ticket_data is None:
                st.session_state.ticket_data = None
            if 'chunked_data' not in st.session_state:
                st.session_state.chunked_data = []
            if 'insights' not in st.session_state:
                st.session_state.insights = None
            if 'automation_suggestions' not in st.session_state:
                st.session_state.automation_suggestions = None
            if 'qualitative_answers' not in st.session_state:
                st.session_state.qualitative_answers = None
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
                
            # Initialize supervisor agent
            supervisor = init_agents()
            
            # Extract column hints if provided
            column_hints_list = column_hints.split("\n") if column_hints else []
            
            # Load and process data through supervisor
            loaded_data = supervisor.process_data(
                uploaded_file, 
                chunk_size=500, 
                column_hints=column_hints_list
            )
            
            if loaded_data is not None and len(loaded_data) > 0:
                # Only update session state if we successfully loaded data
                st.session_state.ticket_data = loaded_data
                st.sidebar.success(f"Successfully loaded {len(loaded_data)} records")
                
                # This is now handled inside the supervisor
                st.session_state.insights = supervisor.generate_insights()
            else:
                st.sidebar.error("Failed to load data. Please check the file format and try again.")
                # Reset state on error
                st.session_state.ticket_data = None 
                st.session_state.insights = None
                st.session_state.automation_suggestions = None
                st.session_state.qualitative_answers = None
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            # Reset state on error
            st.session_state.ticket_data = None
            st.session_state.insights = None
            st.session_state.automation_suggestions = None
            st.session_state.qualitative_answers = None

# Main content area with tabs
# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Visualization", 
    "Qualitative Questions", 
    "Automation Suggestions",
    "💬 Chat Interface"
])

# Tab 1: Data Visualization
with tab1:
    st.header("Data Visualization")
    
    if st.session_state.ticket_data is not None:
        # Replace data overview table with insights summary
        st.subheader("Overall Ticket Data Insights")
        
        # Create a container for the insights
        insights_container = st.container()
        
        with insights_container:
            try:
                supervisor = init_agents()
                
                # Ensure supervisor has loaded data
                if supervisor.data is None and st.session_state.ticket_data is not None:
                    supervisor.data = st.session_state.ticket_data
                
                # Get insights if they're not already in the session state
                if st.session_state.insights is None:
                    st.session_state.insights = supervisor.generate_insights()
                
                # Format and display key insights
                if st.session_state.insights and "insights" in st.session_state.insights:
                    insights = st.session_state.insights["insights"]
                    
                    # Display volume insights
                    if "volume_insights" in insights and insights["volume_insights"]:
                        st.markdown("### Volume Analysis")
                        for insight in insights["volume_insights"][:3]:  # Limit to 3 key insights
                            st.markdown(f"• {insight}")
                    
                    # Display time insights
                    if "time_insights" in insights and insights["time_insights"]:
                        st.markdown("### ⏱️ Time Patterns")
                        for insight in insights["time_insights"][:3]:
                            st.markdown(f"• {insight}")
                    
                    # Display category insights
                    if "category_insights" in insights and insights["category_insights"]:
                        st.markdown("### Category Distribution")
                        for insight in insights["category_insights"][:3]:
                            st.markdown(f"• {insight}")
                    
                    # Display efficiency insights
                    if "efficiency_insights" in insights and insights["efficiency_insights"]:
                        st.markdown("### ⚡ Efficiency Metrics")
                        for insight in insights["efficiency_insights"][:3]:
                            st.markdown(f"• {insight}")
                    
                    # Add a separator before visualizations
                    st.markdown("---")
                else:
                    st.info("Generating insights from your data...")
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
        # Display charts from visualization agent
        st.subheader("Key Visualizations")
        with st.spinner("Generating visualizations..."):
            try:
                supervisor = init_agents()
                
                # Ensure supervisor has loaded data
                if supervisor.data is None and st.session_state.ticket_data is not None:
                    supervisor.data = st.session_state.ticket_data
                
                charts = supervisor.generate_visualizations()
                
                if charts and len(charts) > 0:
                    # Display charts in a grid layout
                    col1, col2 = st.columns(2)
                    
                    for i, chart in enumerate(charts):
                        if chart:  # Ensure chart is not None
                            with col1 if i % 2 == 0 else col2:
                                st.pyplot(chart)
                else:
                    st.info("No charts could be generated from this data. This might occur if the data doesn't contain date fields or categorical data suitable for visualization.")
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
    else:
        st.info("Please upload ticket data to see visualizations and insights")

# Tab 2: Qualitative Questions
with tab2:
    st.header("Qualitative Questions")
    
    if st.session_state.ticket_data is not None:
        # Add custom CSS for better question display
        st.markdown("""
        <style>
        .large-bold-question {
            font-size: 24px !important;
            font-weight: bold !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize necessary session state variables if they don't exist
        if "questions_to_process" not in st.session_state:
            st.session_state.questions_to_process = []
        if "question_answers" not in st.session_state:
            st.session_state.question_answers = {}
        if "questions_initialized" not in st.session_state:
            st.session_state.questions_initialized = False
        
        # First-time initialization: Get all questions
        if not st.session_state.questions_initialized:
            with st.spinner("Identifying key questions for analysis..."):
                try:
                    supervisor = init_agents()
                    
                    # Ensure supervisor has loaded data
                    if supervisor.data is None and st.session_state.ticket_data is not None:
                        supervisor.data = st.session_state.ticket_data
                    
                    # Get all questions first
                    questions = supervisor.qa_agent._generate_dataset_specific_questions(
                        st.session_state.ticket_data,
                        supervisor.analysis_results or {}
                    )
                    
                    # Store questions in session state (limit to 10)
                    st.session_state.questions_to_process = questions[:10]
                    st.session_state.questions_initialized = True
                    
                    # Add progress tracking
                    st.session_state.total_questions = len(st.session_state.questions_to_process)
                    st.session_state.questions_completed = 0
                    
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    # Set initialized to true to avoid getting stuck in an error loop
                    st.session_state.questions_initialized = True
        
        # Show progress
        if st.session_state.questions_initialized:
            # Calculate progress
            total = getattr(st.session_state, 'total_questions', 10)
            completed = len(st.session_state.question_answers)
            
            # Display progress bar
            if completed < total:
                st.progress(completed / total)
                st.write(f"Analyzed {completed} of {total} questions")
            else:
                st.success("All questions analyzed!")
            
            # Display already processed questions
            for i, (question, answer) in enumerate(st.session_state.question_answers.items()):
                st.markdown(f"<div class='large-bold-question'>{question}</div>", unsafe_allow_html=True)
                with st.expander("See Answer", expanded=(i < 2)):
                    st.write(f"**Answer**: {answer['answer']}")
                    st.write(f"**Automation Scope**: {answer['automation_scope']}")
                    st.write(f"**Justification**: {answer['justification']}")
                    st.write(f"**Automation Type**: {answer['automation_type']}")
                    st.write(f"**Implementation Plan**: {answer['implementation_plan']}")
            
            # Process next question if there are still questions to process
            if st.session_state.questions_to_process:
                next_question = st.session_state.questions_to_process[0]
                
                # Display what's being processed
                st.markdown("---")
                st.markdown(f"<div class='large-bold-question'>Currently analyzing: {next_question}</div>", unsafe_allow_html=True)
                status_placeholder = st.empty()
                status_placeholder.info("Processing... this may take up to 30 seconds due to API rate limits")
                
                try:
                    supervisor = init_agents()
                    
                    # Ensure supervisor has loaded data
                    if supervisor.data is None and st.session_state.ticket_data is not None:
                        supervisor.data = st.session_state.ticket_data
                    
                    # Prepare shared context once
                    shared_context = supervisor.qa_agent._prepare_shared_context(
                        st.session_state.ticket_data,
                        supervisor.analysis_results or {}
                    )
                    
                    # Process just this question with rate limiting
                    def process_question():
                        return supervisor.qa_agent._answer_question_with_context(
                            next_question, 
                            shared_context
                        )
                    
                    # Use rate limiter to handle API limits
                    answer = supervisor.qa_agent.rate_limiter.execute_with_retry(process_question)
                    
                    # Store the answer
                    st.session_state.question_answers[next_question] = answer
                    
                    # Remove this question from the queue
                    st.session_state.questions_to_process.pop(0)
                    
                    # Wait 2 seconds before processing the next question
                    time.sleep(2)
                    
                    # Force a rerun to show the new answer
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    status_placeholder.error(f"Error processing question: {error_msg}")
                    
                    # If it's a rate limit error, add a longer wait
                    if "rate limit" in error_msg.lower() or "429" in error_msg:
                        st.warning("Rate limit reached. Waiting 60 seconds before trying the next question...")
                        time.sleep(60)
                    else:
                        # Create fallback answer
                        fallback = {
                            "answer": f"Based on the available data, we cannot provide a specific answer to '{next_question}' at this time.",
                            "automation_scope": "Improved data collection and analysis capabilities",
                            "justification": "Current analysis limitations prevent a detailed answer to this question",
                            "automation_type": "Enhanced data processing",
                            "implementation_plan": "Implement more robust data analysis techniques and fallback mechanisms"
                        }
                        st.session_state.question_answers[next_question] = fallback
                        
                        # Remove this question from the queue
                        st.session_state.questions_to_process.pop(0)
                        
                        # Wait briefly before rerunning
                        time.sleep(1)
                    
                    # Force a rerun to continue or show the fallback answer
                    st.rerun()
                
        else:
            st.info("Preparing qualitative questions for analysis...")
    else:
        st.info("Please upload ticket data to see qualitative analysis")

# Tab 3: Automation Suggestions
with tab3:
    st.header("Automation Suggestions")
    
    if st.session_state.ticket_data is not None:
        if st.session_state.automation_suggestions is None:
            with st.spinner("Identifying automation opportunities..."):
                try:
                    supervisor = init_agents()
                    
                    # Ensure supervisor has loaded data
                    if supervisor.data is None and st.session_state.ticket_data is not None:
                        supervisor.data = st.session_state.ticket_data
                    
                    st.session_state.automation_suggestions = supervisor.generate_automation_suggestions()
                except Exception as e:
                    st.error(f"Error generating automation suggestions: {str(e)}")
        
        # Display automation suggestions
        if st.session_state.automation_suggestions and len(st.session_state.automation_suggestions) > 0:
            for i, suggestion in enumerate(st.session_state.automation_suggestions):
                with st.expander(f"Opportunity {i+1}: {suggestion['title']}", expanded=(i == 0)):
                    st.write(f"**Automation Scope**: {suggestion['scope']}")
                    st.write(f"**Justification**: {suggestion['justification']}")
                    st.write(f"**Automation Type**: {suggestion['type']}")
                    st.write(f"**Implementation Plan**: {suggestion['implementation']}")
                    st.write(f"**Impact**: {suggestion['impact']}")
        else:
            st.info("No automation suggestions could be generated. This might occur if the system couldn't identify clear automation patterns in the data.")
    else:
        st.info("Please upload ticket data to identify automation opportunities")

# Tab 4: Chat Interface
with tab4:
    st.header("Chat with Your Ticket Data")
    
    if st.session_state.ticket_data is not None:
        # Add chat input inside this tab (in the middle of the page)
        user_query = st.text_input("Ask a question about your ticket data...", key="chat_query")
        st.button("Send", key="send_button", 
                 on_click=lambda: st.session_state.update({"submit_query": True}) 
                 if user_query else None)
        
        # Process the query when submitted
        if st.session_state.get("submit_query", False) and st.session_state.get("chat_query", ""):
            user_query = st.session_state.chat_query
            
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Reset submission flag
            st.session_state.submit_query = False
            
            # Clear input field
            st.session_state.chat_query = ""
            
            # Generate response with spinner
            with st.spinner("Thinking..."):
                try:
                    supervisor = init_agents()
                    
                    # Ensure supervisor has loaded data
                    if supervisor.data is None and st.session_state.ticket_data is not None:
                        supervisor.data = st.session_state.ticket_data
                    
                    response = supervisor.chat_response(user_query)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"I'm sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_message)
                    
                    # Add error message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Display chat history
        st.subheader("Conversation")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    else:
        st.info("Please upload ticket data to start a conversation about it")

# Add footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: The more complete and meaningful your ticket data is, the more insightful the analysis will be."
)