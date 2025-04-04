import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import time
import threading
import traceback
from dotenv import load_dotenv

# Import custom JSON utilities
from utils.json_utils import make_json_serializable, safe_json_dumps, safe_json_loads

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")  # Default if not specified

# Page setup once
st.set_page_config(
    page_title="Ticket Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Qualitative Ticket Analysis")
st.markdown("---")

# Session state initialization - only for new sessions
if 'initialization_complete' not in st.session_state:
    st.session_state.ticket_data = None
    st.session_state.chunked_data = []
    st.session_state.insights = None
    st.session_state.automation_suggestions = None
    st.session_state.qualitative_answers = None
    st.session_state.chat_history = []
    st.session_state.initialization_complete = True
    st.session_state.is_processing = False
    # Initialize processing status flags
    st.session_state.qualitative_processing = False
    st.session_state.automation_processing = False
    st.session_state.visualization_processing = False

# Cache expensive operations
@st.cache_resource
def get_llm():
    """Cached LLM initialization to avoid repeated API setup"""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please add it to your .env file.")
        return None
        
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=0.2,
            max_tokens=8192
        )
    except Exception as e:
        st.error(f"Failed to initialize GROQ LLM: {str(e)}")
        return None
# Import agent modules (lazy)
def import_agent_modules():
    """Import agent modules only when needed"""
    if 'agent_modules_loaded' not in st.session_state:
        try:
            # Import in local scope to avoid loading at startup
            from agents.data_agent import DataProcessingAgent
            from agents.analysis_agent import AnalysisAgent
            from agents.visualization_agent import VisualizationAgent
            from agents.automation_agent import AutomationRecommendationAgent
            from agents.qa_agent import QualitativeAnswerAgent
            from agents.chat_agent import ChatAgent
            from agents.supervisor_agent import SupervisorAgent
            
            # Store classes in session state for later use
            st.session_state.DataProcessingAgent = DataProcessingAgent
            st.session_state.AnalysisAgent = AnalysisAgent
            st.session_state.VisualizationAgent = VisualizationAgent
            st.session_state.AutomationRecommendationAgent = AutomationRecommendationAgent
            st.session_state.QualitativeAnswerAgent = QualitativeAnswerAgent
            st.session_state.ChatAgent = ChatAgent
            st.session_state.SupervisorAgent = SupervisorAgent
            
            st.session_state.agent_modules_loaded = True
            return True
        except ImportError as e:
            st.error(f"Failed to import agent modules: {str(e)}")
            st.write("Please check that all agent modules are in the 'agents' directory.")
            return False
    return True

# Lazy supervisor initialization
def get_supervisor():
    """Get or initialize supervisor agent lazily"""
    if 'supervisor' not in st.session_state:
        # Import agent modules if not already done
        if not import_agent_modules():
            return None
            
        # Get LLM instance
        llm = get_llm()
        if llm is None:
            return None
            
        try:
            # Initialize only data agent first (for data loading)
            DataProcessingAgent = st.session_state.DataProcessingAgent
            data_agent = DataProcessingAgent(llm)
            
            # Create supervisor with minimal agents
            SupervisorAgent = st.session_state.SupervisorAgent
            supervisor = SupervisorAgent(
                llm,
                data_agent,
                None,  # analysis_agent
                None,  # visualization_agent
                None,  # automation_agent
                None,  # qa_agent
                None   # chat_agent
            )
            
            st.session_state.supervisor = supervisor
        except Exception as e:
            st.error(f"Failed to initialize supervisor: {str(e)}")
            st.write(traceback.format_exc())
            return None
    
    # Sync session state data with supervisor if needed
    supervisor = st.session_state.supervisor
    if supervisor is not None and supervisor.data is None and st.session_state.ticket_data is not None:
        supervisor.data = st.session_state.ticket_data
        print("Data loaded into supervisor from session state")
    
    return supervisor

# Helper functions to lazily initialize agents when needed
def ensure_analysis_agent(supervisor):
    """Ensure analysis agent is initialized"""
    if supervisor is None:
        return None
        
    if supervisor.analysis_agent is None:
        AnalysisAgent = st.session_state.AnalysisAgent
        supervisor.analysis_agent = AnalysisAgent(supervisor.llm)
    return supervisor

def ensure_visualization_agent(supervisor):
    """Ensure visualization agent is initialized"""
    if supervisor is None:
        return None
        
    if supervisor.visualization_agent is None:
        VisualizationAgent = st.session_state.VisualizationAgent
        supervisor.visualization_agent = VisualizationAgent(supervisor.llm)
    return supervisor

def ensure_automation_agent(supervisor):
    """Ensure automation agent is initialized"""
    if supervisor is None:
        return None
        
    if supervisor.automation_agent is None:
        AutomationRecommendationAgent = st.session_state.AutomationRecommendationAgent
        supervisor.automation_agent = AutomationRecommendationAgent(supervisor.llm)
    return supervisor

def ensure_qa_agent(supervisor):
    """Ensure qualitative answer agent is initialized"""
    if supervisor is None:
        return None
        
    if supervisor.qa_agent is None:
        QualitativeAnswerAgent = st.session_state.QualitativeAnswerAgent
        supervisor.qa_agent = QualitativeAnswerAgent(supervisor.llm)
    return supervisor

def ensure_chat_agent(supervisor):
    """Ensure chat agent is initialized"""
    if supervisor is None:
        return None
        
    if supervisor.chat_agent is None:
        ChatAgent = st.session_state.ChatAgent
        supervisor.chat_agent = ChatAgent(supervisor.llm)
    return supervisor


# Generate qualitative answers in background
def generate_qualitative_answers_bg():
    try:
        st.session_state.qualitative_processing = True
        supervisor = get_supervisor()
        if supervisor is None:
            st.error("Failed to initialize supervisor. Please reload the page and try again.")
            st.session_state.qualitative_processing = False
            return
            
        supervisor = ensure_qa_agent(supervisor)
        if supervisor is None:
            st.error("Failed to initialize QA agent. Please reload the page and try again.")
            st.session_state.qualitative_processing = False
            return
            
        # Generate qualitative answers
        st.session_state.qualitative_answers = supervisor.generate_qualitative_answers()
        st.session_state.qualitative_processing = False
        print("Qualitative answers generated successfully")
    except Exception as e:
        st.session_state.qualitative_processing = False
        print(f"Error generating qualitative answers: {str(e)}")
        traceback.print_exc()



# Generate automation suggestions in background
def generate_automation_suggestions_bg():
    try:
        st.session_state.automation_processing = True
        supervisor = get_supervisor()
        if supervisor is None:
            st.error("Failed to initialize supervisor. Please reload the page and try again.")
            st.session_state.automation_processing = False
            return
            
        supervisor = ensure_automation_agent(supervisor)
        if supervisor is None:
            st.error("Failed to initialize Automation agent. Please reload the page and try again.")
            st.session_state.automation_processing = False
            return
            
        # Generate automation suggestions
        st.session_state.automation_suggestions = supervisor.generate_automation_suggestions()
        st.session_state.automation_processing = False
        print("Automation suggestions generated successfully")
    except Exception as e:
        st.session_state.automation_processing = False
        print(f"Error generating automation suggestions: {str(e)}")
        traceback.print_exc()

# Create sidebar
st.sidebar.title("File Upload")
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
if uploaded_file is not None and not st.session_state.is_processing:
    # Set processing flag to avoid reprocessing
    st.session_state.is_processing = True
    
    with st.spinner("Processing your data..."):
        try:
            # Initialize supervisor agent - this will load only what's needed
            supervisor = get_supervisor()
            
            if supervisor is None:
                st.sidebar.error("Failed to initialize the analysis system.")
                st.session_state.is_processing = False
            else:
                # Extract column hints if provided
                column_hints_list = [hint.strip() for hint in column_hints.split("\n") if hint.strip()] if column_hints else []
                
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
                    
                    # Start basic analysis in background (don't wait for completion)
                    # This avoids blocking the UI while still preparing data
                    ensure_analysis_agent(supervisor)
                    st.session_state.is_processing = False
                else:
                    st.sidebar.error("Failed to load data. Please check the file format and try again.")
                    # Reset state on error
                    st.session_state.ticket_data = None 
                    st.session_state.insights = None
                    st.session_state.automation_suggestions = None
                    st.session_state.qualitative_answers = None
                    st.session_state.is_processing = False
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            st.sidebar.write("Here are more details about the error:")
            st.sidebar.code(traceback.format_exc())
            # Reset state on error
            st.session_state.ticket_data = None
            st.session_state.insights = None
            st.session_state.automation_suggestions = None
            st.session_state.qualitative_answers = None
            st.session_state.is_processing = False


st.sidebar.markdown("---")
st.sidebar.header("💬 Chat Interface")

if st.session_state.ticket_data is not None:
    # Chat History in Sidebar
    chat_history_container = st.sidebar.container()
    
    # Display Chat History
    with chat_history_container:
        if 'chat_interface_history' in st.session_state and st.session_state.chat_interface_history:
            for message in st.session_state.chat_interface_history:
                st.sidebar.chat_message(message["role"]).write(message["content"])
        else:
            st.sidebar.info("Your AI assistant is ready!")
    
    # Chat Input in Sidebar
    user_query = st.sidebar.text_input(
        "Ask a question about your ticket data", 
        key="sidebar_chat_query",
        placeholder="Type your question here..."
    )
    
    # Send Button
    send_button = st.sidebar.button("Send Query", key="sidebar_send_chat_query")
    
    # Process User Query
    if (user_query or send_button) and user_query:
        try:
            # Initialize chat history if not exists
            if 'chat_interface_history' not in st.session_state:
                st.session_state.chat_interface_history = []
            
            # Add user message to chat history
            st.session_state.chat_interface_history.append({
                "role": "user", 
                "content": user_query
            })
            
            # Initialize supervisor and ensure chat agent
            supervisor = get_supervisor()
            supervisor = ensure_chat_agent(supervisor)
            
            # Ensure data is loaded into supervisor
            if supervisor.data is None and st.session_state.ticket_data is not None:
                supervisor.data = st.session_state.ticket_data
            
            # Generate response
            response = supervisor.chat_response(user_query)
            
            # Add AI response to chat history
            st.session_state.chat_interface_history.append({
                "role": "assistant", 
                "content": response
            })
            
            # Clear the input after sending
            st.session_state.sidebar_chat_query = ""
            
            # Rerun to refresh the view
            st.experimental_rerun()
        
        except Exception as e:
            error_message = f"I apologize, but I encountered an error processing your question: {str(e)}"
            st.session_state.chat_interface_history.append({
                "role": "assistant", 
                "content": error_message
            })
            st.experimental_rerun()

else:
    st.sidebar.warning("Upload ticket data to start chatting")            

# Main content area with tabs
tab1, tab2, tab3 = st.tabs([
    "Data Visualization", 
    "Qualitative Questions", 
    "Automation Suggestions",
])

# Initialize active tab in session state if it doesn't exist
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Visualization"



# Tab 1: Data Visualization
with tab1:
    st.header("Data Visualization")
    
    if st.session_state.ticket_data is not None:
        # Add overall insight section
        with st.container():
            st.subheader("💡 Overall Insights")
            
            # Only generate insights when needed instead of upfront
            if st.session_state.insights is None:
                # Show generating message
                insights_status = st.info("Generating insights from your data...")
                
                try:
                    # Ensure the analysis agent is initialized
                    supervisor = get_supervisor()
                    supervisor = ensure_analysis_agent(supervisor)
                    
                    # Generate insights
                    st.session_state.insights = supervisor.generate_insights()
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
            
            if st.session_state.insights and "insights" in st.session_state.insights:
                insights = st.session_state.insights["insights"]
                
                # Display volume insights
                if "volume_insights" in insights and insights["volume_insights"]:
                    with st.expander("Volume Analysis", expanded=True):
                        for insight in insights["volume_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display time insights
                if "time_insights" in insights and insights["time_insights"]:
                    with st.expander("⏱Time Patterns", expanded=True):
                        for insight in insights["time_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display category insights
                if "category_insights" in insights and insights["category_insights"]:
                    with st.expander("Category Distribution", expanded=True):
                        for insight in insights["category_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display efficiency insights
                if "efficiency_insights" in insights and insights["efficiency_insights"]:
                    with st.expander("⚡ Efficiency Metrics", expanded=True):
                        for insight in insights["efficiency_insights"][:3]:
                            st.markdown(f"• {insight}")
        
        # Interactive visualization section
        st.subheader("Create Custom Visualizations")
        st.write("Select columns to visualize and the type of chart you want to see.")
        
        # Initialize the visualization agent only when user interacts with visualization
        if "viz_agent_initialized" not in st.session_state:
            with st.spinner("Preparing visualization tools..."):
                supervisor = get_supervisor()
                supervisor = ensure_visualization_agent(supervisor)
                st.session_state.viz_agent_initialized = True
                
        # Column type detection for better suggestions
        df = st.session_state.ticket_data
        
        # Cache column type detection
        if "column_types" not in st.session_state:
            # Identify column types
            categorical_cols = []
            numeric_cols = []
            date_cols = []
            
            for col in df.columns:
                # Check if it's a numeric column
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                # Check if it's a datetime column or has date-like name
                elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(term in col.lower() for term in ['date', 'time', 'created', 'resolved']):
                    date_cols.append(col)
                # Check if it's a categorical column with reasonable cardinality
                elif df[col].nunique() < 50:
                    categorical_cols.append(col)
            
            st.session_state.categorical_cols = categorical_cols
            st.session_state.numeric_cols = numeric_cols
            st.session_state.date_cols = date_cols
        else:
            categorical_cols = st.session_state.categorical_cols
            numeric_cols = st.session_state.numeric_cols
            date_cols = st.session_state.date_cols
        
        # Create tabs for different chart types
        chart_tabs = st.tabs(["Distribution Charts", "Time Series", "Correlation", "Custom"])
        
        # Tab 1: Distribution Charts (Pie, Bar)
        with chart_tabs[0]:
            st.write("Visualize the distribution of categorical variables")
            
            # Column selection for distribution charts
            dist_col = st.selectbox(
                "Select column to visualize:",
                options=categorical_cols,
                index=0 if categorical_cols else None,
                help="Choose a categorical column to see its distribution"
            )
            
            # Chart type selection
            dist_chart_type = st.radio(
                "Chart type:",
                options=["Pie Chart", "Bar Chart"],
                horizontal=True
            )
            
            # Optional filter
            use_filter = st.checkbox("Add filter to focus on specific data")
            filter_col = None
            filter_val = None
            
            if use_filter:
                filter_col = st.selectbox(
                    "Filter by column:",
                    options=[c for c in categorical_cols if c != dist_col],
                    index=0 if len(categorical_cols) > 1 else None
                )
                
                if filter_col:
                    filter_values = df[filter_col].dropna().unique().tolist()
                    filter_val = st.selectbox(
                        f"Select {filter_col} value:",
                        options=filter_values
                    )
            
            # Generate button
            if st.button("Generate Distribution Chart", key="dist_chart_btn"):
                with st.spinner("Creating visualization..."):
                    try:
                        # Apply filter if selected
                        plot_df = df
                        if use_filter and filter_col and filter_val:
                            plot_df = df[df[filter_col] == filter_val]
                        
                        # Count values
                        value_counts = plot_df[dist_col].value_counts().nlargest(10)
                        
                        # Create appropriate chart
                        if dist_chart_type == "Pie Chart":
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Create pie chart
                            wedges, texts, autotexts = ax.pie(
                                value_counts, 
                                labels=value_counts.index, 
                                autopct='%1.1f%%',
                                textprops={'fontsize': 10},
                                wedgeprops={'width': 0.5, 'edgecolor': 'w'}
                            )
                            
                            # Improve text visibility
                            for text in texts:
                                text.set_fontsize(9)
                            for autotext in autotexts:
                                autotext.set_fontsize(9)
                                autotext.set_fontweight('bold')
                            
                            # Add a title
                            title = f'Distribution of {dist_col}'
                            if use_filter and filter_col and filter_val:
                                title += f' (Filtered by {filter_col}={filter_val})'
                            ax.set_title(title)
                            
                            # Add a circle at the center to create a donut chart
                            centre_circle = plt.Circle((0, 0), 0.25, fc='white')
                            ax.add_patch(centre_circle)
                            
                        else:  # Bar Chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            value_counts.plot(kind='bar', ax=ax)
                            
                            # Add a title
                            title = f'Distribution of {dist_col}'
                            if use_filter and filter_col and filter_val:
                                title += f' (Filtered by {filter_col}={filter_val})'
                            ax.set_title(title)
                            
                            # Rotate labels for better readability
                            plt.xticks(rotation=45, ha='right')
                            
                            # Add value labels on top of bars
                            for i, v in enumerate(value_counts):
                                ax.text(i, v + 0.1, str(v), ha='center')
                        
                        st.pyplot(fig)
                        
                        # Add download button for the chart
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        btn = st.download_button(
                            label="Download Chart",
                            data=buf.getvalue(),
                            file_name=f"{dist_col}_distribution.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error creating chart: {str(e)}")
                        st.write(traceback.format_exc())


# Tab 2: Time Series Charts
        with chart_tabs[1]:
            st.write("Visualize data changes over time")
            
            # Check if we have date columns
            if not date_cols:
                st.warning("No date columns detected in your data. Time series visualization requires date columns.")
            else:
                # Column selections
                time_col = st.selectbox(
                    "Select date/time column:",
                    options=date_cols,
                    index=0,
                    help="Choose a date column for the X-axis"
                )
                
                # Select what to measure
                measure_options = ["Count of tickets"]
                measure_options.extend([f"Average {col}" for col in numeric_cols])
                measure = st.selectbox(
                    "What to measure:",
                    options=measure_options,
                    index=0
                )
                
                # Time grouping
                time_group = st.selectbox(
                    "Group by time period:",
                    options=["Day", "Week", "Month", "Quarter", "Year"],
                    index=2  # Default to Month
                )
                
                # Optional category breakdown
                use_category = st.checkbox("Break down by category")
                category_col = None
                
                if use_category:
                    category_col = st.selectbox(
                        "Select category column for breakdown:",
                        options=categorical_cols,
                        index=0 if categorical_cols else None
                    )
                
                # Generate button
                if st.button("Generate Time Series Chart", key="time_chart_btn"):
                    with st.spinner("Creating time series visualization..."):
                        try:
                            # Create a copy of the dataframe for manipulation
                            plot_df = df.copy()
                            
                            # Convert time column to datetime if it's not already
                            if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                                plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors='coerce')
                            
                            # Group by time period
                            if time_group == "Day":
                                plot_df['time_group'] = plot_df[time_col].dt.date
                            elif time_group == "Week":
                                plot_df['time_group'] = plot_df[time_col].dt.to_period('W').dt.start_time
                            elif time_group == "Month":
                                plot_df['time_group'] = plot_df[time_col].dt.to_period('M').dt.start_time
                            elif time_group == "Quarter":
                                plot_df['time_group'] = plot_df[time_col].dt.to_period('Q').dt.start_time
                            else:  # Year
                                plot_df['time_group'] = plot_df[time_col].dt.year
                            
                            # Drop rows with NaT in time_group
                            plot_df = plot_df.dropna(subset=['time_group'])
                            
                            if len(plot_df) == 0:
                                st.error(f"No valid dates found in column '{time_col}'. Please check your data.")
                            else:
                                # Prepare the data based on measure selection
                                if measure == "Count of tickets":
                                    if use_category and category_col:
                                        # Group by time and category, count rows
                                        result = plot_df.groupby(['time_group', category_col]).size().unstack(fill_value=0)
                                    else:
                                        # Just group by time, count rows
                                        result = plot_df.groupby('time_group').size()
                                else:
                                    # Extract the numeric column name from the measure
                                    numeric_col = measure.replace("Average ", "")
                                    
                                    if use_category and category_col:
                                        # Group by time and category, calculate mean
                                        result = plot_df.groupby(['time_group', category_col])[numeric_col].mean().unstack(fill_value=0)
                                    else:
                                        # Just group by time, calculate mean
                                        result = plot_df.groupby('time_group')[numeric_col].mean()
                                
                                # Create figure and plot
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                if use_category and category_col:
                                    # Plot each category as a separate line
                                    result.plot(kind='line', ax=ax, marker='o')
                                    
                                    # Add legend with better styling
                                    ax.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                                else:
                                    # Plot single line
                                    result.plot(kind='line', ax=ax, marker='o', color='steelblue')
                                
                                # Set title and labels
                                title = f"{measure} by {time_group}"
                                ax.set_title(title)
                                ax.set_xlabel(time_group)
                                ax.set_ylabel(measure)
                                
                                # Improve x-axis display for dates
                                if time_group != "Year":
                                    plt.xticks(rotation=45, ha='right')
                                
                                # Add grid for better readability
                                ax.grid(True, linestyle='--', alpha=0.7)
                                
                                st.pyplot(fig)
                                
                                # Add download button for the chart
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                btn = st.download_button(
                                    label="Download Chart",
                                    data=buf.getvalue(),
                                    file_name=f"time_series_{time_group}.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.error(f"Error creating time series chart: {str(e)}")
                            st.write(traceback.format_exc())

        # Tab 3: Correlation Charts
        with chart_tabs[2]:
            st.write("Explore relationships between variables")
            
            if len(numeric_cols) < 2:
                st.warning("Correlation analysis requires at least two numeric columns. Your data doesn't have enough numeric columns.")
            else:
                # Choose correlation type
                corr_type = st.radio(
                    "Correlation type:",
                    options=["Numeric Correlation", "Category Relationship"],
                    horizontal=True
                )
                
                if corr_type == "Numeric Correlation":
                    # Multi-select for numeric columns
                    selected_num_cols = st.multiselect(
                        "Select numeric columns to correlate:",
                        options=numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))]  # Default to first 4 or fewer
                    )
                    
                    # Visualization type
                    viz_type = st.radio(
                        "Visualization:",
                        options=["Correlation Heatmap", "Scatter Plot"],
                        horizontal=True
                    )
                    
                    if viz_type == "Scatter Plot" and len(selected_num_cols) >= 2:
                        # For scatter plot, select x and y axes
                        x_col = st.selectbox("X-axis:", options=selected_num_cols, index=0)
                        y_col = st.selectbox("Y-axis:", options=[c for c in selected_num_cols if c != x_col], index=0)
                        
                        # Optional color by category
                        color_by = st.checkbox("Color by category")
                        color_col = None
                        
                        if color_by:
                            color_col = st.selectbox("Select category for coloring:", options=categorical_cols)
                    
                    if st.button("Generate Correlation Visualization", key="corr_chart_btn"):
                        with st.spinner("Creating correlation visualization..."):
                            try:
                                if viz_type == "Correlation Heatmap" and len(selected_num_cols) >= 2:
                                    # Calculate correlation matrix
                                    corr_matrix = df[selected_num_cols].corr()
                                    
                                    # Create heatmap
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                                    
                                    # Custom diverging colormap
                                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                                    
                                    # Plot heatmap
                                    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                                square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
                                    
                                    ax.set_title("Correlation Heatmap")
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    
                                    # Add download button for the chart
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name="correlation_heatmap.png",
                                        mime="image/png"
                                    )
                                
                                elif viz_type == "Scatter Plot" and len(selected_num_cols) >= 2:
                                    # Create scatter plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    if color_by and color_col:
                                        # For better colors, use categorical colormap and limit to top categories
                                        if df[color_col].nunique() > 10:
                                            # If too many categories, limit to top 10
                                            top_cats = df[color_col].value_counts().nlargest(10).index
                                            plot_df = df[df[color_col].isin(top_cats)]
                                            title_extra = " (top 10 categories only)"
                                        else:
                                            plot_df = df
                                            title_extra = ""
                                        
                                        # Create scatter plot with categories
                                        scatter = sns.scatterplot(x=x_col, y=y_col, hue=color_col, data=plot_df, ax=ax)
                                        
                                        # Set title and labels
                                        ax.set_title(f"Relationship between {x_col} and {y_col} by {color_col}{title_extra}")
                                        ax.set_xlabel(x_col)
                                        ax.set_ylabel(y_col)
                                        
                                        # Improve legend placement
                                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_col)
                                    else:
                                        # Simple scatter plot without categories
                                        scatter = sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
                                        
                                        # Add regression line
                                        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, ax=ax, color='red')
                                        
                                        # Set title and labels
                                        ax.set_title(f"Relationship between {x_col} and {y_col}")
                                        ax.set_xlabel(x_col)
                                        ax.set_ylabel(y_col)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add download button for the chart
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name=f"correlation_{x_col}_{y_col}.png",
                                        mime="image/png"
                                    )
                            except Exception as e:
                                st.error(f"Error creating correlation visualization: {str(e)}")
                                st.write(traceback.format_exc())


                else:  # Category Relationship
                    # Select two categorical columns
                    if len(categorical_cols) < 2:
                        st.warning("Category relationship analysis requires at least two categorical columns.")
                    else:
                        cat_col1 = st.selectbox("First category:", options=categorical_cols, index=0)
                        cat_col2 = st.selectbox("Second category:", options=[c for c in categorical_cols if c != cat_col1], index=0)
                        
                        # Optional: Normalize option
                        normalize = st.radio(
                            "Calculation method:",
                            options=["Count", "Percentage by row", "Percentage by column"],
                            horizontal=True
                        )
                        
                        if st.button("Generate Category Relationship", key="cat_rel_btn"):
                            with st.spinner("Creating visualization..."):
                                try:
                                    # Create crosstab
                                    if normalize == "Count":
                                        ct = pd.crosstab(df[cat_col1], df[cat_col2])
                                        title = f"Count of tickets by {cat_col1} and {cat_col2}"
                                    elif normalize == "Percentage by row":
                                        ct = pd.crosstab(df[cat_col1], df[cat_col2], normalize='index') * 100
                                        title = f"Percentage distribution of {cat_col2} within each {cat_col1} (row %)"
                                    else:  # Percentage by column
                                        ct = pd.crosstab(df[cat_col1], df[cat_col2], normalize='columns') * 100
                                        title = f"Percentage distribution of {cat_col1} within each {cat_col2} (column %)"
                                    
                                    # Limit to top categories if there are too many
                                    row_limit = 15
                                    col_limit = 10
                                    
                                    if len(ct.index) > row_limit:
                                        # Get top categories by total frequency
                                        top_rows = df[cat_col1].value_counts().nlargest(row_limit).index
                                        ct = ct.loc[top_rows]
                                        title += f" (top {row_limit} {cat_col1} categories)"
                                    
                                    if len(ct.columns) > col_limit:
                                        # Get top categories by total frequency
                                        top_cols = df[cat_col2].value_counts().nlargest(col_limit).index
                                        ct = ct[top_cols]
                                        title += f" (top {col_limit} {cat_col2} categories)"
                                    
                                    # Create heatmap
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    # Adjust formatting based on values
                                    if normalize != "Count":
                                        # For percentages, use one decimal place
                                        sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
                                    else:
                                        # For counts, use integer format
                                        sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                                    
                                    ax.set_title(title)
                                    
                                    # Rotate x-axis labels if there are many categories
                                    if len(ct.columns) > 5:
                                        plt.xticks(rotation=45, ha='right')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add download button for the chart
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name=f"category_{cat_col1}_{cat_col2}.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.error(f"Error creating category relationship visualization: {str(e)}")
                                    st.write(traceback.format_exc())

        # Tab 4: Custom Visualization
        with chart_tabs[3]:
            st.write("Create custom visualizations based on your specific needs")
            
            # Visualization type selection
            custom_viz_type = st.selectbox(
                "Select visualization type:",
                options=[
                    "Boxplot (Distribution by Category)",
                    "Histogram (Value Distribution)",
                    "Count Plot (Frequency by Category)",
                    "Line Plot (Values over Index)"
                ]
            )
            
            if custom_viz_type == "Boxplot (Distribution by Category)":
                if not numeric_cols:
                    st.warning("Boxplots require at least one numeric column.")
                elif not categorical_cols:
                    st.warning("Boxplots require at least one categorical column.")
                else:
                    # Select columns
                    box_num_col = st.selectbox("Select value to plot:", options=numeric_cols)
                    box_cat_col = st.selectbox("Group by category:", options=categorical_cols)
                    
                    # Optional settings
                    box_orient = st.radio("Orientation:", options=["Vertical", "Horizontal"], horizontal=True)
                    
                    # Option to limit categories
                    limit_cats = st.checkbox("Limit number of categories")
                    cat_limit = None
                    if limit_cats:
                        cat_limit = st.slider("Number of top categories to show:", 3, 15, 8)
                    
                    if st.button("Generate Boxplot", key="boxplot_btn"):
                        with st.spinner("Creating boxplot..."):
                            try:
                                # Prepare data
                                plot_df = df[[box_cat_col, box_num_col]].dropna()
                                
                                # Limit categories if requested
                                if limit_cats and cat_limit:
                                    top_cats = df[box_cat_col].value_counts().nlargest(cat_limit).index
                                    plot_df = plot_df[plot_df[box_cat_col].isin(top_cats)]
                                
                                if len(plot_df) == 0:
                                    st.error("No valid data found after filtering.")
                                else:
                                    # Create figure
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    # Create boxplot
                                    orient = 'h' if box_orient == "Horizontal" else 'v'
                                    sns.boxplot(
                                        x=box_num_col if orient == 'h' else box_cat_col,
                                        y=box_cat_col if orient == 'h' else box_num_col,
                                        data=plot_df,
                                        ax=ax,
                                        orient=orient
                                    )
                                    
                                    # Add title
                                    title = f"Distribution of {box_num_col} by {box_cat_col}"
                                    if limit_cats and cat_limit:
                                        title += f" (top {cat_limit} categories)"
                                    ax.set_title(title)
                                    
                                    # Adjust layout
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add download button
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name=f"boxplot_{box_num_col}_by_{box_cat_col}.png",
                                        mime="image/png"
                                    )
                            except Exception as e:
                                st.error(f"Error creating boxplot: {str(e)}")
                                st.write(traceback.format_exc())
            
            elif custom_viz_type == "Histogram (Value Distribution)":
                if not numeric_cols:
                    st.warning("Histograms require at least one numeric column.")
                else:
                    # Select column
                    hist_col = st.selectbox("Select value to plot distribution:", options=numeric_cols)
                    
                    # Optional settings
                    hist_bins = st.slider("Number of bins:", 5, 50, 20)
                    kde_line = st.checkbox("Add density curve (KDE)", value=True)
                    
                    # Filter outliers option
                    filter_outliers = st.checkbox("Filter outliers")
                    
                    if st.button("Generate Histogram", key="hist_btn"):
                        with st.spinner("Creating histogram..."):
                            try:
                                # Prepare data
                                plot_series = df[hist_col].dropna()
                                
                                if len(plot_series) == 0:
                                    st.error("No valid data found.")
                                else:
                                    # Filter outliers if requested
                                    if filter_outliers:
                                        q1 = plot_series.quantile(0.25)
                                        q3 = plot_series.quantile(0.75)
                                        iqr = q3 - q1
                                        lower_bound = q1 - 1.5 * iqr
                                        upper_bound = q3 + 1.5 * iqr
                                        plot_series = plot_series[(plot_series >= lower_bound) & (plot_series <= upper_bound)]
                                    
                                    # Create figure
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Create histogram
                                    sns.histplot(
                                        plot_series,
                                        bins=hist_bins,
                                        kde=kde_line,
                                        ax=ax
                                    )
                                    
                                    # Add title and labels
                                    title = f"Distribution of {hist_col}"
                                    if filter_outliers:
                                        title += " (outliers filtered)"
                                    ax.set_title(title)
                                    ax.set_xlabel(hist_col)
                                    ax.set_ylabel("Frequency")
                                    
                                    # Add mean and median lines
                                    plt.axvline(plot_series.mean(), color='red', linestyle='--', label=f'Mean: {plot_series.mean():.2f}')
                                    plt.axvline(plot_series.median(), color='green', linestyle='-.', label=f'Median: {plot_series.median():.2f}')
                                    plt.legend()
                                    
                                    # Adjust layout
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add download button
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name=f"histogram_{hist_col}.png",
                                        mime="image/png"
                                    )
                            except Exception as e:
                                st.error(f"Error creating histogram: {str(e)}")
                                st.write(traceback.format_exc())
            
            elif custom_viz_type == "Count Plot (Frequency by Category)":
                if not categorical_cols:
                    st.warning("Count plots require at least one categorical column.")
                else:
                    # Select columns
                    count_col = st.selectbox("Select category to count:", options=categorical_cols)
                    
                    # Optional settings
                    count_orient = st.radio("Orientation:", options=["Vertical", "Horizontal"], horizontal=True, index=1)
                    sort_bars = st.checkbox("Sort bars by count", value=True)
                    show_pct = st.checkbox("Show percentages", value=False)
                    
                    # Limit number of categories
                    max_cats = st.slider("Maximum categories to display:", 5, 30, 15)
                    
                    if st.button("Generate Count Plot", key="count_btn"):
                        with st.spinner("Creating count plot..."):
                            try:
                                # Get value counts
                                value_counts = df[count_col].value_counts()
                                
                                # Limit categories
                                if len(value_counts) > max_cats:
                                    value_counts = value_counts.nlargest(max_cats)
                                
                                # Sort if requested
                                if sort_bars:
                                    value_counts = value_counts.sort_values(ascending=False)
                                
                                # Create percentage values if needed
                                if show_pct:
                                    pct_values = (value_counts / value_counts.sum() * 100).round(1)
                                
                                # Create figure
                                fig, ax = plt.subplots(figsize=(12, 8) if count_orient == "Horizontal" else (10, 6))
                                
                                # Create plot
                                if count_orient == "Horizontal":
                                    bars = value_counts.plot(kind='barh', ax=ax)
                                    
                                    # Add value labels
                                    for i, v in enumerate(value_counts):
                                        label = f"{v}"
                                        if show_pct:
                                            label += f" ({pct_values.iloc[i]}%)"
                                        ax.text(v + 0.1, i, label, va='center')
                                else:
                                    bars = value_counts.plot(kind='bar', ax=ax)
                                    
                                    # Add value labels
                                    for i, v in enumerate(value_counts):
                                        label = f"{v}"
                                        if show_pct:
                                            label += f" ({pct_values.iloc[i]}%)"
                                        ax.text(i, v + 0.1, label, ha='center')
                                
                                # Set title and labels
                                ax.set_title(f"Count of tickets by {count_col}")
                                
                                if count_orient == "Vertical":
                                    plt.xticks(rotation=45, ha='right')
                                
                                # Adjust layout
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Add download button
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                btn = st.download_button(
                                    label="Download Chart",
                                    data=buf.getvalue(),
                                    file_name=f"countplot_{count_col}.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Error creating count plot: {str(e)}")
                                st.write(traceback.format_exc())
                                
            elif custom_viz_type == "Line Plot (Values over Index)":
                if not numeric_cols:
                    st.warning("Line plots require at least one numeric column.")
                else:
                    # Select columns - can select multiple
                    line_cols = st.multiselect(
                        "Select values to plot:",
                        options=numeric_cols,
                        default=[numeric_cols[0]] if numeric_cols else None
                    )
                    
                    # Optional settings
                    show_markers = st.checkbox("Show markers", value=True)
                    
                    if st.button("Generate Line Plot", key="line_btn"):
                        if not line_cols:
                            st.warning("Please select at least one column to plot.")
                        else:
                            with st.spinner("Creating line plot..."):
                                try:
                                    # Create figure
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    
                                    # Get subset of dataframe with selected columns
                                    plot_df = df[line_cols].copy()
                                    
                                    # Reset index to create a sequence for x-axis
                                    plot_df = plot_df.reset_index(drop=True)
                                    
                                    # Plot each column
                                    for col in line_cols:
                                        plot_df[col].plot(
                                            kind='line',
                                            ax=ax,
                                            marker='o' if show_markers else None,
                                            markersize=4 if show_markers else None,
                                            alpha=0.7
                                        )
                                    
                                    # Set title and labels
                                    ax.set_title("Line plot of selected values")
                                    ax.set_xlabel("Index")
                                    ax.set_ylabel("Value")
                                    
                                    # Add legend
                                    ax.legend(title="Variables")
                                    
                                    # Add grid
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    
                                    # Adjust layout
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add download button
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                    btn = st.download_button(
                                        label="Download Chart",
                                        data=buf.getvalue(),
                                        file_name="line_plot.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.error(f"Error creating line plot: {str(e)}")
                                    st.write(traceback.format_exc())
                                
    else:
        st.info("Please upload ticket data to see visualizations")                                            


# Tab 2: Qualitative Questions
with tab2:
    st.header("Qualitative Questions")
    
    if st.session_state.ticket_data is not None:
        # Add an explanation
        st.write("""
        This section provides in-depth qualitative analysis of your ticket data based on key questions 
        that reveal patterns, bottlenecks, and opportunities. Each analysis is generated specifically 
        from your data and includes concrete automation recommendations.
        """)
        
        # Check if QA agent needs to be initialized
        if "qa_agent_initialized" not in st.session_state:
            with st.spinner("Preparing qualitative analysis tools..."):
                try:
                    supervisor = get_supervisor()
                    supervisor = ensure_qa_agent(supervisor)
                    st.session_state.qa_agent_initialized = True
                except Exception as e:
                    st.error(f"Error initializing qualitative analysis: {str(e)}")
        
        # Create categories for organizing questions
        categories = {
            "Process Efficiency": [0, 3, 7],  # Question indices about process and efficiency
            "Resource Optimization": [1, 4, 9],  # Questions about resource allocation and staffing
            "Knowledge Management": [2, 5, 6, 8]  # Questions about knowledge and communication
        }
        
        # Add a button to generate qualitative answers
        if st.session_state.qualitative_answers is None and not st.session_state.qualitative_processing:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("Analysis will answer key questions about process efficiency, resource optimization, and knowledge management based on your specific ticket data.")
            
            with col2:
                if st.button("Generate Analysis", key="gen_qual_btn", use_container_width=True):
                    try:
                        # Set processing flag
                        st.session_state.qualitative_processing = True
                        
                        # Generate answers directly instead of using background thread
                        with st.spinner("Generating qualitative analysis... This may take a few minutes."):
                            supervisor = get_supervisor()
                            if supervisor is None:
                                st.error("Failed to initialize supervisor. Please reload the page and try again.")
                                st.session_state.qualitative_processing = False
                            else:
                                supervisor = ensure_qa_agent(supervisor)
                                if supervisor is None:
                                    st.error("Failed to initialize QA agent. Please reload the page and try again.")
                                    st.session_state.qualitative_processing = False
                                else:
                                    # Generate qualitative answers directly
                                    st.session_state.qualitative_answers = supervisor.generate_qualitative_answers()
                                    st.session_state.qualitative_processing = False
                                    st.rerun()  # Force refresh to show results
                    except Exception as e:
                        st.error(f"Error generating qualitative answers: {str(e)}")
                        st.code(traceback.format_exc())
                        st.session_state.qualitative_processing = False
        
        # Show processing message
        elif st.session_state.qualitative_processing:
            st.info("Generating qualitative analysis... This may take a few minutes.")
            # Add a placeholder for a progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.progress(0.5, text="Analyzing ticket data patterns...")
        
        # Show qualitative answers if available
        if st.session_state.qualitative_answers is not None:
            # Check if we have valid answers
            if len(st.session_state.qualitative_answers) > 0:
                # Create tabs for categories
                category_tabs = st.tabs(list(categories.keys()))
                
                # Display questions by category
                for tab_idx, (category, question_indices) in enumerate(categories.items()):
                    with category_tabs[tab_idx]:
                        # Add accordion/expander for each question in this category
                        for idx in question_indices:
                            if idx < len(st.session_state.qualitative_answers):
                                answer = st.session_state.qualitative_answers[idx]
                                
                                # Format question as a proper question with question mark if needed
                                question = answer.get("question", f"Question {idx+1}")
                                
                                with st.expander(f"**{question}**", expanded=(idx == question_indices[0])):
                                    # Analysis section
                                    st.markdown("### Analysis")
                                    st.write(answer.get("answer", "No analysis available."))
                                    
                                    # Automation section
                                    st.markdown("### Automation Opportunity")
                                    
                                    # Check if automation scope exists
                                    if "automation_scope" in answer and answer["automation_scope"]:
                                        st.markdown("**Scope:**")
                                        st.write(answer["automation_scope"])
                                    
                                    # Use columns for justification and automation type
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if "justification" in answer and answer["justification"]:
                                            st.markdown("**Justification:**")
                                            st.write(answer["justification"])
                                    
                                    with col2:
                                        if "automation_type" in answer and answer["automation_type"]:
                                            st.markdown("**Automation Type:**")
                                            st.write(answer["automation_type"])
                                    
                                    # Implementation section
                                    if "implementation_plan" in answer and answer["implementation_plan"]:
                                        st.markdown("**Implementation Plan:**")
                                        st.write(answer["implementation_plan"])
            else:
                st.warning("No qualitative analysis could be generated. This might be due to limited data or processing issues.")
    else:
        st.info("Please upload ticket data to see qualitative analysis")



# Tab 3: Automation Suggestions
with tab3:
    st.header("Automation Suggestions")
    
    # Ensure ticket data is loaded
    if st.session_state.ticket_data is not None:
        # Lazy initialization for Automation agent
        if "automation_agent_initialized" not in st.session_state:
            with st.spinner("Preparing automation analysis tools..."):
                supervisor = get_supervisor()
                supervisor = ensure_automation_agent(supervisor)
                st.session_state.automation_agent_initialized = True
        
        # Get all columns from the dataset
        all_columns = list(st.session_state.ticket_data.columns)
        
        # Categorize columns by type
        numeric_cols = st.session_state.ticket_data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = st.session_state.ticket_data.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = st.session_state.ticket_data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Column Selection with Categorization
        st.subheader("Select Columns for Automation Analysis")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Categorical Columns**")
            categorical_selection = st.multiselect(
                "Select categorical columns", 
                options=categorical_cols,
                key="cat_columns_auto"
            )
        
        with col2:
            st.markdown("**Numeric Columns**")
            numeric_selection = st.multiselect(
                "Select numeric columns", 
                options=numeric_cols,
                key="numeric_columns_auto"
            )
        
        with col3:
            st.markdown("**Date Columns**")
            date_selection = st.multiselect(
                "Select date columns", 
                options=date_cols,
                key="date_columns_auto"
            )
        
        # Combine all selected columns
        selected_columns = categorical_selection + numeric_selection + date_selection
        
        # Generate Automation Suggestions Button
        if st.button("Generate Automation Suggestions", key="generate_auto_suggestions"):
            if selected_columns:
                with st.spinner("Generating targeted automation suggestions..."):
                    try:
                        # Ensure supervisor and automation agent are ready
                        supervisor = get_supervisor()
                        supervisor = ensure_automation_agent(supervisor)
                        
                        # Generate targeted automation suggestions
                        automation_suggestions = supervisor.generate_targeted_automation_suggestions(
                            selected_columns
                        )
                        
                        # Display automation suggestions
                        if automation_suggestions and len(automation_suggestions) > 0:
                            st.subheader("Automation Opportunities")
                            
                            for i, suggestion in enumerate(automation_suggestions, 1):
                                with st.expander(f"Suggestion {i}"):
                                    # Directly render the markdown-formatted suggestion
                                    st.markdown(suggestion)
                        else:
                            st.warning("No automation suggestions could be generated. Please check the selected columns or try again.")
                    
                    except Exception as e:
                        # Detailed error logging
                        st.error(f"Error generating automation suggestions: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.warning("Please select at least one column for automation analysis.")
        
        # Informational section
        st.markdown("""
        ### 💡 How to Use Automation Suggestions
        
        1. Select relevant columns from different categories
        2. Click "Generate Automation Suggestions"
        3. Review AI-generated automation opportunities
        4. Expand each suggestion to see detailed implementation plan
        
        *Note: Suggestions are based on AI analysis of selected columns*
        """)
    
    else:
        st.info("Please upload ticket data to identify automation opportunities")


# Add footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: The more complete and meaningful your ticket data is, the more insightful the analysis will be."
)
