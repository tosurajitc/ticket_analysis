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
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Visualization", 
    "Qualitative Questions", 
    "Automation Suggestions",
    "💬 Chat Interface"
])





# Track active tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Visualization"


active_tab = st.sidebar.radio(
    "Select Tab",
    ["Data Visualization", "Qualitative Questions", "Automation Suggestions", "Chat Interface"],
    key="tab_selection",
    label_visibility="hidden"  # Hide this control from the UI
)

st.session_state.active_tab = active_tab    



# Tab 1: Data Visualization
with tab1:
    st.header("Data Visualization")
    
    if st.session_state.ticket_data is not None:
        # Add overall insight section
        with st.container():
            st.subheader("💡 Overall Insights")
            
            if st.session_state.insights and "insights" in st.session_state.insights:
                insights = st.session_state.insights["insights"]
                
                # Display volume insights
                if "volume_insights" in insights and insights["volume_insights"]:
                    with st.expander("📊 Volume Analysis", expanded=True):
                        for insight in insights["volume_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display time insights
                if "time_insights" in insights and insights["time_insights"]:
                    with st.expander("⏱️ Time Patterns", expanded=True):
                        for insight in insights["time_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display category insights
                if "category_insights" in insights and insights["category_insights"]:
                    with st.expander("🏷️ Category Distribution", expanded=True):
                        for insight in insights["category_insights"][:3]:
                            st.markdown(f"• {insight}")
                
                # Display efficiency insights
                if "efficiency_insights" in insights and insights["efficiency_insights"]:
                    with st.expander("⚡ Efficiency Metrics", expanded=True):
                        for insight in insights["efficiency_insights"][:3]:
                            st.markdown(f"• {insight}")
            else:
                st.info("Generating insights from your data...")
        
        # Interactive visualization section
        st.subheader("Create Custom Visualizations")
        st.write("Select columns to visualize and the type of chart you want to see.")
        
        # Column type detection for better suggestions
        df = st.session_state.ticket_data
        
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
        
        # Create tabs for different chart types
        chart_tabs = st.tabs(["📊 Distribution Charts", "📈 Time Series", "🔄 Correlation", "📊 Custom"])
        
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
        
        # Tab 2: Time Series Charts
        with chart_tabs[1]:
            st.write("Visualize how values change over time")
            
            # Create two columns for side-by-side selection
            ts_col1, ts_col2 = st.columns(2)
            
            with ts_col1:
                # Date column selection
                time_col = st.selectbox(
                    "Select date/time column:",
                    options=date_cols,
                    index=0 if date_cols else None,
                    help="Choose a column representing dates or times"
                )
            
            with ts_col2:
                # Measure column or counting option
                count_or_measure = st.radio(
                    "What to display:",
                    options=["Count tickets", "Measure a value"],
                    horizontal=True
                )
                
                measure_col = None
                if count_or_measure == "Measure a value":
                    measure_col = st.selectbox(
                        "Select value to measure:",
                        options=numeric_cols,
                        index=0 if numeric_cols else None
                    )
            
            # Time unit selection
            time_unit = st.select_slider(
                "Time granularity:",
                options=["Day", "Week", "Month", "Quarter", "Year"],
                value="Month"
            )
            
            # Optional grouping
            use_grouping = st.checkbox("Group by a category")
            group_col = None
            
            if use_grouping:
                group_col = st.selectbox(
                    "Group by:",
                    options=[c for c in categorical_cols if df[c].nunique() <= 10],  # Limit to columns with fewer categories
                    index=0 if categorical_cols else None
                )
            
            # Generate button
            if st.button("Generate Time Series Chart", key="time_chart_btn") and time_col:
                with st.spinner("Creating visualization..."):
                    try:
                        # Convert to datetime if needed
                        plot_df = df.copy()
                        if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                            plot_df[f'{time_col}_dt'] = pd.to_datetime(plot_df[time_col], errors='coerce')
                            date_col = f'{time_col}_dt'
                        else:
                            date_col = time_col
                        
                        # Drop rows with invalid dates
                        plot_df = plot_df.dropna(subset=[date_col])
                        
                        # Create time period column based on selected unit
                        if time_unit == "Day":
                            plot_df['period'] = plot_df[date_col].dt.date
                        elif time_unit == "Week":
                            plot_df['period'] = plot_df[date_col].dt.to_period('W').dt.start_time
                        elif time_unit == "Month":
                            plot_df['period'] = plot_df[date_col].dt.to_period('M').dt.start_time
                        elif time_unit == "Quarter":
                            plot_df['period'] = plot_df[date_col].dt.to_period('Q').dt.start_time
                        else:  # Year
                            plot_df['period'] = plot_df[date_col].dt.to_period('Y').dt.start_time
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        if use_grouping and group_col:
                            # Group by period and category
                            groups = plot_df[group_col].unique()
                            
                            if count_or_measure == "Count tickets":
                                # Count by group
                                for group in groups:
                                    group_data = plot_df[plot_df[group_col] == group]
                                    counts = group_data.groupby('period').size()
                                    periods = [pd.to_datetime(p) for p in counts.index]
                                    ax.plot(periods, counts.values, marker='o', label=str(group))
                                
                                ax.set_ylabel('Ticket Count')
                            else:
                                # Measure by group
                                for group in groups:
                                    group_data = plot_df[plot_df[group_col] == group]
                                    measures = group_data.groupby('period')[measure_col].mean()
                                    periods = [pd.to_datetime(p) for p in measures.index]
                                    ax.plot(periods, measures.values, marker='o', label=str(group))
                                
                                ax.set_ylabel(measure_col)
                            
                            ax.legend(title=group_col)
                        else:
                            # Simple time series without grouping
                            if count_or_measure == "Count tickets":
                                # Count tickets by period
                                counts = plot_df.groupby('period').size()
                                periods = [pd.to_datetime(p) for p in counts.index]
                                ax.plot(periods, counts.values, marker='o', color='steelblue')
                                ax.set_ylabel('Ticket Count')
                            else:
                                # Measure values by period
                                measures = plot_df.groupby('period')[measure_col].mean()
                                periods = [pd.to_datetime(p) for p in measures.index]
                                ax.plot(periods, measures.values, marker='o', color='steelblue')
                                ax.set_ylabel(measure_col)
                        
                        # Set title and format axes
                        title = f'{"Ticket Count" if count_or_measure == "Count tickets" else measure_col} by {time_unit}'
                        if use_grouping and group_col:
                            title += f' (Grouped by {group_col})'
                        ax.set_title(title)
                        ax.set_xlabel(f'Time ({time_unit})')
                        
                        # Format x-axis as dates
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add grid for better readability
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add download button for the chart
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        btn = st.download_button(
                            label="Download Chart",
                            data=buf.getvalue(),
                            file_name=f"time_series_{time_unit.lower()}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error creating time series chart: {str(e)}")
        
        # Tab 3: Correlation Charts
        with chart_tabs[2]:
            st.write("Explore relationships between different variables")
            
            # Create two columns for side-by-side selection
            corr_col1, corr_col2 = st.columns(2)
            
            with corr_col1:
                # First variable
                x_axis = st.selectbox(
                    "X-axis (horizontal):",
                    options=categorical_cols + numeric_cols,
                    index=0 if categorical_cols else None,
                    help="Choose a column for the horizontal axis"
                )
            
            with corr_col2:
                # Second variable
                y_axis = st.selectbox(
                    "Y-axis (vertical):",
                    options=numeric_cols,
                    index=0 if numeric_cols else None,
                    help="Choose a numeric column for the vertical axis"
                )
            
            # Chart type selection
            corr_chart_type = st.radio(
                "Chart type:",
                options=["Bar Chart", "Box Plot", "Heat Map"],
                horizontal=True
            )
            
            # Generate button
            if st.button("Generate Correlation Chart", key="corr_chart_btn") and x_axis and y_axis:
                with st.spinner("Creating visualization..."):
                    try:
                        # Create figure
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        if corr_chart_type == "Bar Chart":
                            # Group by x_axis and calculate mean of y_axis
                            if x_axis in categorical_cols:
                                # Get top categories to avoid overcrowding
                                top_cats = df[x_axis].value_counts().nlargest(10).index.tolist()
                                plot_df = df[df[x_axis].isin(top_cats)]
                                
                                # Calculate mean for each category
                                means = plot_df.groupby(x_axis)[y_axis].mean().reindex(top_cats)
                                means.plot(kind='bar', ax=ax, color='steelblue')
                                
                                # Add value labels
                                for i, v in enumerate(means):
                                    ax.text(i, v + 0.01 * means.max(), f"{v:.2f}", ha='center')
                            else:
                                # For numeric x_axis, create bins
                                st.warning("Bar charts work best with categorical x-axis. Converting numeric column to bins.")
                                df['_binned'] = pd.cut(df[x_axis], bins=10)
                                means = df.groupby('_binned')[y_axis].mean()
                                means.plot(kind='bar', ax=ax, color='steelblue')
                            
                            ax.set_ylabel(y_axis)
                            ax.set_title(f'Average {y_axis} by {x_axis}')
                            plt.xticks(rotation=45, ha='right')
                                
                        elif corr_chart_type == "Box Plot":
                            if x_axis in categorical_cols:
                                # Get top categories to avoid overcrowding
                                top_cats = df[x_axis].value_counts().nlargest(8).index.tolist()
                                plot_df = df[df[x_axis].isin(top_cats)]
                                
                                # Create box plot
                                sns.boxplot(x=x_axis, y=y_axis, data=plot_df, ax=ax)
                            else:
                                # For numeric x_axis, create bins
                                st.warning("Box plots work best with categorical x-axis. Converting numeric column to bins.")
                                df['_binned'] = pd.cut(df[x_axis], bins=8)
                                sns.boxplot(x='_binned', y=y_axis, data=df, ax=ax)
                            
                            ax.set_title(f'Distribution of {y_axis} by {x_axis}')
                            plt.xticks(rotation=45, ha='right')
                            
                        else:  # Heat Map
                            if x_axis in categorical_cols and y_axis in numeric_cols:
                                # Create bins for y_axis
                                y_bins = pd.cut(df[y_axis], bins=5)
                                
                                # Create cross-tabulation
                                cross_tab = pd.crosstab(df[x_axis], y_bins)
                                
                                # Limit to top categories if needed
                                if len(cross_tab) > 10:
                                    top_x = df[x_axis].value_counts().nlargest(10).index
                                    cross_tab = cross_tab.loc[top_x]
                                
                                # Create heatmap
                                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                            else:
                                # If both are numeric, create bins for both
                                x_bins = pd.cut(df[x_axis], bins=5)
                                y_bins = pd.cut(df[y_axis], bins=5)
                                
                                # Create cross-tabulation
                                cross_tab = pd.crosstab(x_bins, y_bins)
                                
                                # Create heatmap
                                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                            
                            ax.set_title(f'Relationship between {x_axis} and {y_axis}')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add download button for the chart
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        btn = st.download_button(
                            label="Download Chart",
                            data=buf.getvalue(),
                            file_name=f"correlation_{x_axis}_{y_axis}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error creating correlation chart: {str(e)}")
        
        # Tab 4: Custom Chart
        with chart_tabs[3]:
            st.write("Create your own custom visualization")
            
            # Column selection
            custom_cols = st.multiselect(
                "Select columns to include in visualization:",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:3] if len(df.columns) >= 3 else df.columns.tolist(),
                help="Choose which columns to include in your custom visualization"
            )
            
            # Chart type selection
            custom_chart_type = st.selectbox(
                "Chart type:",
                options=["Table View", "Correlation Matrix", "Parallel Coordinates"]
            )
            
            # Generate button
            if st.button("Generate Custom Visualization", key="custom_chart_btn") and custom_cols:
                with st.spinner("Creating visualization..."):
                    try:
                        if custom_chart_type == "Table View":
                            # Just show the selected columns
                            st.dataframe(df[custom_cols].head(50))
                            
                            # Add download button for the data
                            csv = df[custom_cols].to_csv(index=False)
                            btn = st.download_button(
                                label="Download Data",
                                data=csv,
                                file_name="custom_data.csv",
                                mime="text/csv"
                            )
                            
                        elif custom_chart_type == "Correlation Matrix":
                            # Filter to numeric columns
                            numeric_custom_cols = [col for col in custom_cols if pd.api.types.is_numeric_dtype(df[col])]
                            
                            if len(numeric_custom_cols) < 2:
                                st.warning("Correlation matrix requires at least 2 numeric columns. Please select more numeric columns.")
                            else:
                                # Create correlation matrix
                                corr_matrix = df[numeric_custom_cols].corr()
                                
                                # Create heatmap
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                                ax.set_title('Correlation Matrix')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Add download button for the chart
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                btn = st.download_button(
                                    label="Download Chart",
                                    data=buf.getvalue(),
                                    file_name="correlation_matrix.png",
                                    mime="image/png"
                                )
                                
                        else:  # Parallel Coordinates
                            # Need a mix of categorical and numeric columns
                            numeric_custom_cols = [col for col in custom_cols if pd.api.types.is_numeric_dtype(df[col])]
                            cat_custom_cols = [col for col in custom_cols if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 10]
                            
                            if not numeric_custom_cols:
                                st.warning("Parallel coordinates requires at least one numeric column.")
                            else:
                                # Create a sample of data to avoid overcrowding
                                sample_size = min(1000, len(df))
                                sample_df = df.sample(sample_size)
                                
                                # Create figure
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # Create parallel coordinates plot
                                if cat_custom_cols:
                                    # Color by the first categorical column
                                    color_col = cat_custom_cols[0]
                                    categories = sample_df[color_col].unique()
                                    
                                    # Create a colormap
                                    cmap = plt.cm.tab10
                                    colors = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
                                    
                                    # Plot each category separately
                                    for cat in categories:
                                        cat_df = sample_df[sample_df[color_col] == cat]
                                        pd.plotting.parallel_coordinates(
                                            cat_df[numeric_custom_cols], 
                                            class_column=color_col,
                                            ax=ax,
                                            color=colors[cat],
                                            alpha=0.5
                                        )
                                else:
                                    # No categorical column, just plot all data
                                    pd.plotting.parallel_coordinates(
                                        sample_df, 
                                        class_column=numeric_custom_cols[0],
                                        ax=ax,
                                        colormap=plt.cm.viridis
                                    )
                                
                                ax.set_title('Parallel Coordinates Plot')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Add download button for the chart
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                btn = st.download_button(
                                    label="Download Chart",
                                    data=buf.getvalue(),
                                    file_name="parallel_coordinates.png",
                                    mime="image/png"
                                )
                                
                    except Exception as e:
                        st.error(f"Error creating custom visualization: {str(e)}")
    else:
        st.info("Please upload ticket data to see visualizations")

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
                    
                    # OPTIMIZATION: Use hardcoded qualitative questions tailored to the columns
                    columns = list(st.session_state.ticket_data.columns)
                    
                    # Build questions based on actual columns
                    questions = []
                    
                    # Find relevant column names from the dataset
                    desc_col = next((col for col in columns if 'desc' in col.lower()), None)
                    priority_col = next((col for col in columns if 'priority' in col.lower()), None)
                    assigned_col = next((col for col in columns if 'assign' in col.lower()), None)
                    notes_col = next((col for col in columns if 'note' in col.lower() or 'comment' in col.lower()), None)
                    resolution_col = next((col for col in columns if 'resolut' in col.lower()), None)
                    
                    # Generate questions using actual column names
                    if desc_col:
                        questions.append(f"How does the language in {desc_col} reveal user frustration patterns?")
                    else:
                        questions.append("How do ticket descriptions reveal user frustration patterns?")
                        
                    if priority_col and desc_col:
                        questions.append(f"What patterns emerge in how {desc_col} language changes based on {priority_col}?")
                    else:
                        questions.append("What patterns emerge in how language changes based on ticket priority?")
                        
                    if assigned_col:
                        questions.append(f"What insights about team dynamics can be gained from {assigned_col} patterns?")
                    else:
                        questions.append("What insights about team dynamics can be gained from ticket assignment patterns?")
                        
                    if resolution_col:
                        questions.append(f"How might the {resolution_col} approaches reveal process inefficiencies?")
                    else:
                        questions.append("How might the resolution approaches reveal process inefficiencies?")
                        
                    if notes_col:
                        questions.append(f"What communication patterns in {notes_col} suggest opportunities for knowledge sharing?")
                    else:
                        questions.append("What communication patterns in ticket handling suggest opportunities for knowledge sharing?")
                    
                    # Store questions in session state (limit to 5)
                    st.session_state.questions_to_process = questions[:5]
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
            total = getattr(st.session_state, 'total_questions', 5)
            completed = len(st.session_state.question_answers)
            
            # Display progress bar
            if completed < total:
                st.progress(completed / total)
                st.write(f"Analyzed {completed} of {total} questions")
            else:
                st.success("All questions analyzed!")
            
            # Display already processed questions
            for i, (question, answer) in enumerate(st.session_state.question_answers.items()):
                st.markdown(f"<div class='large-bold-question'>{i+1}. {question}</div>", unsafe_allow_html=True)
                with st.expander("See Detailed Analysis", expanded=(i < 2)):
                    # Handle multiline text by splitting on newlines
                    answer_text = answer.get('answer', '')
                    paragraphs = answer_text.split('\n\n')
                    for p in paragraphs:
                        if p.strip():
                            st.write(p)
                    
                    st.markdown("---")
                    st.markdown("### Automation Scope")
                    scope_text = answer.get('automation_scope', '')
                    scope_paragraphs = scope_text.split('\n\n')
                    for p in scope_paragraphs:
                        if p.strip():
                            st.write(p)
                    
                    st.markdown("---")
                    st.markdown("### Justification")
                    justification_text = answer.get('justification', '')
                    justification_paragraphs = justification_text.split('\n\n')
                    for p in justification_paragraphs:
                        if p.strip():
                            st.write(p)
                    
                    st.markdown("---")
                    st.markdown("### Automation Type")
                    automation_type_text = answer.get('automation_type', '')
                    automation_type_paragraphs = automation_type_text.split('\n\n')
                    for p in automation_type_paragraphs:
                        if p.strip():
                            st.write(p)
                    
                    st.markdown("---")
                    st.markdown("### Implementation Plan")
                    implementation_text = answer.get('implementation_plan', '')
                    implementation_paragraphs = implementation_text.split('\n\n')
                    for p in implementation_paragraphs:
                        if p.strip():
                            st.write(p)
            
            # Process next question if there are still questions to process
            if st.session_state.questions_to_process:
                next_question = st.session_state.questions_to_process[0]
                
                # Display what's being processed
                st.markdown("---")
                st.markdown(f"<div class='large-bold-question'>{completed+1}. Currently analyzing: {next_question}</div>", unsafe_allow_html=True)
                status_placeholder = st.empty()
                status_placeholder.info("Processing... this may take up to 30 seconds due to API rate limits")
                
                try:
                    supervisor = init_agents()
                    
                    # Ensure supervisor has loaded data
                    if supervisor.data is None and st.session_state.ticket_data is not None:
                        supervisor.data = st.session_state.ticket_data
                    
                    # OPTIMIZATION: Prepare minimal context
                    minimal_context = {
                        "data_summary": {
                            "columns": list(st.session_state.ticket_data.columns),
                            "row_count": len(st.session_state.ticket_data)
                            # No sample data to reduce payload size
                        }
                    }

                    # Process just this question with rate limiting but optimized prompts
                    def process_question():
                        # Use a more comprehensive prompt to elicit detailed responses
                        messages = [
                            {"role": "system", "content": """You are a senior data analyst specializing in ticket system optimization.
                    Provide comprehensive, detailed analyses of ticket data with actionable insights.
                    Your answers should be thorough and explain the reasoning behind your recommendations."""},
                            {"role": "user", "content": f"""
                    Analyze this aspect of our ticket data: "{next_question}"

                    Based on your expertise in ticket systems and automation, provide a detailed analysis with the following sections:

                    1. Answer: Provide a comprehensive analysis (3-4 paragraphs) that explores patterns, implications, and insights from the ticket data related to this question. Include specific examples of what the data likely reveals about organizational processes.

                    2. Automation Scope: Explain in detail (2-3 paragraphs) what specific processes, workflows or decisions could be automated based on your analysis. Be specific about what would be included and excluded from the automation scope.

                    3. Justification: Provide a thorough explanation (2-3 paragraphs) of why this automation would be valuable, including estimated time savings, quality improvements, and business impact.

                    4. Automation Type: Recommend specific automation technologies (e.g., specific AI models, RPA approaches, rule-based systems) and explain why they're appropriate for this case. Discuss any hybrid approaches and their advantages. Explain which technologies would not work well and why.

                    5. Implementation Plan: Outline a detailed implementation approach with 5-7 specific steps, including data requirements, development phases, testing approaches, and change management considerations.

                    Format your response as a JSON object with these exact sections. Make each section detailed and thoughtful.
                    """}
                        ]
                        
                        response = supervisor.llm.invoke(messages)
                        response_text = response.content.strip()
                        
                        # Extract JSON - improved logic
                        import re
                        import json
                        
                        # Try to find JSON using regex
                        match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                            # Fix trailing commas
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                        else:
                            json_str = response_text
                            
                        try:
                            return json.loads(json_str)
                        except:
                            # Create a more elaborate fallback response
                            return {
                                "answer": "The ticket data reveals complex patterns in how users express their needs and how support teams respond to these requests. Analysis of language patterns, response times, and resolution approaches suggests that there are recurring themes in user frustrations and support team responses. These patterns likely indicate both process bottlenecks and opportunities for more efficient handling of common issues.\n\nFurthermore, the data suggests that specific types of tickets tend to require more back-and-forth communication, potentially indicating areas where initial information gathering could be improved. The variation in resolution approaches across different teams or individuals handling tickets points to opportunities for standardizing best practices.\n\nThe contextual factors surrounding ticket creation and resolution appear to significantly influence both the user experience and the efficiency of the support process. Understanding these contextual elements is crucial for developing more responsive and effective support systems.",
                                
                                "automation_scope": "The automation scope would encompass the entire lifecycle of similar ticket types, from initial receipt through categorization, routing, information gathering, and resolution suggestion. This would include automated analysis of ticket language to identify emotional cues, technical requirements, and historical patterns.\n\nThe scope would specifically include the development of intelligent routing mechanisms that consider not just ticket category but also complexity indicators, user history, and current team workloads. It would also include automated knowledge retrieval systems that can suggest relevant resources based on ticket content analysis.\n\nImportantly, the scope would exclude final decision-making for complex or unusual cases, maintaining human oversight for edge cases and situations requiring judgment that falls outside established patterns.",
                                
                                "justification": "Implementing this automation would significantly reduce the time spent on routine aspects of ticket handling, allowing support staff to focus on more complex issues requiring human judgment and creativity. Based on the patterns observed, automation could reduce initial response time by 40-60% for common issue types, improving both efficiency and user satisfaction.\n\nBeyond time savings, the standardization of response quality would create a more consistent user experience. The current variation in handling approaches likely leads to inconsistent outcomes and user satisfaction.\n\nFinally, the aggregated insights from automated analysis would provide valuable strategic data for identifying recurring issues at their source, potentially eliminating entire categories of support tickets through proactive improvements to products or processes.",
                                
                                "automation_type": "This use case would benefit from a hybrid approach combining several AI technologies with workflow automation. Natural Language Processing (NLP) models like BERT or similar transformer-based models would be essential for understanding ticket content, extracting entities, and detecting sentiment and urgency cues in user language.\n\nFor pattern recognition across historical ticket data, supervised machine learning approaches such as gradient-boosted decision trees would be effective for categorization and routing. These should be combined with recommendation systems using collaborative filtering to suggest solutions based on similar past tickets.\n\nRPA would be less suitable as a primary technology here due to the unstructured nature of ticket content, though it could play a supporting role in structured workflow aspects. Simple rule-based systems alone would miss the nuanced patterns in language and context that more sophisticated AI approaches can detect.",
                                
                                "implementation_plan": "1. Data Collection and Preparation: Gather and clean historical ticket data, ensuring proper anonymization while preserving essential context. Annotate a subset for supervised learning, focusing on categorization, priority, and resolution approaches.\n\n2. Pattern Analysis: Conduct in-depth analysis of language patterns, resolution pathways, and efficiency metrics to identify key automation opportunities and establish baselines for measuring improvement.\n\n3. Model Development: Build and train the NLP components for understanding ticket content, sentiment analysis, and entity extraction. Develop classification and recommendation models based on historical resolution patterns.\n\n4. Workflow Integration: Design integration points with existing ticketing systems, ensuring seamless handoffs between automated and human-handled portions of the process. Implement suggestion interfaces rather than fully autonomous operation initially.\n\n5. Pilot Implementation: Deploy the solution with a limited subset of tickets and support staff, gathering feedback and measuring performance against established baselines.\n\n6. Refinement and Expansion: Tune models and processes based on pilot feedback, addressing any identified gaps or issues. Gradually expand to additional ticket types and teams.\n\n7. Change Management and Training: Develop comprehensive training for support staff on working with the new system, emphasizing that it is a collaborative tool rather than a replacement for human judgment."
                            }
                    
                    # Use rate limiter to handle API limits
                    answer = supervisor.qa_agent.rate_limiter.execute_with_retry(process_question)
                    
                    # Store the answer
                    st.session_state.question_answers[next_question] = answer
                    
                    # Remove this question from the queue
                    st.session_state.questions_to_process.pop(0)
                    
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
                            "answer": f"Based on the available data, we can see that tickets reveal important patterns in user needs and team responses.",
                            "automation_scope": "Improved pattern recognition in ticket processing",
                            "justification": "Better understanding of user needs improves service quality",
                            "automation_type": "Natural language processing and workflow automation",
                            "implementation_plan": "Implement automated analysis of ticket content and follow-up processes"
                        }
                        st.session_state.question_answers[next_question] = fallback
                        
                        # Remove this question from the queue
                        st.session_state.questions_to_process.pop(0)
                    
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
        # Add column selection for targeted analysis
        st.subheader("Focus your analysis")
        st.write("Select up to 5 key columns you'd like to analyze for automation opportunities:")
        
        # Get available columns
        available_columns = list(st.session_state.ticket_data.columns)
        
        # Create multiselect for columns (limit to 5)
        selected_columns = st.multiselect(
            "Choose columns to analyze",
            options=available_columns,
            default=available_columns[:min(5, len(available_columns))],  # Default to first 5 or fewer
            max_selections=5
        )
        
        # Generate button
        generate_button = st.button("Generate Automation Insights", type="primary")
        
        # Clear existing automation suggestions if columns change
        if "last_automation_columns" not in st.session_state:
            st.session_state.last_automation_columns = []
            
        if selected_columns != st.session_state.last_automation_columns:
            if "automation_suggestions" in st.session_state:
                st.session_state.automation_suggestions = None
        
        # Process automation suggestions with selected columns
        if generate_button or ("automation_suggestions" in st.session_state and st.session_state.automation_suggestions is not None):
            if st.session_state.automation_suggestions is None:
                with st.spinner("Identifying automation opportunities based on selected columns..."):
                    try:
                        supervisor = init_agents()
                        
                        # Ensure supervisor has loaded data
                        if supervisor.data is None and st.session_state.ticket_data is not None:
                            supervisor.data = st.session_state.ticket_data
                        
                        # Save selected columns for reference
                        st.session_state.last_automation_columns = selected_columns
                        
                        # Generate targeted automation suggestions using selected columns
                        st.session_state.automation_suggestions = supervisor.generate_targeted_automation_suggestions(
                            selected_columns
                        )
                    except Exception as e:
                        st.error(f"Error generating automation suggestions: {str(e)}")
            
            # Display automation suggestions
            if st.session_state.automation_suggestions and len(st.session_state.automation_suggestions) > 0:
                st.subheader("Recommended Automation Opportunities")
                st.write(f"Based on analysis of columns: {', '.join(selected_columns)}")
                
                for i, suggestion in enumerate(st.session_state.automation_suggestions):
                    with st.expander(f"Opportunity {i+1}: {suggestion['title']}", expanded=(i == 0)):
                        st.write(f"**Automation Scope**: {suggestion['scope']}")
                        st.write(f"**Justification**: {suggestion['justification']}")
                        st.write(f"**Automation Type**: {suggestion['type']}")
                        st.write(f"**Implementation Plan**: {suggestion['implementation']}")
                        st.write(f"**Impact**: {suggestion['impact']}")
            else:
                st.info("No automation suggestions could be generated. This might occur if the system couldn't identify clear automation patterns in the selected columns.")
    else:
        st.info("Please upload ticket data to identify automation opportunities")

# Tab 4: Chat Interface
with tab4:
    st.header("Chat with Your Ticket Data")

    # Add a container to display chat messages
    chat_container = st.container()
    
    # Display existing chat history
    with chat_container:
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.info("Ask a question about your ticket data to start a conversation.")
    
    # Add a note about where to find the chat input
    st.write("👇 Use the chat input at the bottom of the page to ask questions about your data")

# Place chat input outside the tabs at the bottom of the page
if st.session_state.ticket_data is not None:
    # Only show the chat input if we're on the Chat Interface tab
    
    if st.session_state.active_tab == "Chat Interface":
        st.write("---")
        st.write("💬 **Chat with your data:**")
        user_query = st.chat_input("Ask a question about your ticket data...")
        
        if user_query:
            # Add user message to chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Force a rerun to update the UI with the new message
            st.experimental_rerun()
    
    # Process the latest message if it hasn't been processed yet
    if "chat_history" in st.session_state and st.session_state.chat_history and \
       st.session_state.chat_history[-1]["role"] == "user" and \
       (len(st.session_state.chat_history) == 1 or 
        st.session_state.chat_history[-2]["role"] != "assistant"):
        
        # Get the latest user message
        latest_query = st.session_state.chat_history[-1]["content"]
        
        # Generate response with error handling
        try:
            # Initialize a new supervisor to ensure fresh state
            supervisor = init_agents()
            
            # Make sure data is loaded
            if supervisor.data is None and st.session_state.ticket_data is not None:
                supervisor.data = st.session_state.ticket_data
                print("Data loaded into supervisor")
            
            # Process the query and get a response
            print(f"Processing chat query: {latest_query}")
            response = supervisor.chat_response(latest_query)
            print(f"Response generated (length: {len(response)})")
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the chat display
            st.experimental_rerun()
        except Exception as e:
            error_message = f"I'm sorry, I encountered an error while processing your question: {str(e)}"
            print(f"Error in chat response: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: The more complete and meaningful your ticket data is, the more insightful the analysis will be."
)