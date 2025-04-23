"""
Conversational interface module for the Incident Management Analytics application.

This module provides a chat-based interface for users to interact with and query 
their incident data through natural language.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List
from models.llm_manager import LLMManager
from ui.utils.session_state import initialize_session_state

logger = logging.getLogger(__name__)

class ConversationPage:
    def __init__(self, data_loader, config=None):
        """
        Initialize Conversational Analytics Page
        
        Args:
            data_loader: Data loading and preprocessing utility
            config: Optional configuration
        """
        self.data_loader = data_loader
        self.config = config or {}
        
        # Initialize LLM manager for generating responses
        self.llm_manager = LLMManager(self.config)
        
        # Initialize session state for chat history if not already present
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def render_page(self, data: pd.DataFrame = None, is_data_sufficient: bool = False):
        """
        Render the Conversational Analytics page
        
        Args:
            data: Incident data DataFrame
            is_data_sufficient: Flag indicating if data is sufficient for analysis
        """
        st.header("💬 Conversational Incident Insights")
        
        # Load incident data if not provided
        if data is None:
            try:
                data = self.data_loader.load_processed_data()
            except Exception as e:
                st.error(f"Error loading incident data: {e}")
                return

        # Validate data sufficiency
        if data is None or len(data) < 50:
            st.warning("Insufficient data to enable conversational insights. "
                     "Please upload more incident tickets to start a conversation.")
            return

        # Render conversation interface
        self._render_conversation_interface(data)

    def _render_conversation_interface(self, data: pd.DataFrame):
        """
        Render the conversational interface for incident insights
        
        Args:
            data: Incident data DataFrame
        """
        # Display conversation history
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        # User input
        if prompt := st.chat_input("Ask a question about your incident data"):
            # Add user message to conversation history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': prompt
            })

            # Display user message
            with st.chat_message('user'):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message('assistant'):
                with st.spinner("Analyzing your data..."):
                    # Generate response based on the question
                    response = self._generate_response(data, prompt)
                    st.markdown(response)

            # Add AI response to conversation history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })

    def _generate_response(self, data: pd.DataFrame, query: str) -> str:
        """
        Generate response based on incident data and user query
        with comprehensive error handling.
        
        Args:
            data: Incident data DataFrame
            query: User's natural language query
        
        Returns:
            String response to the user's query
        """
        try:
            # Validate query
            if not query or len(query.strip()) < 3:
                return "Please ask a more specific question about the incident data."

            # Validate data availability
            if data is None or data.empty:
                return "No incident data is available. Please upload incident data first."

            # Validate data sufficiency
            if len(data) < 10:
                return "Insufficient incident data to generate meaningful insights. Please upload more data."

            # Prepare metadata about the data
            metadata = self._prepare_data_metadata(data)
            
            # Generate insights using LLM
            try:
                insights = self.llm_manager.generate_insights(
                    data=data,
                    metadata=metadata,
                    insight_type="conversational_query",
                    user_query=query
                )
                
                # Check if generation was successful
                if not insights.get("success", False):
                    error_msg = insights.get("error", "Unknown error generating response")
                    logger.error(f"Error generating insights: {error_msg}")
                    return f"I encountered an issue analyzing your data: {error_msg}"
                
                # Extract the answer from the insights
                if "answer" in insights:
                    return insights["answer"]
                else:
                    # Fallback to raw response if available
                    return insights.get("raw_response", "I couldn't generate a proper response to your query.")
                
            except Exception as insights_err:
                logger.error(f"Error generating insights: {str(insights_err)}")
                
                # Generate a simpler response based on basic data analysis
                return self._generate_fallback_response(data, query)

        except Exception as e:
            logger.error(f"Unexpected error in response generation: {str(e)}")
            
            return (
                "I apologize, but I encountered an unexpected error while processing your query. "
                "Please try again or rephrase your question."
            )
    
    def _prepare_data_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare metadata about the data for the LLM
        
        Args:
            data: Incident data DataFrame
            
        Returns:
            Dictionary with metadata about the data
        """
        metadata = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns)
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            metadata["numeric_stats"] = {}
            for col in numeric_cols:
                try:
                    stats = data[col].describe().to_dict()
                    metadata["numeric_stats"][col] = stats
                except:
                    pass
        
        # Add basic statistics for categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            metadata["categorical_stats"] = {}
            for col in cat_cols:
                try:
                    # Get top 5 categories
                    value_counts = data[col].value_counts().head(5).to_dict()
                    metadata["categorical_stats"][col] = value_counts
                except:
                    pass
        
        # Add date range if timestamp column is detected
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            try:
                for col in date_cols:
                    if pd.api.types.is_datetime64_any_dtype(data[col]):
                        min_date = data[col].min()
                        max_date = data[col].max()
                        metadata["date_range"] = {
                            "column": col,
                            "min_date": min_date.strftime("%Y-%m-%d") if not pd.isna(min_date) else None,
                            "max_date": max_date.strftime("%Y-%m-%d") if not pd.isna(max_date) else None
                        }
                        break
            except:
                pass
        
        return metadata
    
    def _generate_fallback_response(self, data: pd.DataFrame, query: str) -> str:
        """
        Generate a fallback response based on basic data analysis when LLM fails
        
        Args:
            data: Incident data DataFrame
            query: User's query
            
        Returns:
            Fallback response
        """
        response_parts = []
        
        # Add basic data summary
        response_parts.append(f"I analyzed your data with {len(data)} incidents across {len(data.columns)} dimensions.")
        
        # Try to identify what the query is about
        query = query.lower()
        
        # Check for time-related queries
        if any(term in query for term in ['time', 'when', 'date', 'period', 'month', 'year', 'day']):
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                col = date_cols[0]
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    min_date = data[col].min()
                    max_date = data[col].max()
                    date_range = (max_date - min_date).days if not pd.isna(min_date) and not pd.isna(max_date) else 0
                    
                    response_parts.append(f"Your data spans {date_range} days from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
                    
                    # If data has time component, add time distribution
                    if not all(data[col].dt.time == pd.Timestamp('00:00:00').time()):
                        hour_counts = data[col].dt.hour.value_counts().sort_index()
                        peak_hour = hour_counts.idxmax()
                        response_parts.append(f"The peak hour for incidents is {peak_hour}:00.")
        
        # Check for priority-related queries
        if any(term in query for term in ['priority', 'severity', 'urgent', 'critical']):
            priority_cols = [col for col in data.columns if 'priority' in col.lower() or 'severity' in col.lower()]
            if priority_cols:
                col = priority_cols[0]
                priority_counts = data[col].value_counts()
                top_priority = priority_counts.index[0] if not priority_counts.empty else "unknown"
                top_count = priority_counts.iloc[0] if not priority_counts.empty else 0
                top_pct = (top_count / len(data)) * 100 if len(data) > 0 else 0
                
                response_parts.append(f"The most common priority is '{top_priority}' ({top_pct:.1f}% of all incidents).")
        
        # Check for category-related queries
        if any(term in query for term in ['category', 'type', 'class']):
            category_cols = [col for col in data.columns if 'category' in col.lower() or 'type' in col.lower()]
            if category_cols:
                col = category_cols[0]
                category_counts = data[col].value_counts()
                top_category = category_counts.index[0] if not category_counts.empty else "unknown"
                top_count = category_counts.iloc[0] if not category_counts.empty else 0
                top_pct = (top_count / len(data)) * 100 if len(data) > 0 else 0
                
                response_parts.append(f"The most common category is '{top_category}' ({top_pct:.1f}% of all incidents).")
        
        # Check for resolution time queries
        if any(term in query for term in ['resolution', 'resolve', 'fixed', 'solved', 'time to']):
            resolution_cols = [col for col in data.columns if 'resolution' in col.lower() or 'time' in col.lower()]
            if resolution_cols:
                resolution_found = False
                for col in resolution_cols:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        mean_time = data[col].mean()
                        median_time = data[col].median()
                        if not pd.isna(mean_time) and not pd.isna(median_time):
                            response_parts.append(f"The average resolution time is {mean_time:.2f} hours, with a median of {median_time:.2f} hours.")
                            resolution_found = True
                            break
                
                if not resolution_found:
                    response_parts.append("I found potential resolution time columns but couldn't extract meaningful statistics.")
        
        # If we couldn't generate a specific response, add a generic one
        if len(response_parts) <= 1:
            response_parts.append("I couldn't generate a detailed response for your specific query. Try asking about incident counts, priorities, categories, or resolution times.")
        
        # Add a closing note
        response_parts.append("For more detailed insights, you might want to explore the other analysis tabs in the dashboard.")
        
        return "\n\n".join(response_parts)

def render_conversation_page(data_loader, config=None, data=None, is_data_sufficient=False):
    """
    Render the Conversational Insights page
    
    Args:
        data_loader: Data loading utility
        config: Optional configuration
        data: Optional preprocessed data 
        is_data_sufficient: Flag indicating if data is sufficient
    """
    # Ensure session state is initialized
    initialize_session_state()
    
    # Create and render the conversation page
    conversation_page = ConversationPage(data_loader, config)
    conversation_page.render_page(data, is_data_sufficient)