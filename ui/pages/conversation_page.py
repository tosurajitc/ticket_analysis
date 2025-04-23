import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from models.llm_manager import LLMManager
from analysis.insights_generator import InsightsGenerator

class ConversationPage:
    def __init__(self, data_loader, config=None):
        """
        Initialize Conversational Insights Page
        
        Args:
            data_loader: Data loading and preprocessing utility
        """
        self.data_loader = data_loader
        self.config = config or {}
        self.llm_interface = LLMManager(self.config)
        self.insights_generator = InsightsGenerator()
        
        # Initialize session state for conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Initialize session state for incident data
        if 'incident_data' not in st.session_state:
            st.session_state.incident_data = None

    def render_page(self):
        """
        Render the Conversational Insights page
        """
        st.header("💬 Conversational Incident Insights")
        
        # Load incident data
        try:
            # Load data only if not already loaded
            if st.session_state.incident_data is None:
                st.session_state.incident_data = self.data_loader.load_processed_data()
        except Exception as e:
            st.error(f"Error loading incident data: {e}")
            return

        # Validate data sufficiency
        if (st.session_state.incident_data is None or 
            len(st.session_state.incident_data) < 50):
            st.warning("Insufficient data to enable conversational insights. "
                       "Please upload more incident tickets to start a conversation.")
            return

        # Render conversation interface
        self._render_conversation_interface()

    def _render_conversation_interface(self):
        """
        Render the conversational interface for incident insights
        """
        # Display conversation history
        for message in st.session_state.conversation_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        # User input
        if prompt := st.chat_input("Ask a question about your incident data"):
            # Add user message to conversation history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': prompt
            })

            # Display user message
            with st.chat_message('user'):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message('assistant'):
                # Generate dynamic insights based on the question
                response = self._generate_ai_response(prompt)
                st.markdown(response)

            # Add AI response to conversation history
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response
            })

    def _generate_ai_response(self, query: str) -> str:
        """
        Generate AI response based on incident data and user query
        
        Args:
            query (str): User's natural language query
        
        Returns:
            str: AI-generated response
        """
        try:
            # Validate query
            if not query or len(query.strip()) < 3:
                return "Please ask a more specific question about the incident data."

            # Analyze query intent
            query_intent = self.llm_interface.classify_query_intent(query)

            # Generate context-aware insights
            insights = self.insights_generator.generate_contextual_insights(
                st.session_state.incident_data,
                query_intent,
                query
            )

            # Format response
            if not insights:
                return "I couldn't find specific insights related to your query. Could you please rephrase or ask a different question?"

            return insights

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm sorry, but I encountered an error while processing your query."

def render_conversation_page(data_loader, config=None):
    """
    Render the Conversational Insights page
    
    Args:
        data_loader: Data loading utility
        config: Configuration dictionary (optional)
    """
    conversation_page = ConversationPage(data_loader, config)
    conversation_page.render_page()