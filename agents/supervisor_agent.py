import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any, Optional, Tuple
import os
import io
import traceback  # For better error reporting
from utils.json_utils import make_json_serializable, safe_json_dumps, safe_json_loads
from utils.rate_limiter import RateLimiter

class SupervisorAgent:
    """
    Supervisor agent that coordinates all other agents and validates their responses.
    """
    
    def __init__(self, llm, data_agent, analysis_agent, visualization_agent, 
                 automation_agent, qa_agent, chat_agent):
        self.llm = llm
        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.visualization_agent = visualization_agent
        self.automation_agent = automation_agent
        self.qa_agent = qa_agent
        self.chat_agent = chat_agent
        self.rate_limiter = RateLimiter(base_delay=2.0, max_retries=2, max_delay=30.0)
        
        # Shared state
        self.data = None
        self.chunked_data = []
        self.analysis_results = None
        self.insights = None
        self.visualizations = None
        self.automation_suggestions = None
        self.qualitative_answers = None
    
    def process_data(self, uploaded_file, chunk_size: int = 500, column_hints: List[str] = None) -> pd.DataFrame:
        """
        Process uploaded data file using the data agent
        """
        try:
            # Load data through data agent
            self.data = self.data_agent.load_data(uploaded_file, column_hints)
            
            if self.data is not None and len(self.data) > 0:
                # Chunk the data for processing
                self.chunked_data = self.data_agent.chunk_data(self.data, chunk_size)
                
                # If no column hints were provided, use heuristics to suggest important columns
                if not column_hints or len(column_hints) == 0:
                    try:
                        suggested_columns = self.data_agent.suggest_important_columns(self.data)
                        print(f"Suggested important columns: {suggested_columns}")
                    except Exception as e:
                        print(f"Error suggesting columns: {str(e)}")
                
                # Reset analysis states when new data is loaded
                self._reset_state()
                
                # Pre-generate basic analysis but don't let failures stop us
                try:
                    print("Pre-generating analysis...")
                    self.analysis_results = self.analysis_agent.analyze_data(self.data)
                    print("Analysis complete")
                    
                    # Try to pre-generate insights if analysis was successful
                    if self.analysis_results:
                        try:
                            print("Pre-generating insights...")
                            self.insights = self.analysis_agent.generate_insights(self.analysis_results)
                            print("Initial insights generated")
                        except Exception as e:
                            print(f"Error pre-generating insights: {str(e)}")
                            self.insights = self._create_empty_insights()
                except Exception as e:
                    print(f"Error pre-generating analysis: {str(e)}")
                    self.analysis_results = {}
                    self.insights = self._create_empty_insights()
                
                return self.data
            else:
                print("Error: Data loading returned empty dataset")
                self._reset_state()
                return None
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            self._reset_state()
            return None
    
    def _reset_state(self):
        """Reset all state variables when data loading fails"""
        # Leave self.data as is - it's what we're loading
        self.chunked_data = []
        self.analysis_results = {}
        self.insights = self._create_empty_insights()
        self.visualizations = []
        self.automation_suggestions = []
        self.qualitative_answers = []
    
    def _create_empty_insights(self):
        """Create empty insights structure"""
        return {
            "volume_insights": [],
            "time_insights": [],
            "category_insights": [],
            "efficiency_insights": [],
            "automation_insights": [],
        }
        
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights from the data
        """
        if self.data is None:
            print("No data loaded. Call process_data first.")
            return {
                "analysis": {},
                "insights": self._create_empty_insights()
            }
        
        try:
            # If we already have analysis results from pre-generation, use those
            if not self.analysis_results:
                print("Generating analysis...")
                self.analysis_results = self.analysis_agent.analyze_data(self.data)
                print("Analysis complete")
            
            # Generate insights from analysis results
            if self.analysis_results:
                try:
                    print("Generating insights...")
                    # We'll try to generate insights separately for each category to isolate errors
                    basic_structure = self._create_empty_insights()
                    
                    try:
                        basic_structure["volume_insights"] = self.analysis_agent._generate_volume_insights(self.analysis_results)
                        print("Volume insights generated")
                    except Exception as e:
                        print(f"Error generating volume insights: {str(e)}")
                    
                    try:
                        basic_structure["time_insights"] = self.analysis_agent._generate_time_insights(self.analysis_results)
                        print("Time insights generated")
                    except Exception as e:
                        print(f"Error generating time insights: {str(e)}")
                        
                    try:
                        basic_structure["category_insights"] = self.analysis_agent._generate_category_insights(self.analysis_results)
                        print("Category insights generated")
                    except Exception as e:
                        print(f"Error generating category insights: {str(e)}")
                        
                    try:
                        basic_structure["efficiency_insights"] = self.analysis_agent._generate_efficiency_insights(self.analysis_results)
                        print("Efficiency insights generated")
                    except Exception as e:
                        print(f"Error generating efficiency insights: {str(e)}")
                        
                    try:
                        basic_structure["automation_insights"] = self.analysis_agent._generate_automation_insights(self.analysis_results)
                        print("Automation insights generated")
                    except Exception as e:
                        print(f"Error generating automation insights: {str(e)}")
                    
                    self.insights = basic_structure
                    print("Insights complete")
                except Exception as e:
                    print(f"Error in overall insight generation: {str(e)}")
                    if not self.insights:
                        self.insights = self._create_empty_insights()
            else:
                # Create empty insights structure if no analysis
                self.insights = self._create_empty_insights()
            
            # Return combined results
            results = {
                "analysis": self.analysis_results or {},
                "insights": self.insights or self._create_empty_insights()
            }
            
            return results
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            # Return empty results instead of None to prevent cascading errors
            return {
                "analysis": {},
                "insights": self._create_empty_insights()
            }
    
    def generate_visualizations(self) -> List[plt.Figure]:
        """
        Generate visualizations of the data
        """
        if self.data is None:
            print("No data loaded. Call process_data first.")
            return []
        
        if self.analysis_results is None:
            # Try to generate insights first, or use empty dict if that fails
            result = self.generate_insights()
            if result is None:
                self.analysis_results = {}
        
        try:
            # Generate visualizations
            # Make sure analysis results are serializable
            serializable_analysis = make_json_serializable(self.analysis_results or {})
            
            self.visualizations = self.visualization_agent.generate_visualizations(self.data, serializable_analysis)
            return self.visualizations
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return []
    
    def generate_automation_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate automation suggestions based on the data
        """
        if self.data is None:
            print("No data loaded. Call process_data first.")
            return []
        
        if self.analysis_results is None:
            # Try to generate insights first, or use empty dict if that fails
            result = self.generate_insights()
            if result is None:
                self.analysis_results = {}
        
        try:
            # Generate automation suggestions
            print("Generating automation suggestions...")
            # Ensure analysis results are JSON serializable
            serializable_analysis = make_json_serializable(self.analysis_results or {})
            
            self.automation_suggestions = self.automation_agent.identify_automation_opportunities(
                self.data, serializable_analysis
            )
            print("Automation suggestions generated")
            
            # Validate and enhance suggestions
            self.automation_suggestions = self._validate_automation_suggestions(self.automation_suggestions)
            
            return self.automation_suggestions
        except Exception as e:
            print(f"Error generating automation suggestions: {str(e)}")
            return []
    
    def generate_qualitative_answers(self) -> List[Dict[str, Any]]:
        """
        Generate answers to qualitative questions
        """
        if self.data is None:
            print("No data loaded. Call process_data first.")
            return []
        
        if self.analysis_results is None:
            # Try to generate insights first, or use empty dict if that fails
            result = self.generate_insights()
            if result is None:
                self.analysis_results = {}
        
        try:
            # Generate qualitative answers
            print("Generating qualitative answers...")
            
            # Estimate and log the time
            question_count = 10  # Default question count
            avg_time_per_question = 15  # Average seconds per question
            estimated_seconds = question_count * avg_time_per_question
            print(f"Estimated processing time: {estimated_seconds} seconds")
            
            # Ensure analysis results are JSON serializable
            serializable_analysis = make_json_serializable(self.analysis_results or {})
            
            # Make a copy of the dataframe to avoid timestamp issues
            df_copy = self.data.copy()
            
            # Convert any timestamp columns to strings in the copy
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    print(f"Converting timestamp column '{col}' to string format")
                    df_copy[col] = df_copy[col].astype(str)
            
            start_time = time.time()
            self.qualitative_answers = self.qa_agent.generate_qualitative_answers(
                df_copy, serializable_analysis
            )
            
            actual_time = time.time() - start_time
            print(f"Qualitative answers generated in {actual_time:.2f} seconds")
            
            # Validate and enhance answers
            self.qualitative_answers = self._validate_qualitative_answers(self.qualitative_answers)
            
            return self.qualitative_answers
        except Exception as e:
            print(f"Error generating qualitative answers: {str(e)}")
            traceback.print_exc()
            return []
    
    def chat_response(self, query: str) -> str:
        """
        Process a chat query and generate a response
        """
        if self.data is None:
            return "Please upload ticket data before asking questions."
        
        if self.analysis_results is None:
            self.generate_insights()
        
        try:
            # Make analysis results JSON serializable
            serializable_analysis = make_json_serializable(self.analysis_results)
            
            # Process query through chat agent
            print("Processing chat query...")
            response = self.chat_agent.process_query(query, self.data, serializable_analysis)
            print("Chat response generated")
            
            # Validate response
            validated_response = self._validate_chat_response(query, response)
            
            return validated_response
        except Exception as e:
            print(f"Error processing chat query: {str(e)}")
            return f"I'm sorry, I encountered an issue while processing your query: {str(e)}"
    
    def _validate_automation_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and enhance automation suggestions
        """
        if not suggestions or len(suggestions) == 0:
            return []
        
        # Format validation
        validated_suggestions = []
        for suggestion in suggestions:
            # Ensure all required fields are present
            if "title" not in suggestion:
                suggestion["title"] = "Untitled Automation Opportunity"
            
            required_fields = ["scope", "justification", "type", "implementation", "impact"]
            for field in required_fields:
                if field not in suggestion:
                    suggestion[field] = f"No {field} information provided."
            
            validated_suggestions.append(suggestion)
        
        return validated_suggestions
    
    def _validate_qualitative_answers(self, answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and enhance qualitative answers
        """
        if not answers or len(answers) == 0:
            return []
        
        # Format validation
        validated_answers = []
        for answer in answers:
            # Ensure all required fields are present
            if "question" not in answer:
                continue
            
            required_fields = ["answer", "automation_scope", "justification", "automation_type", "implementation_plan"]
            for field in required_fields:
                if field not in answer:
                    answer[field] = f"No {field} information provided."
            
            validated_answers.append(answer)
        
        return validated_answers
    
    def _validate_chat_response(self, query: str, response: str) -> str:
        """
        Validate and enhance chat responses
        """
        if not response or len(response.strip()) == 0:
            return "I'm sorry, I couldn't generate a response based on the available data. Please try a different question."
        
        # For simple validation, just return the response without additional API calls
        # This avoids unnecessary rate limit issues
        return response