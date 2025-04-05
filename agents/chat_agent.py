import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from utils.json_utils import make_json_serializable
from utils.rate_limiter import RateLimiter

class ChatAgent:
    """
    Agent responsible for handling user chat queries about the ticket data.
    Enhanced with support for column hints to prioritize important columns.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.rate_limiter = RateLimiter(base_delay=2.0, max_retries=3, max_delay=60.0)
        # Store column hints for prioritization
        self.column_hints = []
        # Store data and analysis results
        self.data = None
        self.analysis_results = None
    
    def set_column_hints(self, column_hints: List[str]):
        """
        Set the list of important columns to focus on during analysis
        
        Args:
            column_hints (List[str]): List of column names to prioritize
        """
        self.column_hints = column_hints if column_hints else []
        print(f"Set {len(self.column_hints)} column hints for chat agent")
    
    def _reduce_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce the size of analysis results to avoid token limits
        
        Args:
            analysis_results (Dict[str, Any]): Original analysis results
        
        Returns:
            Dict[str, Any]: Reduced analysis results
        """
        if not analysis_results:
            return {}
        
        reduced_results = {}
        try:
            # Limit depth and size of nested structures
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    # Truncate dictionary values
                    reduced_results[key] = {
                        k: (v[:50] if isinstance(v, list) else str(v)[:200]) 
                        for k, v in value.items()
                    }
                elif isinstance(value, list):
                    # Limit list to first 20 items
                    reduced_results[key] = value[:20]
                else:
                    # Convert other types to strings and limit length
                    reduced_results[key] = str(value)[:500]
        except Exception as e:
            print(f"Error reducing analysis results: {e}")
            return analysis_results  # Fallback to original if reduction fails
        
        return reduced_results
    
    def process_query(self, query: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """
        Process a user query about the ticket data with focus on important columns
        
        Args:
            query (str): User's query about the data
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            str: Natural language response to the query
        """
        # Store data for possible future use
        self.data = df
        self.analysis_results = analysis_results
        
        # Limit data size for processing
        if len(df) > 1000:
            df = df.sample(1000)

        # Convert datetime columns to strings for easier processing
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        # Ensure analysis results are JSON serializable
        analysis_results = make_json_serializable(analysis_results)
        
        # Analyze the query to determine the type of question
        query_category = self._categorize_query(query)
        
        # Extract relevant data for the query, considering column hints
        relevant_data = self._extract_relevant_data(query, query_category, df, analysis_results)
        
        # Generate response with rate limiting
        def generate_response():
            return self._generate_response(query, query_category, relevant_data, df, analysis_results)
        
        try:
            return self.rate_limiter.execute_with_retry(generate_response)
        except Exception as e:
            print(f"Error processing chat query with rate limiting: {str(e)}")
            return f"I'm sorry, I encountered an issue while processing your query: {str(e)}. This might be due to system limitations. Could you try again with a simpler question?"
    
    def _categorize_query(self, query: str) -> str:
        """
        Categorize the type of query to determine how to process it
        
        Args:
            query (str): User's query
            
        Returns:
            str: Query category (statistics, trends, specific_data, etc.)
        """
        # Use LLM to categorize the query with rate limiting
        def categorize():
            messages = [
                {"role": "system", "content": "You are an expert in categorizing questions about ticket data."},
                {"role": "user", "content": f"""
    Categorize the following user query about ticket data into one of these categories:
    - statistics: Questions about counts, averages, distributions
    - trends: Questions about changes over time or patterns
    - specific_data: Questions about specific tickets or values
    - comparison: Questions comparing different categories or attributes
    - recommendation: Questions asking for suggestions or recommendations
    - explanation: Questions asking for explanations of data patterns

    Query: {query}

    Respond with just the category name, no explanation.
    """}
            ]
            
            response = self.llm.invoke(messages)
            return response
        
        try:
            response = self.rate_limiter.execute_with_retry(categorize)
            category = response.content.strip().lower()
            
            # Validate category
            valid_categories = ["statistics", "trends", "specific_data", "comparison", "recommendation", "explanation"]
            if category in valid_categories:
                return category
            else:
                return "statistics"  # Default category
        except Exception as e:
            print(f"Error categorizing query: {str(e)}")
            return "statistics"  # Default category
    
    def _extract_relevant_data(self, query: str, query_category: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data relevant to the query, prioritizing important columns
        
        Args:
            query (str): User's query
            query_category (str): Type of query
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Relevant data for answering the query
        """
        relevant_data = {}
        
        # Process based on query category
        if query_category == "statistics":
            relevant_data = self._extract_statistical_data(query, df, analysis_results)
        elif query_category == "trends":
            relevant_data = self._extract_trend_data(query, df, analysis_results)
        elif query_category == "specific_data":
            relevant_data = self._extract_specific_data(query, df)
        elif query_category == "comparison":
            relevant_data = self._extract_comparison_data(query, df, analysis_results)
        elif query_category == "recommendation":
            relevant_data = self._extract_recommendation_data(query, analysis_results)
        elif query_category == "explanation":
            relevant_data = self._extract_explanation_data(query, analysis_results)
        
        # Add column hints to relevant data for context
        if self.column_hints:
            relevant_data["important_columns"] = self.column_hints
        
        return relevant_data
    
    def _extract_statistical_data(self, query: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract statistical data relevant to the query, prioritizing important columns
        
        Args:
            query (str): User's query
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Statistical data relevant to the query
        """
        # First prioritize column hints if they exist
        prioritized_columns = []
        if self.column_hints:
            # Check which column hints are relevant to this query
            columns_str = ", ".join(self.column_hints)
            
            messages = [
                {"role": "system", "content": "You are a data analysis expert. Identify which columns are relevant for a query."},
                {"role": "user", "content": f"""
Given these important columns: {columns_str}
Which of these columns would be relevant to answer this query: "{query}"?
List only the relevant column names, comma-separated, no explanations.
If none are relevant, respond with "none".
"""}
            ]
            
            try:
                response = self.llm.invoke(messages)
                if response.content.strip().lower() != "none":
                    prioritized_columns = [col.strip() for col in response.content.split(',') if col.strip() in self.column_hints]
            except Exception as e:
                print(f"Error prioritizing columns: {str(e)}")
        
        # Use LLM to identify additional columns needed for this query
        messages = [
            {"role": "system", "content": "You are a data analysis expert. Identify columns needed to answer statistical queries."},
            {"role": "user", "content": f"""
Given the following columns in a ticket dataset: {list(df.columns)}

Which columns would be needed to answer this statistical query: "{query}"?
List only the column names, comma-separated, no explanations.
"""}
        ]
        
        try:
            response = self.llm.invoke(messages)
            suggested_columns = [col.strip() for col in response.content.split(',')]
            
            # Combine prioritized columns with suggested columns
            columns = prioritized_columns + [col for col in suggested_columns if col not in prioritized_columns]
            
            # Filter to include only valid columns
            valid_columns = [col for col in columns if col in df.columns]
            
            # Get basic statistics for these columns
            stats = {}
            for col in valid_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        "mean": make_json_serializable(df[col].mean()),
                        "median": make_json_serializable(df[col].median()),
                        "min": make_json_serializable(df[col].min()),
                        "max": make_json_serializable(df[col].max())
                    }
                else:
                    # For categorical columns, get top values distribution
                    value_counts = df[col].value_counts(normalize=True)
                    top_values = value_counts.head(10).to_dict()
                    stats[col] = {
                        "top_values": make_json_serializable(top_values),
                        "unique_count": df[col].nunique()
                    }
            
            # Include relevant analysis results
            category_analysis = {k: v for k, v in analysis_results.get("category_analysis", {}).items() 
                              if any(col in k for col in valid_columns)}
            
            priority_analysis = {k: v for k, v in analysis_results.get("priority_analysis", {}).items() 
                              if any(col in k for col in valid_columns)}
            
            return make_json_serializable({
                "query_type": "statistical",
                "relevant_columns": valid_columns,
                "prioritized_columns": prioritized_columns,  # Include which columns were prioritized
                "statistics": stats,
                "category_analysis": category_analysis,
                "priority_analysis": priority_analysis
            })
        except Exception as e:
            print(f"Error extracting statistical data: {str(e)}")
            return {"query_type": "statistical", "error": str(e)}
    
    def _extract_trend_data(self, query: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract trend data relevant to the query, prioritizing important columns
        
        Args:
            query (str): User's query
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Trend data relevant to the query
        """
        # Identify prioritized time columns
        prioritized_time_cols = []
        if self.column_hints:
            for col in self.column_hints:
                if col in df.columns and any(time_term in col.lower() for time_term in ['date', 'time', 'created', 'updated', 'resolved']):
                    prioritized_time_cols.append(col)
        
        # Identify additional time-related columns
        time_cols = prioritized_time_cols.copy()
        for col in df.columns:
            if col not in time_cols and any(time_term in col.lower() for time_term in ['date', 'time', 'created', 'updated', 'resolved']):
                time_cols.append(col)
        
        # Try to convert time columns to datetime
        time_data = {}
        for col in time_cols:
            try:
                df[f'{col}_dt'] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_month'] = df[f'{col}_dt'].dt.to_period('M')
                
                # Get monthly counts
                monthly_counts = df.groupby(f'{col}_month').size().to_dict()
                time_data[col] = {str(k): v for k, v in monthly_counts.items()}
                
                # If there are prioritized columns for analysis
                if prioritized_time_cols and col in prioritized_time_cols:
                    # Get more detailed trend analysis for prioritized columns
                    # For example, get weekly counts instead of just monthly
                    df[f'{col}_week'] = df[f'{col}_dt'].dt.to_period('W')
                    weekly_counts = df.groupby(f'{col}_week').size().to_dict()
                    time_data[f"{col}_weekly"] = {str(k): v for k, v in weekly_counts.items()}
                
                # Clean up temporary columns
                df.drop([f'{col}_dt', f'{col}_month'], axis=1, errors='ignore', inplace=True)
                if f'{col}_week' in df.columns:
                    df.drop([f'{col}_week'], axis=1, errors='ignore', inplace=True)
            except Exception as e:
                print(f"Error processing time column {col}: {str(e)}")
        
        # Include time analysis from analysis results
        time_analysis = analysis_results.get("time_analysis", {})
        
        return {
            "query_type": "trend",
            "time_columns": time_cols,
            "prioritized_time_columns": prioritized_time_cols,
            "time_data": time_data,
            "time_analysis": time_analysis
        }
    
    def _extract_specific_data(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract specific data points relevant to the query, prioritizing important columns
        
        Args:
            query (str): User's query
            df (pd.DataFrame): Ticket data
            
        Returns:
            Dict[str, Any]: Specific data relevant to the query
        """
        # Use LLM to identify potential filter conditions, with emphasis on important columns
        filter_prompt = f"""
Given the following columns in a ticket dataset: {list(df.columns)}

"""
        # Add emphasis on prioritized columns if available
        if self.column_hints:
            filter_prompt += f"With special attention to these important columns: {self.column_hints}\n\n"
        
        filter_prompt += f"""
Extract filter conditions from this query: "{query}"
Format your response as a JSON object with column names as keys and filter values as values.
Example: {{"priority": "high", "status": "open"}}
"""
        
        messages = [
            {"role": "system", "content": "You are a data filtering expert. Extract filter conditions from natural language queries."},
            {"role": "user", "content": filter_prompt}
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Extract JSON from response
            response_text = response.content
            
            # Handle case where response is wrapped in markdown
            if "```json" in response_text and "```" in response_text.split("```json")[1]:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text and "```" in response_text.split("```")[1]:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            filter_conditions = json.loads(json_str)
            
            # Apply filters to dataframe
            filtered_df = df.copy()
            filter_applied = False
            applied_filters = {}
            
            for col, value in filter_conditions.items():
                if col in filtered_df.columns:
                    # Convert value to list if it's not already
                    if not isinstance(value, list):
                        value = [value]
                    
                    # Apply filter
                    original_count = len(filtered_df)
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.lower().isin([str(v).lower() for v in value])]
                    filter_applied = True
                    
                    # Record how many rows this filter removed
                    applied_filters[col] = {
                        "values": value,
                        "removed": original_count - len(filtered_df)
                    }
            
            # Get sample of filtered data
            if filter_applied and len(filtered_df) > 0:
                # Get additional information for important columns
                enhanced_sample = filtered_df.head(min(5, len(filtered_df)))
                
                # Add extra statistics for important columns in the filtered data
                enhanced_stats = {}
                if self.column_hints:
                    for col in self.column_hints:
                        if col in filtered_df.columns:
                            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                                enhanced_stats[col] = {
                                    "mean": make_json_serializable(filtered_df[col].mean()),
                                    "median": make_json_serializable(filtered_df[col].median())
                                }
                            else:
                                value_counts = filtered_df[col].value_counts(normalize=True)
                                enhanced_stats[col] = {
                                    "top_values": make_json_serializable(value_counts.head(3).to_dict())
                                }
                
                sample = make_json_serializable(enhanced_sample.to_dict(orient='records'))
                count = len(filtered_df)
            else:
                sample = []
                count = 0
                enhanced_stats = {}
            
            return {
                "query_type": "specific_data",
                "filter_conditions": filter_conditions,
                "applied_filters": applied_filters,
                "matched_count": count,
                "sample": sample,
                "enhanced_stats": enhanced_stats,
                "total_count": len(df)
            }
        except Exception as e:
            print(f"Error extracting specific data: {str(e)}")
            return {"query_type": "specific_data", "error": str(e)}
    
    def _extract_comparison_data(self, query: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comparison data relevant to the query, prioritizing important columns
        
        Args:
            query (str): User's query
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Comparison data relevant to the query
        """
        # First check if any important columns are relevant for comparison
        suggested_group_by = None
        suggested_measure = None
        
        if self.column_hints and len(self.column_hints) >= 2:
            # Try to identify categorical and numeric columns from hints
            categorical_hints = []
            numeric_hints = []
            
            for col in self.column_hints:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_hints.append(col)
                    else:
                        categorical_hints.append(col)
            
            # If we have both types, suggest them for comparison
            if categorical_hints and numeric_hints:
                suggested_group_by = categorical_hints[0]
                suggested_measure = numeric_hints[0]
        
        # Use LLM to identify columns for comparison, with suggestions
        comparison_prompt = f"""
Given the following columns in a ticket dataset: {list(df.columns)}

"""
        if suggested_group_by and suggested_measure:
            comparison_prompt += f"Based on the data structure, '{suggested_group_by}' and '{suggested_measure}' might be good candidates for comparison.\n\n"
        
        comparison_prompt += f"""
For this comparison query: "{query}"
Identify:
1. The column to group by (category)
2. The column to measure/compare
Format your response as a JSON object with two keys: "group_by" and "measure"
Example: {{"group_by": "category", "measure": "resolution_time"}}
"""
        
        messages = [
            {"role": "system", "content": "You are a data analysis expert. Identify columns for comparison from queries."},
            {"role": "user", "content": comparison_prompt}
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Extract JSON from response
            response_text = response.content
            
            # Handle case where response is wrapped in markdown
            if "```json" in response_text and "```" in response_text.split("```json")[1]:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text and "```" in response_text.split("```")[1]:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            comparison_info = json.loads(json_str)
            
            group_by = comparison_info.get("group_by", "")
            measure = comparison_info.get("measure", "")
            
            comparison_data = {}
            
            # Check if columns exist
            if group_by in df.columns:
                if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                    # Calculate multiple statistics by group
                    comparison_data = df.groupby(group_by)[measure].agg(['mean', 'median', 'count', 'std']).to_dict(orient='index')
                    
                    # Check if this is a prioritized column for deeper analysis
                    if self.column_hints and (group_by in self.column_hints or measure in self.column_hints):
                        # Enhance with percentile information for deeper understanding
                        percentiles = df.groupby(group_by)[measure].quantile([0.25, 0.75]).unstack().to_dict()
                        for group, values in percentiles.items():
                            if group in comparison_data:
                                comparison_data[group].update({f"percentile_{int(p*100)}": v for p, v in values.items()})
                else:
                    # Just do counts by group
                    comparison_data = df[group_by].value_counts().to_dict()
            
            # Include relevant cross-tabulations from analysis
            category_analysis = analysis_results.get("category_analysis", {})
            cross_tabs = {k: v for k, v in category_analysis.items() if "_by_" in k and 
                         (group_by in k or measure in k)}
            
            # Flag if these were prioritized columns
            is_prioritized = (group_by in self.column_hints or measure in self.column_hints)
            
            return {
                "query_type": "comparison",
                "group_by": group_by,
                "measure": measure,
                "is_prioritized": is_prioritized,
                "comparison_data": comparison_data,
                "cross_tabs": cross_tabs
            }
        except Exception as e:
            print(f"Error extracting comparison data: {str(e)}")
            return {"query_type": "comparison", "error": str(e)}
    
    def _extract_recommendation_data(self, query: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract recommendation-related data, prioritizing important columns
        
        Args:
            query (str): User's query
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Recommendation data relevant to the query
        """
        # Include relevant insights from analysis
        insights = analysis_results.get("insights", {})
        
        # Include automation insights
        automation_insights = insights.get("automation_insights", [])
        
        # Include efficiency insights
        efficiency_insights = insights.get("efficiency_insights", [])
        
        # Prioritize insights related to important columns
        prioritized_insights = []
        if self.column_hints:
            # Extract insights specifically related to important columns
            for insight_type, insight_list in insights.items():
                for insight in insight_list:
                    if any(col in str(insight) for col in self.column_hints):
                        prioritized_insights.append({
                            "type": insight_type,
                            "insight": insight,
                            "related_columns": [col for col in self.column_hints if col in str(insight)]
                        })
        
        return {
            "query_type": "recommendation",
            "automation_insights": automation_insights,
            "efficiency_insights": efficiency_insights,
            "prioritized_insights": prioritized_insights,
            "important_columns": self.column_hints
        }
    
    def _extract_explanation_data(self, query: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract explanation-related data, prioritizing important columns
        
        Args:
            query (str): User's query
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Explanation data relevant to the query
        """
        # Include all insights
        insights = analysis_results.get("insights", {})
        
        # Include text analysis for context
        text_analysis = analysis_results.get("text_analysis", {})
        
        # Prioritize insights and analysis related to important columns
        prioritized_data = {}
        if self.column_hints:
            for col in self.column_hints:
                # Extract relevant insights
                col_insights = []
                for insight_type, insight_list in insights.items():
                    for insight in insight_list:
                        if col in str(insight):
                            col_insights.append({
                                "type": insight_type,
                                "insight": insight
                            })
                
                # Extract relevant analysis
                col_analysis = {k: v for k, v in analysis_results.items() if col in str(k)}
                
                # Store the prioritized data
                if col_insights or col_analysis:
                    prioritized_data[col] = {
                        "insights": col_insights,
                        "analysis": col_analysis
                    }
        
        return {
            "query_type": "explanation",
            "insights": insights,
            "text_analysis": text_analysis,
            "prioritized_data": prioritized_data,
            "important_columns": self.column_hints
        }
    
    def _generate_response(self, query: str, query_category: str, relevant_data: Dict[str, Any], df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a natural language response to the user query with focus on important columns
        
        Args:
            query (str): User's query
            query_category (str): Type of query
            relevant_data (Dict[str, Any]): Relevant data for answering the query
            df (pd.DataFrame): Ticket data
            analysis_results (Dict[str, Any]): Pre-computed analysis results
            
        Returns:
            str: Natural language response to the query
        """
        # Create a context object for the LLM
        # Make all data serializable
        context = make_json_serializable({
            "query": query,
            "query_category": query_category,
            "relevant_data": relevant_data,
            "total_tickets": len(df),
            "column_names": list(df.columns),
            "important_columns": self.column_hints  # Add important columns for context
        })
        
        # Determine how much focus should be on important columns
        prioritization_guidance = ""
        if self.column_hints:
            # Check if the relevant data includes information about prioritized columns
            if "prioritized_columns" in relevant_data or "is_prioritized" in relevant_data:
                prioritization_guidance = """
Pay special attention to the prioritized or important columns in your response.
These columns contain the most valuable information for the user and should be emphasized.
"""
        
        # Use LLM to generate a response with rate limiting applied by calling function
        messages = [
            {"role": "system", "content": f"""You are a ticket data analysis assistant. 
Generate helpful, concise responses to queries about ticket data.
Base your answers on the data provided, but make them conversational and easy to understand.
Include specific numbers and percentages where appropriate.
{prioritization_guidance}"""},
            {"role": "user", "content": f"""
Answer the following query about ticket data:

QUERY: {query}

Base your answer on this data context:
{json.dumps(context, indent=2)}

Your response should be:
1. Directly answering the query with specific data points
2. Concise (3-5 sentences)
3. In a conversational tone
4. Including specific metrics where relevant
5. Emphasizing insights from important columns when available

Respond in plain text, not JSON format.
"""}
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def chat_response(self, query: str) -> str:
        """
        Process a chat query and generate a response
        
        Args:
            query (str): User's query
            
        Returns:
            str: Response to the query
        """
        if self.data is None:
            return "Please upload ticket data before asking questions."
        
        try:
            # Make sure we have analysis results
            if self.analysis_results is None:
                print("Error: Analysis results not available for chat")
                return "I'm sorry, but I don't have analysis results available. Please reload the data or try again."
            
            # Make analysis results JSON serializable
            serializable_analysis = make_json_serializable(self.analysis_results or {})
            
            # Print debug info
            print(f"Sending query to chat agent: {query}")
            print(f"Data shape: {self.data.shape}")
            print(f"Prioritizing columns: {self.column_hints}")
            
            # Process query through chat agent
            print("Processing chat query...")
            response = self.process_query(query, self.data, serializable_analysis)
            print("Chat response generated")
            
            return response
        except Exception as e:
            print(f"Error processing chat query: {str(e)}")
            return f"I'm sorry, I encountered an issue while processing your query: {str(e)}"