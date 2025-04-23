#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Manager module for the Incident Management Analytics application.
This module handles interactions with LLM APIs for generating insights.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import hashlib
import requests
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Class for managing interactions with LLM APIs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLMManager with application configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config or {}
        self.config = config
        self.llm_config = config["llm"]
        self.api_key = self.llm_config["api_key"]
        self.model_name = self.llm_config["model_name"]
        self.max_tokens = self.llm_config["max_tokens"]
        self.temperature = self.llm_config["temperature"]
        self.api_timeout = self.llm_config["api_timeout"]
        self.cache_ttl = self.llm_config["cache_ttl"]
        
        # Create cache directory if it doesn't exist
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key to use for LLM requests.
        
        Args:
            api_key: API key to use
        """
        self.api_key = api_key
    
    def generate_insights(self, 
                         data: pd.DataFrame, 
                         metadata: Dict[str, Any], 
                         insight_type: str,
                         user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate insights from incident data using LLMs.
        
        Args:
            data: DataFrame with incident data
            metadata: Metadata about the data
            insight_type: Type of insights to generate
            user_query: Optional user query for conversational insights
            
        Returns:
            Dictionary with generated insights
        """
        # Validate API key
        if not self.api_key:
            return {
                "success": False,
                "error": "No API key provided. Please provide an API key in the settings or through the UI."
            }
        
        # Check if we have a cached response
        cache_key = self._generate_cache_key(data, metadata, insight_type, user_query)
        cached_response = self._get_cached_response(cache_key)
        
        if cached_response:
            logger.info(f"Using cached response for {insight_type}")
            return cached_response
        
        # Prepare prompt based on insight type
        try:
            prompt = self._prepare_prompt(data, metadata, insight_type, user_query)
            
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Parse and validate response
            insights = self._parse_llm_response(llm_response, insight_type)
            
            # Cache the response
            self._cache_response(cache_key, insights)
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error generating insights: {str(e)}",
                "insight_type": insight_type
            }
    
    def _prepare_prompt(self, 
                       data: pd.DataFrame, 
                       metadata: Dict[str, Any], 
                       insight_type: str,
                       user_query: Optional[str] = None) -> str:
        """
        Prepare a prompt for the LLM based on the data and insight type.
        
        Args:
            data: DataFrame with incident data
            metadata: Metadata about the data
            insight_type: Type of insights to generate
            user_query: Optional user query for conversational insights
            
        Returns:
            Prompt string
        """
        # Create a data summary for the LLM
        data_summary = self._create_data_summary(data, metadata)
        
        # Generate prompt based on insight type
        if insight_type == "root_cause_analysis":
            return self._prepare_root_cause_prompt(data_summary, metadata)
        elif insight_type == "automation_opportunity":
            return self._prepare_automation_prompt(data_summary, metadata)
        elif insight_type == "resource_optimization":
            return self._prepare_resource_prompt(data_summary, metadata)
        elif insight_type == "incident_insights":
            return self._prepare_general_insights_prompt(data_summary, metadata)
        elif insight_type == "predictive_analysis":
            return self._prepare_predictive_prompt(data_summary, metadata)
        elif insight_type == "conversational_query":
            if not user_query:
                raise ValueError("User query is required for conversational insights")
            return self._prepare_conversational_prompt(data_summary, user_query)
        else:
            raise ValueError(f"Unknown insight type: {insight_type}")
    
    def _create_data_summary(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """
        Create a concise summary of the data for the LLM.
        
        Args:
            data: DataFrame with incident data
            metadata: Metadata about the data
            
        Returns:
            Data summary string
        """
        # Create a summary of the data that doesn't exceed token limits
        # but provides enough context for the LLM to generate insights
        
        summary = []
        
        # Basic dataset info
        summary.append(f"Dataset contains {len(data)} incidents.")
        
        # Date range
        if 'date_range' in metadata.get('data_summary', {}):
            date_range = metadata['data_summary']['date_range']
            if date_range.get('start') and date_range.get('end'):
                summary.append(f"Date range: {date_range['start']} to {date_range['end']}")
        
        # Available columns
        summary.append(f"Available columns: {', '.join(data.columns.tolist())}")
        
        # Add statistical summaries for key metrics
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            summary.append("\nKey numeric metrics:")
            # Limit to at most 5 numeric columns to avoid token explosion
            for col in numeric_cols[:5]:
                stats = data[col].describe()
                summary.append(f"  {col}: mean={stats['mean']:.2f}, median={stats['50%']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        # Add categorical distributions
        cat_cols = []
        for col in ['priority', 'status', 'category', 'assignee', 'affected_system']:
            if col in data.columns:
                cat_cols.append(col)
        
        if cat_cols:
            summary.append("\nCategory distributions:")
            # Limit to at most 3 categorical columns
            for col in cat_cols[:3]:
                # Get top 5 categories
                top_cats = data[col].value_counts().head(5)
                cat_summary = ", ".join([f"{cat}: {count} ({count/len(data)*100:.1f}%)" for cat, count in top_cats.items()])
                summary.append(f"  {col}: {cat_summary}")
        
        # Add time-based metrics if available
        if 'resolution_time_hours' in data.columns:
            valid_times = data['resolution_time_hours'].dropna()
            if len(valid_times) > 0:
                avg_res_time = valid_times.mean()
                median_res_time = valid_times.median()
                summary.append(f"\nResolution time: avg={avg_res_time:.2f} hours, median={median_res_time:.2f} hours")
        
        # Include metadata insights
        for analysis_type, analysis_data in metadata.items():
            if analysis_type in ['time_analysis', 'priority_analysis', 'resource_analysis', 'category_analysis', 'system_analysis']:
                if isinstance(analysis_data, dict) and analysis_data:
                    summary.append(f"\n{analysis_type.replace('_', ' ').title()} summary:")
                    
                    # Limit to a few key insights per analysis type
                    insight_count = 0
                    for key, value in analysis_data.items():
                        if insight_count >= 3:  # Limit to 3 insights per analysis type
                            break
                            
                        if isinstance(value, dict):
                            # Handle nested dictionaries
                            for subkey, subvalue in list(value.items())[:2]:  # Limit to 2 sub-insights
                                if isinstance(subvalue, (int, float, str, bool)):
                                    summary.append(f"  {key} - {subkey}: {subvalue}")
                                    insight_count += 1
                        elif isinstance(value, (int, float, str, bool)):
                            summary.append(f"  {key}: {value}")
                            insight_count += 1
        
        return "\n".join(summary)
    
    def _prepare_root_cause_prompt(self, data_summary: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare a prompt for root cause analysis.
        
        Args:
            data_summary: Summary of incident data
            metadata: Metadata about the data
            
        Returns:
            Prompt string
        """
        prompt = f"""
        You are an expert incident management analyst tasked with identifying root causes for incident patterns.
        
        DATA SUMMARY:
        {data_summary}
        
        TASK:
        Based on the incident data provided, identify the most likely root causes of recurring incidents and patterns. 
        
        In your analysis, please:
        1. Identify the top 3-5 potential root causes for recurring incidents, based strictly on the data provided
        2. For each root cause, provide evidence from the data that supports this hypothesis
        3. Estimate the potential impact of addressing each root cause (in terms of incident reduction)
        4. Suggest specific investigation steps to confirm each root cause
        5. Recommend potential mitigation strategies for each identified root cause
        
        IMPORTANT GUIDELINES:
        - Base your analysis ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If the data is insufficient to identify strong root cause patterns, acknowledge this limitation and suggest what additional data would be helpful.
        - Focus on systemic root causes rather than individual incidents.
        - Prioritize root causes that affect high-impact or frequent incident categories.
        - Be specific in your recommendations - avoid generic advice that could apply to any incident data.
        
        Please format your response as a structured JSON object with the following keys:
        - root_causes: Array of root cause objects, each containing:
          - cause: Brief description of the root cause
          - evidence: Evidence from the data supporting this as a root cause
          - impact: Estimated impact of addressing this root cause
          - investigation: Specific steps to confirm this root cause
          - mitigation: Recommended strategies to address this root cause
        - limitations: Any limitations in the analysis due to data constraints
        - additional_data_needed: Specific additional data that would improve the analysis
        """
        
        return prompt
    
    def _prepare_automation_prompt(self, data_summary: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare a prompt for automation opportunity analysis.
        
        Args:
            data_summary: Summary of incident data
            metadata: Metadata about the data
            
        Returns:
            Prompt string
        """
        # Get the maximum number of automation suggestions
        max_suggestions = min(
            self.config["analysis"].get("max_automation_suggestions", 5),
            5  # Hard limit to avoid token explosion
        )
        
        prompt = f"""
        You are an automation expert specializing in IT service management.
        
        DATA SUMMARY:
        {data_summary}
        
        TASK:
        Based on the incident data provided, identify the top {max_suggestions} opportunities for automation that would reduce manual effort, improve resolution time, or prevent incidents.
        
        In your analysis, please:
        1. Identify specific recurring incident patterns that are candidates for automation
        2. For each automation opportunity, provide evidence from the data that supports its value
        3. Estimate the potential benefit of each automation (time saved, incidents prevented, etc.)
        4. Suggest a high-level implementation approach for each automation
        5. Consider both remediation automation (fixing issues) and preventive automation (avoiding issues)
        
        IMPORTANT GUIDELINES:
        - Base your analysis ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If the data is insufficient to identify strong automation candidates, acknowledge this limitation and suggest what additional data would be helpful.
        - Prioritize automation opportunities that address high-volume or time-consuming incidents.
        - Be specific about what processes should be automated - avoid generic suggestions that could apply to any environment.
        - Consider both simple automations (scripts, scheduled tasks) and more complex solutions (self-healing systems).
        
        Please format your response as a structured JSON object with the following keys:
        - automation_opportunities: Array of automation opportunity objects, each containing:
          - name: Brief descriptive name for the automation
          - description: What process would be automated
          - evidence: Data points supporting this as a good automation candidate
          - benefit: Estimated benefit (quantitative if possible)
          - approach: High-level implementation approach
          - complexity: Implementation complexity (low/medium/high)
        - insufficient_data: Boolean indicating if there's insufficient data for confident recommendations
        - additional_data_needed: Specific additional data that would improve the analysis
        """
        
        return prompt
    
    def _prepare_resource_prompt(self, data_summary: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare a prompt for resource optimization analysis.
        
        Args:
            data_summary: Summary of incident data
            metadata: Metadata about the data
            
        Returns:
            Prompt string
        """
        prompt = f"""
        You are a resource optimization expert for IT operations teams.
        
        DATA SUMMARY:
        {data_summary}
        
        TASK:
        Based on the incident data provided, analyze the current resource allocation patterns and provide recommendations for optimization.
        
        In your analysis, please:
        1. Identify resource allocation patterns (e.g., workload distribution, specialization)
        2. Highlight potential resource bottlenecks and underutilization
        3. Analyze skill gaps and opportunities for cross-training
        4. Recommend specific changes to resource allocation
        5. Suggest metrics to track resource efficiency
        
        IMPORTANT GUIDELINES:
        - Base your analysis ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If the data is insufficient for confident resource recommendations, acknowledge this limitation.
        - Consider both quantity (number of incidents) and quality (resolution time, escalations) metrics.
        - Focus on actionable, specific recommendations rather than generic best practices.
        - Consider different resource optimization strategies (specialization vs. balanced workload).
        
        Please format your response as a structured JSON object with the following keys:
        - current_state: Summary of current resource allocation patterns
        - bottlenecks: Identified resource bottlenecks and constraints
        - underutilization: Areas where resources appear underutilized
        - skill_gaps: Identified skill gaps or training opportunities
        - recommendations: Array of specific resource optimization recommendations
        - metrics: Recommended metrics to track resource efficiency
        - insufficient_data: Boolean indicating if there's insufficient data for confident recommendations
        - additional_data_needed: Specific additional data that would improve the analysis
        """
        
        return prompt
    
    def _prepare_general_insights_prompt(self, data_summary: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare a prompt for general incident insights.
        
        Args:
            data_summary: Summary of incident data
            metadata: Metadata about the data
            
        Returns:
            Prompt string
        """
        prompt = f"""
        You are an incident management expert with deep analytical skills.
        
        DATA SUMMARY:
        {data_summary}
        
        TASK:
        Based on the incident data provided, generate meaningful insights about incident patterns, trends, and areas for improvement.
        
        In your analysis, please cover:
        1. Significant trends over time in incident volume, categories, or resolution metrics
        2. Notable patterns in incident distribution (by priority, category, system, time, etc.)
        3. Key performance indicators and their trends
        4. Anomalies or outliers worth investigating
        5. Specific areas for process improvement
        6. Early warning indicators for potential future issues
        
        IMPORTANT GUIDELINES:
        - Base your insights ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If the data is insufficient for certain analyses, acknowledge these limitations.
        - Prioritize insights that are actionable and can lead to tangible improvements.
        - Be specific and quantitative where possible (e.g., "Category X incidents increased by 25%").
        - Avoid general statements that could apply to any incident data.
        - Focus on insights that would be most valuable for improving operational efficiency and reducing incidents.
        
        Please format your response as a structured JSON object with the following keys:
        - key_trends: Array of identified trends and patterns
        - performance_indicators: Key metrics and their status
        - anomalies: Notable outliers or unusual patterns
        - improvement_areas: Specific areas for process improvement
        - warning_indicators: Early warning signs to monitor
        - insufficient_data: Boolean indicating if there's insufficient data for confident insights
        - additional_analyses: Suggestions for additional analyses that might yield valuable insights
        """
        
        return prompt
    
    def _prepare_predictive_prompt(self, data_summary: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare a prompt for predictive analysis.
        
        Args:
            data_summary: Summary of incident data
            metadata: Metadata about the data
            
        Returns:
            Prompt string
        """
        # Get the forecast horizon
        forecast_horizon = self.config["analysis"].get("forecast_horizon", 14)
        
        prompt = f"""
        You are a predictive analytics expert for IT service management.
        
        DATA SUMMARY:
        {data_summary}
        
        TASK:
        Based on the historical incident data provided, generate predictions for future incidents and identify leading indicators.
        
        In your predictions, please provide:
        1. Projected incident volume for the next {forecast_horizon} days
        2. Anticipated peak periods or seasonality patterns
        3. Categories likely to see significant changes in volume
        4. Systems at risk of increased incidents
        5. Leading indicators that should be monitored
        6. Confidence level in these predictions
        
        IMPORTANT GUIDELINES:
        - Base your predictions ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If the data is insufficient for confident predictions, acknowledge this limitation.
        - Clearly indicate the level of confidence in each prediction.
        - Explain the reasoning behind your predictions.
        - Focus on specific, actionable predictions rather than general statements.
        - Consider seasonality, trends, and patterns in the historical data.
        
        Please format your response as a structured JSON object with the following keys:
        - volume_prediction: Predicted incident volume (with confidence level)
        - peak_periods: Anticipated peak periods or seasonal patterns
        - category_predictions: Predicted changes in incident categories
        - system_risks: Systems likely to experience increased incidents
        - leading_indicators: Metrics to monitor for early warning
        - confidence_assessment: Overall confidence in predictions
        - insufficient_data: Boolean indicating if there's insufficient data for confident predictions
        - additional_data_needed: Specific additional data that would improve predictions
        """
        
        return prompt
    
    def _prepare_conversational_prompt(self, data_summary: str, user_query: str) -> str:
        """
        Prepare a prompt for conversational analysis based on user query.
        
        Args:
            data_summary: Summary of incident data
            user_query: User's query about the data
            
        Returns:
            Prompt string
        """
        prompt = f"""
        You are an incident management analytics assistant.
        
        DATA SUMMARY:
        {data_summary}
        
        USER QUERY:
        {user_query}
        
        TASK:
        Provide a clear, concise, and accurate answer to the user's question based on the incident data provided.
        
        IMPORTANT GUIDELINES:
        - Base your answer ONLY on the data provided. Do not make assumptions beyond what is supported by the data.
        - If you cannot answer the question with the data provided, explain what additional information would be needed.
        - Be specific and use numbers/percentages where possible.
        - Format your answer in a conversational way that directly addresses the user's question.
        - If the question is ambiguous, provide the most likely interpretation and answer.
        - Keep your response focused and to the point.
        
        Please format your response as a structured JSON object with the following keys:
        - answer: Your direct answer to the user's question
        - confidence: Your confidence level in the answer (high/medium/low)
        - limitations: Any limitations in answering due to data constraints
        - follow_up: Suggested follow-up questions the user might want to ask
        """
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: Prompt string to send to the LLM
            
        Returns:
            LLM response string
        """
        # Set up API request
        # Using GROQ API as specified in requirements
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant specialized in incident management analytics."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        # Make API request
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=self.api_timeout
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            
            # Extract the message content
            message_content = response_json["choices"][0]["message"]["content"]
            
            return message_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise ValueError(f"Error calling LLM API: {str(e)}")
    
    def _parse_llm_response(self, response: str, insight_type: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response.
        
        Args:
            response: LLM response string
            insight_type: Type of insights to generate
            
        Returns:
            Dictionary with parsed insights
        """
        # Try to extract JSON from the response
        try:
            # First, try to parse the entire response as JSON
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if json_match:
                    insights = json.loads(json_match.group(1))
                else:
                    # If that fails, try to extract anything that looks like JSON
                    json_match = re.search(r"({[\s\S]*})", response)
                    if json_match:
                        insights = json.loads(json_match.group(1))
                    else:
                        # If all JSON extraction attempts fail, return the raw response
                        logger.warning("Could not parse JSON from LLM response")
                        insights = {
                            "raw_response": response,
                            "parsing_error": "Could not extract JSON from response"
                        }
            
            # Add metadata to the response
            insights["insight_type"] = insight_type
            insights["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insights["success"] = True
            
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "success": False,
                "error": f"Error parsing insights: {str(e)}",
                "raw_response": response,
                "insight_type": insight_type
            }
    
    def _generate_cache_key(self, 
                           data: pd.DataFrame, 
                           metadata: Dict[str, Any], 
                           insight_type: str,
                           user_query: Optional[str] = None) -> str:
        """
        Generate a cache key for the given parameters.
        
        Args:
            data: DataFrame with incident data
            metadata: Metadata about the data
            insight_type: Type of insights to generate
            user_query: Optional user query for conversational insights
            
        Returns:
            Cache key string
        """
        # Create a string representation of the parameters
        # For the DataFrame, use a hash of the shape and column names
        df_hash = hashlib.md5(f"{data.shape}_{','.join(data.columns)}".encode()).hexdigest()
        
        # For metadata, use a hash of the JSON string
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        # Combine with insight type and user query
        key_parts = [df_hash, metadata_hash, insight_type]
        
        if user_query:
            # For user query, use the query directly
            key_parts.append(hashlib.md5(user_query.encode()).hexdigest())
        
        # Join and hash again
        cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
        
        return cache_key
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for the given cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Cached response dictionary or None if not found
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                # Check if the cache file is still valid
                file_mtime = os.path.getmtime(cache_file)
                file_age = time.time() - file_mtime
                
                if file_age <= self.cache_ttl:
                    # Cache is still valid
                    with open(cache_file, "r") as f:
                        return json.load(f)
                else:
                    # Cache is expired
                    logger.info(f"Cache expired for key {cache_key}")
                    return None
                
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
                return None
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for the given cache key.
        
        Args:
            cache_key: Cache key string
            response: Response dictionary to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, "w") as f:
                json.dump(response, f)
                
            logger.info(f"Cached response for key {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")