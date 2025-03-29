import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from utils.json_utils import make_json_serializable
from utils.rate_limiter import RateLimiter

class AutomationRecommendationAgent:
    """
    Agent responsible for identifying automation opportunities in ticket data.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.rate_limiter = RateLimiter(base_delay=3.0, max_retries=3, max_delay=120.0)
    
    def identify_automation_opportunities(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential automation opportunities based on the ticket data and analysis results
        """
        # Ensure all data is JSON serializable
        analysis_results = make_json_serializable(analysis_results)
        
        # Extract key information for automation analysis
        schema_info = self._extract_schema_info(df)
        
        # Identify patterns in the data that suggest automation opportunities
        category_patterns = self._identify_category_patterns(df, analysis_results)
        text_patterns = self._identify_text_patterns(df, analysis_results)
        time_patterns = self._identify_time_patterns(df, analysis_results)
        workflow_patterns = self._identify_workflow_patterns(df, analysis_results)
        
        # Combine all patterns
        all_patterns = {
            "schema_info": schema_info,
            "category_patterns": category_patterns,
            "text_patterns": text_patterns,
            "time_patterns": time_patterns,
            "workflow_patterns": workflow_patterns
        }
        
        # Make sure all patterns are serializable
        all_patterns = make_json_serializable(all_patterns)
        
        # Generate automation recommendations using LLM
        automation_recommendations = self._generate_recommendations(all_patterns)
        
        return automation_recommendations
    
    def _extract_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic schema information from the dataframe"""
        schema_info = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_values": make_json_serializable({col: df[col].sample(min(3, len(df))).tolist() for col in df.columns}),
        }
        return schema_info
    
    def _identify_category_patterns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in category distributions"""
        patterns = {}
        
        # Check for dominant categories
        category_analysis = analysis_results.get("category_analysis", {})
        
        for key, value in category_analysis.items():
            if key.endswith("_distribution") and isinstance(value, dict):
                category_col = key.replace("_distribution", "")
                sorted_cats = sorted(value.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_cats:
                    # Store top categories
                    patterns[f"top_{category_col}"] = sorted_cats[:5]
                    
                    # Check for highly repetitive categories
                    total = sum(count for _, count in sorted_cats)
                    top_percentage = sorted_cats[0][1] / total if total > 0 else 0
                    
                    if top_percentage > 0.3:  # If top category represents >30% of tickets
                        patterns[f"dominant_{category_col}"] = {
                            "category": sorted_cats[0][0],
                            "count": sorted_cats[0][1],
                            "percentage": top_percentage
                        }
        
        return patterns
    
    def _identify_text_patterns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in text fields"""
        patterns = {}
        
        # Check text analysis results
        text_analysis = analysis_results.get("text_analysis", {})
        
        for key, value in text_analysis.items():
            if key.endswith("_themes") and isinstance(value, dict):
                text_col = key.replace("_themes", "")
                
                # Extract relevant patterns for automation
                if "common_keywords" in value:
                    patterns[f"{text_col}_keywords"] = value["common_keywords"]
                
                if "main_categories" in value:
                    patterns[f"{text_col}_categories"] = value["main_categories"]
                
                if "automation_opportunities" in value:
                    patterns[f"{text_col}_opportunities"] = value["automation_opportunities"]
        
        return patterns
    
    def _identify_time_patterns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in time-related fields"""
        patterns = {}
        
        # Check for time-based patterns
        time_analysis = analysis_results.get("time_analysis", {})
        
        # Look for seasonality or cyclical patterns
        for key, value in time_analysis.items():
            if key.endswith("_month") and isinstance(value, dict):
                date_col = key.replace("_month", "")
                months_data = sorted(value.items(), key=lambda x: int(x[0]))
                
                if months_data:
                    # Check for seasonal patterns (peaks during certain months)
                    month_values = [count for _, count in months_data]
                    total = sum(month_values)
                    
                    if total > 0:
                        month_percentages = [count/total for count in month_values]
                        
                        # Check if any month has >20% of tickets (seasonal peak)
                        max_percent = max(month_percentages)
                        if max_percent > 0.2:
                            max_month = months_data[month_percentages.index(max_percent)][0]
                            patterns[f"{date_col}_seasonal_peak"] = {
                                "month": max_month,
                                "percentage": max_percent
                            }
        
        # Look for patterns in resolution time
        basic_stats = analysis_results.get("basic_stats", {})
        numeric_stats = basic_stats.get("numeric_stats", {})
        
        for col, stats in numeric_stats.items():
            if any(time_term in col.lower() for time_term in ['time', 'duration', 'resolution', 'hours', 'days']):
                if 'mean' in stats and 'std' in stats:
                    patterns[f"{col}_stats"] = {
                        "mean": stats['mean'],
                        "std": stats['std'],
                        "min": stats.get('min', 0),
                        "max": stats.get('max', 0)
                    }
        
        return patterns
    
    def _identify_workflow_patterns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in workflow or process"""
        patterns = {}
        
        # Check for status transitions or workflow patterns
        status_analysis = analysis_results.get("status_analysis", {})
        
        # Look for common status values
        for key, value in status_analysis.items():
            if key.endswith("_distribution") and isinstance(value, dict):
                status_col = key.replace("_distribution", "")
                sorted_statuses = sorted(value.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_statuses:
                    # Get common terminal statuses (typically "closed", "resolved", "completed")
                    terminal_keywords = ["closed", "complete", "resolved", "done", "fixed"]
                    terminal_statuses = [status for status, count in sorted_statuses 
                                        if any(keyword in str(status).lower() for keyword in terminal_keywords)]
                    
                    if terminal_statuses:
                        patterns[f"{status_col}_terminal"] = terminal_statuses
                    
                    # Get common intermediate statuses
                    in_progress_keywords = ["progress", "open", "pending", "working", "assigned"]
                    in_progress_statuses = [status for status, count in sorted_statuses 
                                           if any(keyword in str(status).lower() for keyword in in_progress_keywords)]
                    
                    if in_progress_statuses:
                        patterns[f"{status_col}_in_progress"] = in_progress_statuses
        
        # Look for assignment or ownership patterns
        assignment_cols = []
        for col in df.columns:
            if any(assign_term in col.lower() for assign_term in ['assign', 'owner', 'responsible', 'tech', 'agent']):
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    assignment_cols.append(col)
        
        if assignment_cols:
            patterns["assignment_columns"] = assignment_cols
            
            # Check for assignment distribution
            for col in assignment_cols:
                if col in df.columns:
                    value_counts = df[col].value_counts()
                    top_assignees = value_counts.nlargest(5).to_dict()
                    
                    if top_assignees:
                        patterns[f"{col}_top_assignees"] = top_assignees
        
        return patterns
    
    def _simplify_patterns_for_llm(self, all_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify patterns to reduce token usage for LLM"""
        simplified = {}
        
        # Copy schema info but reduce sample values
        if "schema_info" in all_patterns:
            schema = all_patterns["schema_info"]
            simplified["schema_info"] = {
                "columns": schema.get("columns", []),
                "dtypes": schema.get("dtypes", {})
            }
        
        # Simplify category patterns
        if "category_patterns" in all_patterns:
            cat_patterns = all_patterns["category_patterns"]
            simplified["category_patterns"] = {}
            
            # Keep only top categories for each field
            for key, value in cat_patterns.items():
                if "top_" in key and isinstance(value, list):
                    # Keep only top 3 entries
                    simplified["category_patterns"][key] = value[:3]
                elif "dominant_" in key:
                    # Keep dominant categories as they're important
                    simplified["category_patterns"][key] = value
        
        # Simplify text patterns - these are usually important for automation
        if "text_patterns" in all_patterns:
            simplified["text_patterns"] = all_patterns["text_patterns"]
        
        # Simplify time patterns
        if "time_patterns" in all_patterns:
            time_patterns = all_patterns["time_patterns"]
            simplified["time_patterns"] = {}
            
            # Keep seasonal peaks and time stats
            for key, value in time_patterns.items():
                if "_seasonal_peak" in key or "_stats" in key:
                    simplified["time_patterns"][key] = value
        
        # Simplify workflow patterns - these are crucial for automation
        if "workflow_patterns" in all_patterns:
            workflow = all_patterns["workflow_patterns"]
            simplified["workflow_patterns"] = {}
            
            # Keep most important workflow information
            if "assignment_columns" in workflow:
                simplified["workflow_patterns"]["assignment_columns"] = workflow["assignment_columns"]
            
            # Keep terminal and in-progress statuses
            for key, value in workflow.items():
                if "_terminal" in key or "_in_progress" in key:
                    simplified["workflow_patterns"][key] = value
                elif "_top_assignees" in key and isinstance(value, dict):
                    # Keep only top 3 assignees
                    sorted_assignees = sorted(value.items(), key=lambda x: x[1], reverse=True)[:3]
                    simplified["workflow_patterns"][key] = dict(sorted_assignees)
        
        return simplified
    
    def _generate_recommendations(self, all_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automation recommendations using LLM"""
        # Simplify the patterns to reduce token usage
        simplified_patterns = self._simplify_patterns_for_llm(all_patterns)
        
        # Prepare input for LLM
        messages = [
            {"role": "system", "content": """You are an automation expert specializing in ticket systems and workflow optimization. 
Your task is to identify the top 5 automation opportunities based on the data patterns provided.
Focus on high-impact, feasible automation that would reduce manual effort and improve efficiency."""},
            {"role": "user", "content": f"""
Based on the following ticket data patterns, identify the top 5 automation opportunities.
For each opportunity, provide:
1. A title for the automation opportunity
2. The automation scope (what will be automated)
3. Justification (why this should be automated)
4. Type of automation (AI, RPA, rule-based, etc.)
5. Implementation plan (high-level steps)
6. Expected impact

Here are the patterns found in the ticket data:
{json.dumps(simplified_patterns, indent=2)}

Format your response as a JSON array with 5 objects, each containing the fields: 
title, scope, justification, type, implementation, impact.
"""}
        ]
        
        try:
            # Use rate limiter to handle API limits
            def make_llm_call():
                return self.llm.invoke(messages)
            
            response = self.rate_limiter.execute_with_retry(make_llm_call)
            
            # Extract JSON array from response
            response_text = response.content
            
            # Handle case where response is wrapped in markdown
            if "```json" in response_text and "```" in response_text.split("```json")[1]:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text and "```" in response_text.split("```")[1]:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            recommendations = json.loads(json_str)
            
            # Ensure we have at most 5 recommendations
            results = make_json_serializable(recommendations[:5])
            return results
        except Exception as e:
            print(f"Error generating automation recommendations: {str(e)}")
            # Return default recommendations as fallback
            return [
                {
                    "title": "Automated Ticket Categorization",
                    "scope": "Automatically categorize incoming tickets based on content",
                    "justification": "Reduces manual categorization effort and improves consistency",
                    "type": "AI/ML",
                    "implementation": "Implement ML-based text classification model trained on historical ticket data",
                    "impact": "30-40% reduction in ticket triage time, improved routing accuracy"
                },
                {
                    "title": "Resolution Time Prediction",
                    "scope": "Predict expected resolution time for tickets",
                    "justification": "Helps set expectations and prioritize critical tickets",
                    "type": "AI/ML",
                    "implementation": "Develop predictive model based on historical resolution patterns",
                    "impact": "Improved SLA adherence and resource allocation"
                },
                {
                    "title": "Automated Status Updates",
                    "scope": "Automatically update ticket status based on actions taken",
                    "justification": "Reduces manual status updates and improves tracking",
                    "type": "RPA",
                    "implementation": "Implement workflow rules to detect actions and update status accordingly",
                    "impact": "Improved data accuracy and reduced administrative overhead"
                },
                {
                    "title": "Knowledge Base Enhancement",
                    "scope": "Automatically suggest knowledge base articles for common issues",
                    "justification": "Speeds up resolution by leveraging existing solutions",
                    "type": "AI/NLP",
                    "implementation": "Implement semantic search to match tickets with KB articles",
                    "impact": "Faster resolution times and knowledge reuse"
                },
                {
                    "title": "SLA Monitoring Alerts",
                    "scope": "Proactive notification for tickets approaching SLA breach",
                    "justification": "Prevents SLA violations and improves customer satisfaction",
                    "type": "Rule-based",
                    "implementation": "Set up alert system based on ticket age and priority",
                    "impact": "Reduced SLA violations and improved prioritization"
                }
            ]