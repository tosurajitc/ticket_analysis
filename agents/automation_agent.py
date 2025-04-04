import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import re  # Add this import
from utils.json_utils import make_json_serializable
from utils.rate_limiter import RateLimiter

class AutomationRecommendationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.rate_limiter = RateLimiter(base_delay=3.0, max_retries=3, max_delay=120.0)


    def identify_automation_opportunities(
        self, 
        df: pd.DataFrame, 
        analysis_results: Dict[str, Any], 
        selected_columns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify potential automation opportunities based on ticket data
        """
        # Validate inputs
        if df is None or len(df) == 0 or not selected_columns:
            return []
        
        # Prepare detailed column analysis
        column_details = {}
        for col in selected_columns:
            # Detailed column analysis
            value_counts = df[col].value_counts()
            unique_values = value_counts.to_dict()
            
            column_details[col] = {
                'dtype': str(df[col].dtype),
                'unique_count': len(unique_values),
                'top_values': value_counts.nlargest(5).to_dict(),
                'sample_values': df[col].sample(min(5, len(df))).tolist()
            }
        
        # Create a clean JSON template without f-string interpolation
        json_template = '''[
    {
        "title": "Uniquely Descriptive Title Derived from Data Insights",
        "automation_opportunity": "Specific Opportunity Name",
        "scope": {
            "overview": "Concise description",
            "detailed_description": "Comprehensive explanation",
            "justification": "Detailed rationale"
        },
        "automation_type": {
            "primary": "Specific Automation Approach",
            "techniques": ["Detailed Techniques"]
        },
        "implementation_plan": ["Specific Steps"],
        "impact": {
            "quantitative": {
                "efficiency_gain": "Percentage",
                "cost_reduction": "Percentage Range"
            },
            "qualitative": ["Benefit Descriptions"]
        }
    }
]'''
        
        # Prepare comprehensive messages for LLM
        prompt_content = f"""
Analyze automation opportunities for the following columns: {selected_columns}

Detailed Column Analysis:
{json.dumps(column_details, indent=2)}

Sample Data Overview:
{df[selected_columns].head().to_string()}

Guidelines for Automation Suggestions:
1. Generate 5 highly specific, data-driven automation suggestions
2. Each suggestion must:
- Be directly derived from the actual data characteristics
- Provide a comprehensive and unique automation approach
- Include a clear, descriptive title
- Explain detailed scope with strong justification
- Specify precise automation techniques
- Outline a concrete implementation plan
- Estimate quantifiable business impact

Response Format (Strict JSON):
```json
{json_template}
```

IMPORTANT: Ensure each suggestion is 100% relevant to the specific columns and data provided.
"""
        
        messages = [
            {
                "role": "system", 
                "content": "You are an advanced AI assistant specializing in identifying unique automation opportunities in ticket management systems."
            },
            {
                "role": "user", 
                "content": prompt_content
            }
        ]
        
        # Execute LLM call
        try:
            # Invoke LLM 
            response = self.rate_limiter.execute_with_retry(
                self._generate_suggestions, 
                messages
            )
            
            # Parse and validate suggestions
            return self._validate_suggestions(response)
        
        except Exception as e:
            print(f"Error in automation opportunity identification: {str(e)}")
            return []

    def format_suggestion_for_display(self, suggestion: Dict[str, Any]) -> str:
        """
        Format automation suggestion for display
        """
        try:
            # Construct a comprehensive markdown representation
            full_suggestion = f"## {suggestion.get('title', 'Automation Opportunity')}\n\n"
            
            # Scope Section
            full_suggestion += "### Scope\n"
            
            # Handle different scope structures (string or dict)
            if isinstance(suggestion['scope'], dict):
                if 'overview' in suggestion['scope']:
                    full_suggestion += f"**Overview:** {suggestion['scope']['overview']}\n\n"
                
                if 'detailed_description' in suggestion['scope']:
                    full_suggestion += f"**Detailed Description:** {suggestion['scope']['detailed_description']}\n\n"
                
                if 'justification' in suggestion['scope']:
                    full_suggestion += f"**Justification:** {suggestion['scope']['justification']}\n\n"
            else:
                full_suggestion += f"{suggestion['scope']}\n\n"
            
            # Automation Type
            full_suggestion += "### Automation Type\n"
            
            if isinstance(suggestion['automation_type'], dict):
                if 'primary' in suggestion['automation_type']:
                    full_suggestion += f"**Primary Approach:** {suggestion['automation_type']['primary']}\n"
                
                if 'techniques' in suggestion['automation_type'] and isinstance(suggestion['automation_type']['techniques'], list):
                    full_suggestion += "**Techniques:**\n"
                    for technique in suggestion['automation_type']['techniques']:
                        full_suggestion += f"- {technique}\n"
            else:
                full_suggestion += f"{suggestion['automation_type']}\n"
            
            # Implementation Plan
            full_suggestion += "\n### Implementation Plan\n"
            if isinstance(suggestion['implementation_plan'], list):
                for step in suggestion['implementation_plan']:
                    full_suggestion += f"- {step}\n"
            else:
                full_suggestion += f"{suggestion['implementation_plan']}\n"
            
            # Impact
            full_suggestion += "\n### Expected Impact\n"
            
            if isinstance(suggestion['impact'], dict):
                if 'quantitative' in suggestion['impact'] and isinstance(suggestion['impact']['quantitative'], dict):
                    full_suggestion += "**Quantitative Metrics:**\n"
                    for metric, value in suggestion['impact']['quantitative'].items():
                        full_suggestion += f"- {metric.replace('_', ' ').title()}: {value}\n"
                
                if 'qualitative' in suggestion['impact'] and isinstance(suggestion['impact']['qualitative'], list):
                    full_suggestion += "\n**Qualitative Benefits:**\n"
                    for benefit in suggestion['impact']['qualitative']:
                        full_suggestion += f"- {benefit}\n"
            else:
                full_suggestion += f"{suggestion['impact']}\n"
            
            return full_suggestion
        
        except Exception as e:
            print(f"Error formatting suggestion: {e}")
            return "Unable to format suggestion"

    def _extract_common_keywords(self, series: pd.Series) -> List[str]:
        """
        Extract common keywords from text series
        
        Args:
            series (pd.Series): Text series to analyze
        
        Returns:
            List[str]: Most common keywords
        """
        # Combine all text
        combined_text = ' '.join(series.dropna().astype(str))
        
        # Tokenize and count words
        from collections import Counter
        import re
        
        # Remove common stop words and punctuation
        words = re.findall(r'\b\w+\b', combined_text.lower())
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Filter out stop words and very short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get top keywords
        return [word for word, count in Counter(filtered_words).most_common(10)]

    def _generate_categorical_suggestions(
        self, 
        df: pd.DataFrame, 
        column_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed automation suggestions for categorical columns
        """
        suggestions = []
        
        for col, insights in column_insights.items():
            if insights['type'] == 'categorical':
                # Detailed suggestion with comprehensive scope and justification
                top_category = list(insights['top_values'].keys())[0]
                top_category_percentage = (insights['top_values'][top_category] / len(df)) * 100
                
                suggestion = {
                    "automation_opportunity": f"Intelligent Ticket Routing for {col} Category",
                    "scope": {
                        "overview": f"Develop a sophisticated automated routing system for tickets based on the '{col}' categorization",
                        "detailed_description": f"""
                        Comprehensive ticket routing automation that leverages the '{col}' categorization to:
                        - Automatically classify and route tickets with {top_category_percentage:.2f}% of tickets falling under the '{top_category}' category
                        - Reduce manual intervention in ticket assignment
                        - Optimize support team resource allocation
                        """,
                        "justification": f"""
                        Current Challenges:
                        - Manual ticket routing is time-consuming and error-prone
                        - Inconsistent ticket assignment leads to inefficient resource utilization
                        - The '{top_category}' category represents a significant portion of tickets, indicating a prime opportunity for automation

                        Potential Benefits:
                        - Standardize ticket routing process
                        - Reduce human error in ticket assignment
                        - Improve response times for critical ticket categories
                        - Enable more efficient team productivity
                        """
                    },
                    "automation_type": {
                        "primary": "AI-Powered Rule-Based Automation",
                        "techniques": [
                            "Machine Learning Classification",
                            "Rule-Based Decision Making",
                            "Predictive Routing"
                        ]
                    },
                    "implementation_plan": [
                        "Develop a machine learning model to classify tickets based on historical data",
                        f"Create a comprehensive mapping of {col} categories to appropriate support teams",
                        "Implement a rule-based routing engine with AI-driven decision support",
                        "Develop a feedback loop for continuous model improvement",
                        "Create a monitoring dashboard to track routing effectiveness"
                    ],
                    "impact": {
                        "quantitative": {
                            "efficiency_gain": f"{min(80, insights['unique_count'] * 10)}%",
                            "cost_reduction": "15-25%",
                            "response_time_improvement": "30-40%"
                        },
                        "qualitative": [
                            "Enhanced support team productivity",
                            "Improved ticket resolution consistency",
                            "More accurate resource allocation"
                        ]
                    }
                }
                suggestions.append(suggestion)
        
        return suggestions

    def _generate_text_suggestions(
        self, 
        df: pd.DataFrame, 
        column_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed automation suggestions for text columns
        """
        suggestions = []
        
        for col, insights in column_insights.items():
            if insights.get('text_analysis'):
                keywords = insights['text_analysis'].get('common_keywords', [])
                avg_length = insights['text_analysis'].get('avg_length', 0)
                
                if keywords:
                    suggestion = {
                        "automation_opportunity": f"Advanced Text Analytics for {col}",
                        "scope": {
                            "overview": f"Implement an intelligent text processing system for {col}",
                            "detailed_description": f"""
                            Develop a comprehensive text analytics solution that:
                            - Automatically extracts and categorizes key information from {col}
                            - Identifies patterns and trends in textual data
                            - Provides actionable insights from ticket descriptions

                            Key Characteristics:
                            - Average text length: {avg_length:.2f} characters
                            - Dominant keywords: {', '.join(keywords[:5])}
                            """,
                            "justification": """
                            Current Challenges:
                            - Manual text analysis is time-consuming and subjective
                            - Difficulty in extracting meaningful insights from text data
                            - Inconsistent interpretation of ticket descriptions

                            Potential Benefits:
                            - Standardize text interpretation
                            - Enable quick identification of critical issues
                            - Improve knowledge management
                            - Enhance decision-making capabilities
                            """
                        },
                        "automation_type": {
                            "primary": "AI-Driven Natural Language Processing",
                            "techniques": [
                                "Advanced NLP",
                                "Machine Learning Text Classification",
                                "Semantic Analysis"
                            ]
                        },
                        "implementation_plan": [
                            "Develop a sophisticated NLP model for text classification",
                            "Create a comprehensive taxonomy based on identified keywords",
                            "Implement machine learning algorithms for semantic understanding",
                            "Build an automated tagging and categorization system",
                            "Develop a knowledge extraction and indexing mechanism"
                        ],
                        "impact": {
                            "quantitative": {
                                "efficiency_gain": f"{min(70, len(keywords) * 5)}%",
                                "processing_time_reduction": "40-50%",
                                "insight_generation_speed": "60-70% faster"
                            },
                            "qualitative": [
                                "Enhanced information retrieval",
                                "Improved decision support",
                                "Consistent text interpretation"
                            ]
                        }
                    }
                    suggestions.append(suggestion)
        
        return suggestions

    def _generate_workflow_suggestions(
        self, 
        df: pd.DataFrame, 
        column_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate workflow-related automation suggestions
        
        Args:
            df (pd.DataFrame): Input dataframe
            column_insights (Dict[str, Any]): Column analysis insights
        
        Returns:
            List[Dict[str, Any]]: Workflow automation suggestions
        """
        suggestions = []
        
        # Look for potential workflow optimization opportunities
        if len(column_insights) > 1:
            suggestion = {
                "title": "Cross-Column Workflow Optimization",
                "automation_opportunity": "Cross-Column Workflow Optimization",
                "scope": {
                    "overview": "Develop an integrated workflow system that leverages insights from multiple columns",
                    "detailed_description": "Create a comprehensive workflow system that analyzes relationships between different data points",
                    "justification": "Current manual processes fail to capture cross-column insights"
                },
                "implementation_plan": [
                    "Create a correlation analysis between different columns",
                    "Develop a decision tree for automated workflow routing",
                    "Implement a dynamic workflow management system",
                    "Create predictive models for process optimization"
                ],
                "automation_type": {
                    "primary": "Predictive Analytics and Workflow Automation",
                    "techniques": ["Machine Learning", "Process Mining", "Decision Trees"]
                },
                "impact": {
                    "quantitative": {
                        "efficiency_gain": "40-50%",
                        "cost_reduction": "20-30%"
                    },
                    "qualitative": [
                        "Improved process consistency",
                        "Enhanced decision making",
                        "Better resource allocation"
                    ]
                }
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_suggestions(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate suggestions using the LLM
        
        Args:
            messages (List[Dict[str, Any]]): Prepared messages for LLM
        
        Returns:
            str: Raw response content
        """
        try:
            # Invoke LLM (adjust based on your specific LLM implementation)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"LLM Invocation Error: {e}")
            raise
        
    def _validate_suggestions(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Validate and parse LLM suggestions with enhanced error handling
        
        Args:
            response_content (str): Raw response from LLM
        
        Returns:
            List[Dict[str, Any]]: Validated automation suggestions
        """
        try:
            # Log the raw response for debugging
            print(f"Raw response length: {len(response_content)}")
            print(f"Response preview: {response_content[:200]}...")
            
            # Multiple parsing strategies
            suggestions = []
            
            # Clean the response content
            cleaned_content = response_content.strip()
            
            # Strategy 1: Direct JSON parsing
            try:
                suggestions = json.loads(cleaned_content)
                print("Successfully parsed JSON directly")
            except json.JSONDecodeError as e:
                print(f"Direct JSON parsing failed: {e}")
                
                # Strategy 2: Extract JSON from markdown code block
                import re
                
                # Try to extract JSON between ```json and ```
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_content, re.DOTALL)
                if json_match:
                    try:
                        json_content = json_match.group(1).strip()
                        print(f"Extracted JSON from code block: {json_content[:100]}...")
                        suggestions = json.loads(json_content)
                        print("Successfully parsed JSON from code block")
                    except json.JSONDecodeError as e2:
                        print(f"JSON code block parsing failed: {e2}")
                        
                        # Try with more aggressive cleaning
                        try:
                            # Replace any potential problematic characters
                            cleaned_json = json_content.replace('\n', ' ').replace('\r', '')
                            # Fix common issues with trailing commas
                            cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                            cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                            
                            suggestions = json.loads(cleaned_json)
                            print("Successfully parsed JSON after cleaning")
                        except json.JSONDecodeError as e3:
                            print(f"Cleaned JSON parsing failed: {e3}")
                            
                            # Strategy 3: Try to find array pattern
                            try:
                                array_match = re.search(r'\[\s*{[\s\S]*}\s*\]', cleaned_content, re.DOTALL)
                                if array_match:
                                    array_json = array_match.group(0)
                                    print(f"Extracted array JSON: {array_json[:100]}...")
                                    suggestions = json.loads(array_json)
                                    print("Successfully parsed array JSON")
                            except Exception as e4:
                                print(f"Array extraction failed: {e4}")
                else:
                    print("No JSON code block found")
                    
                    # Try to find any JSON array in the text
                    try:
                        array_match = re.search(r'\[\s*{[\s\S]*}\s*\]', cleaned_content, re.DOTALL)
                        if array_match:
                            array_json = array_match.group(0)
                            print(f"Extracted array JSON: {array_json[:100]}...")
                            suggestions = json.loads(array_json)
                            print("Successfully parsed array JSON")
                    except Exception as e4:
                        print(f"Array extraction failed: {e4}")
            
            # If still no suggestions, try to create a default one
            if not suggestions:
                print("All JSON parsing strategies failed. Creating default suggestion.")
                # Create a single default suggestion
                suggestions = [{
                    "title": "Automated Processing System",
                    "automation_opportunity": "Process Automation",
                    "scope": {
                        "overview": "Develop an automated system to process ticket data",
                        "detailed_description": "Create a system that can automatically categorize and route tickets based on their content",
                        "justification": "Manual processing is time-consuming and error-prone"
                    },
                    "automation_type": {
                        "primary": "Rule-based Automation with ML",
                        "techniques": ["Natural Language Processing", "Classification Algorithms"]
                    },
                    "implementation_plan": [
                        "Analyze ticket data patterns",
                        "Develop classification models",
                        "Implement routing rules",
                        "Create feedback mechanism"
                    ],
                    "impact": {
                        "quantitative": {
                            "efficiency_gain": "30-40%",
                            "cost_reduction": "20-25%"
                        },
                        "qualitative": [
                            "Improved response times",
                            "More consistent handling",
                            "Better resource allocation"
                        ]
                    }
                }]
            
            # Validate suggestions structure
            validated_suggestions = []
            for suggestion in suggestions:
                # Ensure all required keys are present and of correct type
                try:
                    # Validate and normalize scope
                    if isinstance(suggestion.get('scope'), str):
                        suggestion['scope'] = {
                            'overview': suggestion['scope'],
                            'detailed_description': '',
                            'justification': ''
                        }
                    elif not isinstance(suggestion.get('scope'), dict):
                        suggestion['scope'] = {
                            'overview': "Automated process improvement",
                            'detailed_description': '',
                            'justification': ''
                        }
                    
                    # Validate title
                    if 'title' not in suggestion:
                        suggestion['title'] = suggestion.get('automation_opportunity', 'Automation Opportunity')
                    
                    # Validate implementation plan
                    if not isinstance(suggestion.get('implementation_plan'), list):
                        if suggestion.get('implementation_plan'):
                            suggestion['implementation_plan'] = [str(suggestion['implementation_plan'])]
                        else:
                            suggestion['implementation_plan'] = ["Analyze data", "Design solution", "Implement automation", "Monitor and improve"]
                    
                    # Validate impact
                    if 'impact' not in suggestion:
                        suggestion['impact'] = {
                            'quantitative': {'efficiency_gain': '25-30%'},
                            'qualitative': ['Improved efficiency', 'Better user experience']
                        }
                    elif isinstance(suggestion['impact'], str):
                        impact_text = suggestion['impact']
                        suggestion['impact'] = {
                            'quantitative': {'estimated_improvement': impact_text},
                            'qualitative': ['Improved efficiency']
                        }
                    elif not isinstance(suggestion['impact'], dict):
                        suggestion['impact'] = {
                            'quantitative': {'efficiency_gain': '25-30%'},
                            'qualitative': ['Improved efficiency', 'Better user experience']
                        }
                    
                    # If impact is a dict but doesn't have the right structure
                    if isinstance(suggestion['impact'], dict):
                        if 'quantitative' not in suggestion['impact']:
                            suggestion['impact']['quantitative'] = {'efficiency_gain': '25-30%'}
                        if 'qualitative' not in suggestion['impact']:
                            suggestion['impact']['qualitative'] = ['Improved efficiency']
                    
                    # Validate automation type
                    if 'automation_type' not in suggestion:
                        suggestion['automation_type'] = {
                            'primary': 'AI-Powered Automation',
                            'techniques': ['Machine Learning', 'Natural Language Processing']
                        }
                    elif isinstance(suggestion['automation_type'], str):
                        automation_type_text = suggestion['automation_type']
                        suggestion['automation_type'] = {
                            'primary': automation_type_text,
                            'techniques': ['Automation Technology']
                        }
                    elif not isinstance(suggestion['automation_type'], dict):
                        suggestion['automation_type'] = {
                            'primary': 'AI-Powered Automation',
                            'techniques': ['Machine Learning', 'Natural Language Processing']
                        }
                    
                    # If automation_type is a dict but doesn't have the right structure
                    if isinstance(suggestion['automation_type'], dict):
                        if 'primary' not in suggestion['automation_type']:
                            suggestion['automation_type']['primary'] = 'AI-Powered Automation'
                        if 'techniques' not in suggestion['automation_type'] or not isinstance(suggestion['automation_type']['techniques'], list):
                            suggestion['automation_type']['techniques'] = ['Machine Learning', 'Process Automation']
                    
                    # Add the suggestion to validated list
                    validated_suggestions.append(suggestion)
                    print(f"Successfully validated suggestion: {suggestion.get('title')}")
                    
                except Exception as e:
                    print(f"Error validating suggestion: {str(e)}")
                    # Don't add invalid suggestions
            
            # Ensure we have at least one suggestion
            if not validated_suggestions:
                print("No valid suggestions created. Adding a default one.")
                validated_suggestions = [{
                    "title": "Automated Processing System",
                    "automation_opportunity": "Process Automation",
                    "scope": {
                        "overview": "Develop an automated system to process ticket data",
                        "detailed_description": "Create a system that can automatically categorize and route tickets based on their content",
                        "justification": "Manual processing is time-consuming and error-prone"
                    },
                    "automation_type": {
                        "primary": "Rule-based Automation with ML",
                        "techniques": ["Natural Language Processing", "Classification Algorithms"]
                    },
                    "implementation_plan": [
                        "Analyze ticket data patterns",
                        "Develop classification models",
                        "Implement routing rules",
                        "Create feedback mechanism"
                    ],
                    "impact": {
                        "quantitative": {
                            "efficiency_gain": "30-40%",
                            "cost_reduction": "20-25%"
                        },
                        "qualitative": [
                            "Improved response times",
                            "More consistent handling",
                            "Better resource allocation"
                        ]
                    }
                }]
            
            return validated_suggestions
        
        except Exception as e:
            print(f"Fatal error in _validate_suggestions: {str(e)}")
            # Return a default suggestion in case of complete failure
            return [{
                "title": "Automated Processing System",
                "automation_opportunity": "Process Automation",
                "scope": {
                    "overview": "Develop an automated system to process ticket data",
                    "detailed_description": "Create a system that can automatically categorize and route tickets based on their content",
                    "justification": "Manual processing is time-consuming and error-prone"
                },
                "automation_type": {
                    "primary": "Rule-based Automation with ML",
                    "techniques": ["Natural Language Processing", "Classification Algorithms"]
                },
                "implementation_plan": [
                    "Analyze ticket data patterns",
                    "Develop classification models",
                    "Implement routing rules",
                    "Create feedback mechanism"
                ],
                "impact": {
                    "quantitative": {
                        "efficiency_gain": "30-40%",
                        "cost_reduction": "20-25%"
                    },
                    "qualitative": [
                        "Improved response times",
                        "More consistent handling",
                        "Better resource allocation"
                    ]
                }
            }]
    
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