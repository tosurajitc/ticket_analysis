import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time
from utils.json_utils import make_json_serializable
from utils.rate_limiter import RateLimiter
from utils.batch_processor import BatchProcessor

class QualitativeAnswerAgent:
    """
    Agent responsible for answering qualitative questions about the ticket data.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.rate_limiter = RateLimiter(base_delay=3.0, max_retries=3, max_delay=120.0)
        self.batch_processor = BatchProcessor(concurrent_tasks=1, rate_limiter=self.rate_limiter)
        
        # We'll dynamically generate default questions based on the dataset rather than 
        # using fixed questions that might not apply to all ticket datasets
    
    def generate_qualitative_answers(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate answers to qualitative questions about the ticket data
        """
        # Ensure analysis_results is JSON serializable
        analysis_results = make_json_serializable(analysis_results)
        
        # Generate questions based on the actual dataset structure
        all_questions = self._generate_dataset_specific_questions(df, analysis_results)
        
        # Keep only the first 8 questions (reduced from 10 to manage rate limits better)
        questions = all_questions[:8]
        
        print(f"Processing {len(questions)} qualitative questions...")
        
        # Estimate time - assuming average of 15 seconds per question with rate limiting
        estimated_time = len(questions) * 15
        print(f"Estimated processing time: {estimated_time} seconds")
        
        # Prepare shared data for all questions to reduce redundancy
        shared_context = self._prepare_shared_context(df, analysis_results)
        
        # First generate local statistical answers as a fallback
        local_answers = self._generate_local_answers(df, questions)
        
        # Process 3 questions at a time to avoid rate limits
        answers = []
        question_batches = [questions[i:i + 3] for i in range(0, len(questions), 3)]
        
        for batch_idx, question_batch in enumerate(question_batches):
            print(f"Processing batch {batch_idx+1} of {len(question_batches)}...")
            
            # Sleep between batches to avoid rate limits
            if batch_idx > 0:
                print(f"Sleeping for 30 seconds to avoid rate limits...")
                time.sleep(30)
            
            # Process questions in this batch
            batch_questions_with_context = [(q, shared_context) for q in question_batch]
            
            def process_question(question_with_context):
                question, context = question_with_context
                # Try to get answer from LLM
                try:
                    result = {
                        "question": question,
                        "estimated_time": estimated_time,
                        **self._answer_question_with_context(question, context)
                    }
                    return result
                except Exception as e:
                    print(f"Error from LLM for question '{question}', using local fallback: {str(e)}")
                    # If LLM fails, use the local answer
                    return next((ans for ans in local_answers if ans["question"] == question), 
                               self._create_fallback_answer(question, estimated_time))
            
            # Fallback function for failed questions
            def fallback_answer(question_with_context, error):
                question, _ = question_with_context
                print(f"Error generating answer for question '{question}': {str(e)}")
                # Use pre-generated local answer as fallback
                return next((ans for ans in local_answers if ans["question"] == question), 
                           self._create_fallback_answer(question, estimated_time))
            
            # Process batch with rate limiting and fallbacks
            batch_answers = self.batch_processor.process_batch(
                batch_questions_with_context, 
                process_question,
                fallback_answer
            )
            
            answers.extend(batch_answers)
            
        # Ensure answers are JSON serializable
        return make_json_serializable(answers)
        
    def _create_fallback_answer(self, question, estimated_time):
        """Create a fallback answer when all else fails"""
        return {
            "question": question,
            "estimated_time": estimated_time,
            "answer": "Based on statistical analysis of the data, this question requires more context than is available.",
            "automation_scope": "Data collection and standardization",
            "justification": "Current data doesn't provide enough information to fully answer this question",
            "automation_type": "Structured data collection",
            "implementation_plan": "Implement more comprehensive data collection and categorization"
        }
        
    def _generate_local_answers(self, df, questions) -> List[Dict[str, Any]]:
        """Generate basic statistical answers without using LLM API"""
        answers = []
        
        # Get basic statistics about the dataset
        total_tickets = len(df)
        columns = list(df.columns)
        
        # Identify key column types
        categorical_cols = [col for col in columns if df[col].dtype == 'object' and df[col].nunique() < 100]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in columns if 
                    pd.api.types.is_datetime64_any_dtype(df[col]) or
                    any(term in col.lower() for term in ['date', 'day', 'time', 'created', 'resolved'])]
        
        # For each question, generate a basic answer based on data statistics
        for question in questions:
            q_lower = question.lower()
            
            # Initialize answer components
            answer_text = ""
            automation_scope = ""
            justification = ""
            automation_type = ""
            implementation_plan = ""
            
            try:
                # Determine what the question is asking about
                if any(term in q_lower for term in ['most frequent', 'most common', 'highest volume']):
                    # Find most common values in categorical columns
                    if categorical_cols:
                        col = categorical_cols[0]
                        # Find which column is mentioned in the question
                        for potential_col in categorical_cols:
                            if potential_col.lower() in q_lower:
                                col = potential_col
                                break
                        
                        # Get top value
                        top_value = df[col].value_counts().idxmax()
                        count = df[col].value_counts().max()
                        percent = (count / total_tickets) * 100
                        
                        answer_text = f"The most frequent {col} is '{top_value}' with {count} occurrences ({percent:.1f}% of tickets)."
                        automation_scope = f"Automated categorization of {col} values"
                        justification = f"High frequency of '{top_value}' suggests opportunity for standardized handling"
                        automation_type = "Machine learning classification"
                        implementation_plan = f"Train a classifier on historical ticket data to predict {col} values"
                
                elif any(term in q_lower for term in ['average', 'mean', 'typical']):
                    # Calculate averages for numeric columns
                    if numeric_cols:
                        col = numeric_cols[0]
                        # Find which column is mentioned in the question
                        for potential_col in numeric_cols:
                            if potential_col.lower() in q_lower:
                                col = potential_col
                                break
                        
                        mean_val = df[col].mean()
                        median_val = df[col].median()
                        
                        answer_text = f"The average {col} is {mean_val:.2f} with a median of {median_val:.2f}."
                        automation_scope = f"Prediction of expected {col} values"
                        justification = "Accurate predictions can help with resource planning"
                        automation_type = "Predictive analytics"
                        implementation_plan = f"Develop regression model to predict {col} based on ticket attributes"
                
                elif any(term in q_lower for term in ['time', 'duration', 'hours', 'days']):
                    # Time-based analysis
                    if date_cols and len(date_cols) >= 2:
                        # Try to find resolution time between two date columns
                        start_col = date_cols[0]
                        end_col = date_cols[1]
                        
                        answer_text = f"Time-based analysis shows patterns in {start_col} and {end_col} that could be optimized."
                        automation_scope = "Time-based workflow optimization"
                        justification = "Reducing resolution time improves customer satisfaction"
                        automation_type = "Workflow automation"
                        implementation_plan = "Implement automated ticket routing and follow-up based on time thresholds"
                    
                    elif date_cols:
                        # Single date column analysis
                        date_col = date_cols[0]
                        answer_text = f"Analysis of the {date_col} field shows potential for time-based optimizations."
                        automation_scope = "Time-based ticket prioritization"
                        justification = "Proper prioritization improves overall service levels"
                        automation_type = "Rule-based automation"
                        implementation_plan = "Implement rules engine for ticket prioritization based on time patterns"
                
                elif any(term in q_lower for term in ['distribution', 'breakdown']):
                    # Distribution analysis
                    if categorical_cols:
                        col = categorical_cols[0]
                        # Find which column is mentioned in the question
                        for potential_col in categorical_cols:
                            if potential_col.lower() in q_lower:
                                col = potential_col
                                break
                        
                        top_3 = df[col].value_counts().nlargest(3)
                        top_3_str = ", ".join([f"'{k}' ({v})" for k, v in top_3.items()])
                        
                        answer_text = f"The distribution of {col} shows these top categories: {top_3_str}."
                        automation_scope = f"Specialized handling for common {col} values"
                        justification = "Different categories often need different handling approaches"
                        automation_type = "Category-based workflow automation"
                        implementation_plan = f"Define specialized workflows for each major {col} category"
                
                else:
                    # Generic answer for other questions
                    answer_text = f"Analysis of {total_tickets} tickets shows patterns that can be optimized through automation."
                    automation_scope = "General workflow automation"
                    justification = "Automated processes reduce manual effort and improve consistency"
                    automation_type = "Rule-based automation with ML components"
                    implementation_plan = "Identify repetitive tasks and implement automation rules"
            
            except Exception as e:
                print(f"Error generating local answer for '{question}': {str(e)}")
                answer_text = "Statistical analysis of the dataset shows potential for optimization."
                automation_scope = "Data-driven process improvement"
                justification = "Automated processes reduce manual effort"
                automation_type = "Rule-based automation"
                implementation_plan = "Analyze patterns and implement targeted automation"
            
            # Add the answer
            answers.append({
                "question": question,
                "estimated_time": len(questions) * 15,
                "answer": answer_text,
                "automation_scope": automation_scope,
                "justification": justification,
                "automation_type": automation_type,
                "implementation_plan": implementation_plan
            })
        
        return answers
        
    def _prepare_shared_context(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare shared context for all questions to avoid redundant processing"""
        # Prepare sample data with explicit timestamp handling
        sample_rows = df.head(3)
        # Handle timestamp objects specifically - convert to string
        for col in sample_rows.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_rows[col]):
                sample_rows[col] = sample_rows[col].astype(str)
        
        # Create serializable data summary        
        data_summary = {
            "columns": list(df.columns),
            "row_count": len(df),
            "sample": make_json_serializable(sample_rows.to_dict(orient='records'))
        }
        
        # Create a compact version of the analysis results
        compact_analysis = make_json_serializable({
            "basic_stats": analysis_results.get("basic_stats", {}),
            "insights": analysis_results.get("insights", {})
        })
        
        return {
            "data_summary": data_summary,
            "analysis_results": compact_analysis
        }
        
    def _answer_question_with_context(self, question: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate an answer to a specific qualitative question using shared context"""
        try:
            data_summary = context["data_summary"]
            compact_analysis = context["analysis_results"]
            
            # Use LLM to generate the answer with rate limiting
            def generate_answer():
                messages = [
                    {"role": "system", "content": """You are a data analysis expert specializing in ticket systems and automation.
Provide concise, data-driven answers to questions about ticket data, including automation potential."""},
                    {"role": "user", "content": f"""
Answer the following question about ticket data:

QUESTION: {question}

Your answer should include these components, each limited to 1-3 sentences:
a. Direct answer to the question based on the data
b. Automation scope (what could be automated related to this issue)
c. Justification for the automation (why it would be valuable)
d. What kind of automation would be suitable (AI, RPA, rule-based, etc.)
e. Implementation plan for the automation (high-level steps)

The entire answer must be under 500 words total. Be specific and data-driven.
If there is no automation potential, clearly state that instead of sections b-e.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

ANALYSIS RESULTS:
{json.dumps(compact_analysis, indent=2)}

Format your response as a JSON object with these exact keys:
"answer", "automation_scope", "justification", "automation_type", "implementation_plan"
"""}
                ]
                
                response = self.llm.invoke(messages)
                return response
            
            # Execute with rate limiting
            response = self.rate_limiter.execute_with_retry(generate_answer)
            
            # Extract JSON from response
            response_text = response.content
            
            # Handle case where response is wrapped in markdown
            if "```json" in response_text and "```" in response_text.split("```json")[1]:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text and "```" in response_text.split("```")[1]:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            answer = json.loads(json_str)
            return answer
        except Exception as e:
            print(f"Error generating answer for question '{question}': {str(e)}")
            # Return default response as fallback
            return {
                "answer": f"Based on the available data, it's difficult to provide a definitive answer to '{question}'.",
                "automation_scope": "Data collection and standardization to better track this information.",
                "justification": "Current data doesn't sufficiently capture this information, making analysis difficult.",
                "automation_type": "Data pipeline automation with validation rules.",
                "implementation_plan": "Implement structured data collection, enhance ticket forms, and create reporting dashboards."
            }
    
    def _generate_dataset_specific_questions(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate dataset-specific questions based on the data structure"""
        # First generate some generic questions
        generic_questions = self._generate_generic_questions(8)
        
        # Try to generate dynamic questions based on the dataset
        try:
            dynamic_questions = self._generate_dynamic_questions(df, analysis_results)
            if dynamic_questions and len(dynamic_questions) > 0:
                # Replace some generic questions with dynamic ones
                # but keep a mix to ensure we have enough questions
                combined = dynamic_questions + generic_questions
                return combined[:8]  # Return max 8 questions
        except Exception as e:
            print(f"Error generating dynamic questions: {str(e)}")
        
        # If dynamic question generation failed, just use generic ones
        return generic_questions



    def _generate_dynamic_questions(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate dynamic questions based on the specific dataset"""
        # Extract schema information
        schema_info = {
            "columns": list(df.columns),
            "sample": df.head(5).to_dict(orient='records')
        }
        
        # Define default questions to avoid duplication
        default_questions = [
            "What is the average resolution time for high priority tickets?",
            "Which types of tickets show the highest recurrence pattern?",
            "What percentage of tickets are reopened after initial resolution?",
            "Which ticket categories show the longest average time to first response?",
            "What percentage of tickets require multiple team handoffs?"
        ]
        
        # Use LLM to generate dataset-specific questions
        messages = [
            {"role": "system", "content": """You are a data analysis expert specializing in ticket systems.
    Generate insightful questions about ticket data that would be valuable for understanding automation opportunities."""},
            {"role": "user", "content": f"""
    Based on the following ticket data schema and analysis results, generate 5 specific, insightful questions that would be 
    valuable for understanding the data and identifying automation opportunities.

    The questions should:
    1. Be specific to this particular dataset (use the actual column names)
    2. Focus on opportunities for automation or process improvement
    3. Be answerable based on the data provided
    4. Not repeat these existing questions:
    {default_questions}

    Schema information:
    {json.dumps(schema_info, indent=2)}

    Analysis results:
    {json.dumps({k: v for k, v in analysis_results.items() if k in ['basic_stats', 'category_analysis', 'priority_analysis', 'status_analysis']}, indent=2)}

    Return ONLY a list of 5 questions, one per line. Do not include any explanations or numbering.
    """}
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Split the response into lines and clean up
            question_lines = response.content.strip().split('\n')
            questions = [q.strip() for q in question_lines if q.strip() and q.strip().endswith('?')]
            
            return questions
        except Exception as e:
            print(f"Error generating dynamic questions: {str(e)}")
            # Return empty list as fallback
            return []
    
    def _generate_generic_questions(self, count: int) -> List[str]:
        """Generate generic ticket-related questions"""
        generic_questions = [
            "What is the average resolution time for high priority tickets?",
            "Which types of tickets show the highest recurrence pattern?",
            "What percentage of tickets are reopened after initial resolution?",
            "Which ticket categories show the longest average time to first response?",
            "What percentage of tickets require multiple team handoffs?",
            "Which ticket submitters generate the most high-priority tickets?",
            "What is the distribution of ticket resolution methods?",
            "Which time periods show the highest ticket volume?",
            "What percentage of tickets are resolved within SLA targets?",
            "Which tickets require the most frequent updates from support staff?"
        ]
        
        # Return only the required number of questions
        return generic_questions[:count]
    
    def _answer_question(self, question: str, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate an answer to a specific qualitative question"""
        # Prepare data summary for LLM
        data_summary = {
            "columns": list(df.columns),
            "row_count": len(df),
            "sample": df.head(3).to_dict(orient='records')
        }
        
        # Create a compact version of the analysis results
        compact_analysis = {
            "basic_stats": analysis_results.get("basic_stats", {}),
            "insights": analysis_results.get("insights", {})
        }
        
        # Use LLM to generate the answer
        messages = [
            {"role": "system", "content": """You are a data analysis expert specializing in ticket systems and automation.
Provide concise, data-driven answers to questions about ticket data, including automation potential."""},
            {"role": "user", "content": f"""
Answer the following question about ticket data:

QUESTION: {question}

Your answer should include these components, each limited to 1-3 sentences:
a. Direct answer to the question based on the data
b. Automation scope (what could be automated related to this issue)
c. Justification for the automation (why it would be valuable)
d. What kind of automation would be suitable (AI, RPA, rule-based, etc.)
e. Implementation plan for the automation (high-level steps)

The entire answer must be under 500 words total. Be specific and data-driven.
If there is no automation potential, clearly state that instead of sections b-e.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

ANALYSIS RESULTS:
{json.dumps(compact_analysis, indent=2)}

Format your response as a JSON object with these exact keys:
"answer", "automation_scope", "justification", "automation_type", "implementation_plan"
"""}
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
            
            answer = json.loads(json_str)
            return answer
        except Exception as e:
            print(f"Error generating answer for question '{question}': {str(e)}")
            # Return default response as fallback
            return {
                "answer": f"Based on the available data, it's difficult to provide a definitive answer to '{question}'.",
                "automation_scope": "Data collection and standardization to better track this information.",
                "justification": "Current data doesn't sufficiently capture this information, making analysis difficult.",
                "automation_type": "Data pipeline automation with validation rules.",
                "implementation_plan": "Implement structured data collection, enhance ticket forms, and create reporting dashboards."
            }