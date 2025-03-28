# mainsrc/questions_module.py
# Module for predefined questions tab

import sys
import os
import streamlit as st
import json

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from insight_generator import InsightGenerator

# Import local modules
from .utils import safe_json_dumps, validate_dict_fields

class QuestionsModule:
    """
    Module for generating and displaying predefined questions and answers.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the questions module.
        
        Args:
            groq_client: GROQ API client for LLM
        """
        self.groq_client = groq_client
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.insight_generator = InsightGenerator(groq_client)
    
    def generate_predefined_questions(self, df, stats):
        """
        Generate predefined questions and answers based on statistics.
        
        Args:
            df: Processed DataFrame
            stats: Statistics dictionary
            
        Returns:
            list: List of question dictionaries
        """
        try:
            # Generate questions using the LLM
            model = self.model
            
            prompt = f"""
Based on the following ticket data statistics:
{safe_json_dumps(stats)}

Generate exactly 10 insightful questions that would help analyze this ticket data.

Include these 5 specific questions:
1. Which category of incidents consume the greatest number of hours?
2. Which category of incidents require datafix?
3. Which issues are specific to a particular country?
4. Which incidents indicate escalation?
5. How many customer facing documents failed in the last year?

Pay special attention to geographic information which may be found in columns like "Subcategory 3" or other similar fields. The ticket data may contain regional information (like AMER, APAC, EMEA, GLOBAL) along with country-specific data. Be sure to analyze this geographic data thoroughly when answering questions, especially #3 about country-specific issues.

Then generate 5 additional random questions that cover different aspects of the ticket data such as:
- Priority distribution analysis
- Assignment group workload and efficiency
- Ticket volume trends over time
- Business hours vs. non-business hours analysis
- Common issue identification
- Resolution time patterns
- SLA compliance
- Self-service opportunities
- Team performance comparisons

For each question, provide:
1. A clear, data-driven answer based on the statistics provided
2. Whether there's automation potential (Yes/No) and a detailed explanation of what kind of automation would be suitable

For questions where the data doesn't have enough information to provide a complete answer, acknowledge the limitation and suggest what additional data would be needed.

Format your response as a JSON array with objects like:
[
  {{
    "question": "Which category of incidents consume the greatest number of hours?",
    "answer": "Based on the data, Network-related incidents consume the greatest number of hours with an average resolution time of 28.5 hours per ticket. Database issues follow with 24.3 hours on average.",
    "automation_potential": "Yes - Automated diagnostic tools and predefined resolution workflows could be implemented for network issues to reduce resolution time. Machine learning models could predict resolution time and proactively allocate resources to high-consumption categories."
  }}
]
"""
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in IT service management and ticket analysis. Generate insightful questions and data-driven answers based on ticket statistics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            # Extract and parse the JSON response
            response_text = response.choices[0].message.content
            
            # Extract JSON if it's embedded in markdown or additional text
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    questions = json.loads(json_str)
                    
                    # Validate that each question has all required fields
                    validated_questions = []
                    for q in questions:
                        validated_q = validate_dict_fields(
                            q, 
                            ['question', 'answer', 'automation_potential'],
                            {'question': 'Question not available', 'answer': 'No answer available', 'automation_potential': 'No automation potential information'}
                        )
                        validated_questions.append(validated_q)
                    
                    return validated_questions[:10]  # Ensure we return at most 10 questions
                    
                except json.JSONDecodeError:
                    st.warning("Could not parse the generated questions. Using fallback questions.")
                    return self._generate_fallback_questions(stats)
            else:
                st.warning("No valid JSON found in the response. Using fallback questions.")
                return self._generate_fallback_questions(stats)
            
        except Exception as e:
            st.error(f"Error generating predefined questions: {str(e)}")
            return self._generate_fallback_questions(stats)
    
    def _generate_fallback_questions(self, stats):
        """
        Generate fallback questions when LLM fails.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            list: List of question dictionaries
        """
        # Start with the 5 specific required questions
        specific_questions = [
            {
                "question": "Which category of incidents consume the greatest number of hours?",
                "answer": self._get_highest_time_consumers_answer(stats),
                "automation_potential": "Yes - Time tracking automation and category-based analytics could help identify time-consuming incident types."
            },
            {
                "question": "Which category of incidents require datafix?",
                "answer": self._get_datafix_answer(stats),
                "automation_potential": "Yes - Automated classification of tickets that required datafixes based on resolution notes and actions taken."
            },
            {
                "question": "Which issues are specific to a particular country?",
                "answer": self._get_country_specific_issues_answer(stats),
                "automation_potential": "Yes - Automated geographic tagging and region-specific issue tracking could be implemented."
            },
            {
                "question": "Which incidents indicate escalation?",
                "answer": self._get_escalation_answer(stats),
                "automation_potential": "Yes - Automated escalation detection and workflow management could be implemented to identify and track escalated tickets."
            },
            {
                "question": "How many customer facing documents failed in the last year?",
                "answer": self._get_document_failures_answer(stats),
                "automation_potential": "Yes - Automated document monitoring and failure detection systems could be implemented to track customer-facing document issues."
            }
        ]
        
        # Generate 5 more questions based on available statistics
        additional_questions = []
        
        # Priority question
        if "priority_distribution" in stats:
            priorities = stats["priority_distribution"]
            top_priority = max(priorities.items(), key=lambda x: x[1])
            priority_question = {
                "question": "What is the distribution of ticket priorities?",
                "answer": f"The most common priority is '{top_priority[0]}' with {top_priority[1]} tickets. The full distribution is: " + 
                          ", ".join([f"{p}: {c} tickets ({(c/sum(priorities.values()))*100:.1f}%)" for p, c in priorities.items()]),
                "automation_potential": "Yes - Automatic ticket prioritization based on text analysis could be implemented to ensure consistent priority assignment."
            }
            additional_questions.append(priority_question)
        
        # Assignment group question
        if "top_assignment_groups" in stats:
            groups = stats["top_assignment_groups"]
            top_group = list(groups.items())[0] if groups else ("Unknown", 0)
            group_question = {
                "question": "Which teams handle the most tickets?",
                "answer": f"The team handling the most tickets is '{top_group[0]}' with {top_group[1]} tickets. " + 
                          "Other top teams include: " + ", ".join([f"{g}: {c}" for g, c in list(groups.items())[1:4]]),
                "automation_potential": "Yes - Workload balancing and automatic ticket routing could be implemented to distribute tickets more evenly."
            }
            additional_questions.append(group_question)
        
        # Resolution time question
        if "avg_resolution_time_hours" in stats:
            avg_time = stats["avg_resolution_time_hours"]
            time_question = {
                "question": "What is the average resolution time for tickets?",
                "answer": f"The average resolution time is {avg_time:.2f} hours. " + 
                          (f"25% of tickets are resolved in {stats.get('resolution_time_quartiles', {}).get('25%', 0):.2f} hours or less, " if "resolution_time_quartiles" in stats else "") +
                          (f"while 75% are resolved within {stats.get('resolution_time_quartiles', {}).get('75%', 0):.2f} hours." if "resolution_time_quartiles" in stats else ""),
                "automation_potential": "Yes - Automated SLA monitoring and alerting could be implemented to prevent SLA breaches."
            }
            additional_questions.append(time_question)
        
        # Common issues question
        if "top_issues" in stats:
            issues = stats["top_issues"]
            issues_question = {
                "question": "What are the most common issues in the tickets?",
                "answer": "The most common issues are: " + 
                          ", ".join([f"{i}: {c} tickets" for i, c in list(issues.items())[:5]]),
                "automation_potential": "Yes - Knowledge base articles and automated responses could be created for common issues to speed up resolution."
            }
            additional_questions.append(issues_question)
        
        # Business hours question
        if "business_hours_tickets" in stats and "non_business_hours_tickets" in stats:
            bh = stats["business_hours_tickets"]
            nbh = stats["non_business_hours_tickets"]
            total = bh + nbh
            bh_question = {
                "question": "How many tickets are created outside of business hours?",
                "answer": f"{nbh} tickets ({(nbh/total)*100:.1f}%) were created outside business hours, while {bh} tickets ({(bh/total)*100:.1f}%) were created during business hours.",
                "automation_potential": "Yes - After-hours automated responses and escalation procedures could be implemented for tickets created outside business hours."
            }
            additional_questions.append(bh_question)
        
        # Add more generic questions if needed
        generic_questions = [
            {
                "question": "Is there a pattern in ticket creation by day of week?",
                "answer": "Analysis of day-of-week patterns would require additional data processing to identify peak days.",
                "automation_potential": "Yes - Staff scheduling optimization could be implemented based on historical ticket volume patterns."
            },
            {
                "question": "What percentage of tickets are resolved within SLA?",
                "answer": "SLA compliance data is not explicitly available in the current statistics. This would require mapping each ticket's resolution time against its SLA target.",
                "automation_potential": "Yes - Automated SLA tracking and alerting system could be implemented to monitor and improve compliance."
            },
            {
                "question": "Are there recurring ticket types that could be addressed with self-service?",
                "answer": "Based on the common issues identified, several ticket types appear to be good candidates for self-service solutions.",
                "automation_potential": "Yes - Self-service portal with knowledge base articles for common issues could be implemented to reduce ticket volume."
            }
        ]
        
        # Combine questions
        all_questions = specific_questions + additional_questions
        
        # Add generic questions if needed
        for q in generic_questions:
            if len(all_questions) < 10:
                all_questions.append(q)
        
        return all_questions[:10]  # Return at most 10 questions
    
    def _get_highest_time_consumers_answer(self, stats):
        """Generate answer about highest time-consuming incidents."""
        if "top_issues" in stats and "avg_resolution_time_hours" in stats:
            return f"Based on available data, the average resolution time across all tickets is {stats['avg_resolution_time_hours']:.2f} hours. To identify which categories consume the most time, we would need resolution time broken down by category, which is not directly available in the current dataset."
        return "This analysis requires resolution time data broken down by category or incident type, which is not available in the current dataset."
    
    def _get_datafix_answer(self, stats):
        """Generate answer about datafix incidents."""
        if "datafix_mentions" in stats:
            answer = f"There are {stats['datafix_mentions']} tickets that mention datafix requirements in their description."
            if "datafix_by_region" in stats:
                regions = list(list(stats["datafix_by_region"].values())[0].items())
                if regions:
                    top_region = regions[0]
                    answer += f" The region with the most datafix incidents is {top_region[0]} with {top_region[1]} tickets."
            return answer
        return "There is no explicit mention of 'datafix' in the available ticket data. More detailed categorization or text analysis would be needed to identify tickets requiring data fixes."
    
    def _get_country_specific_issues_answer(self, stats):
        """Generate answer about country-specific issues."""
        if "geographic_columns_found" in stats:
            geo_cols = stats["geographic_columns_found"]
            if "region_specific_issues" in stats:
                region_issues = stats["region_specific_issues"]
                if region_issues:
                    regions = list(region_issues.keys())
                    answer = f"Geographic data was found in columns: {', '.join(geo_cols)}. "
                    answer += f"Analysis shows region-specific issues for: {', '.join(regions[:3])}. "
                    
                    # Add detail for the first region
                    if regions:
                        first_region = regions[0]
                        region_data = region_issues[first_region]
                        col_name = list(region_data.keys())[0]
                        issues = region_data[col_name]
                        answer += f"For {first_region} ({col_name}), the top issues are: " + ", ".join([f"{i}" for i in issues.keys()])
                    
                    return answer
            else:
                return f"Geographic data was found in columns: {', '.join(geo_cols)}, but no significant region-specific issues were identified in the analysis."
        
        # Check for other geographic information
        if "top_regions" in stats:
            regions = list(stats["top_regions"].items())
            return f"The data contains geographic information with top regions: {', '.join([r[0] for r in regions[:3]])}, but no significant region-specific issues were identified."
        
        return "No geographic or country data was found in the dataset. To identify country-specific issues, location information would need to be added to the tickets."
    
    def _get_escalation_answer(self, stats):
        """Generate answer about escalation incidents."""
        if "escalation_mentions" in stats:
            answer = f"There are {stats['escalation_mentions']} tickets that mention escalation in their work notes."
            if "escalation_by_region" in stats:
                regions = list(list(stats["escalation_by_region"].values())[0].items())
                if regions:
                    top_region = regions[0]
                    answer += f" The region with the most escalations is {top_region[0]} with {top_region[1]} tickets."
            return answer
        return "There is no explicit mention of 'escalation' in the available ticket data. More detailed analysis of work notes or state transitions would be needed to identify escalated tickets."
    
    def _get_document_failures_answer(self, stats):
        """Generate answer about document failures."""
        if "document_failure_count" in stats:
            answer = f"There are {stats['document_failure_count']} tickets that mention document failures."
            if "document_failures_by_year" in stats:
                years = stats["document_failures_by_year"]
                current_year = max(years.keys())
                answer += f" In the last year ({current_year}), there were {years[current_year]} document failure incidents."
            return answer
        return "There is no explicit mention of 'document failure' in the available ticket data. Additional categorization or tagging would be needed to identify customer-facing document failures."
    
    def display_questions(self, questions):
        """
        Display predefined questions and answers in the UI.
        
        Args:
            questions: List of question dictionaries
        """
        try:
            if not questions:
                st.info("No predefined questions available. Please upload ticket data first.")
                return
                
            for i, qa_pair in enumerate(questions):
                # Ensure minimal display if any field is missing
                question = qa_pair.get('question', "Question not available")
                
                with st.expander(f"Q{i+1}: {question}"):
                    answer = qa_pair.get('answer', "Answer not available")
                    st.markdown(f"**Answer:** {answer}")
                    
                    automation = qa_pair.get('automation_potential', "Automation potential not available")
                    st.markdown(f"**Automation Potential:** {automation}")
        except Exception as e:
            st.error(f"Error displaying questions: {str(e)}")
            st.info("There was an issue displaying the predefined questions. Please try uploading the data again.")