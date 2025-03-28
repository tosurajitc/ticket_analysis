# mainsrc/automation_module.py
# Enhanced wrapper for automation_analyzer

import sys
import os
import streamlit as st
import json

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing module
from automation_analyzer import AutomationAnalyzer

# Import local modules
from .utils import safe_json_dumps, validate_dict_fields

class AutomationModule:
    """
    Module for identifying automation opportunities from ticket data.
    Enhances the existing AutomationAnalyzer to provide more data-driven suggestions.
    """
    
    def __init__(self, groq_client):
        """
        Initialize the automation module.
        
        Args:
            groq_client: GROQ API client for LLM
        """
        self.automation_analyzer = AutomationAnalyzer(groq_client)
    
    def generate_opportunities(self, df, stats, insights):
        """
        Generate automation opportunities from statistics.
        
        Args:
            df: Processed DataFrame
            stats: Statistics dictionary
            insights: Generated insights text
            
        Returns:
            list: List of automation opportunity dictionaries
        """
        try:
            # Enhance the prompt with specific data points
            enhanced_insights = self._enhance_insights_for_automation(insights, stats)
            
            # Use the existing analyzer but with enhanced data
            opportunities = self.automation_analyzer.analyze(df, enhanced_insights)
            
            # Validate and enhance the opportunities
            validated_opportunities = self._validate_opportunities(opportunities, stats)
            
            return validated_opportunities
        except Exception as e:
            st.error(f"Error generating automation opportunities: {str(e)}")
            # Generate fallback opportunities
            return self._generate_fallback_opportunities(stats)
    
    def _enhance_insights_for_automation(self, insights, stats):
        """
        Enhance insights with specific data points for better automation opportunities.
        
        Args:
            insights: Original insights text
            stats: Statistics dictionary
            
        Returns:
            str: Enhanced insights text
        """
        # Start with original insights
        enhanced_text = insights + "\n\n"
        
        # Add specific automation-related data points
        enhanced_text += "AUTOMATION-SPECIFIC DATA POINTS:\n\n"
        
        # Add information about patterns in descriptions
        if "automation_opportunities_from_descriptions" in stats:
            patterns = stats["automation_opportunities_from_descriptions"]
            enhanced_text += "Patterns found in ticket descriptions:\n"
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                enhanced_text += f"- {pattern}: {count} occurrences\n"
            enhanced_text += "\n"
        
        # Add information about resolution patterns
        if "automation_opportunities_from_resolutions" in stats:
            resolutions = stats["automation_opportunities_from_resolutions"]
            enhanced_text += "Common resolution patterns found:\n"
            for pattern, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
                enhanced_text += f"- {pattern}: {count} occurrences\n"
            enhanced_text += "\n"
        
        # Add information about time statistics
        if "avg_resolution_time_hours" in stats:
            enhanced_text += f"Average resolution time: {stats['avg_resolution_time_hours']:.2f} hours\n"
            if "resolution_time_quartiles" in stats:
                q = stats["resolution_time_quartiles"]
                enhanced_text += f"Resolution time quartiles: 25%={q['25%']:.2f}h, 50%={q['50%']:.2f}h, 75%={q['75%']:.2f}h, 90%={q['90%']:.2f}h\n"
            enhanced_text += "\n"
        
        # Add information about top issues
        if "top_issues" in stats:
            enhanced_text += "Top issues by occurrence:\n"
            for issue, count in list(stats["top_issues"].items())[:5]:
                enhanced_text += f"- {issue}: {count} occurrences\n"
            enhanced_text += "\n"
        
        return enhanced_text
    
    def _validate_opportunities(self, opportunities, stats):
        """
        Validate and enhance automation opportunities.
        
        Args:
            opportunities: List of opportunity dictionaries
            stats: Statistics dictionary
            
        Returns:
            list: Validated and enhanced opportunities
        """
        if not opportunities or not isinstance(opportunities, list):
            return self._generate_fallback_opportunities(stats)
        
        validated_opps = []
        required_fields = ['title', 'scope', 'justification', 'type', 'implementation_plan']
        
        for opp in opportunities:
            if not isinstance(opp, dict):
                continue
                
            # Validate required fields exist
            validated_opp = validate_dict_fields(opp, required_fields)
            
            # Enhance the justification if it seems generic
            if not self._is_data_driven_justification(validated_opp['justification'], stats):
                validated_opp['justification'] = self._enhance_justification(validated_opp['title'], validated_opp['justification'], stats)
            
            validated_opps.append(validated_opp)
        
        # Ensure we have at least 5 opportunities
        while len(validated_opps) < 5:
            fallback_opps = self._generate_fallback_opportunities(stats)
            for opp in fallback_opps:
                if len(validated_opps) < 5 and opp['title'] not in [vo['title'] for vo in validated_opps]:
                    validated_opps.append(opp)
        
        return validated_opps[:5]  # Return at most 5 opportunities
    
    def _is_data_driven_justification(self, justification, stats):
        """
        Check if a justification seems data-driven.
        
        Args:
            justification: Justification text
            stats: Statistics dictionary
            
        Returns:
            bool: True if justification appears data-driven
        """
        # Look for numeric references that match our statistics
        if not justification:
            return False
            
        # Check for specific numbers from our stats
        has_data_references = False
        
        # Check for ticket counts
        if "total_tickets" in stats:
            total_str = str(stats["total_tickets"])
            if total_str in justification:
                has_data_references = True
        
        # Check for percentages
        import re
        percent_patterns = re.findall(r'\d+(?:\.\d+)?%', justification)
        if percent_patterns:
            has_data_references = True
        
        # Check for resolution time references
        if "avg_resolution_time_hours" in stats:
            time_str = f"{stats['avg_resolution_time_hours']:.1f}"
            if time_str in justification or "hours" in justification:
                has_data_references = True
        
        # Check for pattern counts
        if "automation_opportunities_from_descriptions" in stats:
            for pattern, count in stats["automation_opportunities_from_descriptions"].items():
                if str(count) in justification and pattern in justification.lower():
                    has_data_references = True
        
        return has_data_references
    
    def _enhance_justification(self, title, justification, stats):
        """
        Enhance a generic justification with data-driven elements.
        
        Args:
            title: Opportunity title
            justification: Original justification text
            stats: Statistics dictionary
            
        Returns:
            str: Enhanced justification
        """
        # Start with original justification
        enhanced = justification + "\n\n"
        enhanced += "Data-driven justification:\n"
        
        # Add relevant statistics based on the opportunity title
        lower_title = title.lower()
        
        # Password reset automation
        if "password" in lower_title:
            if "automation_opportunities_from_descriptions" in stats and "password_reset" in stats["automation_opportunities_from_descriptions"]:
                count = stats["automation_opportunities_from_descriptions"]["password_reset"]
                enhanced += f"- {count} tickets mention password reset in their description.\n"
            if "automation_opportunities_from_resolutions" in stats and "password_reset_done" in stats["automation_opportunities_from_resolutions"]:
                count = stats["automation_opportunities_from_resolutions"]["password_reset_done"]
                enhanced += f"- {count} tickets were resolved with a password reset.\n"
            if "top_issues" in stats and "password" in stats["top_issues"]:
                count = stats["top_issues"]["password"]
                enhanced += f"- {count} tickets contain password-related issues.\n"
        
        # Access request automation
        elif "access" in lower_title:
            if "automation_opportunities_from_descriptions" in stats and "access_request" in stats["automation_opportunities_from_descriptions"]:
                count = stats["automation_opportunities_from_descriptions"]["access_request"]
                enhanced += f"- {count} tickets involve access requests.\n"
            if "automation_opportunities_from_resolutions" in stats and "provided_access" in stats["automation_opportunities_from_resolutions"]:
                count = stats["automation_opportunities_from_resolutions"]["provided_access"]
                enhanced += f"- {count} tickets were resolved by providing access.\n"
            if "top_issues" in stats and "access" in stats["top_issues"]:
                count = stats["top_issues"]["access"]
                enhanced += f"- {count} tickets contain access-related issues.\n"
        
        # Categorization/routing automation
        elif "categori" in lower_title or "routing" in lower_title:
            if "assignment_group" in stats and stats["top_assignment_groups"]:
                top_group = list(stats["top_assignment_groups"].items())[0]
                enhanced += f"- The most common assignment group is '{top_group[0]}' with {top_group[1]} tickets.\n"
                enhanced += f"- Improving routing could reduce reassignment and decrease resolution time.\n"
                if "avg_resolution_time_hours" in stats:
                    enhanced += f"- Current average resolution time is {stats['avg_resolution_time_hours']:.2f} hours.\n"
        
        # Knowledge base automation
        elif "knowledge" in lower_title or "kb" in lower_title:
            if "top_issues" in stats:
                enhanced += "- Top recurring issues that could benefit from knowledge articles:\n"
                for issue, count in list(stats["top_issues"].items())[:3]:
                    enhanced += f"  * {issue}: {count} tickets\n"
            if "automation_opportunities_from_resolutions" in stats and "knowledge_article" in stats["automation_opportunities_from_resolutions"]:
                count = stats["automation_opportunities_from_resolutions"]["knowledge_article"]
                enhanced += f"- {count} tickets were resolved using knowledge base articles.\n"
        
        # SLA monitoring automation
        elif "sla" in lower_title or "monitor" in lower_title:
            if "avg_resolution_time_hours" in stats:
                enhanced += f"- Average resolution time: {stats['avg_resolution_time_hours']:.2f} hours\n"
                if "resolution_time_quartiles" in stats:
                    q = stats["resolution_time_quartiles"]
                    enhanced += f"- 25% of tickets are resolved in {q['25%']:.2f} hours or less\n"
                    enhanced += f"- 75% of tickets are resolved in {q['75%']:.2f} hours or less\n"
                    enhanced += f"- 10% of tickets take longer than {q['90%']:.2f} hours to resolve\n"
        
        # Chatbot automation
        elif "chat" in lower_title or "bot" in lower_title:
            if "top_issues" in stats:
                enhanced += "- Common issues that could be handled by a chatbot:\n"
                for issue, count in list(stats["top_issues"].items())[:3]:
                    enhanced += f"  * {issue}: {count} tickets\n"
            if "business_hours_percentage" in stats and "non_business_hours_percentage" in stats:
                enhanced += f"- {stats['non_business_hours_percentage']:.1f}% of tickets are created outside business hours\n"
                enhanced += "- A chatbot could provide 24/7 support for these cases\n"
        
        # Add general justification about time savings if resolution time stats are available
        if "avg_resolution_time_hours" in stats and "total_tickets" in stats:
            avg_time = stats["avg_resolution_time_hours"]
            total = stats["total_tickets"]
            
            # Estimate potential time savings (conservatively 20%)
            potential_savings = avg_time * total * 0.2
            enhanced += f"\nPotential time savings: If automation reduces resolution time by 20%, approximately {potential_savings:.0f} staff hours could be saved."
        
        return enhanced
    
    def _generate_fallback_opportunities(self, stats):
        """
        Generate data-driven fallback automation opportunities when LLM fails.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            list: List of opportunity dictionaries
        """
        opportunities = []
        
        # Check for specific opportunities from text analysis
        has_password_issues = (
            "automation_opportunities_from_descriptions" in stats and 
            "password_reset" in stats["automation_opportunities_from_descriptions"]
        ) or (
            "top_issues" in stats and 
            "password" in stats["top_issues"]
        )
        
        has_access_issues = (
            "automation_opportunities_from_descriptions" in stats and 
            "access_request" in stats["automation_opportunities_from_descriptions"]
        ) or (
            "top_issues" in stats and 
            "access" in stats["top_issues"]
        )
        
        has_service_restarts = (
            "automation_opportunities_from_resolutions" in stats and 
            "restarted_service" in stats["automation_opportunities_from_resolutions"]
        )
        
        # 1. Password Reset Automation (if relevant)
        if has_password_issues:
            password_count = stats.get("automation_opportunities_from_descriptions", {}).get("password_reset", 0)
            password_count += stats.get("top_issues", {}).get("password", 0)
            
            password_opportunity = {
                "title": "Automated Password Reset System",
                "scope": "Implement a self-service password reset portal that allows users to securely reset their passwords without IT intervention. The system should integrate with Active Directory and include multi-factor authentication for security.",
                "justification": f"Analysis of ticket data shows {password_count} tickets related to password resets. These follow a standard procedure that could be easily automated, reducing IT staff workload and providing immediate resolution for users.",
                "type": "Self-service portal with RPA integration. Use a web portal for user interaction and RPA to execute the reset procedures in Active Directory.",
                "implementation_plan": "1. Requirements gathering (2 weeks)\n2. System design (2 weeks)\n3. Development of self-service portal (3 weeks)\n4. Integration with Active Directory (2 weeks)\n5. Security testing (1 week)\n6. User acceptance testing (1 week)\n7. Deployment and training (1 week)\n\nRequired resources: 1 Web Developer, 1 Systems Administrator, 1 Security Specialist"
            }
            opportunities.append(password_opportunity)
        
        # 2. Access Request Automation (if relevant)
        if has_access_issues:
            access_count = stats.get("automation_opportunities_from_descriptions", {}).get("access_request", 0)
            access_count += stats.get("top_issues", {}).get("access", 0)
            
            access_opportunity = {
                "title": "Automated Access Request System",
                "scope": "Develop a self-service portal for access requests with automated approval workflows, provisioning, and audit trails.",
                "justification": f"Analysis of ticket data shows {access_count} tickets related to access requests. These follow predictable approval processes that could be streamlined through automation.",
                "type": "Workflow automation with identity management integration. Build a web-based request system integrated with existing identity management systems.",
                "implementation_plan": "1. Requirements and access catalog definition (3 weeks)\n2. Workflow design (2 weeks)\n3. Portal development (4 weeks)\n4. Integration with identity systems (3 weeks)\n5. Testing and security review (2 weeks)\n6. Pilot with select departments (2 weeks)\n7. Full rollout (1 week)\n\nRequired resources: 1 Project Manager, 2 Developers, 1 Security/IAM Specialist"
            }
            opportunities.append(access_opportunity)
        
        # 3. Service Restart Automation (if relevant)
        if has_service_restarts:
            restart_count = stats.get("automation_opportunities_from_resolutions", {}).get("restarted_service", 0)
            
            restart_opportunity = {
                "title": "Automated Service Monitoring and Recovery",
                "scope": "Implement an automated service monitoring system that can detect service failures and automatically attempt recovery by restarting services without human intervention.",
                "justification": f"Analysis of resolution notes shows {restart_count} tickets were resolved by simply restarting services or systems. This is a perfect candidate for automation.",
                "type": "Infrastructure automation using monitoring tools and scripted remediation. Implement with tools like Nagios, Zabbix, or Microsoft System Center with PowerShell/Bash scripts for remediation actions.",
                "implementation_plan": "1. Service inventory and criticality assessment (2 weeks)\n2. Monitoring configuration (2 weeks)\n3. Remediation script development (3 weeks)\n4. Testing in dev environment (2 weeks)\n5. Staged production rollout (3 weeks)\n6. Documentation and handover (1 week)\n\nRequired resources: 1 Systems Engineer, 1 Script Developer, 1 QA Engineer"
            }
            opportunities.append(restart_opportunity)
        
        # 4. Ticket Categorization (always relevant)
        if "top_assignment_groups" in stats:
            top_group = list(stats["top_assignment_groups"].items())[0] if stats["top_assignment_groups"] else ("unknown", 0)
            
            categorization_opportunity = {
                "title": "AI-Powered Ticket Categorization and Routing",
                "scope": "Implement an AI system that automatically categorizes incoming tickets based on their description and routes them to the appropriate assignment group. The system should learn from past ticket assignments to improve accuracy over time.",
                "justification": f"With {stats.get('total_tickets', 0)} tickets in the dataset, manual categorization and routing is time-consuming and prone to errors. The top assignment group '{top_group[0]}' handles {top_group[1]} tickets. Automation would reduce response time and ensure tickets reach the right team immediately.",
                "type": "AI/ML solution using Natural Language Processing. Implement a machine learning model trained on historical ticket data to predict the appropriate category and assignment group.",
                "implementation_plan": "1. Data preparation and cleaning (3 weeks)\n2. Model development and training (4 weeks)\n3. Integration with ticketing system (2 weeks)\n4. Testing and validation (2 weeks)\n5. Pilot deployment (2 weeks)\n6. Full deployment and monitoring (1 week)\n\nRequired resources: 1 Data Scientist, 1 ML Engineer, 1 Systems Integrator"
            }
            opportunities.append(categorization_opportunity)
        
        # 5. Knowledge Base Article Suggestions (relevant if we have common issues)
        if "top_issues" in stats and len(stats["top_issues"]) > 0:
            top_issues_str = ", ".join([f"{issue} ({count} tickets)" for issue, count in list(stats["top_issues"].items())[:3]])
            
            kb_opportunity = {
                "title": "Automated Knowledge Base Article Suggestions",
                "scope": "Develop a system that automatically suggests relevant knowledge base articles to agents based on ticket content, and recommends new KB articles to be created for common issues that lack documentation.",
                "justification": f"Analysis of ticket data shows recurring issues that could be resolved faster with proper knowledge base articles. The most common issues are: {top_issues_str}. This would reduce resolution time and improve consistency in solutions.",
                "type": "AI-powered recommendation system using natural language processing to match ticket text with existing KB articles and identify knowledge gaps.",
                "implementation_plan": "1. KB article inventory and indexing (2 weeks)\n2. Development of text matching algorithm (3 weeks)\n3. Integration with ticketing system (2 weeks)\n4. KB gap analysis functionality (2 weeks)\n5. User interface development (2 weeks)\n6. Testing and refinement (2 weeks)\n7. Deployment and training (1 week)\n\nRequired resources: 1 Knowledge Management Specialist, 1 Developer, 1 UX Designer"
            }
            opportunities.append(kb_opportunity)
        
        # 6. SLA Monitoring (relevant if we have resolution time data)
        if "avg_resolution_time_hours" in stats:
            avg_time = stats["avg_resolution_time_hours"]
            percentiles = stats.get("resolution_time_quartiles", {})
            
            sla_opportunity = {
                "title": "Proactive SLA Monitoring and Alerting System",
                "scope": "Implement an automated system that monitors ticket SLAs in real-time, sends proactive alerts for tickets at risk of breaching SLA, and provides escalation paths based on ticket priority and age.",
                "justification": f"The average ticket resolution time is {avg_time:.2f} hours, with 75% of tickets taking up to {percentiles.get('75%', avg_time*1.5):.2f} hours to resolve. Many tickets likely breach SLA targets. A proactive monitoring system would improve compliance and customer satisfaction.",
                "type": "RPA and Business Rules Engine to monitor tickets and trigger alerts based on configurable rules and thresholds.",
                "implementation_plan": "1. SLA policy definition and mapping (2 weeks)\n2. Alert rules configuration (1 week)\n3. Notification system development (2 weeks)\n4. Dashboard development (2 weeks)\n5. Integration with ticketing system (2 weeks)\n6. Testing across different ticket types (1 week)\n7. Deployment and staff training (1 week)\n\nRequired resources: 1 Business Analyst, 1 Developer, 1 QA Tester"
            }
            opportunities.append(sla_opportunity)
        
        # 7. Chatbot (always a relevant option)
        if "business_hours_tickets" in stats and "non_business_hours_tickets" in stats:
            bh = stats["business_hours_tickets"]
            nbh = stats["non_business_hours_tickets"]
            total = bh + nbh
            nbh_percent = (nbh / total) * 100 if total > 0 else 0
            

            chatbot_opportunity = {
                "title": "IT Support Chatbot for First-Level Resolution",
                "scope": "Deploy an AI chatbot that can handle common IT issues, guide users through basic troubleshooting steps, and create tickets automatically when it cannot resolve the issue.",
                "justification": f"Analysis shows {nbh_percent:.1f}% of tickets ({nbh} tickets) are created outside business hours. Additionally, common issues like {', '.join(list(stats.get('top_issues', {}).keys())[:3])} could be addressed by an automated system. A chatbot can provide 24/7 support and immediate responses for these cases.",
                "type": "Conversational AI using natural language understanding and a decision tree-based resolution framework. Integration with existing ticketing system for seamless escalation.",
                "implementation_plan": "1. Define scope and common issues to address (2 weeks)\n2. Design conversation flows (3 weeks)\n3. Build NLU model (4 weeks)\n4. Develop troubleshooting logic (3 weeks)\n5. Integration with ticketing system (2 weeks)\n6. User testing and refinement (3 weeks)\n7. Pilot deployment (2 weeks)\n8. Full deployment and continuous improvement (ongoing)\n\nRequired resources: 1 Conversational AI Specialist, 1 IT Support SME, 1 Systems Integrator, 1 UX Designer"
            }
            opportunities.append(chatbot_opportunity)
        
        # Ensure we have at least 5 opportunities
        standard_opportunities = [
            {
                "title": "IT Support Chatbot for First-Level Resolution",
                "scope": "Deploy an AI chatbot that can handle common IT issues, guide users through basic troubleshooting steps, and create tickets automatically when it cannot resolve the issue.",
                "justification": "Many common IT issues follow standard troubleshooting patterns that can be automated. A chatbot can provide 24/7 support and immediate responses for these cases.",
                "type": "Conversational AI using natural language understanding and a decision tree-based resolution framework. Integration with existing ticketing system for seamless escalation.",
                "implementation_plan": "1. Define scope and common issues to address (2 weeks)\n2. Design conversation flows (3 weeks)\n3. Build NLU model (4 weeks)\n4. Develop troubleshooting logic (3 weeks)\n5. Integration with ticketing system (2 weeks)\n6. User testing and refinement (3 weeks)\n7. Pilot deployment (2 weeks)\n8. Full deployment and continuous improvement (ongoing)\n\nRequired resources: 1 Conversational AI Specialist, 1 IT Support SME, 1 Systems Integrator, 1 UX Designer"
            },
            {
                "title": "Self-Service Portal for Common Requests",
                "scope": "Create a comprehensive self-service portal where users can submit common requests (software installation, access changes, equipment requests) through structured forms with automated backend workflows.",
                "justification": "Many ticket types follow standard procedures and could be automated through self-service. This would reduce manual handling and provide faster resolution for users.",
                "type": "Web portal with workflow automation. Build a user-friendly interface that connects to backend systems via APIs or RPA.",
                "implementation_plan": "1. Request type inventory (2 weeks)\n2. Portal design and UX (3 weeks)\n3. Frontend development (4 weeks)\n4. Backend workflow development (5 weeks)\n5. Integration testing (2 weeks)\n6. User testing (2 weeks)\n7. Deployment (1 week)\n\nRequired resources: 1 UX Designer, 2 Web Developers, 1 Backend Developer, 1 QA Engineer"
            },
            {
                "title": "Automated Software Distribution System",
                "scope": "Implement an automated system for software deployment that allows users to request software from a catalog and automatically installs approved applications without IT intervention.",
                "justification": "Software installation and updates are common ticket types that follow standard procedures. Automation would reduce IT workload and provide immediate fulfillment.",
                "type": "Software distribution platform with self-service catalog and automated deployment scripts.",
                "implementation_plan": "1. Software inventory and policy creation (3 weeks)\n2. Platform selection and configuration (3 weeks)\n3. Package creation for common applications (4 weeks)\n4. Integration with identity management (2 weeks)\n5. Testing and validation (2 weeks)\n6. Pilot with selected user groups (2 weeks)\n7. Full deployment (1 week)\n\nRequired resources: 1 Systems Administrator, 1 Package Developer, 1 Security Specialist"
            }
        ]
        
        # Add standard opportunities if needed
        for opp in standard_opportunities:
            if len(opportunities) < 5 and opp["title"] not in [o["title"] for o in opportunities]:
                opportunities.append(opp)
        
        return opportunities[:5]  # Return at most 5 opportunities