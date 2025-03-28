# mainsrc/analysis_module.py
# Statistical analysis and pattern detection

import pandas as pd
import numpy as np
import streamlit as st
from .utils import NumpyEncoder, safe_json_dumps

class AnalysisModule:
    """
    Handles statistical analysis and pattern detection in ticket data.
    Extracts key statistics that will be used for insights and automation.
    """
    
    def __init__(self):
        """Initialize the analysis module."""
        # Region keywords for geographic analysis
        self.region_keywords = [
            'amer', 'apac', 'emea', 'global', 'north america', 'europe', 'asia', 
            'africa', 'australia', 'usa', 'uk', 'canada', 'india', 'china', 
            'japan', 'germany', 'france', 'italy', 'spain', 'brazil', 'mexico', 
            'russia', 'singapore', 'hong kong', 'uae', 'south africa'
        ]
    
    def extract_statistics(self, df):
        """
        Extract comprehensive statistics from processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            dict: Dictionary of statistics in JSON-serializable format
        """
        try:
            stats = {}
            
            # Basic counts
            stats["total_tickets"] = int(len(df))
            
            # Extract distributions
            self._extract_basic_distributions(df, stats)
            
            # Geographic analysis
            self._extract_geographic_data(df, stats)
            
            # Time-based statistics
            self._extract_time_statistics(df, stats)
            
            # Issue analysis
            self._extract_issue_patterns(df, stats)
            
            # Custom analysis for specific questions
            self._extract_custom_data(df, stats)
            
            return stats
            
        except Exception as e:
            st.error(f"Error extracting statistics: {str(e)}")
            return {"error": str(e), "total_tickets": len(df) if df is not None else 0}
    
    def _extract_basic_distributions(self, df, stats):
        """Extract basic distributions like priority, state, etc."""
        # Priority distribution if available
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts()
            stats["priority_distribution"] = {str(k): int(v) for k, v in zip(priority_counts.index, priority_counts.values)}
        
        # State distribution if available
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            stats["state_distribution"] = {str(k): int(v) for k, v in zip(state_counts.index, state_counts.values)}
        
        # Assignment group distribution if available
        if 'assignment group' in df.columns:
            # Get top 10 assignment groups
            top_groups = df['assignment group'].value_counts().head(10)
            stats["top_assignment_groups"] = {str(k): int(v) for k, v in zip(top_groups.index, top_groups.values)}
    
    def _extract_geographic_data(self, df, stats):
        """Extract geographic data from the dataset."""
        # First, try to identify columns that might contain geographic data
        geo_columns = []
        
        # Check column names that suggest geographic content
        for col in df.columns:
            if any(term in col.lower() for term in ['region', 'country', 'location', 'geo', 'territory']):
                geo_columns.append(col)
            else:
                # Check if column contains geographic terms
                if df[col].dtype == 'object':  # Only check string columns
                    sample = df[col].dropna().astype(str).str.lower().head(100)
                    geo_matches = sample.str.contains('|'.join(self.region_keywords), regex=True).sum()
                    if geo_matches > 5:  # If at least 5 matches in the sample, consider it a geo column
                        geo_columns.append(col)
        
        # Special check for subcategory columns which often contain geographic data
        for col in df.columns:
            if 'subcategory' in col.lower() and col not in geo_columns:
                if df[col].dtype == 'object':  # Only check string columns
                    sample = df[col].dropna().astype(str).str.lower().head(100)
                    geo_matches = sample.str.contains('|'.join(self.region_keywords), regex=True).sum()
                    if geo_matches > 2:  # Lower threshold for subcategory columns
                        geo_columns.append(col)
        
        # Now extract geographic data from the identified columns
        if geo_columns:
            stats["geographic_columns_found"] = geo_columns
            region_data = {}
            
            for col in geo_columns:
                region_counts = df[col].value_counts().head(15)
                region_data[col] = {str(k): int(v) for k, v in zip(region_counts.index, region_counts.values)}
            
            stats["geographic_distribution"] = region_data
            
            # Try to identify highest volume regions/countries
            all_regions = {}
            for col, regions in region_data.items():
                for region, count in regions.items():
                    region_str = str(region).lower()
                    # Check if this is actually a geographic term
                    if any(keyword in region_str for keyword in self.region_keywords):
                        all_regions[region] = all_regions.get(region, 0) + count
            
            if all_regions:
                top_regions = dict(sorted(all_regions.items(), key=lambda x: x[1], reverse=True)[:10])
                stats["top_regions"] = top_regions
                
            # Analyze region-specific issues if we have geographic and issue data
            self._analyze_region_specific_issues(df, stats, geo_columns)
    
    def _extract_time_statistics(self, df, stats):
        """Extract time-based statistics."""
        # Resolution time statistics
        if 'resolution_time_hours' in df.columns:
            stats["avg_resolution_time_hours"] = float(df['resolution_time_hours'].mean())
            stats["median_resolution_time_hours"] = float(df['resolution_time_hours'].median())
            
            # Add resolution time quartiles
            stats["resolution_time_quartiles"] = {
                "25%": float(df['resolution_time_hours'].quantile(0.25)),
                "50%": float(df['resolution_time_hours'].quantile(0.50)),
                "75%": float(df['resolution_time_hours'].quantile(0.75)),
                "90%": float(df['resolution_time_hours'].quantile(0.90))
            }
        
        # Business hours vs. non-business hours
        if 'is_business_hours' in df.columns:
            stats["business_hours_tickets"] = int(df['is_business_hours'].sum())
            stats["non_business_hours_tickets"] = int((~df['is_business_hours']).sum())
            
            # Add percentage
            total = stats["business_hours_tickets"] + stats["non_business_hours_tickets"]
            if total > 0:
                stats["business_hours_percentage"] = float((stats["business_hours_tickets"] / total) * 100)
                stats["non_business_hours_percentage"] = float((stats["non_business_hours_tickets"] / total) * 100)
        
        # Monthly trends if we have date data
        if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
            monthly_counts = df.groupby(df['opened'].dt.to_period('M')).size()
            stats["monthly_ticket_counts"] = {str(period): int(count) for period, count in monthly_counts.items()}
    
    def _extract_issue_patterns(self, df, stats):
        """Extract patterns from issue descriptions and resolutions."""
        # Common issues from short descriptions
        if 'short description' in df.columns:
            # Analyze short descriptions for automation opportunities
            desc_text = ' '.join(df['short description'].astype(str).tolist())
            desc_text = desc_text.lower()
            
            # Define common automation-related patterns
            automation_patterns = {
                'password_reset': ['password reset', 'reset password', 'forgot password', 'change password'],
                'access_request': ['access request', 'request access', 'need access', 'grant access', 'permission'],
                'system_outage': ['outage', 'down', 'not working', 'unavailable', 'cannot access'],
                'software_install': ['install', 'installation', 'deploy', 'deployment', 'update', 'upgrade'],
                'account_locked': ['account locked', 'locked out', 'unlock account', 'locked account'],
                'email_issues': ['email', 'outlook', 'mailbox', 'mail', 'exchange'],
                'network_issues': ['network', 'wifi', 'connection', 'internet', 'vpn', 'remote access'],
                'printer_issues': ['print', 'printer', 'scanning', 'scanner'],
                'data_request': ['data export', 'report request', 'need report', 'data request'],
                'user_creation': ['new user', 'create user', 'new hire', 'new employee', 'onboarding']
            }
            
            # Count occurrences of each pattern
            automation_counts = {}
            for category, patterns in automation_patterns.items():
                count = 0
                for pattern in patterns:
                    count += desc_text.count(pattern)
                if count > 0:
                    automation_counts[category] = count
            
            # Add to stats if we found patterns
            if automation_counts:
                stats["automation_opportunities_from_descriptions"] = automation_counts
        
        # Extract common issues from the issue flags
        issue_columns = [col for col in df.columns if col.startswith('contains_')]
        if issue_columns:
            issue_counts = {}
            for col in issue_columns:
                issue_name = col.replace('contains_', '')
                issue_counts[issue_name] = int(df[col].sum())
            
            # Get top issues
            top_issues = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            stats["top_issues"] = top_issues
        
        # Analyze closed notes for resolution patterns
        closed_notes_column = None
        for col_name in ['closed notes', 'close notes', 'resolution notes', 'resolution', 'work notes']:
            if col_name in df.columns:
                closed_notes_column = col_name
                break
        
        if closed_notes_column:
            # Combine all notes into one text for analysis
            notes_text = ' '.join(df[closed_notes_column].astype(str).fillna('').tolist())
            notes_text = notes_text.lower()
            
            # Define patterns that indicate resolution methods suitable for automation
            resolution_patterns = {
                'password_reset_done': ['password reset', 'reset user password', 'password changed'],
                'account_unlock': ['account unlocked', 'unlocked user', 'unlocked account'],
                'provided_access': ['granted access', 'access provided', 'gave access', 'added permission'],
                'installed_software': ['installed', 'software installed', 'deployment completed'],
                'restarted_service': ['restarted', 'rebooted', 'service restarted', 'restart resolved'],
                'cleared_cache': ['cleared cache', 'cache cleared', 'deleted temp files'],
                'data_correction': ['data fixed', 'corrected data', 'fixed record', 'database correction'],
                'configuration_change': ['reconfigured', 'changed setting', 'updated configuration', 'config change'],
                'knowledge_article': ['per knowledge article', 'followed kb', 'standard procedure', 'according to documentation'],
                'user_education': ['showed user', 'educated user', 'trained user', 'explained to user']
            }
            
            # Count occurrences of each resolution pattern
            resolution_counts = {}
            for category, patterns in resolution_patterns.items():
                count = 0
                for pattern in patterns:
                    count += notes_text.count(pattern)
                if count > 0:
                    resolution_counts[category] = count
            
            # Add to stats if we found patterns
            if resolution_counts:
                stats["automation_opportunities_from_resolutions"] = resolution_counts
    
    def _analyze_region_specific_issues(self, df, stats, geo_columns):
        """Analyze issues specific to geographic regions."""
        if 'top_regions' in stats and len(geo_columns) > 0:
            # Find columns that represent issues
            issue_columns = [col for col in df.columns if col.startswith('contains_')]
            
            if len(issue_columns) > 0:
                region_issues = {}
                
                # For each geo column, analyze issues by region
                for geo_col in geo_columns:
                    # Get unique regions with at least 5 tickets
                    regions = df[geo_col].value_counts()
                    regions = regions[regions >= 5].index.tolist()
                    
                    for region in regions:
                        region_df = df[df[geo_col] == region]
                        region_issue_counts = {}
                        
                        # Count issues for this region
                        for issue_col in issue_columns:
                            issue_name = issue_col.replace('contains_', '')
                            issue_count = int(region_df[issue_col].sum())
                            if issue_count > 0:
                                region_issue_counts[issue_name] = issue_count
                        
                        # Add to region issues if significant
                        if len(region_issue_counts) > 0:
                            # Get top 3 issues
                            top_issues = dict(sorted(region_issue_counts.items(), key=lambda x: x[1], reverse=True)[:3])
                            if region not in region_issues:
                                region_issues[str(region)] = {}
                            region_issues[str(region)][geo_col] = top_issues
                
                if region_issues:
                    stats["region_specific_issues"] = region_issues
    
    def _extract_custom_data(self, df, stats):
        """Extract data for specific predefined questions."""
        # Datafix analysis
        if 'short description' in df.columns:
            datafix_keywords = ['datafix', 'data fix', 'db fix', 'database fix']
            datafix_count = 0
            for keyword in datafix_keywords:
                datafix_count += df['short description'].str.lower().str.contains(keyword, na=False).sum()
            if datafix_count > 0:
                stats["datafix_mentions"] = int(datafix_count)
                
                # If we have geographic data, check for region-specific datafixes
                if 'geographic_columns_found' in stats:
                    datafix_by_region = {}
                    for col in stats["geographic_columns_found"]:
                        region_counts = {}
                        for region in df[col].dropna().unique():
                            region_df = df[df[col] == region]
                            region_datafix_count = 0
                            for keyword in datafix_keywords:
                                if 'short description' in region_df.columns:
                                    region_datafix_count += region_df['short description'].str.lower().str.contains(keyword, na=False).sum()
                            if region_datafix_count > 0:
                                region_counts[str(region)] = int(region_datafix_count)
                        
                        if region_counts:
                            datafix_by_region[col] = dict(sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                    if datafix_by_region:
                        stats["datafix_by_region"] = datafix_by_region
        
        # Escalation analysis
        if 'work notes' in df.columns:
            escalation_keywords = ['escalate', 'escalation', 'elevated', 'raised to']
            escalation_count = 0
            for keyword in escalation_keywords:
                escalation_count += df['work notes'].str.lower().str.contains(keyword, na=False).sum()
            if escalation_count > 0:
                stats["escalation_mentions"] = int(escalation_count)
                
                # If we have geographic data, check for region-specific escalations
                if 'geographic_columns_found' in stats:
                    escalation_by_region = {}
                    for col in stats["geographic_columns_found"]:
                        region_counts = {}
                        for region in df[col].dropna().unique():
                            region_df = df[df[col] == region]
                            region_escalation_count = 0
                            for keyword in escalation_keywords:
                                if 'work notes' in region_df.columns:
                                    region_escalation_count += region_df['work notes'].str.lower().str.contains(keyword, na=False).sum()
                            if region_escalation_count > 0:
                                region_counts[str(region)] = int(region_escalation_count)
                        
                        if region_counts:
                            escalation_by_region[col] = dict(sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                    if escalation_by_region:
                        stats["escalation_by_region"] = escalation_by_region
        
        # Document failure analysis
        if 'short description' in df.columns:
            doc_keywords = ['document failure', 'report failure', 'failed document', 'document error']
            doc_count = 0
            for keyword in doc_keywords:
                doc_count += df['short description'].str.lower().str.contains(keyword, na=False).sum()
            if doc_count > 0:
                stats["document_failure_count"] = int(doc_count)
                
                # Check for document failures by year if we have date data
                if 'opened' in df.columns and pd.api.types.is_datetime64_dtype(df['opened']):
                    doc_failures_by_year = {}
                    for keyword in doc_keywords:
                        doc_failures = df[df['short description'].str.lower().str.contains(keyword, na=False)]
                        if not doc_failures.empty:
                            years = doc_failures['opened'].dt.year.value_counts().to_dict()
                            for year, count in years.items():
                                doc_failures_by_year[int(year)] = doc_failures_by_year.get(int(year), 0) + int(count)
                    
                    if doc_failures_by_year:
                        stats["document_failures_by_year"] = doc_failures_by_year
    
    def extract_query_stats(self, df, query, base_stats):
        """
        Extract additional statistics relevant to a specific user query.
        
        Args:
            df: Processed DataFrame
            query: User query string
            base_stats: Base statistics already computed
            
        Returns:
            dict: Query-specific statistics
        """
        # Implementation similar to the existing extract_query_stats function...
        # This would be a detailed implementation based on the query keywords
        return base_stats  # For brevity, returning base stats for now