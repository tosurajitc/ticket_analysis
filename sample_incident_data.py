"""
Generate sample incident data in CSV and Excel formats for testing data_loader.py
"""

import pandas as pd
import numpy as np
import datetime
import random
import os
from pathlib import Path

# Define a function to generate sample incident data
def generate_sample_incident_data(n_samples=200):
    """Generate a sample dataset of incident tickets."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create timestamps with a realistic distribution (last 180 days)
    base_date = datetime.datetime.now() - datetime.timedelta(days=180)
    dates = []
    for i in range(n_samples):
        # Add a random number of days (0-180 days)
        random_days = np.random.randint(0, 180)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        dates.append(base_date + datetime.timedelta(days=random_days, 
                                                   hours=random_hours,
                                                   minutes=random_minutes))
    
    # Sort dates in ascending order
    dates.sort()
    
    # Calculate resolved dates (some still open)
    resolved_dates = []
    for i, created_date in enumerate(dates):
        if random.random() < 0.8:  # 80% of incidents are resolved
            # Resolution takes between 1 hour and 7 days
            resolution_hours = np.random.exponential(24)  # Average ~24 hours
            resolution_hours = min(resolution_hours, 168)  # Cap at 7 days
            resolved_date = created_date + datetime.timedelta(hours=resolution_hours)
            resolved_dates.append(resolved_date)
        else:
            # Still open
            resolved_dates.append(None)
    
    # Define sample data for categories
    categories = ['Network', 'Server', 'Application', 'Database', 'Security', 'Hardware', 'User Access']
    subcategories = {
        'Network': ['Connectivity', 'Firewall', 'VPN', 'Router', 'Switch'],
        'Server': ['Windows', 'Linux', 'VMware', 'Hardware', 'Performance'],
        'Application': ['Error', 'Performance', 'Configuration', 'Integration'],
        'Database': ['Query', 'Performance', 'Backup', 'Corruption'],
        'Security': ['Access', 'Malware', 'Policy', 'Audit'],
        'Hardware': ['Desktop', 'Laptop', 'Printer', 'Mobile', 'Peripheral'],
        'User Access': ['New Account', 'Password Reset', 'Permission', 'Termination']
    }
    
    priorities = ['P1', 'P2', 'P3', 'P4']
    priority_weights = [0.1, 0.2, 0.4, 0.3]  # P3 most common, P1 least common
    
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed', 'Pending']
    
    assignees = ['John Smith', 'Jane Doe', 'Robert Johnson', 'Lisa Wong', 
                'David Miller', 'Maria Garcia', 'Unassigned']
    
    assignment_groups = ['Network Team', 'Server Team', 'Application Support', 
                       'Database Team', 'Security Team', 'Help Desk']
    
    sources = ['Email', 'Phone', 'Web Portal', 'Monitoring System', 'Chat']
    
    systems = ['CRM System', 'ERP System', 'Email Server', 'Web Application', 
             'Database Server', 'Network Infrastructure', 'VPN Service',
             'Active Directory', 'File Server']
    
    # Generate random data
    incident_data = []
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        
        # Generate descriptions based on category and subcategory
        description_templates = [
            f"{category} issue: {subcategory} problem reported by user",
            f"User experiencing {subcategory} issues with {category}",
            f"{subcategory} failure in {category} system",
            f"{category} {subcategory} not functioning as expected"
        ]
        description = np.random.choice(description_templates)
        
        # Resolution notes for resolved incidents
        if resolved_dates[i] is not None:
            resolution_templates = [
                f"Fixed {subcategory} configuration in {category}",
                f"Restarted {category} service to resolve {subcategory} issue",
                f"Applied patch to fix {subcategory} vulnerability",
                f"User error resolved with additional training",
                f"Known issue with workaround applied"
            ]
            resolution_notes = np.random.choice(resolution_templates)
        else:
            resolution_notes = None
        
        # Determine status based on resolution
        if resolved_dates[i] is None:
            status = np.random.choice(['Open', 'In Progress', 'Pending'])
        else:
            status = np.random.choice(['Resolved', 'Closed'], p=[0.3, 0.7])
        
        # Priority influenced by category (Security more likely to be high priority)
        if category == 'Security':
            priority = np.random.choice(priorities, p=[0.3, 0.3, 0.3, 0.1])
        else:
            priority = np.random.choice(priorities, p=priority_weights)
        
        # Assignment based on category
        if category == 'Network':
            assignment_group = 'Network Team'
        elif category == 'Server':
            assignment_group = 'Server Team'
        elif category == 'Application':
            assignment_group = 'Application Support'
        elif category == 'Database':
            assignment_group = 'Database Team'
        elif category == 'Security':
            assignment_group = 'Security Team'
        else:
            assignment_group = 'Help Desk'
        
        # 10% chance of being unassigned
        if random.random() < 0.1:
            assignee = 'Unassigned'
        else:
            # Remove "Unassigned" from the list for this selection
            valid_assignees = [a for a in assignees if a != 'Unassigned']
            assignee = np.random.choice(valid_assignees)
        
        # System affected based on category
        if category == 'Network':
            system = np.random.choice(['Network Infrastructure', 'VPN Service'])
        elif category == 'Server':
            system = np.random.choice(['Email Server', 'File Server'])
        elif category == 'Application':
            system = np.random.choice(['CRM System', 'ERP System', 'Web Application'])
        elif category == 'Database':
            system = 'Database Server'
        elif category == 'Security':
            system = np.random.choice(['VPN Service', 'Active Directory'])
        else:
            system = np.random.choice(systems)
        
        # Create incident record
        incident = {
            'Incident ID': f'INC{i+1:06d}',
            'Created Date': dates[i],
            'Resolved Date': resolved_dates[i],
            'Priority': priority,
            'Status': status,
            'Category': category,
            'Subcategory': subcategory,
            'Assignee': assignee,
            'Assignment Group': assignment_group,
            'Source': np.random.choice(sources),
            'Affected System': system,
            'Description': description,
            'Resolution Notes': resolution_notes
        }
        
        incident_data.append(incident)
    
    # Convert to DataFrame
    df = pd.DataFrame(incident_data)
    
    return df

def save_sample_data(output_dir="sample_data"):
    """Generate and save sample data in multiple formats."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    df = generate_sample_incident_data()
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "incident_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV file to: {csv_path}")
    
    # Save as Excel
    excel_path = os.path.join(output_dir, "incident_data.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved Excel file to: {excel_path}")
    
    # Save a file with non-standard column names to test column mapping
    df_non_standard = df.copy()
    df_non_standard.columns = [
        'Ticket Number',
        'Date Opened',
        'Date Closed',
        'Severity',
        'Current State',
        'Issue Type',
        'Issue Subtype',
        'Handled By',
        'Support Team',
        'Reported Via',
        'System Impacted',
        'Issue Details',
        'Fix Applied'
    ]
    nonstandard_path = os.path.join(output_dir, "incident_data_nonstandard.xlsx")
    df_non_standard.to_excel(nonstandard_path, index=False)
    print(f"Saved non-standard column file to: {nonstandard_path}")
    
    return {
        'csv_path': csv_path,
        'excel_path': excel_path,
        'nonstandard_path': nonstandard_path
    }

# Run the data generation
if __name__ == "__main__":
    file_paths = save_sample_data()
    print("\nSample data statistics:")
    df = pd.read_csv(file_paths['csv_path'])
    print(f"Total incidents: {len(df)}")
    print(f"Date range: {df['Created Date'].min()} to {df['Created Date'].max()}")
    print(f"Priority distribution: {df['Priority'].value_counts().to_dict()}")
    print(f"Category distribution: {df['Category'].value_counts().to_dict()}")