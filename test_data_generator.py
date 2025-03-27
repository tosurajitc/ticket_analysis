# test_data_generator.py
# Script to generate test ticket data for the analysis system

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_data(num_tickets=200, output_file="test_ticket_data.csv"):
    """
    Generate test ticket data for the analysis system.
    
    Args:
        num_tickets (int): Number of tickets to generate
        output_file (str): Name of the output CSV file
    """
    # Define seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define possible values for categorical fields
    priorities = ['Critical', 'High', 'Medium', 'Low']
    priority_weights = [0.1, 0.25, 0.45, 0.2]  # Probabilities for each priority
    
    states = ['New', 'Open', 'In Progress', 'Resolved', 'Closed', 'Cancelled']
    state_weights = [0.05, 0.15, 0.2, 0.1, 0.45, 0.05]
    
    assignment_groups = [
        'Network Support', 'Desktop Support', 'Server Team', 'Security Team',
        'Database Admin', 'Application Support', 'Infrastructure', 'Cloud Team',
        'Service Desk', 'Email Team'
    ]
    
    subcategories = [
        'Hardware', 'Software', 'Network', 'Access', 'Email', 'Database',
        'Application', 'Security', 'Account', 'Password Reset', 'Performance',
        'Connectivity', 'Printer', 'Mobile Device'
    ]
    
    users = [
        'John Smith', 'Mary Johnson', 'David Williams', 'Lisa Brown',
        'Robert Jones', 'Susan Miller', 'Michael Davis', 'Jennifer Wilson',
        'William Moore', 'Jessica Taylor', 'James Anderson', 'Patricia Thomas',
        'Daniel Jackson', 'Barbara White', 'Joseph Harris'
    ]
    
    # Common issues for ticket descriptions
    common_issues = [
        'Cannot access {system}',
        'Need access to {system}',
        '{system} is running slowly',
        'Error when trying to {action}',
        'Password reset required for {system}',
        '{system} is not responding',
        'Unable to login to {system}',
        'Need assistance with {system}',
        '{system} needs updating',
        'Issue with {system} after recent update',
        '{system} showing error message: {error_code}',
        'New user setup for {system}',
        '{system} account locked out',
        'Configuration change needed for {system}',
        'Printer {action} issue'
    ]
    
    systems = [
        'email', 'CRM', 'database', 'VPN', 'file share', 'SharePoint',
        'ERP system', 'website', 'internal portal', 'customer portal',
        'accounting software', 'HR system', 'mobile app', 'desktop computer',
        'laptop', 'printer', 'network drive', 'video conferencing'
    ]
    
    actions = [
        'login', 'access', 'update', 'connect', 'print', 'save', 'upload',
        'download', 'configure', 'install', 'setup', 'reset', 'restart'
    ]
    
    error_codes = [
        'ERR001', 'ERR404', 'SEC7892', 'SYS201', 'DB5310', 'AUTH502', 
        'NET6621', 'APP9981', '0x80004005', '0x0000007B', 'BSOD0x124',
        'NW189', '500', '403', 'NULL REF'
    ]
    
    # Generate random dates within the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate ticket data
    ticket_data = []
    
    for i in range(1, num_tickets + 1):
        # Generate opened date
        opened_date = start_date + (end_date - start_date) * random.random()
        
        # Determine priority with weighted selection
        priority = random.choices(priorities, weights=priority_weights)[0]
        
        # Determine state with weighted selection
        state = random.choices(states, weights=state_weights)[0]
        
        # Generate other ticket fields
        assignment_group = random.choice(assignment_groups)
        subcategory = random.choice(subcategories)
        opened_by = random.choice(users)
        assigned_to = random.choice(users)
        
        # Generate short description
        issue = random.choice(common_issues)
        system = random.choice(systems)
        action = random.choice(actions)
        error_code = random.choice(error_codes)
        
        description = issue.format(system=system, action=action, error_code=error_code)
        
        # Generate closed date based on priority and state
        if state in ['Resolved', 'Closed', 'Cancelled']:
            # Higher priority tickets get resolved faster on average
            if priority == 'Critical':
                max_days = 2
            elif priority == 'High':
                max_days = 5
            elif priority == 'Medium':
                max_days = 10
            else:  # Low
                max_days = 15
                
            # Add some variability
            resolution_days = max(0.1, random.expovariate(1.0 / (max_days / 2)))
            closed_date = opened_date + timedelta(days=resolution_days)
            
            # Ensure closed date isn't in the future
            if closed_date > end_date:
                closed_date = end_date
        else:
            closed_date = None
        
        # Create ticket record
        ticket = {
            'Number': f'INC{100000 + i}',
            'Priority': priority,
            'Opened': opened_date,
            'Assignment Group': assignment_group,
            'Subcategory': subcategory,
            'Subcategory 2': random.choice(subcategories) if random.random() < 0.7 else None,
            'Subcategory 3': random.choice(subcategories) if random.random() < 0.3 else None,
            'Short description': description,
            'State': state,
            'Opened by': opened_by,
            'Assigned to': assigned_to if random.random() < 0.9 else None,  # Some tickets unassigned
            'Closed': closed_date,
            'Work notes': f'Troubleshooting steps for {system}' if random.random() < 0.8 else None,
            'External Ref#': f'EXT-{random.randint(1000, 9999)}' if random.random() < 0.3 else None
        }
        
        ticket_data.append(ticket)
    
    # Create DataFrame
    df = pd.DataFrame(ticket_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Test data generated and saved to {output_file}")
    return df

if __name__ == "__main__":
    generate_test_data()