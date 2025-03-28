
# Utility functions and classes for the application

import pandas as pd
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy and Pandas data types.
    Use this for all JSON serialization to avoid errors.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Period):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def safe_json_dumps(obj, indent=2):
    """
    Safely convert an object to JSON string, handling NumPy and Pandas types.
    
    Args:
        obj: The object to convert
        indent: Indentation level for JSON formatting
        
    Returns:
        str: JSON string representation
    """
    return json.dumps(obj, indent=indent, cls=NumpyEncoder)

def validate_dict_fields(obj, required_fields, default_values=None):
    """
    Ensure all required fields exist in a dictionary.
    
    Args:
        obj (dict): Dictionary to validate
        required_fields (list): List of required field names
        default_values (dict, optional): Dictionary of default values for missing fields
        
    Returns:
        dict: Validated dictionary with all required fields
    """
    if not isinstance(obj, dict):
        # If not a dictionary, create an empty one
        obj = {}
    
    if default_values is None:
        default_values = {}
        
    # Ensure all required fields exist
    for field in required_fields:
        if field not in obj:
            obj[field] = default_values.get(field, f"{field} not available")
    
    return obj