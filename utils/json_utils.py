import json
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from decimal import Decimal
import uuid
from typing import Any, Dict, List, Union

class JSONSerializer:
    """
    Custom JSON serializer to handle common non-serializable types in pandas and numpy.
    """
    
    @staticmethod
    def serialize(obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable types.
        """
        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle datetime types
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)
        
        # Handle pandas Series
        if isinstance(obj, pd.Series):
            return obj.to_list()
        
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            # Convert any timestamp columns to strings first
            df_copy = obj.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype(str)
            return df_copy.to_dict(orient='records')
            
        # Handle pandas NaT (Not a Time)
        if pd.isna(obj):
            return None
        
        # Handle other custom types as needed
        return str(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely convert any object to a JSON string, handling common non-serializable types.
    
    Args:
        obj: The object to convert to JSON
        **kwargs: Additional arguments to pass to json.dumps
    
    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, default=JSONSerializer.serialize, **kwargs)

def safe_json_loads(json_str: str, **kwargs) -> Any:
    """
    Safely load a JSON string.
    
    Args:
        json_str: The JSON string to load
        **kwargs: Additional arguments to pass to json.loads
    
    Returns:
        Python object represented by the JSON string
    """
    return json.loads(json_str, **kwargs)

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert a nested structure to be JSON serializable.
    
    Args:
        obj: The object to make JSON serializable
    
    Returns:
        A JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return set(make_json_serializable(item) for item in obj)
    elif isinstance(obj, pd.DataFrame):
        # Handle DataFrame specifically
        df_copy = obj.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
        return df_copy.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(obj):
            return obj.astype(str).to_list()
        return obj.to_list()
    else:
        try:
            # Try standard serialization
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # If it fails, use our custom serializer
            return JSONSerializer.serialize(obj)

# Monkey patch json module to always use our serializer
_original_dumps = json.dumps

def patched_dumps(obj, *args, **kwargs):
    """Patched version of json.dumps that handles non-serializable types"""
    if 'default' not in kwargs:
        kwargs['default'] = JSONSerializer.serialize
    return _original_dumps(obj, *args, **kwargs)

# Replace the standard json.dumps with our patched version
json.dumps = patched_dumps