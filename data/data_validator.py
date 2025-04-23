"""
Data validation module for the Incident Management Analytics application.
This module validates uploaded incident data to ensure it meets the requirements
for analysis.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class for validating incident data to ensure it meets analysis requirements.
    """
    
    def __init__(self):
        """
        Initialize the DataValidator.
        """
        pass
    
    def validate(self, df: pd.DataFrame, mandatory_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate incident data to ensure it meets the requirements for analysis.
        
        Args:
            df: DataFrame with incident data
            mandatory_columns: List of column names that must be present
            
        Returns:
            Dictionary with validation results including:
                - is_valid: Boolean indicating if the data is valid
                - errors: List of error messages if any
                - warnings: List of warning messages if any
                - column_status: Dictionary with status of each mandatory column
        """
        # Use default mandatory columns if none are provided
        if mandatory_columns is None:
            # These are placeholders - actual values should come from constants or config
            mandatory_columns = ["incident_id", "timestamp", "priority", "status"]
        
        return validate_incident_data(df, mandatory_columns)
        
    def check_data_sufficiency(self, df: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """
        Check if the data is sufficient for a specific type of analysis.
        
        Args:
            df: DataFrame with incident data
            analysis_type: Type of analysis to check for
            
        Returns:
            Dictionary with sufficiency results and recommendations
        """
        return check_data_sufficiency(df, analysis_type)


def validate_incident_data(df: pd.DataFrame, mandatory_columns: List[str]) -> Dict[str, Any]:
    """
    Validate incident data to ensure it meets the requirements for analysis.
    
    Args:
        df: DataFrame with incident data
        mandatory_columns: List of column names that must be present
        
    Returns:
        Dictionary with validation results including:
            - is_valid: Boolean indicating if the data is valid
            - errors: List of error messages if any
            - warnings: List of warning messages if any
            - column_status: Dictionary with status of each mandatory column
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "column_status": {},
        "data_quality": {},
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("The uploaded file contains no data")
        return validation_results
    
    # Check if DataFrame has at least one column
    if len(df.columns) < 1:
        validation_results["is_valid"] = False
        validation_results["errors"].append("The uploaded file contains no columns")
        return validation_results
    
    # Check for minimal column requirements
    available_columns = set(df.columns)
    missing_columns = []
    
    # Track column presence
    for column in mandatory_columns:
        if column in available_columns:
            validation_results["column_status"][column] = "present"
        else:
            validation_results["column_status"][column] = "missing"
            missing_columns.append(column)
    
    # Try to identify if we have enough columns for basic analysis
    if "incident_id" not in available_columns:
        # Check if we can identify an ID column
        potential_id_columns = [col for col in available_columns if any(id_term in col.lower() for id_term in ["id", "number", "ticket", "case", "incident"])]
        if potential_id_columns:
            validation_results["warnings"].append(f"No 'incident_id' column found, but {potential_id_columns[0]} might be used instead")
        else:
            validation_results["errors"].append("No column that could be used as incident identifier was found")
            validation_results["is_valid"] = False
    
    # Check if we have any date column
    has_date_column = any(date_term in col.lower() for col in available_columns for date_term in ["date", "time", "created", "opened", "resolved", "closed"])
    if not has_date_column:
        validation_results["errors"].append("No date-related columns found. At least one date column is required for time-based analysis")
        validation_results["is_valid"] = False
    
    # Allow data to be valid if it has at least an ID and a date column, even if other mandatory columns are missing
    if validation_results["is_valid"] and missing_columns:
        missing_columns_str = ", ".join(missing_columns)
        validation_results["warnings"].append(f"The following recommended columns are missing: {missing_columns_str}")
    
    # Data quality checks
    if validation_results["is_valid"]:
        validation_results["data_quality"] = _check_data_quality(df)
        
        # Add data quality warnings
        for metric, value in validation_results["data_quality"].items():
            if metric == "null_percentage" and value > 25:
                validation_results["warnings"].append(f"High percentage of null values ({value:.1f}%) may affect analysis quality")
            elif metric == "duplicate_percentage" and value > 5:
                validation_results["warnings"].append(f"High percentage of duplicate incidents ({value:.1f}%) detected")
    
    return validation_results

def _check_data_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Check the quality of the incident data.
    
    Args:
        df: DataFrame with incident data
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_metrics = {}
    
    # Calculate percentage of null values
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    quality_metrics["null_percentage"] = (null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Check for duplicates if incident_id exists
    if "incident_id" in df.columns:
        duplicates = df.duplicated(subset=["incident_id"]).sum()
        quality_metrics["duplicate_percentage"] = (duplicates / len(df)) * 100 if len(df) > 0 else 0
    
    # Check date range if created_date exists
    if "created_date" in df.columns:
        try:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df["created_date"]):
                dates = pd.to_datetime(df["created_date"], errors="coerce")
            else:
                dates = df["created_date"]
                
            # Calculate date range only if we have valid dates
            if not dates.isnull().all():
                min_date = dates.min()
                max_date = dates.max()
                date_range_days = (max_date - min_date).days
                quality_metrics["date_range_days"] = date_range_days
                
                # Flag if date range is too short or too long
                if date_range_days < 7:
                    quality_metrics["short_date_range"] = True
                elif date_range_days > 730:  # More than 2 years
                    quality_metrics["long_date_range"] = True
                    
                # Check for even distribution
                if date_range_days > 0:
                    expected_per_day = len(df) / date_range_days
                    # Calculate actual distribution
                    date_counts = dates.dt.date.value_counts()
                    std_dev = date_counts.std()
                    mean = date_counts.mean()
                    if mean > 0:
                        quality_metrics["date_distribution_cv"] = std_dev / mean  # Coefficient of variation
        except Exception as e:
            logger.warning(f"Error analyzing date range: {str(e)}")
    
    # Check for priority distribution if priority exists
    if "priority" in df.columns:
        priority_counts = df["priority"].value_counts(normalize=True)
        # Check if single priority dominates (>80%)
        if not priority_counts.empty and priority_counts.max() > 0.8:
            quality_metrics["priority_imbalance"] = True
    
    # Check for categorical columns with too many unique values
    for col in df.select_dtypes(include=["object"]).columns:
        unique_count = df[col].nunique()
        # Convert unique_count to a scalar value and then compare
        if isinstance(unique_count, pd.Series):
            unique_count = unique_count.iloc[0] if len(unique_count) > 0 else 0
        if unique_count > 100:  # Arbitrary threshold for too many categories
            quality_metrics[f"{col}_high_cardinality"] = True
    
    return quality_metrics

def check_data_sufficiency(df: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
    """
    Check if the data is sufficient for a specific type of analysis.
    
    Args:
        df: DataFrame with incident data
        analysis_type: Type of analysis to check for
        
    Returns:
        Dictionary with sufficiency results and recommendations
    """
    result = {
        "is_sufficient": False,
        "reason": "",
        "required_columns": [],
        "missing_columns": [],
        "recommendations": []
    }
    
    # Define required columns for different analysis types
    analysis_requirements = {
        "time_analysis": {
            "required_columns": ["created_date"],
            "min_rows": 10,
            "recommendations": [
                "Need at least one date column (created_date or equivalent)",
                "Recommend minimum 10 incidents"
            ]
        },
        "resolution_time_analysis": {
            "required_columns": ["created_date", "resolved_date"],
            "min_rows": 10,
            "recommendations": [
                "Need both created_date and resolved_date",
                "Dates should be in a proper datetime format",
                "Recommend minimum 10 resolved incidents"
            ]
        },
        "priority_analysis": {
            "required_columns": ["priority"],
            "min_rows": 10,
            "recommendations": [
                "Need priority or severity column",
                "Need at least two different priority levels",
                "Recommend minimum 10 incidents"
            ]
        },
        "resource_analysis": {
            "required_columns": ["assignee", "created_date"],
            "alternative_columns": ["assignment_group"],
            "min_rows": 20,
            "recommendations": [
                "Need assignee or assignment_group column",
                "Need date information for time-based analysis",
                "Recommend minimum 20 incidents",
                "Should have multiple assignees/groups for comparison"
            ]
        },
        "category_analysis": {
            "required_columns": ["category"],
            "alternative_columns": ["subcategory", "type"],
            "min_rows": 15,
            "recommendations": [
                "Need category or type column",
                "Need at least 2 different categories",
                "Recommend minimum 15 incidents"
            ]
        },
        "system_analysis": {
            "required_columns": ["affected_system"],
            "alternative_columns": ["application", "service", "component"],
            "min_rows": 15,
            "recommendations": [
                "Need affected_system or equivalent column",
                "Need multiple systems for comparison",
                "Recommend minimum 15 incidents"
            ]
        },
        "automation_opportunity": {
            "required_columns": ["description"],
            "alternative_columns": ["resolution_notes", "category", "subcategory"],
            "min_rows": 30,
            "recommendations": [
                "Need description field or detailed categorization",
                "Need sufficient repetition of incident types",
                "Recommend minimum 30 incidents",
                "Better results with resolution information"
            ]
        },
        "text_analysis": {
            "required_columns": ["description"],
            "alternative_columns": ["resolution_notes", "comments"],
            "min_rows": 20,
            "recommendations": [
                "Need text fields like description or comments",
                "Text should be detailed enough for analysis",
                "Recommend minimum 20 incidents with text content"
            ]
        }
    }
    
    # Check if the requested analysis type is supported
    if analysis_type not in analysis_requirements:
        result["reason"] = f"Unknown analysis type: {analysis_type}"
        return result
    
    requirements = analysis_requirements[analysis_type]
    result["required_columns"] = requirements["required_columns"].copy()
    
    # Check for required columns
    available_columns = set(df.columns)
    missing_primary_columns = [col for col in requirements["required_columns"] if col not in available_columns]
    
    # If primary columns are missing, check for alternatives
    if missing_primary_columns and "alternative_columns" in requirements:
        for missing_col in missing_primary_columns.copy():
            idx = requirements["required_columns"].index(missing_col)
            for alt_col in requirements.get("alternative_columns", []):
                if alt_col in available_columns:
                    missing_primary_columns.remove(missing_col)
                    result["required_columns"][idx] = alt_col
                    break
    
    result["missing_columns"] = missing_primary_columns
    
    # Check for minimum rows
    if len(df) < requirements["min_rows"]:
        result["reason"] = f"Insufficient data: {len(df)} incidents available, {requirements['min_rows']} required for {analysis_type}"
        result["recommendations"] = requirements["recommendations"]
        return result
    
    # If we still have missing columns
    if missing_primary_columns:
        missing_cols_str = ", ".join(missing_primary_columns)
        result["reason"] = f"Missing required columns: {missing_cols_str}"
        result["recommendations"] = requirements["recommendations"]
        return result
    
    # Additional specific checks for different analysis types
    if analysis_type == "priority_analysis":
        unique_priorities = df["priority"].nunique()
        if unique_priorities < 2:
            result["reason"] = f"Insufficient variety: only {unique_priorities} priority level(s) found"
            return result
    
    elif analysis_type == "resource_analysis":
        resource_col = "assignee" if "assignee" in df.columns else "assignment_group"
        unique_resources = df[resource_col].nunique()
        if unique_resources < 2:
            result["reason"] = f"Insufficient variety: only {unique_resources} {resource_col}(s) found"
            return result
    
    elif analysis_type == "text_analysis":
        text_col = next((col for col in ["description", "resolution_notes", "comments"] if col in df.columns), None)
        if text_col:
            # Check if text is too short on average
            avg_len = df[text_col].astype(str).apply(len).mean()
            if avg_len < 15:  # Arbitrary threshold for too short text
                result["reason"] = f"Text in {text_col} is too short for meaningful analysis (avg {avg_len:.1f} chars)"
                return result
    
    # If we've passed all checks, the data is sufficient
    result["is_sufficient"] = True
    return result