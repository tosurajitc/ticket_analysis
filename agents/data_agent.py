import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import io
import os

class DataProcessingAgent:
    """
    Agent responsible for loading, chunking, and preprocessing ticket data.
    """
    
    def __init__(self, llm):
        self.llm = llm
        
    def load_data(self, uploaded_file, column_hints: List[str] = None) -> pd.DataFrame:
        """
        Load data from uploaded file and perform initial processing
        """
        try:
            # Get file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Read the file based on its type
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Clean the data
            df = self._clean_data(df, column_hints)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _clean_data(self, df: pd.DataFrame, column_hints: List[str] = None) -> pd.DataFrame:
        """
        Clean and preprocess the data
        """
        # Ensure column names are clean (lowercase, no spaces)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        df = df.fillna({col: 'unknown' for col in df.select_dtypes(include=['object', 'string']).columns})
        
        # If numeric columns exist, fill NaNs with mean or 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                if df[col].count() > 0:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(0)
        
        # Use column hints to try to standardize column names if provided
        if column_hints and len(column_hints) > 0:
            df = self._standardize_columns(df, column_hints)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame, column_hints: List[str]) -> pd.DataFrame:
        """
        Attempt to map columns to standard names based on hints
        """
        # Clean column hints
        clean_hints = [hint.lower().replace(' ', '_').strip() for hint in column_hints if hint.strip()]
        
        # Use LLM to try to map columns to standard names
        if clean_hints and len(clean_hints) > 0:
            column_mapping = {}
            df_columns = list(df.columns)
            
            # Create a mapping based on similarity
            for hint in clean_hints:
                # Check if hint already exists in dataframe
                if hint in df_columns:
                    column_mapping[hint] = hint
                    continue
                
                # Otherwise, ask LLM to find the most similar column
                messages = [
                    {"role": "system", "content": "Map columns to standard names based on semantic similarity."},
                    {"role": "user", "content": f"""
                    I have a dataframe with these columns: {df_columns}
                    
                    Which one of these columns is most likely to represent the standard column '{hint}'?
                    Respond with just the column name from the list above, or 'none' if there's no good match.
                    """}
                ]
                
                try:
                    response = self.llm.invoke(messages)
                    matched_column = response.content.strip().lower()
                    
                    # Only use the mapping if the matched column exists in the dataframe
                    if matched_column in df_columns and matched_column != 'none':
                        column_mapping[matched_column] = hint
                except Exception as e:
                    print(f"Error during column mapping: {str(e)}")
            
            # Rename columns based on mapping
            df = df.rename(columns=column_mapping)
        
        return df
    
    def chunk_data(self, df: pd.DataFrame, chunk_size: int = 500) -> List[pd.DataFrame]:
        """
        Split dataframe into chunks for processing
        """
        if len(df) <= chunk_size:
            return [df]
        
        # Calculate number of chunks
        n_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Split into chunks
        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)
        
        return chunks
    
    def extract_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract schema information from the dataframe
        """
        schema_info = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_values": {col: df[col].sample(min(5, len(df))).tolist() for col in df.columns},
            "unique_counts": {col: df[col].nunique() for col in df.columns},
            "missing_values": {col: df[col].isna().sum() for col in df.columns},
        }
        
        return schema_info
    
    def suggest_important_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Use LLM to suggest which columns are likely important for ticket analysis
        """
        # Instead of using LLM for large datasets, use heuristics to identify important columns
        try:
            # Check for columns with common ticket-related terms
            important_columns = []
            
            # Define categories of important columns with keywords
            column_categories = {
                "identifier": ["id", "ticket", "number", "key", "reference"],
                "category": ["category", "type", "group", "class"],
                "priority": ["priority", "severity", "urgency", "importance"],
                "status": ["status", "state", "resolution"],
                "dates": ["date", "created", "opened", "resolved", "closed", "updated", "due"],
                "owner": ["owner", "assigned", "tech", "agent", "responsible"],
                "customer": ["customer", "client", "user", "submitter", "reporter"],
                "content": ["description", "summary", "detail", "comment", "note"]
            }
            
            # Go through all columns and check against keywords
            for col in df.columns:
                col_lower = col.lower()
                for category, keywords in column_categories.items():
                    if any(keyword in col_lower for keyword in keywords):
                        important_columns.append(col)
                        break
            
            # If no columns were found, return some default column names
            if not important_columns:
                important_columns = ['id', 'category', 'priority', 'status', 'created_date', 'resolved_date', 'description']
                
            return important_columns
        except Exception as e:
            print(f"Error suggesting important columns: {str(e)}")
            # Return some common ticket columns as fallback
            return ['id', 'category', 'priority', 'status', 'created_date', 'resolved_date', 'description']