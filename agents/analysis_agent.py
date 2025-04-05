import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import re
import warnings
from utils.json_utils import make_json_serializable, safe_json_dumps, safe_json_loads

class AnalysisAgent:
    """
    Agent responsible for analyzing ticket data and generating insights.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.column_hints = []
    

    def set_column_hints(self, column_hints: List[str]):
        self.column_hints = column_hints


    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of the ticket data
        """
        try:
            # Basic statistics
            stats = self._calculate_basic_stats(df)

            # Prioritize column hints in analysis if available
            prioritized_columns = self.column_hints if self.column_hints else list(df.columns)
            
            # Time-based analysis
            time_analysis = self._analyze_time_dimensions(df, prioritized_columns)
            
            # Category/type analysis
            category_analysis = self._analyze_categories(df, prioritized_columns)
            
            # Priority analysis
            priority_analysis = self._analyze_priority(df, prioritized_columns)
            
            # Status analysis
            status_analysis = self._analyze_status(df, prioritized_columns)
            
            # Text analysis of descriptions (if available)
            text_analysis = self._analyze_text_fields(df, prioritized_columns)
            
            # Combine all analyses
            analysis_results = {
                "basic_stats": stats,
                "time_analysis": time_analysis,
                "category_analysis": category_analysis,
                "priority_analysis": priority_analysis,
                "status_analysis": status_analysis,
                "text_analysis": text_analysis
            }
            
            # Ensure all results are JSON serializable
            return make_json_serializable(analysis_results)
        except Exception as e:
            print(f"Error in analyze_data: {str(e)}")
            # Return empty analysis
            return {
                "basic_stats": {},
                "time_analysis": {},
                "category_analysis": {},
                "priority_analysis": {},
                "status_analysis": {},
                "text_analysis": {}
            }
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics about the dataset"""
        stats = {
            "total_tickets": len(df),
            "column_counts": {col: df[col].count() for col in df.columns},
            "unique_values": {col: df[col].nunique() for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'category'},
        }
        
        # Add numeric column statistics where applicable
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def _analyze_time_dimensions(self, df: pd.DataFrame, prioritized_columns: List[str]) -> Dict[str, Any]:
        """Analyze time-related dimensions in the data"""
        # Try to identify date columns
        date_cols = []
        # First check if any prioritized columns are date columns
        for col in prioritized_columns:
            if col in df.columns and any(date_term in col.lower() for date_term in ['date', 'time', 'created', 'resolved', 'closed', 'updated']):
                try:
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    date_cols.append(col)
                except:
                    pass
        
        # Then check remaining columns if needed
        if not date_cols:
            for col in df.columns:
                if col not in prioritized_columns and any(date_term in col.lower() for date_term in ['date', 'time', 'created', 'resolved', 'closed', 'updated']):
                    try:
                        pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                        date_cols.append(col)
                    except:
                        pass

        for col in df.columns:
            # Check if column name suggests it's a date
            if any(date_term in col.lower() for date_term in ['date', 'time', 'created', 'resolved', 'closed', 'updated']):
                try:
                    # Just check if column can be converted to datetime without storing result
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    date_cols.append(col)
                except:
                    pass
        
        time_analysis = {"date_columns": date_cols}
        
        # If we found date columns, analyze them
        if date_cols:
            # Make a copy of the dataframe to avoid modifying the original
            temp_df = df.copy()
            
            # For each date column, analyze distribution
            for date_col in date_cols:
                try:
                    # Skip columns that have dictionary values
                    if isinstance(temp_df[date_col].iloc[0] if len(temp_df) > 0 else '', dict):
                        continue
                        
                    # With this:
                    # First, determine if we can infer a common date format from the first few non-null values
                    def get_common_date_format(series):
                        # Get first few non-null values
                        sample_values = series.dropna().astype(str).head(5).tolist()
                        
                        # Common format patterns to check
                        formats = [
                            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
                            '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', 
                            '%m/%d/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S'
                        ]
                        
                        # Check each format
                        for fmt in formats:
                            try:
                                # If all samples can be parsed with this format, use it
                                all_valid = all(pd.to_datetime(val, format=fmt, errors='raise') for val in sample_values)
                                if all_valid:
                                    return fmt
                            except:
                                continue
                        
                        return None

                    date_format = get_common_date_format(temp_df[date_col])
                    if date_format:
                        temp_df[f'{date_col}_dt'] = pd.to_datetime(temp_df[date_col], format=date_format, errors='coerce')
                    else:
                        temp_df[f'{date_col}_dt'] = pd.to_datetime(temp_df[date_col], errors='coerce')
                    
                    # Get valid date values only
                    valid_dates = temp_df.dropna(subset=[f'{date_col}_dt'])
                    
                    # Skip if no valid dates
                    if len(valid_dates) == 0:
                        continue
                    
                    # Extract time components as Python built-in types to avoid serialization issues
                    def safe_int(x):
                        try:
                            return int(x)
                        except:
                            return 0
                            
                    # Create dictionaries with string keys and integer values
                    years = valid_dates[f'{date_col}_dt'].dt.year.value_counts().to_dict()
                    year_dict = {str(k): safe_int(v) for k, v in years.items()}
                    
                    months = valid_dates[f'{date_col}_dt'].dt.month.value_counts().to_dict()
                    month_dict = {str(k): safe_int(v) for k, v in months.items()}
                    
                    days = valid_dates[f'{date_col}_dt'].dt.day.value_counts().to_dict()
                    day_dict = {str(k): safe_int(v) for k, v in days.items()}
                    
                    weekdays = valid_dates[f'{date_col}_dt'].dt.dayofweek.value_counts().to_dict()
                    weekday_dict = {str(k): safe_int(v) for k, v in weekdays.items()}
                    
                    # Store results
                    time_analysis[f"{date_col}_year"] = year_dict
                    time_analysis[f"{date_col}_month"] = month_dict
                    time_analysis[f"{date_col}_day"] = day_dict
                    time_analysis[f"{date_col}_weekday"] = weekday_dict
                
                except Exception as e:
                    print(f"Error analyzing date column {date_col}: {str(e)}")
        
        return time_analysis
    
    def _analyze_categories(self, df: pd.DataFrame, prioritized_columns: List[str]) -> Dict[str, Any]:
        """Analyze category or type related fields"""
        # Identify potential category columns
        category_cols = []
        # First check if any prioritized columns are date columns
        for col in prioritized_columns:
            if col in df.columns and any(date_term in col.lower() for date_term in ['category', 'type', 'group', 'class']):
                try:
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    category_cols.append(col)
                except:
                    pass
        
        # Then check remaining columns if needed
        if not category_cols:
            for col in df.columns:
                if col not in prioritized_columns and any(date_term in col.lower() for date_term in ['category', 'type', 'group', 'class']):
                    try:
                        pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                        category_cols.append(col)
                    except:
                        pass

        for col in df.columns:
            if any(cat_term in col.lower() for cat_term in ['category', 'type', 'group', 'class']):
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    category_cols.append(col)
        
        category_analysis = {"category_columns": category_cols}
        
        # Analyze distribution for each category column
        for cat_col in category_cols:
            try:
                category_counts = df[cat_col].value_counts().to_dict()
                # Ensure keys are strings and values are integers for JSON serialization
                category_analysis[f"{cat_col}_distribution"] = {str(k): int(v) for k, v in category_counts.items()}
                
                # Try to find numeric columns to analyze by category
                numeric_cols = df.select_dtypes(include=['number']).columns
                for num_col in numeric_cols:
                    try:
                        means_by_category = df.groupby(cat_col)[num_col].mean().to_dict()
                        # Ensure keys are strings and values are floats for JSON serialization
                        category_analysis[f"{cat_col}_{num_col}_mean"] = {str(k): float(v) for k, v in means_by_category.items()}
                    except Exception as e:
                        print(f"Error analyzing {num_col} by {cat_col}: {str(e)}")
            except Exception as e:
                print(f"Error analyzing category column {cat_col}: {str(e)}")
        
        return category_analysis

    
    def _analyze_priority(self, df: pd.DataFrame, prioritized_columns: List[str]) -> Dict[str, Any]:
        """Analyze priority-related fields"""
        # Identify potential priority columns
        priority_cols = []

        # First check if any prioritized columns are date columns
        for col in prioritized_columns:
            if col in df.columns and any(date_term in col.lower() for date_term in ['priority', 'severity', 'urgency', 'importance']):
                try:
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    priority_cols.append(col)
                except:
                    pass
        
        # Then check remaining columns if needed
        if not priority_cols:
            for col in df.columns:
                if col not in prioritized_columns and any(date_term in col.lower() for date_term in ['priority', 'severity', 'urgency', 'importance']):
                    try:
                        pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                        priority_cols.append(col)
                    except:
                        pass

        for col in df.columns:
            if any(prio_term in col.lower() for prio_term in ['priority', 'severity', 'urgency', 'importance']):
                priority_cols.append(col)
        
        priority_analysis = {"priority_columns": priority_cols}
        
        # Analyze priority distributions
        for prio_col in priority_cols:
            try:
                priority_counts = df[prio_col].value_counts().to_dict()
                # Ensure keys are strings for JSON serialization
                priority_analysis[f"{prio_col}_distribution"] = {str(k): int(v) for k, v in priority_counts.items()}
                
                # See if we can correlate with other dimensions
                for other_col in df.columns:
                    if other_col != prio_col and (df[other_col].dtype == 'object' or df[other_col].dtype == 'category'):
                        try:
                            # Create a cross-tabulation
                            cross_tab = pd.crosstab(df[prio_col], df[other_col])
                            
                            # Convert to nested dict with string keys
                            cross_tab_dict = {}
                            for idx, row in cross_tab.iterrows():
                                idx_str = str(idx)
                                cross_tab_dict[idx_str] = {str(col): int(val) for col, val in row.items()}
                            
                            priority_analysis[f"{prio_col}_by_{other_col}"] = cross_tab_dict
                        except Exception as e:
                            print(f"Error cross-tabulating {prio_col} with {other_col}: {str(e)}")
            except Exception as e:
                print(f"Error analyzing priority column {prio_col}: {str(e)}")
        
        return priority_analysis
    
    def _analyze_status(self, df: pd.DataFrame, prioritized_columns: List[str]) -> Dict[str, Any]:
        """Analyze status-related fields"""
        # Identify potential status columns
        status_cols = []
        # First check if any prioritized columns are date columns
        for col in prioritized_columns:
            if col in df.columns and any(date_term in col.lower() for date_term in ['status', 'state', 'resolution']):
                try:
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    status_cols.append(col)
                except:
                    pass
        
        # Then check remaining columns if needed
        if not status_cols:
            for col in df.columns:
                if col not in prioritized_columns and any(date_term in col.lower() for date_term in ['status', 'state', 'resolution']):
                    try:
                        pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                        status_cols.append(col)
                    except:
                        pass

        for col in df.columns:
            if any(status_term in col.lower() for status_term in ['status', 'state', 'resolution']):
                status_cols.append(col)
        
        status_analysis = {"status_columns": status_cols}
        
        # Analyze status distributions
        for status_col in status_cols:
            try:
                status_counts = df[status_col].value_counts().to_dict()
                # Ensure keys are strings for JSON serialization
                status_analysis[f"{status_col}_distribution"] = {str(k): int(v) for k, v in status_counts.items()}
            except Exception as e:
                print(f"Error analyzing status column {status_col}: {str(e)}")
        
        return status_analysis
    
    def _analyze_text_fields(self, df: pd.DataFrame, prioritized_columns: List[str]) -> Dict[str, Any]:
        """Analyze text fields like descriptions"""
        # Identify potential text columns
        text_cols = []
        # First check if any prioritized columns are date columns
        for col in prioritized_columns:
            if col in df.columns and any(date_term in col.lower() for date_term in ['description', 'comment', 'note', 'detail', 'summary']):
                try:
                    pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                    text_cols.append(col)
                except:
                    pass
        
        # Then check remaining columns if needed
        if not text_cols:
            for col in df.columns:
                if col not in prioritized_columns and any(date_term in col.lower() for date_term in ['description', 'comment', 'note', 'detail', 'summary']):
                    try:
                        pd.to_datetime(df[col].iloc[0] if len(df) > 0 else '', errors='coerce')
                        text_cols.append(col)
                    except:
                        pass
        for col in df.columns:
            if any(text_term in col.lower() for text_term in ['description', 'comment', 'note', 'detail', 'summary']):
                if df[col].dtype == 'object':
                    # Check if it contains long text (average length > 50 chars)
                    avg_len = df[col].astype(str).apply(len).mean()
                    if avg_len > 50:
                        text_cols.append(col)
        
        text_analysis = {"text_columns": text_cols}
        
        # Get sample of text for LLM analysis if there are text columns
        if text_cols and len(text_cols) > 0:
            text_samples = {}
            for text_col in text_cols:
                # Get a sample of non-empty text values
                non_empty = df[df[text_col].notnull() & (df[text_col].astype(str).str.len() > 0)]
                if len(non_empty) > 0:
                    sample = non_empty[text_col].sample(min(50, len(non_empty))).tolist()
                    text_samples[text_col] = sample
            
            # If we have samples, use LLM to extract common themes
            if text_samples:
                try:
                    # For each text column, analyze themes
                    for text_col, samples in text_samples.items():
                        themes = self._extract_themes_from_text(text_col, samples)
                        text_analysis[f"{text_col}_themes"] = themes
                except Exception as e:
                    print(f"Error analyzing text themes: {str(e)}")
        
        return text_analysis
    
    def _extract_themes_from_text(self, column_name: str, text_samples: List[str]) -> Dict[str, Any]:
        # Limit the number of samples and their length
        limited_samples = text_samples[:10]  # Reduce sample count
        
        # Truncate each sample to a maximum length
        truncated_samples = []
        for sample in limited_samples:
            # Limit each sample to 500 characters
            truncated_sample = sample[:500]
            truncated_samples.append(truncated_sample)
        
        messages = [
            {"role": "system", "content": "Extract concise themes from ticket descriptions."},
            {"role": "user", "content": f"""
            Analyze themes for '{column_name}' column with these samples:
            
            {truncated_samples}
            
            Provide a very concise JSON response with:
            1. Top 3-5 keywords
            2. Main issue categories
            3. Brief automation opportunities
            """}
        ]
        
        # Implement additional error handling
        try:
            response = self.llm.invoke(messages)
            # Rest of the existing parsing logic...
        except Exception as e:
            print(f"Theme extraction error: {e}")
            return self._get_default_themes()


    def _intelligent_text_sampling(self, df: pd.DataFrame, text_col: str, max_samples: int = 20, max_length: int = 500) -> List[str]:
        """
        Intelligently sample text with consideration for diversity and length
        """
        # Remove null or empty entries
        valid_texts = df[df[text_col].notna() & (df[text_col].str.len() > 0)][text_col]
        
        # Stratified sampling across different categories or priorities
        try:
            # Try to get diverse samples
            sample_strategy = []
            
            # Add representative samples from different categories if possible
            if 'category' in df.columns:
                sample_strategy = (
                    valid_texts.groupby(df['category'])
                    .apply(lambda x: x.sample(min(3, len(x))))
                    .reset_index(drop=True)
                )
            
            # If stratified sampling fails, fall back to random sampling
            if len(sample_strategy) == 0:
                sample_strategy = valid_texts.sample(min(max_samples, len(valid_texts)))
            
            # Truncate and prepare samples
            samples = [
                str(text)[:max_length] 
                for text in sample_strategy
            ]
            
            return samples
        
        except Exception as e:
            print(f"Sampling error: {e}")
            # Fallback to simple random sampling
            return list(valid_texts.sample(min(max_samples, len(valid_texts))).str[:max_length])



    def _get_default_themes(self):
        """Return default themes when LLM extraction fails"""
        return {
            "common_keywords": ["error", "issue", "problem"],
            "main_categories": ["technical issue", "user error", "system failure"],
            "automation_opportunities": ["automatic categorization", "sentiment analysis"]
        }
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate high-level insights based on the analysis results
        """
        # Ensure we have valid analysis results
        if not analysis_results:
            return {
                "volume_insights": [],
                "time_insights": [],
                "category_insights": [],
                "efficiency_insights": [],
                "automation_insights": [],
            }
            
        try:
            insights = {
                "volume_insights": self._generate_volume_insights(analysis_results),
                "time_insights": self._generate_time_insights(analysis_results),
                "category_insights": self._generate_category_insights(analysis_results),
                "efficiency_insights": self._generate_efficiency_insights(analysis_results),
                "automation_insights": self._generate_automation_insights(analysis_results),
            }
            
            # Ensure all insights are strings, not dicts or other complex types
            for category, insight_list in insights.items():
                insights[category] = [str(item) if not isinstance(item, str) else item for item in insight_list]
            
            return insights
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            # Return empty insights on error
            return {
                "volume_insights": [],
                "time_insights": [],
                "category_insights": [],
                "efficiency_insights": [],
                "automation_insights": [],
            }
    
    def _generate_volume_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights about ticket volumes"""
        insights = []
        
        # Extract basic stats
        basic_stats = analysis_results.get("basic_stats", {})
        total_tickets = basic_stats.get("total_tickets", 0)
        
        if total_tickets > 0:
            insights.append(f"Total of {total_tickets} tickets in the dataset.")
        
        # Check category distribution if available
        category_analysis = analysis_results.get("category_analysis", {})
        for key, value in category_analysis.items():
            if key.endswith("_distribution") and isinstance(value, dict):
                category_name = key.replace("_distribution", "")
                top_categories = sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_categories:
                    top_category = top_categories[0]
                    insights.append(f"Most common {category_name}: '{top_category[0]}' with {top_category[1]} tickets ({round(top_category[1]/total_tickets*100, 1)}% of total).")
                    
                    if len(top_categories) >= 3:
                        top_3_count = sum(count for _, count in top_categories[:3])
                        insights.append(f"Top 3 {category_name} categories account for {round(top_3_count/total_tickets*100, 1)}% of tickets.")
        
        return insights
    
    def _generate_time_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights about time patterns"""
        insights = []
        
        time_analysis = analysis_results.get("time_analysis", {})
        
        # Check if there are date columns analyzed
        date_columns = time_analysis.get("date_columns", [])
        
        for date_col in date_columns:
            # Look for patterns in month distribution
            month_key = f"{date_col}_month"
            if month_key in time_analysis:
                month_data = time_analysis[month_key]
                if month_data:
                    # Make sure both key and value are converted to the same type for comparison
                    # Convert all keys to integers for safe comparison
                    peak_month = max(
                        [(int(k) if k.isdigit() else k, v) for k, v in month_data.items()], 
                        key=lambda x: x[1]
                    )
                    insights.append(f"Peak month for {date_col}: Month {peak_month[0]} with {peak_month[1]} tickets.")
            
            # Look for patterns in weekday distribution
            weekday_key = f"{date_col}_weekday"
            if weekday_key in time_analysis:
                weekday_data = time_analysis[weekday_key]
                if weekday_data:
                    # Convert all keys to integers for safe comparison
                    peak_day = max(
                        [(int(k) if k.isdigit() else k, v) for k, v in weekday_data.items()], 
                        key=lambda x: x[1]
                    )
                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    day_index = peak_day[0] if isinstance(peak_day[0], int) else int(peak_day[0]) if str(peak_day[0]).isdigit() else -1
                    day_name = weekday_names[day_index] if 0 <= day_index < 7 else f"Day {peak_day[0]}"
                    insights.append(f"Most tickets for {date_col} occur on {day_name} ({peak_day[1]} tickets).")
        
        return insights
    
    def _generate_category_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights about categories and their relationships"""
        insights = []
        
        try:
            category_analysis = analysis_results.get("category_analysis", {})
            priority_analysis = analysis_results.get("priority_analysis", {})
            
            # Look for interesting cross-tabulations
            for key, value in category_analysis.items():
                if key.startswith("category_") and "_by_" in key and isinstance(value, dict):
                    parts = key.split("_by_")
                    category_col = parts[0]
                    other_col = parts[1]
                    
                    # Find the most common combination
                    if value:
                        # Flatten the nested dict
                        combinations = []
                        for cat, inner_dict in value.items():
                            if isinstance(inner_dict, dict):  # Make sure it's a dict
                                for other, count in inner_dict.items():
                                    # Convert count to int if it's a string
                                    if isinstance(count, str):
                                        try:
                                            count = int(count)
                                        except ValueError:
                                            count = 0
                                    
                                    combinations.append((cat, other, count))
                        
                        if combinations:
                            # Convert all count values to int for comparison
                            for i in range(len(combinations)):
                                cat, other, count = combinations[i]
                                if not isinstance(count, int):
                                    try:
                                        count = int(count)
                                    except (ValueError, TypeError):
                                        count = 0
                                combinations[i] = (cat, other, count)
                            
                            top_combo = max(combinations, key=lambda x: x[2])
                            insights.append(f"Most common combination: {category_col} '{top_combo[0]}' with {other_col} '{top_combo[1]}' ({top_combo[2]} tickets).")
            
            # Check for priority insights
            for key, value in priority_analysis.items():
                if key.endswith("_distribution") and isinstance(value, dict):
                    priority_col = key.replace("_distribution", "")
                    if value:
                        # Convert all values to integers for safe calculation
                        converted_values = {}
                        for prio, count in value.items():
                            if isinstance(count, str):
                                try:
                                    count = int(count)
                                except ValueError:
                                    count = 0
                            converted_values[prio] = count
                        
                        total = sum(converted_values.values())
                        high_prio_keywords = ["high", "critical", "urgent", "p1", "p0", "1", "2"]
                        
                        high_prio_count = sum(count for prio, count in converted_values.items() 
                                             if any(keyword in str(prio).lower() for keyword in high_prio_keywords))
                        
                        if high_prio_count > 0 and total > 0:
                            insights.append(f"{round(high_prio_count/total*100, 1)}% of tickets are high priority ({high_prio_count} out of {total}).")
        except Exception as e:
            print(f"Error generating category insights: {str(e)}")
            
        return insights
    
    def _generate_efficiency_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights about efficiency and resolution times"""
        insights = []
        
        # Look for columns related to resolution time or duration
        basic_stats = analysis_results.get("basic_stats", {})
        numeric_stats = basic_stats.get("numeric_stats", {})
        
        for col, stats in numeric_stats.items():
            if any(time_term in col.lower() for time_term in ['time', 'duration', 'resolution', 'hours', 'days']):
                if 'mean' in stats and 'std' in stats:
                    avg_time = stats['mean']
                    std_time = stats['std']
                    unit = "hours" if "hour" in col.lower() else "days" if "day" in col.lower() else "time units"
                    
                    insights.append(f"Average {col} is {round(avg_time, 2)} {unit} with standard deviation of {round(std_time, 2)} {unit}.")
                    
                    # Check for specific insights by category
                    category_analysis = analysis_results.get("category_analysis", {})
                    for cat_key, cat_value in category_analysis.items():
                        if cat_key.endswith(f"_{col}_mean") and isinstance(cat_value, dict):
                            category_col = cat_key.replace(f"_{col}_mean", "")
                            if cat_value:
                                sorted_cats = sorted(cat_value.items(), key=lambda x: x[1])
                                
                                if sorted_cats:
                                    fastest = sorted_cats[0]
                                    slowest = sorted_cats[-1]
                                    
                                    insights.append(f"Fastest {category_col} for {col}: '{fastest[0]}' ({round(fastest[1], 2)} {unit}).")
                                    insights.append(f"Slowest {category_col} for {col}: '{slowest[0]}' ({round(slowest[1], 2)} {unit}).")
        
        return insights
    
    def _generate_automation_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights about automation opportunities"""
        insights = []
        
        try:
            # Check text analysis for automation opportunities
            text_analysis = analysis_results.get("text_analysis", {})
            
            for key, value in text_analysis.items():
                if key.endswith("_themes") and isinstance(value, dict):
                    if "automation_opportunities" in value:
                        auto_opps = value["automation_opportunities"]
                        if auto_opps and isinstance(auto_opps, list) and len(auto_opps) > 0:
                            text_col = key.replace("_themes", "")
                            insights.append(f"Potential automation opportunities based on {text_col}: {', '.join(auto_opps)}.")
            
            # Look for repetitive patterns in categories
            category_analysis = analysis_results.get("category_analysis", {})
            basic_stats = analysis_results.get("basic_stats", {})
            total_tickets = basic_stats.get("total_tickets", 0)
            
            # Ensure total_tickets is a number
            if isinstance(total_tickets, str):
                try:
                    total_tickets = int(total_tickets)
                except ValueError:
                    total_tickets = 0
            
            if total_tickets > 0:
                for key, value in category_analysis.items():
                    if key.endswith("_distribution") and isinstance(value, dict):
                        category_col = key.replace("_distribution", "")
                        if value:
                            # Ensure we have numeric values for comparison
                            converted_values = {}
                            for k, v in value.items():
                                if isinstance(v, str):
                                    try:
                                        v = int(v)
                                    except ValueError:
                                        v = 0
                                converted_values[k] = v
                                
                            # Sort by the converted numeric values    
                            sorted_cats = sorted(converted_values.items(), key=lambda x: x[1], reverse=True)
                            
                            # If a single category dominates (>30% of tickets)
                            if sorted_cats and sorted_cats[0][1] > 0:
                                dominant_cat = sorted_cats[0]
                                percentage = (dominant_cat[1] / total_tickets) * 100 if total_tickets > 0 else 0
                                if percentage > 30:
                                    insights.append(f"'{dominant_cat[0]}' dominates the {category_col} with {round(percentage, 1)}% of tickets - potential for focused automation.")
        except Exception as e:
            print(f"Error generating automation insights: {str(e)}")
            
        return insights