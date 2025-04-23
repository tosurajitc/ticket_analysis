# analysis/resource_optimizer.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Tuple, Union, Optional
import datetime

class ResourceOptimizer:
    """
    Analyzes incident data to provide recommendations for optimal resource allocation.
    Uses machine learning to predict staffing needs and identify skill requirements
    based on historical incident patterns and resolution times.
    """
    
    def __init__(self, min_samples_required: int = 50):
        """
        Initialize the resource optimizer.
        
        Args:
            min_samples_required: Minimum number of incidents required for analysis
        """
        self.min_samples_required = min_samples_required
        self.models = {}
        self.transformers = {}
        self.logger = logging.getLogger(__name__)
        self.last_analysis = None
    
    def _validate_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        Validate if the data is sufficient for resource optimization analysis.
        
        Args:
            df: Incident dataframe
            required_columns: List of columns required for the analysis
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for resource optimization")
            return False
            
        if len(df) < self.min_samples_required:
            self.logger.warning(
                f"Insufficient data for resource optimization. Got {len(df)} incidents, "
                f"need at least {self.min_samples_required}."
            )
            return False
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(
                    f"Missing required columns for resource optimization: {', '.join(missing_cols)}"
                )
                return False
            
        return True
    
    def analyze_workload_distribution(self, df: pd.DataFrame,
                                    timestamp_col: str,
                                    category_col: str = None,
                                    priority_col: str = None,
                                    assignee_col: str = None) -> Dict:
        """
        Analyze the distribution of incidents across time, categories, and assignees.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing incident creation timestamps
            category_col: Column containing incident categories
            priority_col: Column containing incident priorities
            assignee_col: Column containing incident assignees
            
        Returns:
            Dictionary containing workload distribution analysis
        """
        required_cols = [timestamp_col]
        optional_cols = [col for col in [category_col, priority_col, assignee_col] if col is not None]
        
        if not self._validate_data(df, required_cols):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for workload distribution analysis',
                'workload': None
            }
        
        try:
            # Ensure timestamp is in datetime format
            if df[timestamp_col].dtype != 'datetime64[ns]':
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Extract time components
            df_with_time = df.copy()
            df_with_time['hour'] = df_with_time[timestamp_col].dt.hour
            df_with_time['day_of_week'] = df_with_time[timestamp_col].dt.dayofweek
            df_with_time['day_name'] = df_with_time[timestamp_col].dt.day_name()
            
            # Initialize result
            result = {
                'success': True,
                'message': 'Workload distribution analysis completed successfully',
                'workload': {
                    'temporal': {},
                    'categorical': {},
                    'staffing': {}
                }
            }
            
            # Analyze temporal distribution
            # By hour of day
            hourly_counts = df_with_time.groupby('hour').size()
            total_incidents = len(df_with_time)
            hourly_percentages = (hourly_counts / total_incidents * 100).round(1)
            
            result['workload']['temporal']['hourly'] = {
                'counts': hourly_counts.to_dict(),
                'percentages': hourly_percentages.to_dict(),
                'peak_hour': hourly_counts.idxmax(),
                'peak_hour_count': hourly_counts.max(),
                'peak_hour_percentage': f"{hourly_percentages.max():.1f}%"
            }
            
            # By day of week
            daily_counts = df_with_time.groupby(['day_of_week', 'day_name']).size()
            daily_counts_dict = daily_counts.to_dict()
            daily_counts_readable = {day_name: count for (_, day_name), count in daily_counts_dict.items()}
            
            daily_percentages = (daily_counts / total_incidents * 100).round(1)
            daily_percentages_dict = daily_percentages.to_dict()
            daily_percentages_readable = {day_name: f"{pct:.1f}%" for (_, day_name), pct in daily_percentages_dict.items()}
            
            peak_day_idx = daily_counts.idxmax()
            peak_day_name = peak_day_idx[1]
            
            result['workload']['temporal']['daily'] = {
                'counts': daily_counts_readable,
                'percentages': daily_percentages_readable,
                'peak_day': peak_day_name,
                'peak_day_count': daily_counts.max(),
                'peak_day_percentage': f"{daily_percentages.max():.1f}%"
            }
            
            # Analyze categorical distribution (if applicable)
            if category_col and category_col in df.columns:
                category_counts = df[category_col].value_counts()
                category_percentages = (category_counts / total_incidents * 100).round(1)
                
                result['workload']['categorical']['category'] = {
                    'counts': category_counts.to_dict(),
                    'percentages': {cat: f"{pct:.1f}%" for cat, pct in category_percentages.items()},
                    'top_categories': category_counts.nlargest(5).index.tolist(),
                    'top_category': category_counts.idxmax(),
                    'top_category_count': category_counts.max(),
                    'top_category_percentage': f"{category_percentages.max():.1f}%"
                }
            
            if priority_col and priority_col in df.columns:
                priority_counts = df[priority_col].value_counts()
                priority_percentages = (priority_counts / total_incidents * 100).round(1)
                
                result['workload']['categorical']['priority'] = {
                    'counts': priority_counts.to_dict(),
                    'percentages': {pri: f"{pct:.1f}%" for pri, pct in priority_percentages.items()},
                    'distribution': priority_percentages.to_dict()
                }
            
            # Analyze assignee distribution (if applicable)
            if assignee_col and assignee_col in df.columns:
                assignee_counts = df[assignee_col].value_counts()
                assignee_workload_percentages = (assignee_counts / total_incidents * 100).round(1)
                
                # Calculate workload concentration (Gini coefficient)
                assignee_sorted = np.sort(assignee_counts.values)
                n = len(assignee_sorted)
                if n > 1:
                    # Calculate Gini coefficient for workload distribution
                    index = np.arange(1, n + 1)
                    gini = ((2 * index - n - 1) * assignee_sorted).sum() / (n * assignee_sorted.sum())
                else:
                    gini = 0
                
                result['workload']['staffing']['assignee'] = {
                    'total_assignees': len(assignee_counts),
                    'incidents_per_assignee': {
                        'mean': assignee_counts.mean(),
                        'median': assignee_counts.median(),
                        'min': assignee_counts.min(),
                        'max': assignee_counts.max()
                    },
                    'workload_concentration': round(gini, 2),
                    'top_assignees': assignee_counts.nlargest(5).index.tolist(),
                    'distribution_balance': "Balanced" if gini < 0.3 else 
                                           "Moderately imbalanced" if gini < 0.5 else 
                                           "Highly imbalanced"
                }
                
                # Cross-tabulate assignees with categories if both available
                if category_col and category_col in df.columns:
                    # Get top 10 assignees by volume
                    top_assignees = assignee_counts.nlargest(10).index
                    top_assignee_df = df[df[assignee_col].isin(top_assignees)]
                    
                    # Get their category specialization
                    specialization = {}
                    for assignee in top_assignees:
                        assignee_incidents = top_assignee_df[top_assignee_df[assignee_col] == assignee]
                        if len(assignee_incidents) > 0:
                            category_dist = assignee_incidents[category_col].value_counts(normalize=True).round(2)
                            top_category = category_dist.idxmax()
                            specialization[assignee] = {
                                'primary_category': top_category,
                                'primary_percentage': f"{category_dist.max() * 100:.1f}%",
                                'category_distribution': {cat: f"{pct * 100:.1f}%" for cat, pct in category_dist.items()}
                            }
                    
                    result['workload']['staffing']['specialization'] = specialization
            
            # Calculate workload by time and category combinations
            if category_col and category_col in df.columns:
                # Identify peak hours by category
                peak_hours_by_category = {}
                categories = df[category_col].unique()
                
                for category in categories:
                    category_df = df_with_time[df_with_time[category_col] == category]
                    if len(category_df) > 0:
                        cat_hourly_counts = category_df.groupby('hour').size()
                        if not cat_hourly_counts.empty:
                            peak_hour = cat_hourly_counts.idxmax()
                            peak_hours_by_category[category] = {
                                'peak_hour': peak_hour,
                                'count': cat_hourly_counts.max()
                            }
                
                result['workload']['temporal']['peak_hours_by_category'] = peak_hours_by_category
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing workload distribution: {str(e)}")
            return {
                'success': False,
                'message': f'Error during workload distribution analysis: {str(e)}',
                'workload': None
            }
    
    def predict_staffing_needs(self, df: pd.DataFrame,
                             timestamp_col: str,
                             resolution_time_col: str,
                             category_col: str = None,
                             priority_col: str = None,
                             forecast_days: int = 7) -> Dict:
        """
        Predict future staffing needs based on historical incident patterns and resolution times.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing incident creation timestamps
            resolution_time_col: Column containing resolution times
            category_col: Column containing incident categories
            priority_col: Column containing incident priorities
            forecast_days: Number of days to forecast staffing needs
            
        Returns:
            Dictionary containing staffing needs predictions
        """
        required_cols = [timestamp_col, resolution_time_col]
        optional_cols = [col for col in [category_col, priority_col] if col is not None]
        
        if not self._validate_data(df, required_cols):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for staffing prediction',
                'predictions': None
            }
        
        try:
            # Ensure timestamp is in datetime format
            if df[timestamp_col].dtype != 'datetime64[ns]':
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Calculate incident inflow by day
            df_with_time = df.copy()
            df_with_time['date'] = df_with_time[timestamp_col].dt.date
            df_with_time['day_of_week'] = df_with_time[timestamp_col].dt.dayofweek
            
            # Aggregate incidents by date and get count
            daily_incidents = df_with_time.groupby('date').size()
            
            # Calculate average resolution time per incident
            df_with_time['resolution_time'] = pd.to_numeric(df_with_time[resolution_time_col], errors='coerce')
            mean_resolution_time = df_with_time['resolution_time'].mean()
            
            # Calculate average incidents by day of week
            avg_by_day = df_with_time.groupby('day_of_week').size().reset_index()
            avg_by_day.columns = ['day_of_week', 'incident_count']
            day_totals = avg_by_day.groupby('day_of_week')['incident_count'].sum()
            
            # Get count of each day of week in the dataset
            day_counts = df_with_time['day_of_week'].value_counts()
            
            # Calculate average incidents per specific day of week
            avg_incidents_by_day = {}
            for day in range(7):
                if day in day_totals and day in day_counts:
                    avg_incidents_by_day[day] = day_totals[day] / day_counts[day]
                else:
                    avg_incidents_by_day[day] = 0
            
            # Calculate average work hours per incident
            if categorical_features := [col for col in optional_cols if col in df.columns]:
                # Build more sophisticated model that takes categories into account
                X_data = df_with_time[categorical_features + ['day_of_week']]
                y_data = df_with_time['resolution_time']
                
                # Prepare preprocessor
                categorical_cols = [col for col in categorical_features if df_with_time[col].dtype == 'object']
                numeric_cols = [col for col in categorical_features if df_with_time[col].dtype != 'object']
                numeric_cols.append('day_of_week')
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ]
                )
                
                # Build and fit pipeline
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(random_state=42))
                ])
                
                # Fit model
                model.fit(X_data, y_data)
                self.models['resolution_time'] = model
                self.transformers['resolution_features'] = categorical_features + ['day_of_week']
            
            # Calculate staffing needs for future days
            result = {
                'success': True,
                'message': 'Staffing needs prediction completed successfully',
                'predictions': {
                    'by_day': {},
                    'overall': {},
                    'by_category': {}
                }
            }
            
            # Start from tomorrow
            start_date = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
            
            # For each future day
            daily_predictions = []
            for day_offset in range(forecast_days):
                forecast_date = start_date + datetime.timedelta(days=day_offset)
                day_of_week = forecast_date.weekday()
                
                # Predict number of incidents
                predicted_incidents = avg_incidents_by_day.get(day_of_week, daily_incidents.mean())
                
                # Predict staff hours needed
                if 'resolution_time' in self.models:
                    # Use sophisticated model if available
                    # For simplicity, we'll just use average by day of week here
                    # In a real implementation, you'd generate predictions for various incident types
                    staff_hours = predicted_incidents * mean_resolution_time
                else:
                    # Use simple average
                    staff_hours = predicted_incidents * mean_resolution_time
                
                # Store prediction
                day_prediction = {
                    'date': forecast_date,
                    'day_of_week': day_of_week,
                    'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
                    'predicted_incidents': round(predicted_incidents, 1),
                    'predicted_hours': round(staff_hours, 1),
                    'staff_needed': max(1, round(staff_hours / 8))  # Assuming 8-hour workday
                }
                
                daily_predictions.append(day_prediction)
                result['predictions']['by_day'][str(forecast_date)] = day_prediction
            
            # Calculate overall stats
            total_incidents = sum(day['predicted_incidents'] for day in daily_predictions)
            total_hours = sum(day['predicted_hours'] for day in daily_predictions)
            avg_daily_staff = sum(day['staff_needed'] for day in daily_predictions) / forecast_days
            
            result['predictions']['overall'] = {
                'total_predicted_incidents': round(total_incidents, 1),
                'total_predicted_hours': round(total_hours, 1),
                'average_daily_staff': round(avg_daily_staff, 1),
                'forecast_period_days': forecast_days
            }
            
            # If category data is available, break down by category
            if category_col and category_col in df.columns:
                category_percents = df_with_time[category_col].value_counts(normalize=True)
                
                category_predictions = {}
                for category, percent in category_percents.items():
                    category_incidents = total_incidents * percent
                    
                    # Get average resolution time for this category
                    cat_df = df_with_time[df_with_time[category_col] == category]
                    cat_resolution_time = cat_df['resolution_time'].mean() if not cat_df.empty else mean_resolution_time
                    
                    category_hours = category_incidents * cat_resolution_time
                    
                    category_predictions[category] = {
                        'percentage': f"{percent * 100:.1f}%",
                        'predicted_incidents': round(category_incidents, 1),
                        'average_resolution_time': round(cat_resolution_time, 1),
                        'predicted_hours': round(category_hours, 1),
                        'staff_needed': max(1, round(category_hours / (8 * forecast_days)))  # Daily staff
                    }
                
                result['predictions']['by_category'] = category_predictions
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting staffing needs: {str(e)}")
            return {
                'success': False,
                'message': f'Error during staffing prediction: {str(e)}',
                'predictions': None
            }
    
    def get_skill_recommendations(self, df: pd.DataFrame,
                                 resolution_time_col: str,
                                 category_col: str,
                                 assignee_col: str = None,
                                 skill_cols: List[str] = None) -> Dict:
        """
        Generate recommendations for skill development based on incident resolution analysis.
        
        Args:
            df: Incident dataframe
            resolution_time_col: Column containing resolution times
            category_col: Column containing incident categories
            assignee_col: Column containing incident assignees
            skill_cols: Additional columns containing skill information
            
        Returns:
            Dictionary containing skill development recommendations
        """
        required_cols = [resolution_time_col, category_col]
        optional_cols = [assignee_col] + (skill_cols or [])
        optional_cols = [col for col in optional_cols if col is not None]
        
        if not self._validate_data(df, required_cols):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for skill recommendations',
                'recommendations': None
            }
        
        try:
            df_analysis = df.copy()
            
            # Ensure resolution time is numeric
            df_analysis['resolution_time'] = pd.to_numeric(df_analysis[resolution_time_col], errors='coerce')
            
            # Get categories
            categories = df_analysis[category_col].unique()
            
            # Initialize result
            result = {
                'success': True,
                'message': 'Skill recommendations generated successfully',
                'recommendations': {
                    'by_category': {},
                    'team_composition': {},
                    'skill_gaps': []
                }
            }
            
            # Analyze resolution time by category
            res_by_category = df_analysis.groupby(category_col)['resolution_time'].agg(['mean', 'count'])
            
            # Calculate relative difficulty (normalized resolution time)
            if len(res_by_category) > 0:
                overall_mean = df_analysis['resolution_time'].mean()
                res_by_category['relative_difficulty'] = res_by_category['mean'] / overall_mean
                
                for category in categories:
                    if category in res_by_category.index:
                        count = res_by_category.loc[category, 'count']
                        mean_time = res_by_category.loc[category, 'mean']
                        rel_diff = res_by_category.loc[category, 'relative_difficulty']
                        
                        # Skip categories with too few incidents
                        if count < 5:
                            continue
                        
                        category_info = {
                            'incident_count': int(count),
                            'average_resolution_time': round(mean_time, 1),
                            'relative_difficulty': round(rel_diff, 2),
                            'complexity': "High" if rel_diff > 1.3 else 
                                        "Medium" if rel_diff > 0.7 else 
                                        "Low"
                        }
                        
                        result['recommendations']['by_category'][category] = category_info
            
            # Analyze team skill distribution
            if assignee_col and assignee_col in df.columns:
                # Create cross-tabulation of assignees and categories
                assignee_category = pd.crosstab(
                    df_analysis[assignee_col], 
                    df_analysis[category_col],
                    normalize='index'
                )
                
                # Get top categories for each assignee
                assignee_specialties = {}
                for assignee in assignee_category.index:
                    if assignee_category.loc[assignee].max() > 0.5:  # If assignee handles >50% of one category
                        specialty = assignee_category.loc[assignee].idxmax()
                        specialty_pct = assignee_category.loc[assignee].max()
                        assignee_specialties[assignee] = {
                            'primary_category': specialty,
                            'specialization_degree': f"{specialty_pct * 100:.1f}%",
                            'is_specialist': specialty_pct > 0.7
                        }
                
                # Count specialists by category
                specialists_by_category = {}
                for assignee, info in assignee_specialties.items():
                    category = info['primary_category']
                    if category not in specialists_by_category:
                        specialists_by_category[category] = 0
                    
                    if info['is_specialist']:
                        specialists_by_category[category] += 1
                
                # Analyze resolution time by specialist vs non-specialist
                if assignee_specialties:
                    specialist_performance = {}
                    
                    for category in categories:
                        # Skip categories with too few incidents
                        if category not in res_by_category.index or res_by_category.loc[category, 'count'] < 10:
                            continue
                        
                        # Get specialists for this category
                        specialists = [
                            assignee for assignee, info in assignee_specialties.items()
                            if info['primary_category'] == category and info['is_specialist']
                        ]
                        
                        if not specialists:
                            continue
                        
                        # Compare resolution times
                        specialist_incidents = df_analysis[
                            (df_analysis[category_col] == category) & 
                            (df_analysis[assignee_col].isin(specialists))
                        ]
                        
                        nonspecialist_incidents = df_analysis[
                            (df_analysis[category_col] == category) & 
                            (~df_analysis[assignee_col].isin(specialists))
                        ]
                        
                        if len(specialist_incidents) > 5 and len(nonspecialist_incidents) > 5:
                            specialist_time = specialist_incidents['resolution_time'].mean()
                            nonspecialist_time = nonspecialist_incidents['resolution_time'].mean()
                            
                            improvement = (nonspecialist_time - specialist_time) / nonspecialist_time
                            
                            specialist_performance[category] = {
                                'specialist_count': len(specialists),
                                'specialist_resolution_time': round(specialist_time, 1),
                                'nonspecialist_resolution_time': round(nonspecialist_time, 1),
                                'improvement_percentage': f"{improvement * 100:.1f}%",
                                'significant_improvement': improvement > 0.2
                            }
                
                # Store team composition analysis
                result['recommendations']['team_composition'] = {
                    'total_assignees': len(assignee_category.index),
                    'specialists_count': sum(1 for info in assignee_specialties.values() if info['is_specialist']),
                    'generalists_count': len(assignee_category.index) - sum(1 for info in assignee_specialties.values() if info['is_specialist']),
                    'specialist_by_category': specialists_by_category,
                    'specialist_performance': specialist_performance
                }
                
                # Identify skill gaps
                for category, category_info in result['recommendations']['by_category'].items():
                    # Categories with high relative difficulty but few specialists
                    if category_info['complexity'] == "High" and specialists_by_category.get(category, 0) < 2:
                        result['recommendations']['skill_gaps'].append({
                            'category': category,
                            'issue': 'specialist_shortage',
                            'current_specialists': specialists_by_category.get(category, 0),
                            'relative_difficulty': category_info['relative_difficulty'],
                            'recommendation': f"Develop more specialists for {category} incidents, which have {category_info['relative_difficulty']}x longer resolution times than average"
                        })
                    
                    # Categories with performance gap between specialists and non-specialists
                    if category in specialist_performance and specialist_performance[category]['significant_improvement']:
                        result['recommendations']['skill_gaps'].append({
                            'category': category,
                            'issue': 'performance_gap',
                            'improvement_potential': specialist_performance[category]['improvement_percentage'],
                            'recommendation': f"Train more team members in {category} skills, as specialists resolve these incidents {specialist_performance[category]['improvement_percentage']} faster"
                        })
            
            # If no skill gaps were identified
            if not result['recommendations']['skill_gaps']:
                result['recommendations']['skill_gaps'].append({
                    'issue': 'no_clear_gaps',
                    'recommendation': "No clear skill gaps identified based on available data"
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating skill recommendations: {str(e)}")
            return {
                'success': False,
                'message': f'Error during skill recommendation generation: {str(e)}',
                'recommendations': None
            }
    
    def get_resource_optimization_insights(self, workload_result: Dict, 
                                         staffing_result: Dict = None, 
                                         skill_result: Dict = None) -> List[Dict]:
        """
        Generate data-driven insights from resource optimization analysis results.
        
        Args:
            workload_result: Result from analyze_workload_distribution method
            staffing_result: Result from predict_staffing_needs method
            skill_result: Result from get_skill_recommendations method
            
        Returns:
            List of dictionaries with resource optimization insights
        """
        insights = []
        
        # Check if sufficient data for insights
        if not workload_result['success']:
            return [{
                'type': 'error',
                'message': 'Insufficient data to generate resource optimization insights'
            }]
        
        try:
            # Generate temporal insights
            if 'workload' in workload_result and workload_result['workload']:
                temporal = workload_result['workload'].get('temporal', {})
                
                # Peak hour insight
                if 'hourly' in temporal:
                    peak_hour = temporal['hourly']['peak_hour']
                    peak_hour_pct = temporal['hourly']['peak_hour_percentage']
                    
                    insights.append({
                        'type': 'temporal_peak',
                        'subtype': 'hour',
                        'data': {
                            'peak_hour': peak_hour,
                            'percentage': peak_hour_pct
                        },
                        'message': f"Peak incident volume occurs at {peak_hour}:00, accounting for {peak_hour_pct} of all incidents"
                    })
                
                # Peak day insight
                if 'daily' in temporal:
                    peak_day = temporal['daily']['peak_day']
                    peak_day_pct = temporal['daily']['peak_day_percentage']
                    
                    insights.append({
                        'type': 'temporal_peak',
                        'subtype': 'day',
                        'data': {
                            'peak_day': peak_day,
                            'percentage': peak_day_pct
                        },
                        'message': f"{peak_day} has the highest incident volume, accounting for {peak_day_pct} of all incidents"
                    })
                
                # Category-specific peak times
                if 'peak_hours_by_category' in temporal:
                    for category, data in temporal['peak_hours_by_category'].items():
                        insights.append({
                            'type': 'temporal_category_peak',
                            'data': {
                                'category': category,
                                'peak_hour': data['peak_hour']
                            },
                            'message': f"{category} incidents peak at {data['peak_hour']}:00"
                        })
            
            # Generate categorical insights
            if 'workload' in workload_result and 'categorical' in workload_result['workload']:
                categorical = workload_result['workload']['categorical']
                
                # Top category insight
                if 'category' in categorical:
                    top_category = categorical['category']['top_category']
                    top_category_pct = categorical['category']['top_category_percentage']
                    
                    insights.append({
                        'type': 'category_distribution',
                        'data': {
                            'top_category': top_category,
                            'percentage': top_category_pct
                        },

                        'message': f"{top_category} is the most common incident type, representing {top_category_pct} of all incidents"
                    })
            
            # Generate staffing insights
            if 'workload' in workload_result and 'staffing' in workload_result['workload']:
                staffing = workload_result['workload']['staffing']
                
                # Workload distribution insight
                if 'assignee' in staffing:
                    balance = staffing['assignee']['distribution_balance']
                    concentration = staffing['assignee'].get('workload_concentration', 0)
                    
                    insights.append({
                        'type': 'workload_balance',
                        'data': {
                            'balance_status': balance,
                            'concentration_index': concentration
                        },
                        'message': f"Workload distribution is {balance.lower()} with a concentration index of {concentration}"
                    })
                    
                    # Specialization insights
                    if 'specialization' in staffing:
                        specialist_count = sum(1 for assignee, data in staffing['specialization'].items() 
                                         if float(data['primary_percentage'].strip('%')) > 70)
                        
                        if specialist_count > 0:
                            insights.append({
                                'type': 'specialist_distribution',
                                'data': {
                                    'specialist_count': specialist_count
                                },
                                'message': f"Team has {specialist_count} specialists who handle primarily one category of incidents"
                            })
            
            # Generate staffing predictions insights
            if staffing_result and staffing_result['success'] and 'predictions' in staffing_result:
                predictions = staffing_result['predictions']
                
                # Overall staffing needs
                if 'overall' in predictions:
                    overall = predictions['overall']
                    avg_staff = overall['average_daily_staff']
                    total_incidents = overall['total_predicted_incidents']
                    
                    insights.append({
                        'type': 'staffing_prediction',
                        'data': {
                            'avg_daily_staff': avg_staff,
                            'total_incidents': total_incidents,
                            'forecast_days': overall['forecast_period_days']
                        },
                        'message': f"Estimated {avg_staff} staff needed daily to handle {total_incidents} incidents over the next {overall['forecast_period_days']} days"
                    })
                
                # Daily variations
                if 'by_day' in predictions:
                    days = predictions['by_day']
                    # Find day with highest staffing need
                    max_staff_day = max(days.items(), key=lambda x: x[1]['staff_needed'])
                    date = max_staff_day[0]
                    staff = max_staff_day[1]['staff_needed']
                    day_name = max_staff_day[1]['day_name']
                    
                    insights.append({
                        'type': 'peak_staffing_day',
                        'data': {
                            'date': date,
                            'day': day_name,
                            'staff_needed': staff
                        },
                        'message': f"Highest staffing need is on {day_name} ({date}) with {staff} staff required"
                    })
                
                # Category-specific staffing
                if 'by_category' in predictions:
                    categories = predictions['by_category']
                    # Find category with highest staffing need
                    max_staff_category = max(categories.items(), key=lambda x: x[1]['staff_needed'])
                    category = max_staff_category[0]
                    staff = max_staff_category[1]['staff_needed']
                    
                    insights.append({
                        'type': 'category_staffing',
                        'data': {
                            'category': category,
                            'staff_needed': staff
                        },
                        'message': f"{category} incidents require the most resources with {staff} dedicated staff recommended"
                    })
            
            # Generate skill recommendations insights
            if skill_result and skill_result['success'] and 'recommendations' in skill_result:
                recommendations = skill_result['recommendations']
                
                # Skills gaps
                if 'skill_gaps' in recommendations:
                    for gap in recommendations['skill_gaps']:
                        if gap.get('issue') != 'no_clear_gaps':
                            insights.append({
                                'type': 'skill_gap',
                                'data': gap,
                                'message': gap['recommendation']
                            })
                
                # Team composition insights
                if 'team_composition' in recommendations:
                    composition = recommendations['team_composition']
                    
                    if 'specialist_performance' in composition:
                        # Find category with biggest specialist advantage
                        specialist_perf = composition['specialist_performance']
                        best_categories = []
                        
                        for category, perf in specialist_perf.items():
                            if perf.get('significant_improvement'):
                                best_categories.append((category, perf))
                        
                        if best_categories:
                            # Sort by improvement percentage
                            best_categories.sort(key=lambda x: float(x[1]['improvement_percentage'].strip('%')), reverse=True)
                            top_category, top_perf = best_categories[0]
                            
                            insights.append({
                                'type': 'specialist_impact',
                                'data': {
                                    'category': top_category,
                                    'improvement': top_perf['improvement_percentage']
                                },
                                'message': f"Specialists resolve {top_category} incidents {top_perf['improvement_percentage']} faster than non-specialists"
                            })
            
            # If no insights were generated
            if not insights:
                insights.append({
                    'type': 'general',
                    'message': "No specific resource optimization insights available from current data"
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating resource optimization insights: {str(e)}")
            return [{
                'type': 'error',
                'message': f'Error generating resource optimization insights: {str(e)}'
            }]


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    import datetime
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 500
    
    # Create assignees
    assignees = [f'Agent{i:02d}' for i in range(20)]
    
    # Create categories with different patterns
    categories = ['Network', 'Server', 'Application', 'Database', 'Security']
    
    # Create priorities
    priorities = ['P1', 'P2', 'P3', 'P4']
    
    # Create timestamps with weekly patterns
    base_date = datetime.datetime(2023, 1, 1)
    dates = []
    for i in range(n_samples):
        # Add a random number of days (0-90 days)
        random_days = np.random.randint(0, 90)
        # Business hours more likely
        hour = np.random.choice(
            np.concatenate([np.arange(8, 18), np.arange(0, 24)]),  # Business hours more likely
            p=np.concatenate([np.ones(10) * 0.08, np.ones(24) * 0.01])  # Higher weights for business hours
        )
        dates.append(base_date + datetime.timedelta(days=random_days, hours=hour))
    
    # Create synthetic data
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(n_samples)],
        'created_at': dates,
        'category': np.random.choice(categories, n_samples),
        'priority': np.random.choice(priorities, n_samples, p=[0.1, 0.2, 0.4, 0.3]),
        'resolution_time': np.random.exponential(4, n_samples),  # Hours to resolve
        'assignee': np.random.choice(assignees, n_samples)
    }
    
    # Create specialists who handle primarily one category
    specialist_map = {
        'Agent01': 'Network',
        'Agent02': 'Network',
        'Agent03': 'Server',
        'Agent04': 'Server',
        'Agent05': 'Application',
        'Agent06': 'Database',
        'Agent07': 'Security'
    }
    
    # Adjust data to reflect specialist assignments and performance
    for i in range(n_samples):
        assignee = data['assignee'][i]
        
        # If this is a specialist, increase likelihood they handle their specialty
        if assignee in specialist_map:
            if np.random.random() < 0.8:  # 80% chance specialist handles their specialty
                data['category'][i] = specialist_map[assignee]
                
                # Specialists resolve their specialty faster
                data['resolution_time'][i] *= 0.6
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Test the resource optimizer
    optimizer = ResourceOptimizer()
    
    # Analyze workload distribution
    workload_result = optimizer.analyze_workload_distribution(
        df, 
        timestamp_col='created_at',
        category_col='category',
        priority_col='priority',
        assignee_col='assignee'
    )
    
    # Predict staffing needs
    staffing_result = optimizer.predict_staffing_needs(
        df,
        timestamp_col='created_at',
        resolution_time_col='resolution_time',
        category_col='category',
        priority_col='priority'
    )
    
    # Get skill recommendations
    skill_result = optimizer.get_skill_recommendations(
        df,
        resolution_time_col='resolution_time',
        category_col='category',
        assignee_col='assignee'
    )
    
    # Generate insights
    if workload_result['success']:
        print("RESOURCE OPTIMIZATION INSIGHTS:")
        insights = optimizer.get_resource_optimization_insights(
            workload_result,
            staffing_result,
            skill_result
        )
        
        for insight in insights:
            print(f"- {insight['message']}")
    else:
        print(f"Resource optimization analysis failed: {workload_result['message']}")                            