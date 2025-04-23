# models/anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import logging
from typing import Dict, List, Tuple, Union, Optional
import datetime

class IncidentAnomalyDetector:
    """
    Detects anomalous incidents that deviate from normal patterns.
    These could be critical incidents that require special attention or
    indicate emerging problems before they become widespread.
    """
    
    def __init__(self, contamination: float = 0.05, min_samples_required: int = 20):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            min_samples_required: Minimum number of samples needed for reliable anomaly detection
        """
        self.contamination = contamination
        self.min_samples_required = min_samples_required
        self.scaler = None
        self.model = None
        self.pca = None
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None
        
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate if the data is sufficient for anomaly detection.
        
        Args:
            df: Incident dataframe
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for anomaly detection")
            return False
            
        if len(df) < self.min_samples_required:
            self.logger.warning(
                f"Insufficient data for anomaly detection. Got {len(df)} samples, "
                f"need at least {self.min_samples_required}."
            )
            return False
            
        return True
    
    def prepare_features(self, df: pd.DataFrame, 
                         numerical_features: List[str],
                         categorical_features: List[str],
                         datetime_features: List[str] = None) -> Optional[np.ndarray]:
        """
        Prepare features for anomaly detection by scaling numerical features,
        one-hot encoding categorical features, and extracting datetime features.
        
        Args:
            df: Incident dataframe
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            datetime_features: List of datetime feature column names
            
        Returns:
            Combined feature matrix or None if preparation fails
        """
        if not self._validate_data(df):
            return None
        
        try:
            prepared_features = []
            
            # Process numerical features
            if numerical_features and all(col in df.columns for col in numerical_features):
                numerical_df = df[numerical_features].copy()
                # Handle missing values
                numerical_df = numerical_df.fillna(numerical_df.mean())
                self.scaler = StandardScaler()
                numerical_data = self.scaler.fit_transform(numerical_df)
                prepared_features.append(numerical_data)
            
            # Process categorical features
            if categorical_features and all(col in df.columns for col in categorical_features):
                categorical_df = pd.get_dummies(df[categorical_features], drop_first=True)
                prepared_features.append(categorical_df.values)
            
            # Process datetime features
            if datetime_features and all(col in df.columns for col in datetime_features):
                datetime_df = pd.DataFrame()
                
                for col in datetime_features:
                    # Ensure column is in datetime format
                    if df[col].dtype != 'datetime64[ns]':
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            self.logger.warning(f"Could not convert {col} to datetime. Skipping.")
                            continue
                    
                    # Extract useful datetime components
                    datetime_df[f"{col}_hour"] = df[col].dt.hour
                    datetime_df[f"{col}_day"] = df[col].dt.day
                    datetime_df[f"{col}_weekday"] = df[col].dt.weekday
                    datetime_df[f"{col}_month"] = df[col].dt.month
                    
                if not datetime_df.empty:
                    datetime_scaler = StandardScaler()
                    datetime_data = datetime_scaler.fit_transform(datetime_df)
                    prepared_features.append(datetime_data)
            
            if not prepared_features:
                self.logger.warning("No valid features available for anomaly detection")
                return None
                
            # Combine all features
            combined_features = np.hstack(prepared_features)
            
            # Apply PCA for dimension reduction if the feature space is large
            if combined_features.shape[1] > 20:
                n_components = min(20, combined_features.shape[0] - 1, combined_features.shape[1])
                self.pca = PCA(n_components=n_components)
                combined_features = self.pca.fit_transform(combined_features)
                
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for anomaly detection: {str(e)}")
            return None
    
    def detect_anomalies(self, df: pd.DataFrame, 
                         numerical_features: List[str],
                         categorical_features: List[str],
                         datetime_features: List[str] = None,
                         algorithm: str = 'iforest',
                         return_scores: bool = True) -> Dict:
        """
        Detect anomalies in incident data using the specified algorithm.
        
        Args:
            df: Incident dataframe
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            datetime_features: List of datetime feature column names
            algorithm: Algorithm to use ('iforest', 'lof', or 'ocsvm')
            return_scores: Whether to return anomaly scores
            
        Returns:
            Dictionary containing anomaly detection results
        """
        features = self.prepare_features(df, numerical_features, categorical_features, datetime_features)
        
        if features is None:
            return {
                'success': False,
                'message': 'Insufficient or invalid data for anomaly detection',
                'anomalies': None
            }
        
        try:
            # Select and initialize the anomaly detection model
            if algorithm == 'iforest':
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
            elif algorithm == 'lof':
                self.model = LocalOutlierFactor(
                    contamination=self.contamination,
                    n_neighbors=20,
                    novelty=True
                )
            elif algorithm == 'ocsvm':
                self.model = OneClassSVM(
                    nu=self.contamination,
                    kernel="rbf",
                    gamma='auto'
                )
            else:
                return {
                    'success': False,
                    'message': f'Unsupported algorithm: {algorithm}',
                    'anomalies': None
                }
            
            # Fit the model and get anomaly labels
            self.model.fit(features)
            
            # Get predictions (1 for normal, -1 for anomalies)
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(features)
            else:
                # For LOF in non-novelty mode, use fit_predict
                y_pred = self.model.fit_predict(features)
            
            # Convert to boolean mask where True indicates anomaly
            is_anomaly = (y_pred == -1)
            
            # Get anomaly scores if requested and available
            anomaly_scores = None
            if return_scores:
                if hasattr(self.model, 'decision_function'):
                    # For IsolationForest and OneClassSVM
                    scores = self.model.decision_function(features)
                    # Convert to anomaly score where higher value = more anomalous
                    anomaly_scores = -scores
                elif hasattr(self.model, 'score_samples'):
                    # For LOF
                    scores = self.model.score_samples(features)
                    # Convert to anomaly score where higher value = more anomalous
                    anomaly_scores = -scores
            
            # Create dataframe with results
            df_with_anomalies = df.copy()
            df_with_anomalies['is_anomaly'] = is_anomaly
            
            if anomaly_scores is not None:
                df_with_anomalies['anomaly_score'] = anomaly_scores
            
            # Get anomaly samples
            anomaly_samples = df_with_anomalies[is_anomaly]
            
            return {
                'success': True,
                'message': f'Anomaly detection completed with {algorithm}',
                'anomalies': {
                    'df_with_anomalies': df_with_anomalies,
                    'anomaly_samples': anomaly_samples,
                    'anomaly_count': sum(is_anomaly),
                    'anomaly_percentage': f"{sum(is_anomaly) / len(df_with_anomalies):.1%}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return {
                'success': False,
                'message': f'Error during anomaly detection: {str(e)}',
                'anomalies': None
            }
    
    def get_anomaly_insights(self, df: pd.DataFrame, anomaly_result: Dict) -> List[Dict]:
        """
        Generate insights about detected anomalies by comparing statistical
        properties between anomalous and normal incidents.
        
        Args:
            df: Original incident dataframe
            anomaly_result: Result from detect_anomalies method
            
        Returns:
            List of dictionaries with insights about anomalies
        """
        if not anomaly_result['success'] or anomaly_result['anomalies'] is None:
            return [{
                'type': 'error',
                'message': 'Insufficient data to generate anomaly insights'
            }]
        
        df_with_anomalies = anomaly_result['anomalies']['df_with_anomalies']
        anomaly_samples = anomaly_result['anomalies']['anomaly_samples']
        
        if len(anomaly_samples) == 0:
            return [{
                'type': 'information',
                'message': 'No anomalies detected in the current dataset'
            }]
        
        insights = []
        
        # Get normal samples for comparison
        normal_samples = df_with_anomalies[~df_with_anomalies['is_anomaly']]
        
        # Compare numerical features
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if col in df_with_anomalies.columns and col not in ['is_anomaly', 'anomaly_score']:
                try:
                    anomaly_mean = anomaly_samples[col].mean()
                    normal_mean = normal_samples[col].mean()
                    
                    if anomaly_mean != normal_mean:
                        percent_diff = abs((anomaly_mean - normal_mean) / normal_mean) * 100
                        
                        if percent_diff > 20:  # Only report significant differences
                            comparison = "higher" if anomaly_mean > normal_mean else "lower"
                            insights.append({
                                'type': 'numerical_comparison',
                                'feature': col,
                                'anomaly_mean': anomaly_mean,
                                'normal_mean': normal_mean,
                                'percent_difference': percent_diff,
                                'direction': comparison,
                                'message': (
                                    f"Anomalous incidents have {percent_diff:.1f}% {comparison} "
                                    f"{col} than normal incidents "
                                    f"({anomaly_mean:.2f} vs {normal_mean:.2f})"
                                )
                            })
                except Exception as e:
                    self.logger.warning(f"Error comparing numerical feature {col}: {str(e)}")
        
        # Compare categorical features
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col in df_with_anomalies.columns and col not in ['is_anomaly', 'anomaly_score']:
                try:
                    # Calculate value distributions
                    anomaly_counts = anomaly_samples[col].value_counts(normalize=True)
                    normal_counts = normal_samples[col].value_counts(normalize=True)
                    
                    # Find over-represented values in anomalies
                    for value in anomaly_counts.index:
                        if value in normal_counts:
                            anomaly_freq = anomaly_counts[value]
                            normal_freq = normal_counts[value]
                            
                            if anomaly_freq > 2 * normal_freq and anomaly_freq > 0.2:
                                insights.append({
                                    'type': 'categorical_comparison',
                                    'feature': col,
                                    'value': value,
                                    'anomaly_frequency': f"{anomaly_freq:.1%}",
                                    'normal_frequency': f"{normal_freq:.1%}",
                                    'message': (
                                        f"Value '{value}' for '{col}' appears in {anomaly_freq:.1%} of anomalies "
                                        f"compared to only {normal_freq:.1%} of normal incidents "
                                        f"({anomaly_freq/normal_freq:.1f}x more frequent)"
                                    )
                                })
                except Exception as e:
                    self.logger.warning(f"Error comparing categorical feature {col}: {str(e)}")
        
        # If no specific insights, provide general information
        if not insights:
            insights = [{
                'type': 'general',
                'message': (
                    f"Found {len(anomaly_samples)} anomalies ({anomaly_result['anomalies']['anomaly_percentage']}). "
                    f"These incidents deviate from normal patterns but no simple explanation was found."
                )
            }]
        
        # Add overall summary
        insights.insert(0, {
            'type': 'summary',
            'anomaly_count': len(anomaly_samples),
            'anomaly_percentage': anomaly_result['anomalies']['anomaly_percentage'],
            'message': (
                f"Detected {len(anomaly_samples)} anomalous incidents "
                f"({anomaly_result['anomalies']['anomaly_percentage']} of total)"
            )
        })
        
        return insights
    
    def get_temporal_patterns(self, df: pd.DataFrame, anomaly_result: Dict, 
                             timestamp_col: str) -> List[Dict]:
        """
        Analyze temporal patterns of anomalies to identify when they occur more frequently.
        
        Args:
            df: Original incident dataframe
            anomaly_result: Result from detect_anomalies method
            timestamp_col: Column containing timestamp information
            
        Returns:
            List of dictionaries with temporal insights
        """
        if not anomaly_result['success'] or anomaly_result['anomalies'] is None:
            return [{
                'type': 'error',
                'message': 'Insufficient data to generate temporal patterns'
            }]
        
        if timestamp_col not in df.columns:
            return [{
                'type': 'error',
                'message': f'Timestamp column {timestamp_col} not found in data'
            }]
        
        df_with_anomalies = anomaly_result['anomalies']['df_with_anomalies']
        
        # Ensure timestamp column is datetime type
        try:
            if df_with_anomalies[timestamp_col].dtype != 'datetime64[ns]':
                df_with_anomalies[timestamp_col] = pd.to_datetime(df_with_anomalies[timestamp_col])
        except Exception as e:
            return [{
                'type': 'error',
                'message': f'Could not convert {timestamp_col} to datetime: {str(e)}'
            }]
        
        insights = []
        
        try:
            # Add time-based columns
            df_with_anomalies['hour'] = df_with_anomalies[timestamp_col].dt.hour
            df_with_anomalies['day'] = df_with_anomalies[timestamp_col].dt.day
            df_with_anomalies['weekday'] = df_with_anomalies[timestamp_col].dt.weekday
            df_with_anomalies['month'] = df_with_anomalies[timestamp_col].dt.month
            
            # Calculate anomaly rate by hour of day
            hourly_anomaly_rate = []
            for hour in range(24):
                hour_data = df_with_anomalies[df_with_anomalies['hour'] == hour]
                if len(hour_data) >= 5:  # Only include hours with enough data
                    anomaly_rate = hour_data['is_anomaly'].mean()
                    hourly_anomaly_rate.append({
                        'hour': hour,
                        'anomaly_rate': anomaly_rate,
                        'sample_count': len(hour_data)
                    })
            
            # Find hours with significantly higher anomaly rates
            if hourly_anomaly_rate:
                overall_rate = df_with_anomalies['is_anomaly'].mean()
                for hour_data in sorted(hourly_anomaly_rate, key=lambda x: x['anomaly_rate'], reverse=True):
                    if hour_data['anomaly_rate'] > 2 * overall_rate and hour_data['anomaly_rate'] > 0.1:
                        insights.append({
                            'type': 'temporal_hour',
                            'hour': hour_data['hour'],
                            'anomaly_rate': f"{hour_data['anomaly_rate']:.1%}",
                            'overall_rate': f"{overall_rate:.1%}",
                            'message': (
                                f"Hour {hour_data['hour']}:00 has an anomaly rate of {hour_data['anomaly_rate']:.1%}, "
                                f"which is {hour_data['anomaly_rate']/overall_rate:.1f}x higher than overall"
                            )
                        })
            
            # Calculate anomaly rate by weekday
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_anomaly_rate = []
            for weekday in range(7):
                weekday_data = df_with_anomalies[df_with_anomalies['weekday'] == weekday]
                if len(weekday_data) >= 5:  # Only include weekdays with enough data
                    anomaly_rate = weekday_data['is_anomaly'].mean()
                    weekday_anomaly_rate.append({
                        'weekday': weekday,
                        'weekday_name': weekday_names[weekday],
                        'anomaly_rate': anomaly_rate,
                        'sample_count': len(weekday_data)
                    })
            
            # Find weekdays with significantly higher anomaly rates
            if weekday_anomaly_rate:
                for weekday_data in sorted(weekday_anomaly_rate, key=lambda x: x['anomaly_rate'], reverse=True):
                    if weekday_data['anomaly_rate'] > 1.5 * overall_rate and weekday_data['anomaly_rate'] > 0.1:
                        insights.append({
                            'type': 'temporal_weekday',
                            'weekday': weekday_data['weekday'],
                            'weekday_name': weekday_data['weekday_name'],
                            'anomaly_rate': f"{weekday_data['anomaly_rate']:.1%}",
                            'overall_rate': f"{overall_rate:.1%}",
                            'message': (
                                f"{weekday_data['weekday_name']}s have an anomaly rate of {weekday_data['anomaly_rate']:.1%}, "
                                f"which is {weekday_data['anomaly_rate']/overall_rate:.1f}x higher than overall"
                            )
                        })
            
            # If no specific temporal patterns found
            if not insights:
                insights.append({
                    'type': 'temporal_general',
                    'message': "No significant temporal patterns found in anomaly distribution"
                })
        
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {str(e)}")
            insights.append({
                'type': 'error',
                'message': f'Error analyzing temporal patterns: {str(e)}'
            })
        
        return insights


# Add the detect_anomalies function to be compatible with landing_page.py
def detect_anomalies(time_series_data, date_column, value_column, sensitivity=3.0):
    """
    Detect anomalies in time series data.
    
    Args:
        time_series_data: DataFrame with time series data
        date_column: Column name for dates
        value_column: Column name for values to analyze
        sensitivity: Detection sensitivity (lower means more anomalies)
        
    Returns:
        Boolean Series indicating anomalies (True) and normal points (False)
    """
    try:
        # Basic validation
        if time_series_data is None or time_series_data.empty:
            return None
            
        if date_column not in time_series_data.columns or value_column not in time_series_data.columns:
            return None
            
        # Convert sensitivity to contamination (inverse relationship)
        # Higher sensitivity = lower contamination = fewer anomalies
        contamination = min(0.5, max(0.01, 1.0 / sensitivity))
        
        # Create detector
        detector = IncidentAnomalyDetector(contamination=contamination)
        
        # Prepare features
        features = detector.prepare_features(
            time_series_data,
            numerical_features=[value_column],
            categorical_features=[],
            datetime_features=[date_column]
        )
        
        if features is None:
            return None
            
        # Use IsolationForest for detection
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit and predict
        model.fit(features)
        predictions = model.predict(features)
        
        # Convert to boolean mask (True for anomalies)
        anomalies = (predictions == -1)
        
        return pd.Series(anomalies, index=time_series_data.index)
    
    except Exception as e:
        logging.error(f"Error detecting anomalies: {str(e)}")
        return None


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 300
    
    # Create timestamps with anomalies more frequent on weekends
    base_date = datetime.datetime(2023, 1, 1)
    dates = []
    for i in range(n_samples):
        # Add a random number of hours (0-168, representing a week)
        random_hours = np.random.randint(0, 24*30)  # Up to 30 days
        dates.append(base_date + datetime.timedelta(hours=random_hours))
    
    # Create synthetic data with anomalies
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(n_samples)],
        'created_at': dates,
        'priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], n_samples),
        'category': np.random.choice(['Network', 'Server', 'Application', 'Database'], n_samples),
        'resolution_time': np.random.exponential(5, n_samples)  # Hours to resolve
    }
    
    # Inject anomalies
    # 5% of incidents have very long resolution times
    anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.05), replace=False)
    for idx in anomaly_indices:
        data['resolution_time'][idx] = np.random.uniform(20, 48)  # Much longer resolution time
        # More likely to be P1 for anomalies
        data['priority'][idx] = np.random.choice(['P1', 'P2'], 1)[0]
    
    df = pd.DataFrame(data)
    
    # Test the anomaly detection
    detector = IncidentAnomalyDetector()
    result = detector.detect_anomalies(
        df,
        numerical_features=['resolution_time'],
        categorical_features=['priority', 'category'],
        datetime_features=['created_at']
    )
    
    if result['success']:
        print(f"Anomaly detection completed successfully.")
        insights = detector.get_anomaly_insights(df, result)
        print("\nANOMALY INSIGHTS:")
        for insight in insights:
            print(f"- {insight['message']}")
        
        temporal_insights = detector.get_temporal_patterns(df, result, 'created_at')
        print("\nTEMPORAL PATTERNS:")
        for insight in temporal_insights:
            print(f"- {insight['message']}")
    else:
        print(f"Anomaly detection failed: {result['message']}")