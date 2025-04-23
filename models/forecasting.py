# models/forecasting.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
try:
    # Try the new package name first
    from prophet import Prophet
except ImportError:
    try:
        # Fall back to the old package name
        from fbprophet import Prophet
    except ImportError:
        # If neither is available, create a stub to avoid errors
        import logging
        logging.warning("Prophet package not available. Install with 'pip install prophet' for enhanced forecasting capabilities.")
        
        class Prophet:
            """Stub Prophet class to prevent import errors."""
            def __init__(self, **kwargs):
                pass
                
            def fit(self, *args, **kwargs):
                return self
                
            def make_future_dataframe(self, *args, **kwargs):
                import pandas as pd
                return pd.DataFrame()
                
            def predict(self, *args, **kwargs):
                import pandas as pd
                return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Tuple, Union, Optional
import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class IncidentForecaster:
    """
    Forecasts future incident volumes and patterns based on historical data.
    This helps teams anticipate workload and proactively allocate resources.
    """
    
    def __init__(self, min_historical_points: int = 30):
        """
        Initialize the incident forecaster.
        
        Args:
            min_historical_points: Minimum number of data points required for forecasting
        """
        self.min_historical_points = min_historical_points
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.logger = logging.getLogger(__name__)
        self.fitted = False
        self.forecast_models = {}
        self.seasonality_info = None
        
    def _validate_data(self, df: pd.DataFrame, timestamp_col: str) -> bool:
        """
        Validate if the data is sufficient for forecasting.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for forecasting")
            return False
            
        if len(df) < self.min_historical_points:
            self.logger.warning(
                f"Insufficient historical data for forecasting. Got {len(df)} data points, "
                f"need at least {self.min_historical_points}."
            )
            return False
        
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in data")
            return False
            
        # Check if timestamp column is in datetime format or can be converted
        try:
            if df[timestamp_col].dtype != 'datetime64[ns]':
                pd.to_datetime(df[timestamp_col])
        except:
            self.logger.warning(f"Column '{timestamp_col}' cannot be converted to datetime")
            return False
            
        return True
    
    def prepare_time_series(self, df: pd.DataFrame, 
                           timestamp_col: str, 
                           value_col: str = None, 
                           freq: str = 'D',
                           category_col: str = None) -> Dict:
        """
        Prepare time series data for forecasting by resampling to a regular frequency.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            value_col: Column containing the value to forecast (if None, will count incidents)
            freq: Frequency for resampling ('H': hourly, 'D': daily, 'W': weekly, 'M': monthly)
            category_col: Optional column to segment forecasts by category
            
        Returns:
            Dictionary containing prepared time series data
        """
        if not self._validate_data(df, timestamp_col):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for forecasting',
                'time_series': None
            }
        
        try:
            # Copy dataframe and ensure timestamp is in datetime format
            df_copy = df.copy()
            if df_copy[timestamp_col].dtype != 'datetime64[ns]':
                df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
            
            # Set timestamp as index
            df_copy.set_index(timestamp_col, inplace=True)
            
            result = {
                'success': True,
                'message': 'Time series prepared successfully',
                'time_series': {}
            }
            
            # If category column is provided, prepare time series for each category
            if category_col and category_col in df.columns:
                categories = df[category_col].unique()
                
                for category in categories:
                    category_df = df_copy[df_copy[category_col] == category]
                    
                    if len(category_df) < max(10, self.min_historical_points // 3):
                        # Skip categories with too few data points
                        continue
                    
                    if value_col and value_col in category_df.columns:
                        # Resample using the specified value column
                        ts = category_df[value_col].resample(freq).sum()
                    else:
                        # Count incidents per time period
                        ts = category_df.resample(freq).size()
                    
                    # Filter out missing periods if any
                    ts = ts[ts.notna()]
                    
                    if len(ts) >= self.min_historical_points // 2:
                        result['time_series'][str(category)] = {
                            'data': ts,
                            'name': str(category),
                            'count': len(ts)
                        }
            
            # Always prepare overall time series
            if value_col and value_col in df_copy.columns:
                # Resample using the specified value column
                overall_ts = df_copy[value_col].resample(freq).sum()
            else:
                # Count incidents per time period
                overall_ts = df_copy.resample(freq).size()
            
            # Filter out missing periods if any
            overall_ts = overall_ts[overall_ts.notna()]
            
            result['time_series']['overall'] = {
                'data': overall_ts,
                'name': 'Overall',
                'count': len(overall_ts)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing time series: {str(e)}")
            return {
                'success': False,
                'message': f'Error during time series preparation: {str(e)}',
                'time_series': None
            }
    
    def analyze_seasonality(self, time_series: pd.Series) -> Dict:
        """
        Analyze seasonality patterns in the time series.
        
        Args:
            time_series: Pandas Series with datetime index
            
        Returns:
            Dictionary containing seasonality analysis results
        """
        if time_series is None or len(time_series) < self.min_historical_points:
            return {
                'success': False,
                'message': 'Insufficient data for seasonality analysis',
                'seasonality': None
            }
        
        try:
            # Determine period based on frequency
            inferred_freq = pd.infer_freq(time_series.index)
            
            if inferred_freq is None:
                # Try to infer from difference between timestamps
                avg_interval = (time_series.index[-1] - time_series.index[0]) / len(time_series)
                
                if avg_interval <= pd.Timedelta(hours=1):
                    period = 24  # Hourly data, assume daily seasonality
                    seasonality_type = 'hourly'
                elif avg_interval <= pd.Timedelta(days=1):
                    period = 7  # Daily data, assume weekly seasonality
                    seasonality_type = 'daily'
                elif avg_interval <= pd.Timedelta(days=7):
                    period = 4  # Weekly data, assume monthly seasonality
                    seasonality_type = 'weekly'
                else:
                    period = 12  # Monthly data, assume yearly seasonality
                    seasonality_type = 'monthly'
            else:
                if 'H' in inferred_freq:
                    period = 24  # Hourly data, assume daily seasonality
                    seasonality_type = 'hourly'
                elif 'D' in inferred_freq:
                    period = 7  # Daily data, assume weekly seasonality
                    seasonality_type = 'daily'
                elif 'W' in inferred_freq:
                    period = 4  # Weekly data, assume monthly seasonality
                    seasonality_type = 'weekly'
                else:
                    period = 12  # Monthly data, assume yearly seasonality
                    seasonality_type = 'monthly'
            
            # Only perform decomposition if enough periods are available
            min_periods_required = period * 2
            
            if len(time_series) >= min_periods_required:
                # Decompose the time series
                decomposition = seasonal_decompose(
                    time_series,
                    model='additive',
                    period=period,
                    extrapolate_trend='freq'
                )
                
                # Extract components
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
                # Calculate strength of seasonality
                # Using formula: 1 - Var(residual) / Var(detrended)
                detrended = time_series - trend
                var_detrended = np.var(detrended.dropna())
                var_residual = np.var(residual.dropna())
                
                if var_detrended > 0:
                    seasonality_strength = max(0, min(1, 1 - (var_residual / var_detrended)))
                else:
                    seasonality_strength = 0
                
                # Calculate peak seasonal periods
                seasonal_avg = seasonal.groupby(seasonal.index.hour if seasonality_type == 'hourly' else 
                                                seasonal.index.dayofweek if seasonality_type == 'daily' else
                                                seasonal.index.day if seasonality_type == 'weekly' else
                                                seasonal.index.month).mean()
                
                peak_period = seasonal_avg.idxmax()
                low_period = seasonal_avg.idxmin()
                
                # Map peak period to more human-readable format
                if seasonality_type == 'hourly':
                    peak_time = f"{peak_period}:00"
                    low_time = f"{low_period}:00"
                elif seasonality_type == 'daily':
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    peak_time = days[peak_period]
                    low_time = days[low_period]
                elif seasonality_type == 'weekly':
                    peak_time = f"Day {peak_period} of month"
                    low_time = f"Day {low_period} of month"
                else:
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    peak_time = months[peak_period - 1]
                    low_time = months[low_period - 1]
                
                return {
                    'success': True,
                    'message': 'Seasonality analysis completed successfully',
                    'seasonality': {
                        'type': seasonality_type,
                        'period': period,
                        'strength': seasonality_strength,
                        'peak_period': peak_period,
                        'peak_time': peak_time,
                        'low_period': low_period,
                        'low_time': low_time,
                        'components': {
                            'trend': trend,
                            'seasonal': seasonal,
                            'residual': residual
                        }
                    }
                }
            else:
                return {
                    'success': False,
                    'message': f'Insufficient data for seasonality decomposition (need {min_periods_required} points)',
                    'seasonality': None
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonality: {str(e)}")
            return {
                'success': False,
                'message': f'Error during seasonality analysis: {str(e)}',
                'seasonality': None
            }
    
    def forecast_arima(self, time_series: pd.Series, 
                     forecast_periods: int = 14, 
                     return_conf_int: bool = True) -> Dict:
        """
        Generate forecast using ARIMA model.
        
        Args:
            time_series: Time series data to forecast
            forecast_periods: Number of periods to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Dictionary containing forecast results
        """
        if time_series is None or len(time_series) < self.min_historical_points:
            return {
                'success': False,
                'message': 'Insufficient data for ARIMA forecasting',
                'forecast': None
            }
        
        try:
            # Fit ARIMA model
            # Starting with simple parameters
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(
                start=time_series.index[-1] + (time_series.index[1] - time_series.index[0]),
                periods=forecast_periods,
                freq=pd.infer_freq(time_series.index)
            )
            forecast = pd.Series(forecast, index=forecast_index)
            
            result = {
                'success': True,
                'message': 'ARIMA forecast completed successfully',
                'forecast': {
                    'predicted': forecast,
                    'model': 'ARIMA(1,1,1)'
                }
            }
            
            # Add confidence intervals if requested
            if return_conf_int:
                forecast_obj = model_fit.get_forecast(steps=forecast_periods)
                conf_int = forecast_obj.conf_int()
                result['forecast']['lower_bound'] = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index)
                result['forecast']['upper_bound'] = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index)
            
            # Store model for later use
            self.forecast_models['arima'] = model_fit
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating ARIMA forecast: {str(e)}")
            return {
                'success': False,
                'message': f'Error during ARIMA forecasting: {str(e)}',
                'forecast': None
            }
    
    def forecast_prophet(self, time_series: pd.Series, 
                       forecast_periods: int = 14,
                       include_components: bool = True) -> Dict:
        """
        Generate forecast using Facebook Prophet model.
        
        Args:
            time_series: Time series data to forecast
            forecast_periods: Number of periods to forecast
            include_components: Whether to return seasonal components
            
        Returns:
            Dictionary containing forecast results
        """
        if time_series is None or len(time_series) < self.min_historical_points:
            return {
                'success': False,
                'message': 'Insufficient data for Prophet forecasting',
                'forecast': None
            }
        
        try:
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({'ds': time_series.index, 'y': time_series.values})
            
            # Determine appropriate seasonality based on frequency
            inferred_freq = pd.infer_freq(time_series.index)
            
            # Configure Prophet model
            prophet_params = {}
            if inferred_freq and 'H' in inferred_freq:
                prophet_params['daily_seasonality'] = True
            elif inferred_freq and 'D' in inferred_freq:
                prophet_params['weekly_seasonality'] = True
                prophet_params['yearly_seasonality'] = True
            
            # Create and fit model
            model = Prophet(**prophet_params)
            model.fit(df_prophet)
            
            # Create future dataframe and predict
            future = model.make_future_dataframe(
                periods=forecast_periods,
                freq=inferred_freq or 'D'
            )
            forecast = model.predict(future)
            
            # Extract prediction for future periods
            historical_len = len(time_series)
            forecast_result = forecast.iloc[historical_len:, :]
            
            # Create forecast series with datetime index
            forecast_series = pd.Series(
                forecast_result['yhat'].values,
                index=pd.to_datetime(forecast_result['ds'])
            )
            
            result = {
                'success': True,
                'message': 'Prophet forecast completed successfully',
                'forecast': {
                    'predicted': forecast_series,
                    'model': 'Prophet',
                    'lower_bound': pd.Series(
                        forecast_result['yhat_lower'].values,
                        index=pd.to_datetime(forecast_result['ds'])
                    ),
                    'upper_bound': pd.Series(
                        forecast_result['yhat_upper'].values,
                        index=pd.to_datetime(forecast_result['ds'])
                    )
                }
            }
            
            # Add seasonal components if requested
            if include_components:
                components = {}
                if 'yearly' in forecast.columns:
                    components['yearly'] = pd.Series(
                        forecast_result['yearly'].values,
                        index=pd.to_datetime(forecast_result['ds'])
                    )
                if 'weekly' in forecast.columns:
                    components['weekly'] = pd.Series(
                        forecast_result['weekly'].values,
                        index=pd.to_datetime(forecast_result['ds'])
                    )
                if 'daily' in forecast.columns:
                    components['daily'] = pd.Series(
                        forecast_result['daily'].values,
                        index=pd.to_datetime(forecast_result['ds'])
                    )
                
                result['forecast']['components'] = components
            
            # Store model for later use
            self.forecast_models['prophet'] = model
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating Prophet forecast: {str(e)}")
            return {
                'success': False,
                'message': f'Error during Prophet forecasting: {str(e)}',
                'forecast': None
            }
    
    def get_forecast_insights(self, forecasts: Dict, time_series: pd.Series) -> List[Dict]:
        """
        Generate insights from forecasting results.
        
        Args:
            forecasts: Dictionary of forecast results
            time_series: Original time series data
            
        Returns:
            List of dictionaries with forecast insights
        """
        if not forecasts or 'forecast' not in forecasts or not forecasts['success']:
            return [{
                'type': 'error',
                'message': 'Insufficient data to generate forecast insights'
            }]
        
        try:
            insights = []
            forecast_data = forecasts['forecast']['predicted']
            
            # Calculate mean of historical data
            historical_mean = time_series.mean()
            
            # Calculate mean of forecast
            forecast_mean = forecast_data.mean()
            
            # Calculate percent change
            percent_change = ((forecast_mean - historical_mean) / historical_mean) * 100
            
            # Add trend insight
            if abs(percent_change) > 5:
                trend_direction = "increase" if percent_change > 0 else "decrease"
                insights.append({
                    'type': 'trend',
                    'change_percent': f"{percent_change:.1f}%",
                    'direction': trend_direction,
                    'message': (
                        f"Forecast predicts a {abs(percent_change):.1f}% {trend_direction} in "
                        f"incident volume compared to historical average"
                    )
                })
            else:
                insights.append({
                    'type': 'trend',
                    'change_percent': f"{percent_change:.1f}%",
                    'direction': 'stable',
                    'message': "Forecast predicts stable incident volume with minimal change"
                })
            
            # Find peak periods in forecast
            peak_date = forecast_data.idxmax()
            peak_value = forecast_data.max()
            
            insights.append({
                'type': 'peak',
                'date': peak_date,
                'value': peak_value,
                'message': f"Peak incident volume of {peak_value:.1f} expected on {peak_date.strftime('%Y-%m-%d')}"
            })
            
            # Compare with seasonality if available
            if hasattr(self, 'seasonality_info') and self.seasonality_info is not None:
                seasonality = self.seasonality_info.get('seasonality')
                if seasonality:
                    insights.append({
                        'type': 'seasonality',
                        'peak_time': seasonality['peak_time'],
                        'message': f"Historical peak volume occurs on {seasonality['peak_time']}"
                    })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating forecast insights: {str(e)}")
            return [{
                'type': 'error',
                'message': f'Error generating forecast insights: {str(e)}'
            }]

# Add the generate_forecast function at module level
def generate_forecast(time_series_data, date_column, value_column, forecast_horizon):
    """
    Generate a forecast for incident data.
    
    Args:
        time_series_data: DataFrame with time series data
        date_column: Column name for dates
        value_column: Column name for values to forecast
        forecast_horizon: Number of periods to forecast
        
    Returns:
        DataFrame with forecast results or None if forecasting fails
    """
    try:
        import pandas as pd
        import logging
        
        # Basic validation
        if time_series_data is None or time_series_data.empty:
            return None
            
        if date_column not in time_series_data.columns or value_column not in time_series_data.columns:
            return None
        
        # Create an ARIMA-based forecast if we don't have enough data
        # or as a fallback if Prophet fails
        if len(time_series_data) < 30:
            logging.info("Insufficient data points. Using simple moving average forecast.")
            
            # Use a simple moving average for the forecast
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Ensure timestamp is datetime
            if time_series_data[date_column].dtype != 'datetime64[ns]':
                time_series_data[date_column] = pd.to_datetime(time_series_data[date_column])
                
            # Set timestamp as index
            ts_data = time_series_data.copy()
            ts_data.set_index(date_column, inplace=True)
            
            # Get the series to forecast
            values = ts_data[value_column]
            
            # Fit model
            model = ExponentialSmoothing(
                values,
                trend='add',
                seasonal=None,
                damped=True
            ).fit()
            
            # Generate forecast
            last_date = time_series_data[date_column].max()
            date_range = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            
            # Predict future values
            forecast_values = model.forecast(forecast_horizon)
            
            # Create result dataframe
            result_df = pd.DataFrame({
                'ds': date_range,
                'yhat': forecast_values
            })
            
            # Add confidence intervals (simple method)
            result_df['yhat_lower'] = forecast_values * 0.8
            result_df['yhat_upper'] = forecast_values * 1.2
            
            return result_df
        
        # Try using the IncidentForecaster if available
        try:
            forecaster = IncidentForecaster()
            
            # Prepare time series
            ts_result = forecaster.prepare_time_series(
                time_series_data,
                timestamp_col=date_column,
                value_col=value_column
            )
            
            if not ts_result['success'] or 'overall' not in ts_result['time_series']:
                return None
            
            # Get the overall time series
            time_series = ts_result['time_series']['overall']['data']
            
            # Try various forecasting methods in order of preference
            methods = ['prophet', 'arima']
            forecast_result = None
            
            for method in methods:
                try:
                    if method == 'prophet':
                        forecast_result = forecaster.forecast_prophet(
                            time_series, 
                            forecast_periods=forecast_horizon
                        )
                    elif method == 'arima':
                        forecast_result = forecaster.forecast_arima(
                            time_series, 
                            forecast_periods=forecast_horizon
                        )
                        
                    if forecast_result and forecast_result['success']:
                        break
                except Exception as method_error:
                    logging.warning(f"Error with {method} forecast: {str(method_error)}")
            
            if forecast_result and forecast_result['success']:
                # Format result in a DataFrame
                forecast = forecast_result['forecast']['predicted']
                
                # Create DataFrame with forecast
                result_df = pd.DataFrame({
                    'ds': forecast.index,
                    'yhat': forecast.values
                })
                
                # Add lower and upper bounds if available
                if 'lower_bound' in forecast_result['forecast'] and 'upper_bound' in forecast_result['forecast']:
                    result_df['yhat_lower'] = forecast_result['forecast']['lower_bound'].values
                    result_df['yhat_upper'] = forecast_result['forecast']['upper_bound'].values
                else:
                    # Add basic confidence intervals if not provided
                    result_df['yhat_lower'] = result_df['yhat'] * 0.8
                    result_df['yhat_upper'] = result_df['yhat'] * 1.2
                
                return result_df
            
        except Exception as e:
            logging.warning(f"Error using IncidentForecaster: {str(e)}")
        
        # As a final fallback, use a simple forecasting method
        logging.info("Using fallback forecasting method.")
        
        # Create a date range for the forecast
        last_date = time_series_data[date_column].max()
        date_range = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        # Calculate average value for the forecast
        avg_value = time_series_data[value_column].mean()
        
        # Create result dataframe with flat forecast
        result_df = pd.DataFrame({
            'ds': date_range,
            'yhat': [avg_value] * forecast_horizon
        })
        
        # Add simple confidence intervals
        result_df['yhat_lower'] = avg_value * 0.7
        result_df['yhat_upper'] = avg_value * 1.3
        
        return result_df
    
    except Exception as e:
        logging.error(f"Error generating forecast: {str(e)}")
        return None

# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 365  # One year of daily data
    
    # Create timestamps with seasonal patterns
    base_date = datetime.datetime(2022, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(n_samples)]
    
    # Base incident count with weekly and yearly seasonality
    base_count = 10
    weekly_pattern = np.array([1.2, 1.0, 0.9, 0.9, 1.0, 0.7, 0.6])  # Mon-Sun
    monthly_pattern = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7])  # Jan-Dec
    
    # Generate incident counts
    incident_counts = []
    for i, date in enumerate(dates):
        weekday = date.weekday()
        month = date.month - 1
        
        # Apply weekly and monthly patterns

        count = base_count * weekly_pattern[weekday] * monthly_pattern[month]
        
        # Add noise
        count = max(0, int(count + np.random.normal(0, 1)))
        incident_counts.append(count)
    
    # Create data for multiple incidents per day
    all_dates = []
    categories = []
    
    for i, (date, count) in enumerate(zip(dates, incident_counts)):
        for j in range(count):
            # Add random hour
            incident_time = date + datetime.timedelta(hours=np.random.randint(0, 24))
            all_dates.append(incident_time)
            
            # Assign category (with seasonal patterns)
            if date.month in [12, 1, 2]:  # Winter
                cat_probs = [0.4, 0.3, 0.2, 0.1]  # Network more common in winter
            elif date.month in [6, 7, 8]:  # Summer
                cat_probs = [0.2, 0.2, 0.5, 0.1]  # Application more common in summer
            else:
                cat_probs = [0.25, 0.25, 0.25, 0.25]  # Equal distribution
                
            categories.append(np.random.choice(
                ['Network', 'Server', 'Application', 'Database'],
                p=cat_probs
            ))
    
    # Create synthetic data
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(len(all_dates))],
        'created_at': all_dates,
        'category': categories,
        'priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], len(all_dates)),
        'resolution_time': np.random.exponential(3, len(all_dates))  # Hours to resolve
    }
    
    df = pd.DataFrame(data)
    
    # Test the forecaster
    forecaster = IncidentForecaster()
    
    # Example of using the standalone function
    print("Testing standalone generate_forecast function:")
    time_series_data = df.groupby(pd.Grouper(key='created_at', freq='D')).size().reset_index(name='count')
    
    # Convert to DataFrame with columns
    time_series_df = pd.DataFrame({
        'date': time_series_data.index,
        'count': time_series_data.values
    })
    
    forecast_result = generate_forecast(
        time_series_df,
        date_column='date',
        value_column='count',
        forecast_horizon=30
    )
    
    if forecast_result is not None:
        print(f"Forecast generated successfully for the next {len(forecast_result)} days.")
        print(f"Average forecasted incidents: {forecast_result['yhat'].mean():.2f}")
    else:
        print("Failed to generate forecast.")