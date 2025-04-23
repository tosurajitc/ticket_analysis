# Application Metadata
APP_TITLE = "Incident Management Analytics"
APP_DESCRIPTION = """
An AI-powered incident management tool that provides deep insights, 
predictive analytics, and actionable recommendations to optimize 
incident resolution and resource allocation.
"""
VERSION = "1.0.0"

# Data Processing Constants
DATA_SUFFICIENCY_THRESHOLD = 5  # Minimum number of tickets required for meaningful analysis
MAX_TICKET_HISTORY_DAYS = 365  # Maximum historical ticket data to consider

# LLM Configuration
DEFAULT_MODEL_NAME = "deepseek-r1-distill-llama-70b"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_API_TIMEOUT = 30
DEFAULT_CACHE_TTL = 3600

# Supported File Types
SUPPORTED_FILE_TYPES = [
    '.csv', 
    '.xlsx', 
    '.xls', 
    '.json', 
    '.parquet'
]
DEFAULT_CHUNK_SIZE = 1000

# Visualization Constants
DEFAULT_CHART_TYPES = [
    'bar', 
    'line', 
    'pie', 
    'scatter', 
    'heatmap'
]
DEFAULT_THEME = 'light'
DEFAULT_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

# Incident Data Column Configuration
MANDATORY_INCIDENT_COLUMNS = [
    'ticket_id', 
    "incident_id", 
    "created_date"
    'timestamp', 
    'status', 
    'priority'
]
OPTIONAL_INCIDENT_COLUMNS = [
    'category', 
    'subcategory', 
    'assignee', 
    'resolution_time', 
    'impact'
]

# Analysis Dimensions
ANALYSIS_DIMENSIONS = [
    'priority', 
    'category', 
    'assignee', 
    'status', 
    'resolution_time'
]

# Anomaly and Clustering
DEFAULT_MIN_CLUSTER_SIZE = 5
ANOMALY_THRESHOLD = 2.0
SIMILARITY_THRESHOLD = 0.85

# Forecasting
DEFAULT_FORECAST_HORIZON = 14

# Resource Optimization
MAX_AUTOMATION_SUGGESTIONS = 5
ROOT_CAUSE_ANALYSIS_DEPTH = 3

# Logging Configuration
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}
DEFAULT_LOG_LEVEL = 'INFO'

# UI Configuration
DEFAULT_PAGE_SIZE = 10
DEFAULT_CACHE_EXPIRY = 3600