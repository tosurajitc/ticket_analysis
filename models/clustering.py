# models/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import logging
from typing import Dict, List, Tuple, Union, Optional

class IncidentClusterAnalyzer:
    """
    Performs clustering analysis on incident data to identify patterns and group similar incidents.
    This helps in identifying common underlying issues that may not be apparent through standard reporting.
    """
    
    def __init__(self, min_samples_for_clustering: int = 30):
        """
        Initialize the clustering analyzer.
        
        Args:
            min_samples_for_clustering: Minimum number of samples required for meaningful clustering
        """
        self.min_samples_for_clustering = min_samples_for_clustering
        self.clustering_model = None
        self.feature_scaler = None
        self.text_vectorizer = None
        self.pca_model = None
        self.logger = logging.getLogger(__name__)
        self.cluster_centers = None
        self.cluster_labels = None
        self.feature_importance = None
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate if the data is sufficient for clustering.
        
        Args:
            df: Incident dataframe
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for clustering")
            return False
            
        if len(df) < self.min_samples_for_clustering:
            self.logger.warning(
                f"Insufficient data for clustering. Got {len(df)} samples, "
                f"need at least {self.min_samples_for_clustering}."
            )
            return False
            
        return True
    
    def prepare_features(self, df: pd.DataFrame, 
                         numerical_features: List[str],
                         categorical_features: List[str],
                         text_features: List[str] = None) -> Optional[np.ndarray]:
        """
        Prepare features for clustering by converting categorical variables to one-hot encoding,
        scaling numerical features, and vectorizing text features.
        
        Args:
            df: Incident dataframe
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            text_features: List of text feature column names
            
        Returns:
            Combined feature matrix or None if preparation fails
        """
        if not self._validate_data(df):
            return None
        
        try:
            # Process numerical features
            numerical_data = None
            if numerical_features and all(col in df.columns for col in numerical_features):
                numerical_df = df[numerical_features].copy()
                numerical_df = numerical_df.fillna(numerical_df.mean())
                self.feature_scaler = StandardScaler()
                numerical_data = self.feature_scaler.fit_transform(numerical_df)
            
            # Process categorical features
            categorical_data = None
            if categorical_features and all(col in df.columns for col in categorical_features):
                categorical_df = pd.get_dummies(df[categorical_features], drop_first=True)
                categorical_data = categorical_df.values
            
            # Process text features
            text_data = None
            if text_features and all(col in df.columns for col in text_features):
                # Combine all text columns into a single text
                combined_text = df[text_features].fillna('').agg(' '.join, axis=1)
                self.text_vectorizer = TfidfVectorizer(
                    max_features=100, 
                    stop_words='english',
                    min_df=3
                )
                text_data = self.text_vectorizer.fit_transform(combined_text).toarray()
            
            # Combine all feature types
            feature_arrays = [arr for arr in [numerical_data, categorical_data, text_data] if arr is not None]
            
            if not feature_arrays:
                self.logger.warning("No valid features available for clustering")
                return None
                
            # Combine all features
            combined_features = np.hstack(feature_arrays)
            
            # Apply PCA for dimension reduction if the feature space is large
            if combined_features.shape[1] > 50:
                self.pca_model = PCA(n_components=min(50, combined_features.shape[0] - 1))
                combined_features = self.pca_model.fit_transform(combined_features)
                
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for clustering: {str(e)}")
            return None
    
    def find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        """
        Find the optimal number of clusters using the elbow method.
        
        Args:
            features: Feature matrix
            max_k: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        if features is None or features.shape[0] < 3:
            return 0
            
        inertia = []
        k_range = range(2, min(max_k + 1, features.shape[0]))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertia.append(kmeans.inertia_)
            except Exception as e:
                self.logger.warning(f"Error calculating inertia for k={k}: {str(e)}")
                continue
        
        if not inertia:
            return 0
            
        # Find the elbow point using the rate of decrease
        rates = np.diff(inertia) / np.array(inertia)[:-1]
        optimal_idx = np.argmin(rates) + 2  # +2 because range started at 2
        return optimal_idx
        
    def perform_clustering(self, df: pd.DataFrame, 
                          numerical_features: List[str],
                          categorical_features: List[str],
                          text_features: List[str] = None,
                          n_clusters: int = None,
                          method: str = 'kmeans') -> Dict:
        """
        Perform clustering on incident data.
        
        Args:
            df: Incident dataframe
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            text_features: List of text feature column names
            n_clusters: Number of clusters (if None, will be determined automatically)
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Dictionary containing clustering results
        """
        features = self.prepare_features(df, numerical_features, categorical_features, text_features)
        
        if features is None or features.shape[0] < 3:
            return {
                'success': False,
                'message': 'Insufficient or invalid data for clustering',
                'clusters': None
            }
        
        try:
            if method == 'kmeans':
                if n_clusters is None:
                    n_clusters = self.find_optimal_k(features)
                    if n_clusters < 2:
                        n_clusters = min(5, features.shape[0] - 1)
                
                self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.cluster_labels = self.clustering_model.fit_predict(features)
                self.cluster_centers = self.clustering_model.cluster_centers_
                
            elif method == 'dbscan':
                self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
                self.cluster_labels = self.clustering_model.fit_predict(features)
                # DBSCAN doesn't have cluster centers
                self.cluster_centers = None
            
            else:
                return {
                    'success': False,
                    'message': f'Unsupported clustering method: {method}',
                    'clusters': None
                }
            
            # Get cluster distribution
            unique_labels = np.unique(self.cluster_labels)
            cluster_counts = {label: np.sum(self.cluster_labels == label) for label in unique_labels}
            
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = self.cluster_labels
            
            # Generate 2D visualization coordinates for plotting
            tsne = TSNE(n_components=2, random_state=42)
            vis_coords = tsne.fit_transform(features)
            
            return {
                'success': True,
                'message': 'Clustering completed successfully',
                'clusters': {
                    'labels': self.cluster_labels,
                    'centers': self.cluster_centers,
                    'distribution': cluster_counts,
                    'df_with_clusters': df_with_clusters,
                    'visualization_coords': vis_coords,
                    'n_clusters': len(unique_labels)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error performing clustering: {str(e)}")
            return {
                'success': False,
                'message': f'Error during clustering: {str(e)}',
                'clusters': None
            }
    
    def get_cluster_insights(self, df: pd.DataFrame, cluster_result: Dict) -> List[Dict]:
        """
        Generate insights for each cluster based on the most common values and statistics.
        
        Args:
            df: Original incident dataframe
            cluster_result: Result from perform_clustering method
            
        Returns:
            List of dictionaries with insights for each cluster
        """
        if not cluster_result['success'] or cluster_result['clusters'] is None:
            return [{
                'cluster': -1, 
                'insight': 'Insufficient data to generate cluster insights'
            }]
        
        df_with_clusters = cluster_result['clusters']['df_with_clusters']
        insights = []
        
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            # Skip noise points (cluster -1) from DBSCAN
            if cluster_id == -1 and len(df_with_clusters['cluster'].unique()) > 1:
                continue
                
            cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            cluster_size = len(cluster_df)
            
            # Skip clusters that are too small
            if cluster_size < 3:
                continue
            
            # Calculate statistics for numerical columns
            numerical_stats = {}
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if col in cluster_df.columns:
                    numerical_stats[col] = {
                        'mean': cluster_df[col].mean(),
                        'median': cluster_df[col].median(),
                        'std': cluster_df[col].std()
                    }
            
            # Find most common values for categorical columns
            categorical_patterns = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if col in cluster_df.columns and col != 'cluster':
                    value_counts = cluster_df[col].value_counts(normalize=True)
                    if not value_counts.empty:
                        top_values = value_counts.nlargest(3)
                        categorical_patterns[col] = {
                            val: f"{count:.1%}" for val, count in top_values.items()
                        }
            
            insights.append({
                'cluster': cluster_id,
                'size': cluster_size,
                'percentage': f"{cluster_size / len(df_with_clusters):.1%}",
                'numerical_stats': numerical_stats,
                'categorical_patterns': categorical_patterns
            })
        
        return insights


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic data
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(n_samples)],
        'priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], n_samples),
        'category': np.random.choice(['Network', 'Server', 'Application', 'Database'], n_samples),
        'resolution_time': np.concatenate([
            np.random.normal(2, 0.5, n_samples // 2),  # One cluster with ~2 hours
            np.random.normal(8, 1.5, n_samples // 2)   # Another cluster with ~8 hours
        ]),
        'description': [
            f"{'Server crash' if i < n_samples // 2 else 'Network outage'} "
            f"affecting {'production' if i % 3 == 0 else 'development'} environment"
            for i in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Test the clustering
    analyzer = IncidentClusterAnalyzer()
    result = analyzer.perform_clustering(
        df,
        numerical_features=['resolution_time'],
        categorical_features=['priority', 'category'],
        text_features=['description']
    )
    
    if result['success']:
        insights = analyzer.get_cluster_insights(df, result)
        print(f"Found {len(insights)} clusters")
        for i, insight in enumerate(insights):
            print(f"\nCluster {insight['cluster']} ({insight['size']} incidents, {insight['percentage']})")
            print("Numerical statistics:")
            for col, stats in insight['numerical_stats'].items():
                print(f"  {col}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
            print("Categorical patterns:")
            for col, patterns in insight['categorical_patterns'].items():
                pattern_str = ", ".join([f"{k}: {v}" for k, v in patterns.items()])
                print(f"  {col}: {pattern_str}")
    else:
        print(f"Clustering failed: {result['message']}")