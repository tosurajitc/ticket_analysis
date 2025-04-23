# analysis/root_cause_analyzer.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import networkx as nx
import re
import logging
from typing import Dict, List, Tuple, Union, Optional, Set
import datetime

class RootCauseAnalyzer:
    """
    Analyzes incident data to identify underlying root causes and patterns.
    Uses NLP and statistical analysis to discover common themes, correlations,
    and potential systemic issues that may be causing incidents.
    """
    
    def __init__(self, min_samples_required: int = 30, min_term_freq: int = 3):
        """
        Initialize the root cause analyzer.
        
        Args:
            min_samples_required: Minimum number of incidents required for analysis
            min_term_freq: Minimum frequency for terms to be considered significant
        """
        self.min_samples_required = min_samples_required
        self.min_term_freq = min_term_freq
        self.logger = logging.getLogger(__name__)
        self.vectorizers = {}
        self.nlp_models = {}
        self.topic_terms = {}
        self.common_terms = set()


    def sanitize_text_items(self, item):
        """
        Recursively sanitize text items in dictionaries and lists,
        ensuring None values are converted to strings.
        """
        if item is None:
            return "unknown"
        elif isinstance(item, dict):
            return {k: self.sanitize_text_items(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self.sanitize_text_items(i) for i in item]
        elif isinstance(item, tuple):
            return tuple(self.sanitize_text_items(i) for i in item)
        elif isinstance(item, (int, float)):
            return item  # Keep numeric values as is
        else:
            # Convert to string for everything else
            return str(item)




    def _validate_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        Validate if the data is sufficient for root cause analysis.
        
        Args:
            df: Incident dataframe
            required_columns: List of columns required for the analysis
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if df is None or df.empty:
            self.logger.warning("No data provided for root cause analysis")
            return False
            
        if len(df) < self.min_samples_required:
            self.logger.warning(
                f"Insufficient data for root cause analysis. Got {len(df)} incidents, "
                f"need at least {self.min_samples_required}."
            )
            return False
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(
                    f"Missing required columns for root cause analysis: {', '.join(missing_cols)}"
                )
                return False
            
        return True
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP analysis by cleaning and normalizing.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numeric values (often irrelevant for root cause)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_text_features(self, df: pd.DataFrame, 
                             text_columns: List[str],
                             max_features: int = 500) -> Dict:
        """
        Extract and analyze text features from incident descriptions, comments, etc.
        
        Args:
            df: Incident dataframe
            text_columns: Columns containing textual information
            max_features: Maximum number of features to extract
            
        Returns:
            Dictionary containing text analysis results
        """
        if not self._validate_data(df, text_columns):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for text feature extraction',
                'features': None
            }
        
        try:
            # Combine all text columns into a single text field
            df_text = df.copy()
            df_text['combined_text'] = ''
            
            for col in text_columns:
                df_text['combined_text'] += ' ' + df_text[col].fillna('').astype(str)
            
            # Preprocess text
            df_text['processed_text'] = df_text['combined_text'].apply(self.preprocess_text)
            
            # Filter out rows with empty text
            df_text = df_text[df_text['processed_text'].str.strip() != '']
            
            if len(df_text) < self.min_samples_required // 2:
                return {
                    'success': False,
                    'message': 'Insufficient text data after preprocessing',
                    'features': None
                }
            
            # Extract terms using TF-IDF
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=max_features,
                min_df=max(3, len(df_text) // 20),  # Minimum document frequency
                stop_words='english'
            )
            
            tfidf_matrix = self.vectorizers['tfidf'].fit_transform(df_text['processed_text'])
            tfidf_feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            
            # Calculate term importance
            term_importance = {}
            tfidf_sums = tfidf_matrix.sum(axis=0)
            
            for idx, term in enumerate(tfidf_feature_names):
                term_importance[term] = tfidf_sums[0, idx]
            
            # Sort terms by importance
            sorted_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)
            top_terms = [term for term, score in sorted_terms[:min(50, len(sorted_terms))]]
            
            # Store common terms for later analysis
            self.common_terms = set(top_terms)
            
            # Calculate term co-occurrence (which terms appear together)
            self.vectorizers['count'] = CountVectorizer(
                vocabulary=top_terms,
                binary=True  # Only care about presence, not frequency
            )
            
            count_matrix = self.vectorizers['count'].fit_transform(df_text['processed_text'])
            term_cooc = count_matrix.T.dot(count_matrix).toarray()
            
            # Create co-occurrence graph
            cooc_graph = nx.Graph()
            
            for i, term_i in enumerate(top_terms):
                cooc_graph.add_node(term_i, weight=term_importance[term_i])
                
                for j, term_j in enumerate(top_terms):
                    if i < j and term_cooc[i, j] > max(3, len(df_text) // 30):
                        cooc_graph.add_edge(term_i, term_j, weight=term_cooc[i, j])
            
            # Find communities of related terms
            term_communities = []
            try:
                communities = nx.community.greedy_modularity_communities(cooc_graph)
                
                for i, community in enumerate(communities):
                    if len(community) > 1:  # Only include multi-term communities
                        community_terms = list(community)
                        community_weight = sum(term_importance[term] for term in community_terms)
                        
                        term_communities.append({
                            'id': i,
                            'terms': community_terms,
                            'weight': community_weight,
                            'central_term': max(community_terms, key=lambda t: term_importance[t])
                        })
            except:
                # Fallback if community detection fails
                self.logger.warning("Community detection failed, using simple clustering")
                from sklearn.cluster import KMeans
                
                # Convert terms to vectors using their co-occurrence
                term_vectors = []
                for term in top_terms:
                    idx = list(tfidf_feature_names).index(term)
                    term_vectors.append(term_cooc[idx])
                
                # Determine number of clusters
                n_clusters = min(10, len(top_terms) // 5)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(term_vectors)
                    
                    for i in range(n_clusters):
                        cluster_terms = [top_terms[j] for j in range(len(top_terms)) if clusters[j] == i]
                        if cluster_terms:
                            community_weight = sum(term_importance[term] for term in cluster_terms)
                            term_communities.append({
                                'id': i,
                                'terms': cluster_terms,
                                'weight': community_weight,
                                'central_term': max(cluster_terms, key=lambda t: term_importance[t])
                            })
            
            return {
                'success': True,
                'message': 'Text feature extraction completed successfully',
                'features': {
                    'top_terms': [(term, float(score)) for term, score in sorted_terms[:30]],
                    'term_communities': sorted(term_communities, key=lambda x: x['weight'], reverse=True),
                    'tfidf_matrix': tfidf_matrix,
                    'processed_df': df_text,
                    'feature_names': tfidf_feature_names
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting text features: {str(e)}")
            return {
                'success': False,
                'message': f'Error during text feature extraction: {str(e)}',
                'features': None
            }
    
    def extract_topics(self, df: pd.DataFrame, 
                        text_columns: List[str],
                        n_topics: int = 5) -> Dict:
        """
        Extract topics from incident text using topic modeling with enhanced compatibility.
        
        Args:
            df: Incident dataframe
            text_columns: Columns containing textual information
            n_topics: Number of topics to extract
            
        Returns:
            Dictionary containing topic modeling results
        """
        # First extract text features
        feature_result = self.extract_text_features(df, text_columns)
        
        if not feature_result.get('success', False):
            return {
                'success': False,
                'message': feature_result.get('message', 'Feature extraction failed'),
                'topics': None
            }
        
        try:
            features = feature_result.get('features')
            if not features:
                return {
                    'success': False,
                    'message': 'No features extracted',
                    'topics': None
                }
                
            tfidf_matrix = features.get('tfidf_matrix')
            feature_names = features.get('feature_names')
            df_text = features.get('processed_df')
            
            if tfidf_matrix is None or feature_names is None or df_text is None or df_text.empty:
                return {
                    'success': False,
                    'message': 'Missing required feature components',
                    'topics': None
                }
            
            # Ensure we have enough data for the requested number of topics
            actual_n_topics = min(n_topics, max(2, len(df_text) // 20))
            
            # Import NMF with version-aware initialization
            from sklearn.decomposition import NMF, LatentDirichletAllocation
            import inspect

            # Prepare potential NMF initialization parameters
            nmf_param_sets = [
                # Basic initialization
                {"n_components": actual_n_topics, "random_state": 42},
                
                # Versions with different parameter support
                {"n_components": actual_n_topics, "random_state": 42, "init": "nndsvd"},
                {"n_components": actual_n_topics, "random_state": 42, "solver": "cd"},
                {"n_components": actual_n_topics, "random_state": 42, "max_iter": 200},
            ]
            
            # Sparse matrix support for older scikit-learn versions
            if hasattr(tfidf_matrix, 'tocsr'):
                tfidf_matrix = tfidf_matrix.tocsr()
            
            # Try NMF initialization with multiple parameter sets
            nmf_model = None
            for params in nmf_param_sets:
                try:
                    # Filter parameters to those supported by the current NMF version
                    supported_params = {
                        k: v for k, v in params.items() 
                        if k in inspect.signature(NMF.__init__).parameters
                    }
                    
                    # Create NMF model with supported parameters
                    try:
                        nmf_model = NMF(**supported_params)
                        
                        # Attempt to fit the model
                        try:
                            nmf_model.fit(tfidf_matrix)
                            break  # Successful initialization and fitting
                        except Exception as fit_error:
                            logger.warning(f"NMF fit failed with params {supported_params}: {str(fit_error)}")
                            nmf_model = None
                            continue
                    
                    except TypeError as init_error:
                        logger.warning(f"NMF initialization failed with params {supported_params}: {str(init_error)}")
                        continue
                
                except Exception as e:
                    logger.warning(f"Unexpected error in NMF initialization: {str(e)}")
                    continue
            
            # Fallback to LDA if NMF fails completely
            if nmf_model is None:
                logger.warning("NMF initialization failed, falling back to LDA")
                nmf_model = LatentDirichletAllocation(
                    n_components=actual_n_topics,
                    random_state=42,
                    max_iter=10
                )
                
                # Fit LDA model with fallback error handling
                try:
                    nmf_model.fit(tfidf_matrix)
                except Exception as lda_error:
                    logger.error(f"Fallback LDA model failed: {str(lda_error)}")
                    return {
                        'success': False,
                        'message': f'Topic modeling failed: {str(lda_error)}',
                        'topics': None
                    }
            
            # Get top terms for each topic
            topic_terms = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                # Ensure we don't exceed the feature count
                max_features = min(10, len(feature_names))
                top_features_idx = topic.argsort()[:-max_features-1:-1]
                
                # Safely extract features
                top_features = []
                weights = []
                for i in top_features_idx:
                    if i < len(feature_names):
                        feature = feature_names[i]
                        if feature is not None:  # Add None check
                            top_features.append(str(feature))  # Convert to string
                            weights.append(float(topic[i]))  # Convert to float for serialization
                
                # Skip if we couldn't extract features
                if not top_features:
                    continue
                    
                topic_terms.append({
                    'id': topic_idx,
                    'terms': top_features,
                    'weights': weights,
                    'representative_term': top_features[0] if top_features else "unknown",
                    'total_weight': float(sum(weights)) if weights else 0.0  # Convert to float for serialization
                })
            
            # Store topic terms for later use
            self.topic_terms = {topic['id']: topic for topic in topic_terms}
            
            # Safely transform documents to get topic distributions
            doc_topic_matrix = None
            try:
                doc_topic_matrix = nmf_model.transform(tfidf_matrix)
            except Exception as transform_error:
                logger.error(f"Error transforming documents: {str(transform_error)}")
                # Create a fallback matrix
                doc_topic_matrix = np.zeros((tfidf_matrix.shape[0], len(topic_terms)))
            
            # Only proceed if we have valid results
            if doc_topic_matrix is None or len(topic_terms) == 0:
                return {
                    'success': False,
                    'message': "Topic modeling failed to produce valid results",
                    'topics': None
                }
            
            # Safely assign dominant topic to each document
            dominant_topics = np.argmax(doc_topic_matrix, axis=1) if doc_topic_matrix is not None else []
            df_text['dominant_topic'] = [
                t if 0 <= t < len(topic_terms) else 0  # Avoid index out of bounds
                for t in dominant_topics
            ]
            
            # Count documents by dominant topic
            topic_counts = df_text['dominant_topic'].value_counts().to_dict()
            
            # Add document counts to topic info
            for topic in topic_terms:
                topic_id = topic['id']
                topic['document_count'] = int(topic_counts.get(topic_id, 0))  # Convert to int for serialization
                if len(df_text) > 0:
                    topic['document_percentage'] = f"{topic['document_count'] / len(df_text) * 100:.1f}%"
                else:
                    topic['document_percentage'] = "0.0%"
            
            # Get representative documents for each topic (similar to previous implementation)
            representative_docs = {}
            
            # Sort topics by document count
            topic_terms = sorted(topic_terms, key=lambda x: x.get('document_count', 0), reverse=True)
            
            # Compile final result with safety checks
            result = {
                'success': True,
                'message': 'Topic extraction completed successfully',
                'topics': {
                    'topic_list': topic_terms,
                    'doc_topic_matrix': doc_topic_matrix.tolist() if isinstance(doc_topic_matrix, np.ndarray) else None,
                    'representative_docs': representative_docs,
                    'df_with_topics': df_text[['dominant_topic']].to_dict() if not df_text.empty else {}
                }
            }
            
            # Sanitize the entire result to ensure no None values
            return self.sanitize_text_items(result)
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Error during topic extraction: {str(e)}',
                'topics': None
            }
        
    def find_correlations(self, df: pd.DataFrame, 
                        target_col: str, 
                        feature_cols: List[str] = None) -> Dict:
        """
        Find correlations between the target column and other features.
        
        Args:
            df: Incident dataframe
            target_col: Column to analyze correlations for (e.g., resolution_time)
            feature_cols: Columns to check for correlation with target
            
        Returns:
            Dictionary containing correlation analysis results
        """
        if not self._validate_data(df, [target_col]):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for correlation analysis',
                'correlations': None
            }
        
        try:
            # If feature columns not specified, use all appropriate columns
            if feature_cols is None:
                # Exclude the target column and non-useful columns
                exclude_cols = [target_col, 'id', 'incident_id', 'description', 'comments']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Check if we have the target column in correct format
            if target_col not in df.columns:
                return {
                    'success': False,
                    'message': f'Target column {target_col} not found in data',
                    'correlations': None
                }
            
            # Ensure target column is numeric
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                try:
                    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                except:
                    return {
                        'success': False,
                        'message': f'Could not convert {target_col} to numeric',
                        'correlations': None
                    }
            
            # Initialize results
            numeric_correlations = []
            categorical_associations = []
            
            # Analyze numeric correlations
            numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                # Calculate correlation matrix
                corr_matrix = df[[target_col] + numeric_cols].corr(method='spearman')
                
                # Extract correlations with target
                for col in numeric_cols:
                    if col != target_col:
                        correlation = corr_matrix.loc[target_col, col]
                        if not pd.isna(correlation) and abs(correlation) > 0.1:
                            numeric_correlations.append({
                                'feature': col,
                                'correlation': round(correlation, 3),
                                'strength': abs(correlation),
                                'direction': 'positive' if correlation > 0 else 'negative'
                            })
            
            # Analyze categorical associations
            categorical_cols = [col for col in feature_cols if col in df.columns and 
                               not pd.api.types.is_numeric_dtype(df[col])]
            
            for col in categorical_cols:
                # Group by category and calculate mean of target
                try:
                    group_means = df.groupby(col)[target_col].agg(['mean', 'count'])
                    
                    # Filter out categories with too few samples
                    group_means = group_means[group_means['count'] >= 5]
                    
                    if len(group_means) >= 2:  # Need at least 2 categories
                        # Calculate overall mean
                        overall_mean = df[target_col].mean()
                        
                        # Calculate max deviation from mean
                        max_dev_category = group_means['mean'].idxmax()
                        min_dev_category = group_means['mean'].idxmin()
                        max_deviation = (group_means['mean'].max() - group_means['mean'].min()) / overall_mean
                        
                        # Only include if substantial deviation
                        if max_deviation > 0.2:
                            categorical_associations.append({
                                'feature': col,
                                'max_deviation': round(max_deviation, 3),
                                'high_category': max_dev_category,
                                'high_value': round(group_means.loc[max_dev_category, 'mean'], 3),
                                'low_category': min_dev_category,
                                'low_value': round(group_means.loc[min_dev_category, 'mean'], 3),
                                'mean_difference': round(group_means['mean'].max() - group_means['mean'].min(), 3)
                            })
                except Exception as e:
                    self.logger.warning(f"Error analyzing categorical feature {col}: {str(e)}")
            
            return {
                'success': True,
                'message': 'Correlation analysis completed successfully',
                'correlations': {
                    'target_column': target_col,
                    'numeric_correlations': sorted(numeric_correlations, key=lambda x: abs(x['correlation']), reverse=True),
                    'categorical_associations': sorted(categorical_associations, key=lambda x: x['max_deviation'], reverse=True)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error finding correlations: {str(e)}")
            return {
                'success': False,
                'message': f'Error during correlation analysis: {str(e)}',
                'correlations': None
            }
    
    def analyze_recurring_patterns(self, df: pd.DataFrame,
                                 timestamp_col: str,
                                 id_col: str = None,
                                 text_columns: List[str] = None,
                                 category_col: str = None) -> Dict:
        """
        Identify recurring patterns in incidents, such as repeated failures,
        cyclic issues, or incidents with common characteristics.
        
        Args:
            df: Incident dataframe
            timestamp_col: Column containing timestamp information
            id_col: Column containing incident IDs
            text_columns: Columns containing textual information
            category_col: Column containing incident categories
            
        Returns:
            Dictionary containing recurring pattern analysis results
        """
        required_cols = [timestamp_col]
        if id_col:
            required_cols.append(id_col)
            
        if not self._validate_data(df, required_cols):
            return {
                'success': False,
                'message': 'Insufficient or invalid data for recurring pattern analysis',
                'patterns': None
            }
        
        try:
            # Ensure timestamp is in datetime format
            if df[timestamp_col].dtype != 'datetime64[ns]':
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Sort by timestamp
            df_sorted = df.sort_values(timestamp_col)
            
            patterns = {
                'temporal': [],
                'textual': [],
                'categorical': [],
                'sequential': []
            }
            
            # Analyze temporal patterns
            # Get incidents per day
            df_sorted['date'] = df_sorted[timestamp_col].dt.date
            daily_counts = df_sorted.groupby('date').size()
            
            # Calculate mean and std of daily incidents
            mean_daily = daily_counts.mean()
            std_daily = daily_counts.std()
            
            # Find days with abnormally high incident counts
            threshold = mean_daily + 2 * std_daily
            high_incident_days = daily_counts[daily_counts > threshold]
            
            if len(high_incident_days) > 0:
                for date, count in high_incident_days.items():
                    patterns['temporal'].append({
                        'type': 'high_volume_day',
                        'date': date,
                        'incident_count': int(count),
                        'expected_count': round(mean_daily, 1),
                        'deviation': f"{(count - mean_daily) / mean_daily * 100:.1f}%"
                    })
            
            # Analyze hourly patterns
            df_sorted['hour'] = df_sorted[timestamp_col].dt.hour
            hourly_counts = df_sorted.groupby('hour').size()
            
            # Find peak hours
            if len(hourly_counts) > 0:
                peak_hour = hourly_counts.idxmax()
                peak_count = hourly_counts.max()
                hour_mean = hourly_counts.mean()
                
                if peak_count > hour_mean * 1.5:
                    patterns['temporal'].append({
                        'type': 'peak_hour',
                        'hour': int(peak_hour),
                        'incident_count': int(peak_count),
                        'expected_count': round(hour_mean, 1),
                        'deviation': f"{(peak_count - hour_mean) / hour_mean * 100:.1f}%"
                    })
            
            # Analyze text similarity for recurring issues (if text columns provided)
            if text_columns and all(col in df.columns for col in text_columns):
                # Use previously extracted text features if available
                if hasattr(self, 'vectorizers') and 'tfidf' in self.vectorizers:
                    vectorizer = self.vectorizers['tfidf']
                else:
                    # Create and fit new vectorizer
                    combined_text = ''
                    for col in text_columns:
                        combined_text += ' ' + df_sorted[col].fillna('').astype(str)
                    
                    processed_text = [self.preprocess_text(text) for text in combined_text]
                    
                    vectorizer = TfidfVectorizer(
                        max_features=200,
                        min_df=3,
                        stop_words='english'
                    )
                    vectorizer.fit(processed_text)
                
                # Group similar incidents
                if id_col:
                    # Create text vectors
                    incident_text = {}
                    for idx, row in df_sorted.iterrows():
                        incident_id = row[id_col]
                        text = ' '.join([str(row[col]) for col in text_columns if col in row and pd.notna(row[col])])
                        incident_text[incident_id] = self.preprocess_text(text)
                    
                    # Convert to vectors
                    text_vectors = {}
                    for incident_id, text in incident_text.items():
                        if text:
                            try:
                                vector = vectorizer.transform([text])[0]
                                text_vectors[incident_id] = vector
                            except:
                                pass
                    
                    # Calculate similarity between incidents
                    similar_groups = []
                    processed_incidents = set()
                    
                    for incident_id, vector in text_vectors.items():
                        if incident_id in processed_incidents:
                            continue
                        
                        # Find similar incidents
                        similar_incidents = []
                        for other_id, other_vector in text_vectors.items():
                            if incident_id != other_id and other_id not in processed_incidents:
                                similarity = cosine_similarity(vector, other_vector)[0][0]
                                if similarity > 0.7:  # High similarity threshold
                                    similar_incidents.append(other_id)
                        
                        # If we found similar incidents, create a group
                        if similar_incidents:
                            group = [incident_id] + similar_incidents
                            similar_groups.append(group)
                            processed_incidents.update(group)
                    
                    # Analyze time patterns in similar incident groups
                    for i, group in enumerate(similar_groups):
                        if len(group) >= 3:  # Only consider groups with at least 3 incidents
                            group_incidents = df_sorted[df_sorted[id_col].isin(group)]
                            
                            # Calculate time between consecutive incidents
                            if len(group_incidents) >= 2:
                                group_incidents = group_incidents.sort_values(timestamp_col)
                                timestamps = group_incidents[timestamp_col].tolist()
                                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                                             for i in range(len(timestamps)-1)]
                                
                                avg_time_diff = sum(time_diffs) / len(time_diffs)
                                
                                # Check if this is a cyclic pattern
                                is_cyclic = False
                                std_time_diff = np.std(time_diffs) if len(time_diffs) > 1 else 0
                                
                                if std_time_diff / avg_time_diff < 0.3 and avg_time_diff < 168:  # Less than a week
                                    is_cyclic = True
                                
                                # Get text sample
                                sample_incident = group_incidents.iloc[0]
                                text_sample = ' '.join([str(sample_incident[col]) 
                                                      for col in text_columns 
                                                      if col in sample_incident and pd.notna(sample_incident[col])])
                                
                                # Extract key terms
                                if text_sample:
                                    top_terms = []
                                    try:
                                        # Use common terms set from earlier analysis
                                        if hasattr(self, 'common_terms') and self.common_terms:
                                            sample_words = set(self.preprocess_text(text_sample).split())
                                            top_terms = list(sample_words.intersection(self.common_terms))[:5]
                                    except:
                                        # Fallback to simple word counting
                                        words = self.preprocess_text(text_sample).split()
                                        word_counts = Counter(words).most_common(5)
                                        top_terms = [word for word, count in word_counts if len(word) > 3]
                                
                                patterns['textual'].append({
                                    'type': 'recurring_issue',
                                    'group_id': i,
                                    'incident_count': len(group),
                                    'key_terms': top_terms[:5],
                                    'is_cyclic': is_cyclic,
                                    'avg_time_between': round(avg_time_diff, 1),
                                    'time_unit': 'hours',
                                    'first_occurrence': timestamps[0],
                                    'last_occurrence': timestamps[-1]
                                })
            
            # Analyze categorical patterns (if category column provided)
            if category_col and category_col in df.columns:
                # Count incidents by category
                category_counts = df_sorted[category_col].value_counts()
                
                # Analyze trends in each category
                for category in category_counts.index:
                    category_df = df_sorted[df_sorted[category_col] == category]
                    
                    # Skip categories with too few incidents
                    if len(category_df) < 10:
                        continue
                    
                    # Check for increasing trend
                    category_df['month'] = category_df[timestamp_col].dt.to_period('M')
                    monthly_counts = category_df.groupby('month').size()
                    
                    if len(monthly_counts) >= 3:  # Need at least 3 months
                        first_half = monthly_counts[:len(monthly_counts)//2].mean()
                        second_half = monthly_counts[len(monthly_counts)//2:].mean()
                        
                        # Calculate trend
                        trend_percent = 0
                        if first_half > 0:
                            trend_percent = (second_half - first_half) / first_half * 100
                        
                        # Only report significant trends
                        if abs(trend_percent) > 20:
                            trend_direction = "increasing" if trend_percent > 0 else "decreasing"
                            patterns['categorical'].append({
                                'type': 'category_trend',
                                'category': category,
                                'direction': trend_direction,
                                'change_percentage': f"{abs(trend_percent):.1f}%",
                                'first_period_avg': round(first_half, 1),
                                'last_period_avg': round(second_half, 1)
                            })
            
            # If no patterns were found in any category
            for pattern_type, pattern_list in patterns.items():
                if not pattern_list:
                    patterns[pattern_type].append({
                        'type': 'no_pattern',
                        'message': f"No significant {pattern_type} patterns detected in the current dataset"
                    })
            
            return {
                'success': True,
                'message': 'Recurring pattern analysis completed successfully',
                'patterns': patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing recurring patterns: {str(e)}")
            return {
                'success': False,
                'message': f'Error during recurring pattern analysis: {str(e)}',
                'patterns': None
            }
    
    def get_root_cause_insights(self, topics_result: Dict = None, 
                            correlations_result: Dict = None, 
                            patterns_result: Dict = None) -> List[Dict]:
        """
        Generate insights about potential root causes from analysis results.
        
        Args:
            topics_result: Result from extract_topics method
            correlations_result: Result from find_correlations method
            patterns_result: Result from analyze_recurring_patterns method
            
        Returns:
            List of dictionaries with root cause insights
        """
        insights = []
        
        # Check if we have sufficient results for insights
        if not (topics_result and topics_result.get('success', False) or 
                correlations_result and correlations_result.get('success', False) or 
                patterns_result and patterns_result.get('success', False)):
            return [{
                'type': 'error',
                'message': 'Insufficient data to generate root cause insights'
            }]
        
        try:
            # Generate topic-based insights
            if topics_result and topics_result.get('success', False) and topics_result.get('topics'):
                topics = topics_result['topics'].get('topic_list', [])
                
                # Get top topics by document count
                for topic in topics[:min(3, len(topics))]:
                    # Ensure topic has required fields and they are not None
                    if 'terms' not in topic or not topic['terms']:
                        continue
                    
                    # Sanitize the terms to ensure they're all strings
                    sanitized_terms = []
                    for term in topic['terms'][:5]:
                        if term is not None:
                            sanitized_terms.append(str(term))
                        else:
                            sanitized_terms.append("unknown")
                    
                    # Only proceed if we have terms to display
                    if not sanitized_terms:
                        continue
                        
                    topic_terms = ', '.join(sanitized_terms)
                    doc_count = topic.get('document_count', 0)
                    doc_percentage = topic.get('document_percentage', '0%')
                    
                    insights.append({
                        'type': 'topic',
                        'data': {
                            'topic_id': topic.get('id', -1),
                            'top_terms': sanitized_terms,
                            'document_count': doc_count,
                            'document_percentage': doc_percentage
                        },
                        'message': f"Common root cause pattern: '{topic_terms}' appears in {doc_percentage} of incidents ({doc_count} total)"
                    })
            
            # Generate correlation-based insights
            if correlations_result and correlations_result.get('success', False) and correlations_result.get('correlations'):
                correlations = correlations_result['correlations']
                
                # Numeric correlations
                if 'numeric_correlations' in correlations and correlations['numeric_correlations']:
                    for corr in correlations['numeric_correlations'][:min(3, len(correlations['numeric_correlations']))]:
                        direction = "increases" if corr.get('direction', '') == 'positive' else "decreases"
                        target = correlations.get('target_column', 'value')
                        
                        # Ensure feature name is a string
                        feature = str(corr.get('feature', 'unknown feature'))
                        correlation_value = str(corr.get('correlation', 'unknown correlation'))
                        
                        insights.append({
                            'type': 'correlation',
                            'subtype': 'numeric',
                            'data': self.sanitize_text_items(corr),
                            'message': f"When '{feature}' increases, '{target}' {direction} (correlation: {correlation_value})"
                        })
                
                # Categorical associations
                if 'categorical_associations' in correlations and correlations['categorical_associations']:
                    for assoc in correlations['categorical_associations'][:min(3, len(correlations['categorical_associations']))]:
                        target = correlations.get('target_column', 'value')
                        
                        # Ensure all values are strings
                        feature = str(assoc.get('feature', 'unknown feature'))
                        high_category = str(assoc.get('high_category', 'unknown high'))
                        low_category = str(assoc.get('low_category', 'unknown low'))
                        
                        # Handle potential None or non-numeric max_deviation
                        max_deviation = assoc.get('max_deviation', 0)
                        if max_deviation is None:
                            max_deviation = 0
                        
                        insights.append({
                            'type': 'correlation',
                            'subtype': 'categorical',
                            'data': self.sanitize_text_items(assoc),
                            'message': f"'{target}' is {max_deviation * 100:.1f}% higher for '{feature}={high_category}' than '{feature}={low_category}'"
                        })
            
            # Generate pattern-based insights
            if patterns_result and patterns_result.get('success', False) and patterns_result.get('patterns'):
                patterns = patterns_result['patterns']
                
                # Recurring textual patterns
                if 'textual' in patterns and patterns['textual']:
                    for pattern in patterns['textual']:
                        if pattern.get('type') != 'no_pattern':
                            cyclic_text = "occurring cyclically" if pattern.get('is_cyclic') else "recurring sporadically"
                            
                            # Sanitize key_terms
                            key_terms = pattern.get('key_terms', [])
                            if key_terms:
                                sanitized_terms = []
                                for term in key_terms[:3]:
                                    if term is not None:
                                        sanitized_terms.append(str(term))
                                    else:
                                        sanitized_terms.append("unknown")
                                
                                terms = ', '.join(sanitized_terms)
                                
                                # Get incident count, defaulting to 0 if missing
                                incident_count = pattern.get('incident_count', 0)
                                if incident_count is None:
                                    incident_count = 0
                                
                                insights.append({
                                    'type': 'pattern',
                                    'subtype': 'recurring',
                                    'data': self.sanitize_text_items(pattern),
                                    'message': f"Identified recurring issue with terms '{terms}' {cyclic_text}, appearing {incident_count} times"
                                })
                
                # Temporal patterns
                if 'temporal' in patterns and patterns['temporal']:
                    for pattern in patterns['temporal']:
                        if pattern.get('type') == 'high_volume_day':
                            date_val = pattern.get('date')
                            if date_val is not None:
                                date_str = str(date_val)
                            else:
                                date_str = "unknown date"
                                
                            incident_count = pattern.get('incident_count', 0)
                            if incident_count is None:
                                incident_count = 0
                                
                            deviation = pattern.get('deviation', '0%')
                            if deviation is None:
                                deviation = "0%"
                                
                            insights.append({
                                'type': 'pattern',
                                'subtype': 'temporal',
                                'data': self.sanitize_text_items(pattern),
                                'message': f"Abnormal incident spike on {date_str} with {incident_count} incidents ({deviation} above normal)"
                            })
                        elif pattern.get('type') == 'peak_hour':
                            hour = pattern.get('hour', 0)
                            if hour is None:
                                hour = 0
                                
                            deviation = pattern.get('deviation', '0%')
                            if deviation is None:
                                deviation = "0%"
                                
                            insights.append({
                                'type': 'pattern',
                                'subtype': 'temporal',
                                'data': self.sanitize_text_items(pattern),
                                'message': f"Incidents consistently peak at {hour}:00 with {deviation} higher volume than average"
                            })
                
                # Category trends
                if 'categorical' in patterns and patterns['categorical']:
                    for pattern in patterns['categorical']:
                        if pattern.get('type') == 'category_trend':
                            category = str(pattern.get('category', 'unknown category'))
                            direction = str(pattern.get('direction', 'unknown direction'))
                            change_percentage = str(pattern.get('change_percentage', '0%'))
                            
                            insights.append({
                                'type': 'pattern',
                                'subtype': 'trend',
                                'data': self.sanitize_text_items(pattern),
                                'message': f"'{category}' incidents are {direction} by {change_percentage} over time"
                            })
            
            # If no insights were generated
            if not insights:
                insights.append({
                    'type': 'general',
                    'message': "No clear root causes identified in the current dataset"
                })
            
            # Final sanitization of all insights
            sanitized_insights = self.sanitize_text_items(insights)
            return sanitized_insights
            
        except Exception as e:
            self.logger.error(f"Error generating root cause insights: {str(e)}")
            return [{
                'type': 'error',
                'message': f'Error generating root cause insights: {str(e)}'
            }]


# If this module is run directly, it can be tested
if __name__ == "__main__":
    # Simple test code
    import pandas as pd
    import numpy as np
    
    # Create synthetic incident data
    np.random.seed(42)
    n_samples = 200
    
    # Create timestamps
    base_date = datetime.datetime(2023, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(n_samples)]
    
    # Create synthetic causes with specific patterns
    causes = [
        "Network timeout due to high traffic volume",
        "Database connection failure during backup",
        "Memory allocation error in application server",
        "API rate limit exceeded during peak hours",
        "Certificate expiration causing secure connections to fail"
    ]
    
    # Create descriptions with specific root cause patterns
    descriptions = []
    for i in range(n_samples):
        # Create cyclical patterns
        if i % 30 < 5:  # First 5 days of each month
            base_cause = causes[0]
            category = "Network"
            priority = np.random.choice(["P1", "P2"], p=[0.7, 0.3])
            resolution_time = np.random.normal(8, 2)
        elif i % 14 < 2:  # Every other week
            base_cause = causes[1]
            category = "Database"
            priority = np.random.choice(["P1", "P2", "P3"], p=[0.2, 0.5, 0.3])
            resolution_time = np.random.normal(6, 1.5)
        elif i % 7 == 5:  # Every Saturday
            base_cause = causes[2]
            category = "Application"
            priority = np.random.choice(["P2", "P3"], p=[0.6, 0.4])
            resolution_time = np.random.normal(4, 1)
        else:
            base_cause = np.random.choice(causes[3:])
            category = np.random.choice(["Network", "Database", "Application", "Security"])
            priority = np.random.choice(["P1", "P2", "P3", "P4"])
            resolution_time = np.random.normal(3, 1)
        
        # Add some variation to the description
        variations = [
            "User reported ",
            "Alert triggered for ",
            "Monitoring system detected ",
            "Customer complaint about ",
            "Automated test failed due to "
        ]
        
        prefix = np.random.choice(variations)
        suffix = f" on {category.lower()} system {i % 10 + 1}"
        
        descriptions.append(prefix + base_cause + suffix)
    
    # Create synthetic data
    data = {
        'incident_id': [f'INC{i:05d}' for i in range(n_samples)],
        'created_at': dates,
        'description': descriptions,
        'category': [d.split()[0] for d in descriptions],  # Extract first word as category
        'priority': priority,
        'resolution_time': abs(resolution_time),  # Ensure positive
        'resolved_by': np.random.choice(['Team1', 'Team2', 'Team3'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create comments with additional information
    df['comments'] = df.apply(
        lambda row: f"Issue investigation shows {row['category']} problem related to {' '.join(row['description'].split()[-5:])}", 
        axis=1
    )
    
    # Test the root cause analyzer
    analyzer = RootCauseAnalyzer()
    
    # Extract topics
    topics_result = analyzer.extract_topics(
        df,
        text_columns=['description', 'comments']
    )
    
    # Find correlations
    correlations_result = analyzer.find_correlations(
        df,
        target_col='resolution_time',
        feature_cols=['category', 'priority']
    )
    
    # Analyze recurring patterns
    patterns_result = analyzer.analyze_recurring_patterns(
        df,
        timestamp_col='created_at',
        id_col='incident_id',
        text_columns=['description', 'comments'],
        category_col='category'
    )
    
    # Generate insights
    if topics_result['success']:
        print("ROOT CAUSE INSIGHTS:")
        insights = analyzer.get_root_cause_insights(
            topics_result,
            correlations_result,
            patterns_result
        )
        
        for insight in insights:
            print(f"- {insight['message']}")
    else:
        print(f"Root cause analysis failed: {topics_result['message']}")                                       