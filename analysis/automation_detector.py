#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automation detector module for the Incident Management Analytics application.
This module identifies potential automation opportunities in incident data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class AutomationDetector:
    """
    Class for identifying potential automation opportunities in incident data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutomationDetector with application configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.analysis_config = config["analysis"]
        self.min_cluster_size = self.analysis_config.get("min_cluster_size", 5)
        self.similarity_threshold = self.analysis_config.get("similarity_threshold", 0.85)
        self.max_suggestions = self.analysis_config.get("max_automation_suggestions", 5)
        
        # Load English stopwords
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords specific to incident management
        self.stop_words.update([
            'incident', 'ticket', 'issue', 'problem', 'error', 'request', 'reported',
            'user', 'customer', 'client', 'system', 'server', 'application', 'service',
            'please', 'help', 'need', 'thanks', 'thank', 'hello', 'hi', 'regards'
        ])
    
    def detect_automation_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential automation opportunities in incident data.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Dictionary with detected automation opportunities
        """
        logger.info("Detecting automation opportunities")
        
        if len(df) < self.min_cluster_size * 2:
            return {
                "success": False,
                "error": f"Insufficient data for automation detection. Need at least {self.min_cluster_size * 2} incidents.",
                "opportunities": []
            }
        
        opportunities = []
        
        # Try different approaches to identify automation opportunities
        try:
            # 1. Look for repetitive incidents based on categories
            category_opportunities = self._detect_categorical_patterns(df)
            opportunities.extend(category_opportunities)
            
            # 2. Look for text-based patterns in descriptions
            text_opportunities = self._detect_text_patterns(df)
            opportunities.extend(text_opportunities)
            
            # 3. Look for resolution-based patterns
            resolution_opportunities = self._detect_resolution_patterns(df)
            opportunities.extend(resolution_opportunities)
            
            # Deduplicate opportunities (might have overlap between approaches)
            opportunities = self._deduplicate_opportunities(opportunities)
            
            # Calculate priority scores and sort
            opportunities = self._prioritize_opportunities(opportunities, df)
            
            # Limit to max suggestions
            opportunities = opportunities[:min(len(opportunities), self.max_suggestions)]
            
            # Check if we found enough opportunities
            if not opportunities:
                return {
                    "success": True,
                    "message": "No clear automation opportunities detected in the data",
                    "opportunities": []
                }
            
            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error detecting automation opportunities: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error detecting automation opportunities: {str(e)}",
                "opportunities": []
            }
    
    def _detect_categorical_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect automation opportunities based on categorical patterns.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            List of automation opportunities
        """
        opportunities = []
        
        # Check if we have category and subcategory columns
        category_columns = []
        for col in ['category', 'subcategory', 'type', 'issue_type']:
            if col in df.columns:
                category_columns.append(col)
        
        if not category_columns:
            logger.info("No category columns found for categorical pattern detection")
            return opportunities
        
        # Analyze patterns in categories
        if len(category_columns) == 1:
            # Single category column
            cat_col = category_columns[0]
            cat_counts = df[cat_col].value_counts()
            
            # Find frequent categories (more than min_cluster_size and at least 5% of incidents)
            threshold = max(self.min_cluster_size, len(df) * 0.05)
            frequent_cats = cat_counts[cat_counts >= threshold]
            
            for cat, count in frequent_cats.items():
                # Check if this category has consistent resolution patterns
                cat_df = df[df[cat_col] == cat]
                
                # Only consider categories with enough incidents
                if len(cat_df) >= self.min_cluster_size:
                    consistency_score = self._calculate_consistency_score(cat_df)
                    
                    if consistency_score >= 0.7:  # Threshold for consistent categories
                        opportunities.append({
                            "type": "category_pattern",
                            "name": f"Automate {cat} incidents",
                            "category": cat,
                            "count": int(count),
                            "percentage": (count / len(df)) * 100,
                            "consistency_score": consistency_score,
                            "description": self._generate_category_description(cat, count, len(df), consistency_score),
                            "evidence": self._gather_category_evidence(cat_df, cat_col, cat),
                            "implementation_complexity": self._estimate_complexity(consistency_score, count)
                        })
        
        elif len(category_columns) >= 2:
            # Multiple category columns - look for combinations
            cat_col1 = category_columns[0]
            cat_col2 = category_columns[1]
            
            # Group by both columns
            cat_combo_counts = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
            cat_combo_counts = cat_combo_counts.sort_values('count', ascending=False)
            
            # Find frequent combinations
            threshold = max(self.min_cluster_size, len(df) * 0.05)
            frequent_combos = cat_combo_counts[cat_combo_counts['count'] >= threshold]
            
            for _, row in frequent_combos.iterrows():
                cat1 = row[cat_col1]
                cat2 = row[cat_col2]
                count = row['count']
                
                # Get incidents with this combination
                combo_df = df[(df[cat_col1] == cat1) & (df[cat_col2] == cat2)]
                
                # Check consistency
                consistency_score = self._calculate_consistency_score(combo_df)
                
                if consistency_score >= 0.7:  # Threshold for consistent categories
                    opportunities.append({
                        "type": "category_combination",
                        "name": f"Automate {cat1} - {cat2} incidents",
                        "primary_category": cat1,
                        "secondary_category": cat2,
                        "count": int(count),
                        "percentage": (count / len(df)) * 100,
                        "consistency_score": consistency_score,
                        "description": self._generate_combo_description(cat1, cat2, count, len(df), consistency_score),
                        "evidence": self._gather_category_evidence(combo_df, cat_col1, cat1, cat_col2, cat2),
                        "implementation_complexity": self._estimate_complexity(consistency_score, count)
                    })
        
        return opportunities
    
    def _detect_text_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect automation opportunities based on text patterns in descriptions.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            List of automation opportunities
        """
        opportunities = []
        
        # Check if we have text columns
        text_columns = []
        for col in ['description', 'summary', 'title', 'details']:
            if col in df.columns and df[col].dtype == 'object':
                # Check if column contains substantive text
                avg_len = df[col].astype(str).apply(len).mean()
                if avg_len >= 20:  # Only use columns with reasonable text length
                    text_columns.append(col)
        
        if not text_columns:
            logger.info("No suitable text columns found for text pattern detection")
            return opportunities
        
        # Use the first suitable text column
        text_col = text_columns[0]
        
        # Process text data
        df['processed_text'] = df[text_col].fillna('').astype(str).apply(self._preprocess_text)
        
        # Filter out empty texts
        text_df = df[df['processed_text'].str.strip() != ''].copy()
        
        if len(text_df) < self.min_cluster_size:
            logger.info(f"Insufficient text data for clustering: {len(text_df)} valid text entries")
            return opportunities
        
        try:
            # Convert text to TF-IDF features
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words=self.stop_words
            )
            
            # Check if we have enough data
            if len(text_df) < 10:
                logger.info(f"Insufficient data for TF-IDF vectorization: {len(text_df)} entries")
                return opportunities
            
            tfidf_matrix = vectorizer.fit_transform(text_df['processed_text'])
            
            # Cluster similar texts
            eps = 1.0 - self.similarity_threshold  # Convert similarity to distance
            dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size, metric='cosine')
            text_df['cluster'] = dbscan.fit_predict(tfidf_matrix)
            
            # Count samples in each cluster
            cluster_counts = text_df['cluster'].value_counts()
            
            # Get clusters with enough samples (excluding noise cluster -1)
            valid_clusters = cluster_counts[(cluster_counts >= self.min_cluster_size) & (cluster_counts.index != -1)]
            
            # Check if we found any clusters
            if len(valid_clusters) == 0:
                logger.info("No significant text clusters found")
                return opportunities
            
            # Analyze each valid cluster
            for cluster_id in valid_clusters.index:
                cluster_df = text_df[text_df['cluster'] == cluster_id]
                cluster_size = len(cluster_df)
                
                # Extract common terms to describe the cluster
                common_terms = self._extract_common_terms(cluster_df['processed_text'])
                
                # Create a descriptive name based on common terms
                if common_terms:
                    top_terms = list(common_terms.keys())[:3]  # Top 3 terms
                    cluster_name = " ".join(top_terms)
                else:
                    cluster_name = f"Text Cluster {cluster_id}"
                
                # Calculate consistency score
                consistency_score = self._calculate_consistency_score(cluster_df)
                
                # Check if resolution is consistent
                has_consistent_resolution = False
                resolution_pattern = None
                if 'resolution_notes' in cluster_df.columns:
                    resolution_terms = self._extract_common_terms(
                        cluster_df['resolution_notes'].fillna('').astype(str)
                    )
                    if resolution_terms:
                        top_res_terms = list(resolution_terms.keys())[:3]
                        resolution_pattern = " ".join(top_res_terms)
                        
                        # Check resolution consistency
                        if len(resolution_terms) >= 2 and max(resolution_terms.values()) >= cluster_size * 0.6:
                            has_consistent_resolution = True
                
                opportunities.append({
                    "type": "text_pattern",
                    "name": f"Automate '{cluster_name}' incidents",
                    "count": int(cluster_size),
                    "percentage": (cluster_size / len(df)) * 100,
                    "consistency_score": consistency_score,
                    "key_terms": top_terms if 'top_terms' in locals() else [],
                    "description": self._generate_text_description(
                        cluster_name, cluster_size, len(df), consistency_score, has_consistent_resolution
                    ),
                    "evidence": self._gather_text_evidence(
                        cluster_df, text_col, common_terms, resolution_pattern
                    ),
                    "implementation_complexity": self._estimate_complexity(
                        consistency_score, cluster_size, has_consistent_resolution
                    )
                })
            
        except Exception as e:
            logger.error(f"Error in text clustering: {str(e)}", exc_info=True)
        
        return opportunities
    
    def _detect_resolution_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect automation opportunities based on resolution patterns.
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            List of automation opportunities
        """
        opportunities = []
        
        # Check if we have resolution notes
        if 'resolution_notes' not in df.columns:
            logger.info("No resolution notes column found for resolution pattern detection")
            return opportunities
        
        # Preprocess resolution notes
        df['processed_resolution'] = df['resolution_notes'].fillna('').astype(str).apply(self._preprocess_text)
        
        # Filter out empty resolutions
        res_df = df[df['processed_resolution'].str.strip() != ''].copy()
        
        if len(res_df) < self.min_cluster_size:
            logger.info(f"Insufficient resolution data: {len(res_df)} valid resolution entries")
            return opportunities
        
        # Look for common resolution phrases
        resolution_phrases = self._extract_resolution_phrases(res_df['processed_resolution'])
        
        # Check each common phrase
        for phrase, count in resolution_phrases.items():
            # Only consider phrases that appear in enough incidents
            if count >= self.min_cluster_size and len(phrase.split()) >= 2:  # Require at least 2 words
                # Get incidents with this resolution phrase
                phrase_df = res_df[res_df['processed_resolution'].str.contains(phrase, case=False, na=False)]
                
                if len(phrase_df) >= self.min_cluster_size:
                    # Check if these incidents have consistent characteristics
                    consistency_score = self._calculate_consistency_score(phrase_df)
                    
                    # Find the most common category if available
                    common_category = None
                    if 'category' in phrase_df.columns:
                        cat_counts = phrase_df['category'].value_counts()
                        if len(cat_counts) > 0:
                            common_category = cat_counts.index[0]
                            cat_percentage = (cat_counts.iloc[0] / len(phrase_df)) * 100
                    
                    # Only add if there's reasonable consistency
                    if consistency_score >= 0.65:
                        opportunities.append({
                            "type": "resolution_pattern",
                            "name": f"Automate '{phrase[:30]}...' resolution",
                            "resolution_phrase": phrase,
                            "count": int(count),
                            "percentage": (count / len(df)) * 100,
                            "consistency_score": consistency_score,
                            "common_category": common_category,
                            "description": self._generate_resolution_description(
                                phrase, count, len(df), consistency_score, common_category
                            ),
                            "evidence": self._gather_resolution_evidence(phrase_df, phrase),
                            "implementation_complexity": "medium"  # Resolution automation typically medium complexity
                        })
        
        return opportunities
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        
        # Rejoin
        return ' '.join(filtered_tokens)
    
    def _extract_common_terms(self, text_series: pd.Series) -> Dict[str, int]:
        """
        Extract common terms from a series of texts.
        
        Args:
            text_series: Series of text strings
            
        Returns:
            Dictionary of terms and their frequencies
        """
        # Combine all texts
        all_text = ' '.join(text_series)
        
        # Count word frequencies
        words = all_text.split()
        word_counts = Counter(words)
        
        # Remove very common and very rare words
        total_words = len(words)
        filtered_counts = {
            word: count for word, count in word_counts.items()
            if count >= 3 and count <= total_words * 0.9 and len(word) > 2
        }
        
        # Get most common terms
        return dict(Counter(filtered_counts).most_common(10))
    
    def _extract_resolution_phrases(self, text_series: pd.Series) -> Dict[str, int]:
        """
        Extract common resolution phrases.
        
        Args:
            text_series: Series of resolution text strings
            
        Returns:
            Dictionary of phrases and their frequencies
        """
        all_phrases = []
        
        # Extract 2-4 word phrases from each text
        for text in text_series:
            words = text.split()
            
            # 2-word phrases
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) >= 5:  # Minimum phrase length
                        all_phrases.append(phrase)
            
            # 3-word phrases
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if len(phrase) >= 8:
                        all_phrases.append(phrase)
            
            # 4-word phrases
            if len(words) >= 4:
                for i in range(len(words) - 3):
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
                    if len(phrase) >= 10:
                        all_phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(all_phrases)
        
        # Filter to most common phrases
        common_phrases = dict(phrase_counts.most_common(20))
        
        return common_phrases
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a consistency score for a group of incidents.
        Higher scores indicate more consistent patterns suitable for automation.
        
        Args:
            df: DataFrame with a group of incidents
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        scores = []
        
        # Check resolution time consistency if available
        if 'resolution_time_hours' in df.columns:
            valid_times = df['resolution_time_hours'].dropna()
            if len(valid_times) >= 3:
                # Calculate coefficient of variation (lower is more consistent)
                cv = valid_times.std() / valid_times.mean() if valid_times.mean() > 0 else 1.0
                # Convert to score (1.0 for perfectly consistent, 0.0 for highly variable)
                res_time_score = max(0, min(1, 1 - (cv / 2)))
                scores.append(res_time_score)
        
        # Check category consistency
        for cat_col in ['category', 'subcategory', 'type']:
            if cat_col in df.columns:
                # Calculate entropy of category distribution
                cat_counts = df[cat_col].value_counts(normalize=True)
                # If one category dominates, high consistency
                cat_score = cat_counts.iloc[0] if len(cat_counts) > 0 else 0.0
                scores.append(cat_score)
        
        # Check assignee consistency
        if 'assignee' in df.columns:
            assignee_counts = df['assignee'].value_counts(normalize=True)
            assignee_score = assignee_counts.iloc[0] if len(assignee_counts) > 0 else 0.0
            scores.append(assignee_score * 0.7)  # Weight less than other factors
        
        # Check priority consistency
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts(normalize=True)
            priority_score = priority_counts.iloc[0] if len(priority_counts) > 0 else 0.0
            scores.append(priority_score * 0.8)
        
        # Resolution consistency
        if 'resolution_notes' in df.columns:
            # Check if resolution notes contain similar phrases
            df['proc_res'] = df['resolution_notes'].fillna('').astype(str).apply(self._preprocess_text)
            common_terms = self._extract_common_terms(df['proc_res'])
            
            # Calculate term frequency in the documents
            if common_terms:
                top_term, top_count = list(common_terms.items())[0]
                term_freq = top_count / len(' '.join(df['proc_res']))
                term_score = min(1.0, term_freq * 20)  # Scale appropriately
                scores.append(term_score)
        
        # Return average score if we have scores, otherwise 0
        return sum(scores) / len(scores) if scores else 0.0
    
    def _deduplicate_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate automation opportunities.
        
        Args:
            opportunities: List of automation opportunities
            
        Returns:
            Deduplicated list of opportunities
        """
        if not opportunities:
            return []
        
        # Group opportunities by type
        grouped = defaultdict(list)
        for opp in opportunities:
            grouped[opp['type']].append(opp)
        
        deduplicated = []
        
        # Process each type of opportunity
        for opp_type, type_opps in grouped.items():
            if opp_type == 'category_pattern' or opp_type == 'category_combination':
                # Use the highest consistency score for each category
                category_map = {}
                for opp in type_opps:
                    key = opp.get('category', '') if 'category' in opp else f"{opp.get('primary_category', '')}-{opp.get('secondary_category', '')}"
                    
                    if key not in category_map or opp['consistency_score'] > category_map[key]['consistency_score']:
                        category_map[key] = opp
                
                deduplicated.extend(list(category_map.values()))
                
            elif opp_type == 'text_pattern':
                # Deduplicate based on overlapping key terms
                term_map = {}
                for opp in type_opps:
                    key_terms = frozenset(opp.get('key_terms', []))
                    
                    # Skip if no key terms
                    if not key_terms:
                        deduplicated.append(opp)
                        continue
                    
                    # Check for overlaps with existing opportunities
                    overlap_found = False
                    for existing_terms, existing_opp in list(term_map.items()):
                        # If significant overlap, keep the one with higher consistency
                        if len(key_terms.intersection(existing_terms)) >= min(2, len(key_terms) // 2):
                            overlap_found = True
                            if opp['consistency_score'] > existing_opp['consistency_score']:
                                term_map[key_terms] = opp
                                del term_map[existing_terms]
                            break
                    
                    if not overlap_found:
                        term_map[key_terms] = opp
                
                deduplicated.extend(list(term_map.values()))
                
            elif opp_type == 'resolution_pattern':
                # Deduplicate based on resolution phrase similarity
                phrase_map = {}
                for opp in type_opps:
                    phrase = opp.get('resolution_phrase', '')
                    
                    # Skip if no phrase
                    if not phrase:
                        deduplicated.append(opp)
                        continue
                    
                    # Check for overlaps with existing opportunities
                    overlap_found = False
                    for existing_phrase, existing_opp in list(phrase_map.items()):
                        # If phrases are similar, keep the one with higher consistency
                        if self._phrases_similar(phrase, existing_phrase):
                            overlap_found = True
                            if opp['consistency_score'] > existing_opp['consistency_score']:
                                phrase_map[phrase] = opp
                                del phrase_map[existing_phrase]
                            break
                    
                    if not overlap_found:
                        phrase_map[phrase] = opp
                
                deduplicated.extend(list(phrase_map.values()))
                
            else:
                # For other types, just add all
                deduplicated.extend(type_opps)
        
        return deduplicated
    
    def _phrases_similar(self, phrase1: str, phrase2: str) -> bool:
        """
        Check if two phrases are similar.
        
        Args:
            phrase1: First phrase
            phrase2: Second phrase
            
        Returns:
            True if phrases are similar, False otherwise
        """
        # Convert to sets of words
        words1 = set(phrase1.lower().split())
        words2 = set(phrase2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= 0.5  # Threshold for similarity
    
    def _prioritize_opportunities(self, opportunities: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prioritize automation opportunities.
        
        Args:
            opportunities: List of automation opportunities
            df: Original DataFrame with incident data
            
        Returns:
            Prioritized list of opportunities
        """
        if not opportunities:
            return []
        
        # Calculate priority score for each opportunity
        for opp in opportunities:
            # Base score from consistency
            score = opp['consistency_score'] * 40  # 0-40 points for consistency
            
            # Add points for frequency
            frequency_score = min(30, opp['percentage'])  # 0-30 points for frequency
            score += frequency_score
            
            # Add points for potential time savings
            if 'count' in opp and 'resolution_time_hours' in df.columns:
                avg_resolution = df['resolution_time_hours'].mean()
                potential_hours_saved = opp['count'] * avg_resolution
                time_score = min(20, potential_hours_saved / 100)  # Scale appropriately
                score += time_score
            
            # Add extra points for resolution patterns (likely easier to implement)
            if opp['type'] == 'resolution_pattern':
                score += 5
            
            # Add extra points for consistent categories
            if (opp['type'] == 'category_pattern' or opp['type'] == 'category_combination') and opp['consistency_score'] > 0.8:
                score += 10
            
            # Store the score
            opp['priority_score'] = round(score, 2)
        
        # Sort by priority score descending
        return sorted(opportunities, key=lambda x: x['priority_score'], reverse=True)
    
    def _generate_category_description(self, category: str, count: int, total: int, consistency: float) -> str:
        """
        Generate a description for a category-based automation opportunity.
        
        Args:
            category: Category name
            count: Number of incidents
            total: Total number of incidents
            consistency: Consistency score
            
        Returns:
            Description string
        """
        percentage = (count / total) * 100
        
        if consistency >= 0.9:
            consistency_text = "extremely consistent"
        elif consistency >= 0.8:
            consistency_text = "highly consistent"
        elif consistency >= 0.7:
            consistency_text = "fairly consistent"
        else:
            consistency_text = "somewhat consistent"
        
        return f"Automate handling of '{category}' incidents which represent {percentage:.1f}% of all tickets and show {consistency_text} patterns. These incidents appear to follow predictable resolution steps that could be automated to save time and improve consistency."
    
    def _generate_combo_description(self, cat1: str, cat2: str, count: int, total: int, consistency: float) -> str:
        """
        Generate a description for a category combination automation opportunity.
        
        Args:
            cat1: Primary category
            cat2: Secondary category
            count: Number of incidents
            total: Total number of incidents
            consistency: Consistency score
            
        Returns:
            Description string
        """
        percentage = (count / total) * 100
        
        if consistency >= 0.9:
            consistency_text = "extremely consistent"
        elif consistency >= 0.8:
            consistency_text = "highly consistent"
        elif consistency >= 0.7:
            consistency_text = "fairly consistent"
        else:
            consistency_text = "somewhat consistent"
        
        return f"Automate handling of incidents categorized as '{cat1} - {cat2}' which represent {percentage:.1f}% of all tickets and show {consistency_text} patterns. These specific combinations of categories appear to follow predictable resolution steps that could be automated."
    
    def _generate_text_description(self, pattern: str, count: int, total: int, consistency: float, consistent_resolution: bool) -> str:
        """
        Generate a description for a text-based automation opportunity.
        
        Args:
            pattern: Text pattern
            count: Number of incidents
            total: Total number of incidents
            consistency: Consistency score
            consistent_resolution: Whether resolution is consistent
            
        Returns:
            Description string
        """
        percentage = (count / total) * 100
        
        if consistency >= 0.9:
            consistency_text = "extremely consistent"
        elif consistency >= 0.8:
            consistency_text = "highly consistent"
        elif consistency >= 0.7:
            consistency_text = "fairly consistent"
        else:
            consistency_text = "somewhat consistent"
        
        resolution_text = ""
        if consistent_resolution:
            resolution_text = " These incidents also show consistent resolution patterns, suggesting they could be successfully automated."
        
        return f"Automate handling of incidents related to '{pattern}' which represent {percentage:.1f}% of all tickets and show {consistency_text} characteristics.{resolution_text} These incidents appear to be candidates for automated response based on text pattern recognition."
    
    def _generate_resolution_description(self, phrase: str, count: int, total: int, consistency: float, common_category: Optional[str]) -> str:
        """
        Generate a description for a resolution-based automation opportunity.
        
        Args:
            phrase: Resolution phrase
            count: Number of incidents
            total: Total number of incidents
            consistency: Consistency score
            common_category: Common category for these incidents, if any
            
        Returns:
            Description string
        """
        percentage = (count / total) * 100
        
        # Limit phrase length in description
        if len(phrase) > 50:
            phrase = phrase[:47] + "..."
        
        category_text = ""
        if common_category:
            category_text = f" Many of these incidents fall under the '{common_category}' category."
        
        return f"Automate resolution for incidents handled with the pattern '{phrase}' which represents {percentage:.1f}% of all ticket resolutions.{category_text} The consistency in resolution approach suggests a standard procedure that could be automated to save time and improve consistency."
    
    def _gather_category_evidence(self, df: pd.DataFrame, cat_col: str, category: str, cat_col2: Optional[str] = None, category2: Optional[str] = None) -> Dict[str, Any]:
        """
        Gather evidence for a category-based automation opportunity.
        
        Args:
            df: DataFrame with incidents in this category
            cat_col: Category column name
            category: Category value
            cat_col2: Optional second category column
            category2: Optional second category value
            
        Returns:
            Dictionary with evidence
        """
        evidence = {
            "count": len(df),
            "category_column": cat_col,
            "category_value": category
        }
        
        if cat_col2 and category2:
            evidence["secondary_category_column"] = cat_col2
            evidence["secondary_category_value"] = category2
        
        # Add resolution time evidence if available
        if 'resolution_time_hours' in df.columns:
            valid_times = df['resolution_time_hours'].dropna()
            if len(valid_times) > 0:
                evidence["resolution_time"] = {
                    "mean_hours": round(valid_times.mean(), 2),
                    "median_hours": round(valid_times.median(), 2),
                    "std_dev": round(valid_times.std(), 2),
                    "coefficient_of_variation": round(valid_times.std() / valid_times.mean(), 2) if valid_times.mean() > 0 else 0
                }
        
        # Add priority distribution if available
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts()
            evidence["priority_distribution"] = {
                str(priority): int(count) for priority, count in priority_counts.items()
            }
        
        # Add common resolution phrases if available
        if 'resolution_notes' in df.columns:
            df['proc_res'] = df['resolution_notes'].fillna('').astype(str).apply(self._preprocess_text)
            common_terms = self._extract_common_terms(df['proc_res'])
            if common_terms:
                evidence["common_resolution_terms"] = {
                    term: int(count) for term, count in common_terms.items()
                }
        
        return evidence
    
    def _gather_text_evidence(self, df: pd.DataFrame, text_col: str, common_terms: Dict[str, int], resolution_pattern: Optional[str]) -> Dict[str, Any]:
        """
        Gather evidence for a text-based automation opportunity.
        
        Args:
            df: DataFrame with incidents in this text pattern
            text_col: Text column name
            common_terms: Dictionary of common terms
            resolution_pattern: Optional resolution pattern
            
        Returns:
            Dictionary with evidence
        """
        evidence = {
            "count": len(df),
            "text_column": text_col,
            "common_terms": {
                term: int(count) for term, count in common_terms.items()
            }
        }
        
        # Add example texts (truncated)
        examples = []
        for text in df[text_col].head(3):
            if pd.notna(text):
                text_str = str(text)
                if len(text_str) > 100:
                    text_str = text_str[:97] + "..."
                examples.append(text_str)
        
        if examples:
            evidence["examples"] = examples
        
        # Add resolution pattern if available
        if resolution_pattern:
            evidence["resolution_pattern"] = resolution_pattern
        
        # Add resolution time evidence if available
        if 'resolution_time_hours' in df.columns:
            valid_times = df['resolution_time_hours'].dropna()
            if len(valid_times) > 0:
                evidence["resolution_time"] = {
                    "mean_hours": round(valid_times.mean(), 2),
                    "median_hours": round(valid_times.median(), 2),
                    "std_dev": round(valid_times.std(), 2)
                }
        
        # Add category distribution if available
        if 'category' in df.columns:
            cat_counts = df['category'].value_counts()
            evidence["category_distribution"] = {
                str(cat): int(count) for cat, count in cat_counts.head(3).items()
            }
        
        return evidence
    
    def _gather_resolution_evidence(self, df: pd.DataFrame, phrase: str) -> Dict[str, Any]:
        """
        Gather evidence for a resolution-based automation opportunity.
        
        Args:
            df: DataFrame with incidents using this resolution
            phrase: Resolution phrase
            
        Returns:
            Dictionary with evidence
        """
        evidence = {
            "count": len(df),
            "resolution_phrase": phrase
        }
        
        # Add category distribution if available
        for cat_col in ['category', 'subcategory', 'type']:
            if cat_col in df.columns:
                cat_counts = df[cat_col].value_counts()
                if not cat_counts.empty:
                    evidence[f"{cat_col}_distribution"] = {
                        str(cat): int(count) for cat, count in cat_counts.head(3).items()
                    }
        
        # Add resolution time evidence if available
        if 'resolution_time_hours' in df.columns:
            valid_times = df['resolution_time_hours'].dropna()
            if len(valid_times) > 0:
                evidence["resolution_time"] = {
                    "mean_hours": round(valid_times.mean(), 2),
                    "median_hours": round(valid_times.median(), 2),
                    "std_dev": round(valid_times.std(), 2)
                }
        
        # Add priority distribution if available
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts()
            evidence["priority_distribution"] = {
                str(priority): int(count) for priority, count in priority_counts.items()
            }
        
        return evidence
    
    def _estimate_complexity(self, consistency_score: float, count: int, consistent_resolution: bool = False) -> str:
        """
        Estimate implementation complexity for an automation opportunity.
        
        Args:
            consistency_score: Consistency score
            count: Number of incidents
            consistent_resolution: Whether resolution is consistent
            
        Returns:
            Complexity level (low, medium, high)
        """
        # Higher consistency typically means lower complexity
        if consistency_score >= 0.9:
            base_complexity = "low"
        elif consistency_score >= 0.75:
            base_complexity = "medium"
        else:
            base_complexity = "high"
        
        # Adjust based on volume - very low volume might be higher complexity relative to value
        if count < 10:
            # Increase complexity by one level
            if base_complexity == "low":
                return "medium"
            else:
                return "high"
        
        # Adjust based on resolution consistency
        if consistent_resolution and base_complexity == "medium":
            return "low"
        
        return base_complexity