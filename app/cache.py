"""
Semantic Cache Implementation

A custom semantic caching system that recognizes semantically similar queries
even when phrased differently. Built from first principles without Redis or
dedicated caching middleware.

Key Design Decisions:
--------------------
1. Cluster-Aware Indexing: 
   - Cache entries are partitioned by cluster membership
   - Lookup only checks entries in relevant clusters
   - Reduces search space from O(n) to O(k·m) where k=clusters, m=entries per cluster
   
2. Similarity Threshold:
   - The CRITICAL tunable parameter (default: 0.85)
   - Higher values = more strict matching, fewer false positives
   - Lower values = more aggressive caching, higher hit rate
   - 0.85 empirically balances precision and recall
   
3. Soft Cluster Membership:
   - Queries can belong to multiple clusters (with probabilities)
   - Checks all clusters where query has >0.1 probability
   - Captures semantic ambiguity naturally
   
4. Data Structure:
   - Dict[int, List[CacheEntry]] - cluster ID to entries mapping
   - Each entry stores: query, embedding, result, cluster distribution
   - Linear search within cluster (acceptable for moderate cache sizes)
   
5. Eviction Policy:
   - Currently none (unbounded growth)
   - In production: add LRU eviction or size-based limits
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pickle


@dataclass
class CacheEntry:
    """
    Represents a single cached query-result pair.
    
    Attributes:
        query: Original query string
        query_embedding: Dense vector representation (from sentence-transformer)
        result: Computed search results
        cluster_probs: Soft cluster assignments (cluster_id -> probability)
        dominant_cluster: Primary cluster (argmax of cluster_probs)
        timestamp: When this entry was cached
        hit_count: Number of times this entry was returned
    """
    query: str
    query_embedding: np.ndarray
    result: Dict[str, Any]
    cluster_probs: Dict[int, float]
    dominant_cluster: int
    timestamp: datetime = field(default_factory=datetime.now)
    hit_count: int = 0


class SemanticCache:
    """
    Cluster-aware semantic cache for query results.
    
    This cache recognizes that "recent ML advances" and "latest machine learning 
    developments" are semantically equivalent, even though they're phrased differently.
    
    The cluster structure from Part 2 does real work here:
    - When a query comes in, we only check cached entries in relevant clusters
    - This dramatically reduces the number of similarity computations
    - For a cache with 1000 entries across 25 clusters, we check ~40 instead of 1000
    
    Critical Parameter Analysis:
    ---------------------------
    similarity_threshold explores the precision/recall trade-off:
    
    - 0.95: Very strict
      * Pros: High precision, returns only very similar queries
      * Cons: Low hit rate, cache underutilized
      * Use case: When exactness is critical
      
    - 0.85: Balanced (RECOMMENDED)
      * Pros: Good hit rate while maintaining semantic relevance
      * Cons: Occasional false positives (rare)
      * Use case: General semantic search
      
    - 0.75: Aggressive
      * Pros: High hit rate, maximum cache utility
      * Cons: May return results for tangentially related queries
      * Use case: When speed matters more than precision
      
    The "interesting question" from the spec is not which performs best,
    but what each reveals about semantic similarity distributions in the corpus.
    Lower thresholds expose the long tail of related-but-distinct queries.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, min_cluster_prob: float = 0.1):
        """
        Initialize the semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for cache hits (0-1)
                                 This is THE critical tunable parameter.
            min_cluster_prob: Minimum cluster probability to check (0-1)
                             Queries must have >this probability to check that cluster
        """
        # Cluster-partitioned cache: cluster_id -> list of cache entries
        self._cache: Dict[int, List[CacheEntry]] = {}
        
        # Statistics tracking
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._similarity_scores_on_hit: List[float] = []
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.min_cluster_prob = min_cluster_prob
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Why cosine similarity?
        - Invariant to magnitude (focuses on direction)
        - Natural metric for sentence embeddings
        - Fast to compute (single dot product + norms)
        - Range [0, 1] for normalized vectors (easier to interpret)
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score in [0, 1] for normalized vectors
        """
        # Normalize vectors to unit length
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Cosine similarity via dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clamp to [0, 1] range (handles numerical errors)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def get(
        self, 
        query: str, 
        query_embedding: np.ndarray, 
        cluster_probs: Dict[int, float]
    ) -> Optional[Tuple[Dict[str, Any], str, float, int]]:
        """
        Attempt to retrieve a cached result for a semantically similar query.
        
        Cache Lookup Algorithm:
        1. Identify relevant clusters (probability > min_cluster_prob)
        2. For each relevant cluster:
           a. Retrieve all cached entries in that cluster
           b. Compute similarity to each cached entry
           c. Track the best match across all clusters
        3. If best match exceeds threshold, return it (HIT)
        4. Otherwise, return None (MISS)
        
        Complexity: O(k · m) where k=relevant clusters, m=avg entries per cluster
        This is much better than O(n) for the full cache when clusters are effective.
        
        Args:
            query: The query string (for logging/debugging)
            query_embedding: Dense vector representation of query
            cluster_probs: Soft cluster assignments for this query
            
        Returns:
            On cache hit: (result, matched_query, similarity_score, dominant_cluster)
            On cache miss: None
        """
        # Identify clusters to check (those with sufficient probability)
        relevant_clusters = [
            cluster_id 
            for cluster_id, prob in cluster_probs.items() 
            if prob > self.min_cluster_prob
        ]
        
        if not relevant_clusters:
            # Query doesn't belong to any cluster strongly enough
            self._miss_count += 1
            return None
        
        # Track best match across all relevant clusters
        best_match: Optional[CacheEntry] = None
        best_similarity: float = 0.0
        
        # Check each relevant cluster
        for cluster_id in relevant_clusters:
            # Skip if no entries cached for this cluster yet
            if cluster_id not in self._cache:
                continue
            
            # Check all cached entries in this cluster
            for entry in self._cache[cluster_id]:
                similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
        
        # Determine if we have a cache hit
        if best_match is not None and best_similarity >= self.similarity_threshold:
            # Cache HIT
            self._hit_count += 1
            self._similarity_scores_on_hit.append(best_similarity)
            best_match.hit_count += 1
            
            return (
                best_match.result,
                best_match.query,
                best_similarity,
                best_match.dominant_cluster
            )
        else:
            # Cache MISS
            self._miss_count += 1
            return None
    
    def put(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Dict[str, Any],
        cluster_probs: Dict[int, float],
        dominant_cluster: int
    ) -> None:
        """
        Store a new query-result pair in the cache.
        
        Storage Strategy:
        - Entry is added to ALL clusters where query has significant probability
        - This allows future queries in any of those clusters to find this entry
        - Trade-off: storage duplication vs. lookup efficiency
        - For this corpus size, duplication is negligible
        
        Args:
            query: The query string
            query_embedding: Dense vector representation
            result: Computed search results
            cluster_probs: Soft cluster assignments
            dominant_cluster: Primary cluster (for stats)
        """
        # Create cache entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            result=result,
            cluster_probs=cluster_probs,
            dominant_cluster=dominant_cluster
        )
        
        # Add to all relevant clusters (probability > threshold)
        for cluster_id, prob in cluster_probs.items():
            if prob > self.min_cluster_prob:
                if cluster_id not in self._cache:
                    self._cache[cluster_id] = []
                self._cache[cluster_id].append(entry)
    
    def clear(self) -> int:
        """
        Flush all cache entries and reset statistics.
        
        Returns:
            Number of entries deleted
        """
        # Count total entries (accounting for duplication across clusters)
        unique_entries = set()
        for entries in self._cache.values():
            for entry in entries:
                unique_entries.add(id(entry))
        
        entries_deleted = len(unique_entries)
        
        # Clear everything
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._similarity_scores_on_hit.clear()
        
        return entries_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and analysis.
        
        Returns:
            Dictionary containing:
            - total_entries: Number of unique cached queries
            - hit_count: Number of cache hits
            - miss_count: Number of cache misses
            - hit_rate: Percentage of requests served from cache
            - avg_similarity_on_hit: Average similarity score for cache hits
            - cluster_distribution: Number of cached queries per cluster
        """
        # Count unique entries (accounting for duplication)
        unique_entries = set()
        for entries in self._cache.values():
            for entry in entries:
                unique_entries.add(id(entry))
        
        total_entries = len(unique_entries)
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
        
        avg_similarity = (
            np.mean(self._similarity_scores_on_hit) 
            if self._similarity_scores_on_hit 
            else None
        )
        
        # Cluster distribution (count primary cluster assignments)
        cluster_dist: Dict[int, int] = {}
        seen_entries = set()
        for cluster_id, entries in self._cache.items():
            for entry in entries:
                if id(entry) not in seen_entries:
                    seen_entries.add(id(entry))
                    dominant = entry.dominant_cluster
                    cluster_dist[dominant] = cluster_dist.get(dominant, 0) + 1
        
        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "avg_similarity_on_hit": avg_similarity,
            "cluster_distribution": cluster_dist
        }
    
    def save(self, filepath: str) -> None:
        """Save cache to disk for persistence."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'cache': self._cache,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'similarity_scores': self._similarity_scores_on_hit,
                'threshold': self.similarity_threshold,
                'min_cluster_prob': self.min_cluster_prob
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load cache from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self._cache = data['cache']
            self._hit_count = data['hit_count']
            self._miss_count = data['miss_count']
            self._similarity_scores_on_hit = data['similarity_scores']
            self.similarity_threshold = data['threshold']
            self.min_cluster_prob = data['min_cluster_prob']
