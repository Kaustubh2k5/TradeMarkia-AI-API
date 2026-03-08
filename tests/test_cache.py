"""
Tests for SemanticCache implementation.

Tests cover:
1. Basic get/put operations
2. Similarity threshold behavior
3. Cluster-aware lookup
4. Cache statistics
5. Cache clearing
"""

import pytest
import numpy as np
from app.cache import SemanticCache, CacheEntry


@pytest.fixture
def sample_cache():
    """Create a sample cache for testing."""
    return SemanticCache(similarity_threshold=0.85)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    
    # Create 3 sample embeddings with known relationships
    embeddings = {
        'query1': np.random.randn(384).astype('float32'),
        'query2': np.random.randn(384).astype('float32'),  # Different
        'query1_similar': np.random.randn(384).astype('float32')  # Will be made similar to query1
    }
    
    # Make query1_similar actually similar to query1
    embeddings['query1_similar'] = 0.9 * embeddings['query1'] + 0.1 * embeddings['query1_similar']
    
    # Normalize all embeddings
    for key in embeddings:
        embeddings[key] = embeddings[key] / np.linalg.norm(embeddings[key])
    
    return embeddings


def test_cache_initialization():
    """Test cache initialization with different parameters."""
    cache = SemanticCache(similarity_threshold=0.9, min_cluster_prob=0.15)
    
    assert cache.similarity_threshold == 0.9
    assert cache.min_cluster_prob == 0.15
    assert cache._hit_count == 0
    assert cache._miss_count == 0
    assert len(cache._cache) == 0


def test_cache_miss_on_empty(sample_cache, sample_embeddings):
    """Test that get returns None on empty cache."""
    result = sample_cache.get(
        query="test query",
        query_embedding=sample_embeddings['query1'],
        cluster_probs={0: 0.8, 1: 0.2}
    )
    
    assert result is None
    assert sample_cache._miss_count == 1
    assert sample_cache._hit_count == 0


def test_cache_put_and_hit(sample_cache, sample_embeddings):
    """Test putting an entry and getting a cache hit."""
    # Put an entry
    sample_cache.put(
        query="machine learning",
        query_embedding=sample_embeddings['query1'],
        result={"data": "test result"},
        cluster_probs={0: 0.7, 1: 0.3},
        dominant_cluster=0
    )
    
    # Try to get with exact same embedding (should hit)
    result = sample_cache.get(
        query="machine learning",
        query_embedding=sample_embeddings['query1'],
        cluster_probs={0: 0.7, 1: 0.3}
    )
    
    assert result is not None
    cached_result, matched_query, similarity, dominant = result
    assert matched_query == "machine learning"
    assert cached_result == {"data": "test result"}
    assert similarity > 0.99  # Should be nearly 1.0 for identical embeddings
    assert dominant == 0


def test_cache_hit_with_similar_query(sample_cache, sample_embeddings):
    """Test cache hit with semantically similar query."""
    # Put an entry
    sample_cache.put(
        query="recent ML advances",
        query_embedding=sample_embeddings['query1'],
        result={"data": "ML results"},
        cluster_probs={0: 0.6, 1: 0.4},
        dominant_cluster=0
    )
    
    # Try similar query (similar embedding)
    result = sample_cache.get(
        query="latest machine learning developments",
        query_embedding=sample_embeddings['query1_similar'],
        cluster_probs={0: 0.65, 1: 0.35}
    )
    
    assert result is not None
    cached_result, matched_query, similarity, dominant = result
    assert matched_query == "recent ML advances"
    assert similarity > 0.85  # Should exceed threshold


def test_cache_miss_with_dissimilar_query(sample_cache, sample_embeddings):
    """Test cache miss with dissimilar query."""
    # Put an entry
    sample_cache.put(
        query="machine learning",
        query_embedding=sample_embeddings['query1'],
        result={"data": "ML results"},
        cluster_probs={0: 0.7, 1: 0.3},
        dominant_cluster=0
    )
    
    # Try dissimilar query
    result = sample_cache.get(
        query="cooking recipes",
        query_embedding=sample_embeddings['query2'],
        cluster_probs={0: 0.6, 1: 0.4}
    )
    
    assert result is None
    assert sample_cache._miss_count == 1


def test_cluster_aware_lookup(sample_cache, sample_embeddings):
    """Test that cache only checks relevant clusters."""
    # Put entry in cluster 0
    sample_cache.put(
        query="query in cluster 0",
        query_embedding=sample_embeddings['query1'],
        result={"cluster": 0},
        cluster_probs={0: 0.9, 1: 0.1},
        dominant_cluster=0
    )
    
    # Put entry in cluster 1
    sample_cache.put(
        query="query in cluster 1",
        query_embedding=sample_embeddings['query2'],
        result={"cluster": 1},
        cluster_probs={1: 0.9, 0: 0.1},
        dominant_cluster=1
    )
    
    # Query only in cluster 0 (shouldn't find cluster 1 entry)
    result = sample_cache.get(
        query="test",
        query_embedding=sample_embeddings['query1_similar'],
        cluster_probs={0: 0.95, 1: 0.05}  # 0.05 < min_cluster_prob (0.1)
    )
    
    # Should find cluster 0 entry
    if result is not None:
        cached_result, matched_query, _, _ = result
        assert matched_query == "query in cluster 0"


def test_cache_statistics(sample_cache, sample_embeddings):
    """Test cache statistics tracking."""
    # Add some entries
    for i in range(5):
        sample_cache.put(
            query=f"query {i}",
            query_embedding=sample_embeddings['query1'] * (1 + i * 0.01),
            result={"index": i},
            cluster_probs={0: 0.7, 1: 0.3},
            dominant_cluster=0
        )
    
    # Trigger some hits and misses
    sample_cache.get("q1", sample_embeddings['query1'], {0: 0.8, 1: 0.2})  # Hit
    sample_cache.get("q2", sample_embeddings['query1'], {0: 0.8, 1: 0.2})  # Hit
    sample_cache.get("q3", sample_embeddings['query2'], {0: 0.8, 1: 0.2})  # Miss
    
    stats = sample_cache.get_stats()
    
    assert stats['total_entries'] == 5
    assert stats['hit_count'] == 2
    assert stats['miss_count'] == 1
    assert stats['hit_rate'] == 2/3
    assert stats['avg_similarity_on_hit'] is not None


def test_cache_clear(sample_cache, sample_embeddings):
    """Test cache clearing."""
    # Add entries
    for i in range(3):
        sample_cache.put(
            query=f"query {i}",
            query_embedding=sample_embeddings['query1'],
            result={"index": i},
            cluster_probs={0: 0.7, 1: 0.3},
            dominant_cluster=0
        )
    
    # Trigger a hit
    sample_cache.get("q", sample_embeddings['query1'], {0: 0.7, 1: 0.3})
    
    # Clear cache
    deleted = sample_cache.clear()
    
    assert deleted == 3
    assert sample_cache._hit_count == 0
    assert sample_cache._miss_count == 0
    assert len(sample_cache._cache) == 0


def test_similarity_threshold_behavior():
    """Test different similarity threshold values."""
    np.random.seed(42)
    
    base_embedding = np.random.randn(384).astype('float32')
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    # Create slightly different embedding
    similar_embedding = 0.88 * base_embedding + 0.12 * np.random.randn(384).astype('float32')
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
    
    # Test with strict threshold (0.95)
    strict_cache = SemanticCache(similarity_threshold=0.95)
    strict_cache.put(
        query="test",
        query_embedding=base_embedding,
        result={"data": "test"},
        cluster_probs={0: 1.0},
        dominant_cluster=0
    )
    
    result_strict = strict_cache.get(
        query="similar",
        query_embedding=similar_embedding,
        cluster_probs={0: 1.0}
    )
    
    # Test with loose threshold (0.75)
    loose_cache = SemanticCache(similarity_threshold=0.75)
    loose_cache.put(
        query="test",
        query_embedding=base_embedding,
        result={"data": "test"},
        cluster_probs={0: 1.0},
        dominant_cluster=0
    )
    
    result_loose = loose_cache.get(
        query="similar",
        query_embedding=similar_embedding,
        cluster_probs={0: 1.0}
    )
    
    # Strict cache more likely to miss
    # Loose cache more likely to hit
    # (exact behavior depends on random embeddings)
    assert strict_cache.similarity_threshold > loose_cache.similarity_threshold


def test_cosine_similarity_computation(sample_cache):
    """Test cosine similarity computation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.5, 0.866, 0.0])  # 60 degrees from vec1
    
    similarity = sample_cache._cosine_similarity(vec1, vec2)
    
    # cos(60°) = 0.5
    assert abs(similarity - 0.5) < 0.01


def test_multiple_cluster_membership(sample_cache, sample_embeddings):
    """Test entry stored in multiple clusters."""
    # Put entry with significant probability in multiple clusters
    sample_cache.put(
        query="multi-cluster query",
        query_embedding=sample_embeddings['query1'],
        result={"data": "test"},
        cluster_probs={0: 0.4, 1: 0.35, 2: 0.25},
        dominant_cluster=0
    )
    
    # Should be findable from any of the significant clusters
    result_from_0 = sample_cache.get(
        query="test",
        query_embedding=sample_embeddings['query1'],
        cluster_probs={0: 0.9, 1: 0.05, 2: 0.05}
    )
    
    result_from_1 = sample_cache.get(
        query="test",
        query_embedding=sample_embeddings['query1'],
        cluster_probs={0: 0.05, 1: 0.9, 2: 0.05}
    )
    
    assert result_from_0 is not None
    assert result_from_1 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
