"""
Tests for FastAPI endpoints.

Tests cover:
1. Query endpoint (with cache hit/miss)
2. Cache stats endpoint
3. Cache clear endpoint
4. Health check
5. Error handling
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import sys

# Mock the vector database for testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import QueryRequest, QueryResponse, CacheStats


@pytest.fixture
def client():
    """
    Create a test client for the API.
    
    Note: This requires the data/embeddings directory to exist.
    If running tests without prepared data, mock the vector_db.
    """
    # For actual testing, you would mock the vector_db loading
    # Here we assume the data is prepared
    
    from app.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test the detailed health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "vector_db_loaded" in data
    assert "cache_initialized" in data


def test_query_endpoint_valid_request(client):
    """Test query endpoint with valid request."""
    response = client.post(
        "/query",
        json={"query": "machine learning algorithms"}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert "query" in data
    assert "cache_hit" in data
    assert "result" in data
    assert "dominant_cluster" in data
    assert data["query"] == "machine learning algorithms"


def test_query_endpoint_empty_query(client):
    """Test query endpoint with empty query."""
    response = client.post(
        "/query",
        json={"query": ""}
    )
    
    assert response.status_code == 422
    assert "detail" in response.json()


def test_query_endpoint_invalid_json(client):
    """Test query endpoint with invalid JSON."""
    response = client.post(
        "/query",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422  # Unprocessable Entity


def test_query_cache_behavior(client):
    """Test cache hit behavior with repeated queries."""
    query_text = "artificial intelligence research"
    
    # First request (cache miss)
    response1 = client.post("/query", json={"query": query_text})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cache_hit"] == False
    assert data1["matched_query"] is None
    
    # Second request (should be cache hit)
    response2 = client.post("/query", json={"query": query_text})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cache_hit"] == True
    assert data2["matched_query"] == query_text
    assert data2["similarity_score"] > 0.99  # Exact match


def test_similar_query_cache_hit(client):
    """Test cache hit with semantically similar query."""
    # First query
    response1 = client.post(
        "/query",
        json={"query": "recent developments in deep learning"}
    )
    assert response1.status_code == 200
    
    # Similar query (may or may not hit depending on threshold)
    response2 = client.post(
        "/query",
        json={"query": "latest advances in neural networks"}
    )
    assert response2.status_code == 200
    
    data2 = response2.json()
    # If cache hit, similarity should be reasonable
    if data2["cache_hit"]:
        assert data2["similarity_score"] >= 0.85  # Above threshold


def test_cache_stats_endpoint(client):
    """Test cache statistics endpoint."""
    # Make some queries first
    client.post("/query", json={"query": "test query 1"})
    client.post("/query", json={"query": "test query 2"})
    
    response = client.get("/cache/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_entries" in data
    assert "hit_count" in data
    assert "miss_count" in data
    assert "hit_rate" in data
    assert "cluster_distribution" in data
    
    assert isinstance(data["total_entries"], int)
    assert isinstance(data["hit_rate"], float)
    assert 0 <= data["hit_rate"] <= 1


def test_cache_clear_endpoint(client):
    """Test cache clearing endpoint."""
    # Add some queries to cache
    client.post("/query", json={"query": "test query 1"})
    client.post("/query", json={"query": "test query 2"})
    
    # Check stats before clear
    stats_before = client.get("/cache/stats").json()
    entries_before = stats_before["total_entries"]
    
    # Clear cache
    response = client.delete("/cache")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "entries_deleted" in data
    assert data["entries_deleted"] >= 0
    
    # Check stats after clear
    stats_after = client.get("/cache/stats").json()
    assert stats_after["total_entries"] == 0
    assert stats_after["hit_count"] == 0
    assert stats_after["miss_count"] == 0


def test_cluster_info_endpoint(client):
    """Test cluster information endpoint."""
    response = client.get("/clusters/info")
    
    # May return 503 if clustering not loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "n_clusters" in data
        assert "cluster_sizes" in data
        assert "total_documents" in data


def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import concurrent.futures
    
    def make_request(i):
        return client.post("/query", json={"query": f"test query {i}"})
    
    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        responses = [f.result() for f in futures]
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)


def test_long_query(client):
    """Test handling of very long queries."""
    long_query = "artificial intelligence " * 100  # 100 repetitions
    
    response = client.post("/query", json={"query": long_query})
    
    # Should handle long query (may truncate or reject based on limits)
    assert response.status_code in [200, 400, 422]


def test_special_characters_in_query(client):
    """Test handling of special characters."""
    special_query = "What is AI? #MachineLearning @2024 <test>"
    
    response = client.post("/query", json={"query": special_query})
    assert response.status_code == 200


def test_unicode_in_query(client):
    """Test handling of Unicode characters."""
    unicode_query = "machine learning 机器学习 apprentissage automatique"
    
    response = client.post("/query", json={"query": unicode_query})
    assert response.status_code == 200


def test_response_model_validation():
    """Test Pydantic model validation."""
    # Test QueryResponse validation
    valid_response = QueryResponse(
        query="test",
        cache_hit=True,
        matched_query="test",
        similarity_score=0.95,
        result={"data": "test"},
        dominant_cluster=0
    )
    
    assert valid_response.query == "test"
    assert valid_response.cache_hit == True
    
    # Test CacheStats validation
    stats = CacheStats(
        total_entries=10,
        hit_count=5,
        miss_count=5,
        hit_rate=0.5
    )
    
    assert stats.total_entries == 10
    assert stats.hit_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
