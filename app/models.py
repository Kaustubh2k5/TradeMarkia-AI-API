"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class QueryRequest(BaseModel):
    """Request model for semantic search query."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")


class QueryResponse(BaseModel):
    """Response model for semantic search query."""
    query: str = Field(..., description="The original query")
    cache_hit: bool = Field(..., description="Whether the query was served from cache")
    matched_query: Optional[str] = Field(None, description="The cached query that matched (on cache hit)")
    similarity_score: Optional[float] = Field(None, description="Similarity to matched query (0-1)")
    result: Dict[str, Any] = Field(..., description="Search results and cluster information")
    dominant_cluster: int = Field(..., description="The primary cluster for this query")


class CacheStats(BaseModel):
    """Response model for cache statistics."""
    total_entries: int = Field(..., description="Total number of cached queries")
    hit_count: int = Field(..., description="Number of cache hits")
    miss_count: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    avg_similarity_on_hit: Optional[float] = Field(None, description="Average similarity score on cache hits")
    cluster_distribution: Dict[int, int] = Field(default_factory=dict, description="Distribution of cached queries across clusters")


class CacheDeleteResponse(BaseModel):
    """Response model for cache deletion."""
    message: str = Field(..., description="Confirmation message")
    entries_deleted: int = Field(..., description="Number of cache entries deleted")
