"""
FastAPI Service for Semantic Search with Semantic Caching

Endpoints:
----------
1. POST /query - Semantic search with cache
2. GET /cache/stats - Cache statistics
3. DELETE /cache - Clear cache

State Management:
----------------
- Vector database loaded once at startup
- Semantic cache persists across requests (in-memory)
- Thread-safe operations for concurrent requests
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from typing import Dict, Any, List

from app.models import QueryRequest, QueryResponse, CacheStats, CacheDeleteResponse
from app.cache import SemanticCache
from app.vector_db import VectorDatabase


# Global state (loaded at startup)
vector_db: VectorDatabase = None
semantic_cache: SemanticCache = None

def ensure_initialized():
    global vector_db, semantic_cache

    if vector_db is None:
        embeddings_dir = os.path.join("data", "embeddings")

        vector_db = VectorDatabase()
        vector_db.load(embeddings_dir)

    if semantic_cache is None:
        semantic_cache = SemanticCache(similarity_threshold=0.85)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    
    Startup:
    - Load pre-trained vector database
    - Load pre-trained clustering model
    - Initialize semantic cache
    
    Shutdown:
    - Optionally save cache to disk
    """
    global vector_db, semantic_cache
    
    # Startup
    print("=" * 60)
    print("🚀 Starting Semantic Search API")
    print("=" * 60)
    
    # Load vector database
    embeddings_dir = os.path.join('data', 'embeddings')
    if not os.path.exists(embeddings_dir):
        raise RuntimeError(
            f"Embeddings directory not found: {embeddings_dir}\n"
            "Please run: python scripts/prepare_data.py"
        )
    
    print("\n📦 Loading vector database...")
    vector_db = VectorDatabase()
    vector_db.load(embeddings_dir)
    print("✅ Vector database loaded")
    
    # Initialize semantic cache
    # CRITICAL PARAMETER: similarity_threshold
    # - 0.85 is the recommended default
    # - Adjust based on your precision/recall requirements
    # - See app/cache.py for detailed analysis
    print("\n🧠 Initializing semantic cache...")
    semantic_cache = SemanticCache(similarity_threshold=0.85)
    print("✅ Semantic cache initialized (threshold=0.85)")
    
    print("\n" + "=" * 60)
    print("✨ API ready at http://localhost:8000")
    print("📖 Docs available at http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down API...")
    # Optionally save cache to disk here
    # cache_path = os.path.join('data', 'cache.pkl')
    # semantic_cache.save(cache_path)


# Create FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="Semantic search with fuzzy clustering and custom caching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Semantic Search API is running",
        "endpoints": {
            "query": "POST /query",
            "cache_stats": "GET /cache/stats",
            "cache_clear": "DELETE /cache",
            "docs": "GET /docs"
        }
    }


@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Semantic search with cache lookup.
    
    Flow:
    -----
    1. Embed the query
    2. Get cluster assignment
    3. Check semantic cache
       - HIT: Return cached result
       - MISS: Compute result, cache it, return
    
    The cache recognizes semantically similar queries even if phrased differently.
    For example:
    - "recent ML advances" ≈ "latest machine learning developments"
    - "gun control laws" ≈ "firearm legislation"
    
    Args:
        request: Query request with natural language query string
        
    Returns:
        QueryResponse with results and cache information
    """
    query_text = request.query.strip()
    
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    # Step 1: Embed the query
    ensure_initialized()
    
    query_embedding = vector_db.embed_query(query_text)
    
    # Step 2: Get cluster assignment
    cluster_probs, dominant_cluster = vector_db.get_cluster_assignment(query_embedding)
    
    # Step 3: Check cache
    cache_result = semantic_cache.get(query_text, query_embedding, cluster_probs)
    
    if cache_result is not None:
        # CACHE HIT
        result, matched_query, similarity_score, cached_dominant_cluster = cache_result
        
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=matched_query,
            similarity_score=similarity_score,
            result=result,
            dominant_cluster=cached_dominant_cluster
        )
    
    else:
        # CACHE MISS - compute result
        result = _compute_search_result(query_embedding, cluster_probs, dominant_cluster)
        
        # Cache the result for future queries
        semantic_cache.put(
            query=query_text,
            query_embedding=query_embedding,
            result=result,
            cluster_probs=cluster_probs,
            dominant_cluster=dominant_cluster
        )
        
        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result,
            dominant_cluster=dominant_cluster
        )


@app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
async def get_cache_stats() -> CacheStats:
    """
    Get cache statistics.
    
    Returns:
        CacheStats with hit/miss counts, hit rate, and cluster distribution
    """
    stats = semantic_cache.get_stats()
    
    return CacheStats(
        total_entries=stats['total_entries'],
        hit_count=stats['hit_count'],
        miss_count=stats['miss_count'],
        hit_rate=stats['hit_rate'],
        avg_similarity_on_hit=stats['avg_similarity_on_hit'],
        cluster_distribution=stats['cluster_distribution']
    )


@app.delete("/cache", response_model=CacheDeleteResponse, tags=["Cache"])
async def clear_cache() -> CacheDeleteResponse:
    """
    Clear all cache entries and reset statistics.
    
    Returns:
        CacheDeleteResponse with confirmation and count of deleted entries
    """
    entries_deleted = semantic_cache.clear()
    
    return CacheDeleteResponse(
        message="Cache cleared successfully",
        entries_deleted=entries_deleted
    )


# Helper Functions
# ================

def _compute_search_result(
    query_embedding,
    cluster_probs: Dict[int, float],
    dominant_cluster: int,
    k: int = 10
) -> Dict[str, Any]:
    """
    Compute search results for a query.
    
    This is what happens on a cache miss:
    1. Search FAISS index for k nearest neighbors
    2. Retrieve document metadata
    3. Package results with cluster information
    
    Args:
        query_embedding: Query vector
        cluster_probs: Cluster probability distribution
        dominant_cluster: Primary cluster
        k: Number of results to return
        
    Returns:
        Dictionary with search results and metadata
    """
    # Search vector database
    distances, indices = vector_db.search(query_embedding, k=k)
    
    # Retrieve documents
    top_documents = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        doc = vector_db.get_document(int(idx))
        
        # Calculate similarity score (convert L2 distance to similarity)
        # For normalized vectors: similarity ≈ 1 - (distance^2 / 2)
        similarity = max(0.0, 1.0 - (float(dist) ** 2) / 2.0)
        
        top_documents.append({
            'rank': i + 1,
            'document_id': int(idx),
            'text': doc.get('text', '')[:200] + '...',  # Preview
            'category': doc.get('category', 'unknown'),
            'similarity': round(similarity, 4),
            'distance': round(float(dist), 4)
        })
    
    # Format cluster distribution (top 5 clusters)
    sorted_clusters = sorted(
        cluster_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    cluster_distribution = {
        f"cluster_{cid}": round(prob, 4)
        for cid, prob in sorted_clusters
    }
    
    return {
        'top_documents': top_documents,
        'cluster_distribution': cluster_distribution,
        'num_results': len(top_documents)
    }


# Additional utility endpoints
# ============================

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "vector_db_loaded": vector_db is not None,
        "cache_initialized": semantic_cache is not None,
        "num_documents": len(vector_db.documents) if vector_db else 0,
        "num_clusters": vector_db.n_clusters if vector_db else 0,
        "cache_entries": semantic_cache.get_stats()['total_entries'] if semantic_cache else 0
    }


@app.get("/clusters/info", tags=["Clustering"])
async def get_cluster_info():
    """
    Get information about the clustering model.
    
    Returns basic statistics about the cluster structure.
    """
    if vector_db is None or vector_db.gmm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clustering model not loaded"
        )
    
    # Get cluster sizes (approximate)
    cluster_labels = vector_db.gmm.predict(vector_db.embeddings)
    cluster_sizes = {
        int(i): int((cluster_labels == i).sum())
        for i in range(vector_db.n_clusters)
    }
    
    return {
        "n_clusters": vector_db.n_clusters,
        "cluster_sizes": cluster_sizes,
        "total_documents": len(vector_db.documents)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
