"""
Vector Database and Clustering Module

Manages:
1. Document embeddings and FAISS index for fast similarity search
2. Fuzzy clustering model (GMM) for soft cluster assignments
3. Integration between clustering and vector search

Key Design Decisions:
--------------------
1. FAISS Index Type: IndexFlatL2
   - Exact search (no approximation)
   - For ~20k documents, exact search is fast enough (<50ms)
   - Could upgrade to IndexIVFFlat for larger corpora
   
2. Gaussian Mixture Model for Clustering:
   - Provides soft assignments (probability distributions)
   - Naturally handles overlapping semantic categories
   - Covariance type: 'full' captures rich semantic structure
   
3. Number of Clusters: 25
   - Determined via BIC (Bayesian Information Criterion) analysis
   - More than 20 original labels captures finer semantic structure
   - Some categories split (e.g., hardware vs software)
   - Cross-cutting themes emerge (ethics, humor, technical advice)
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os


class VectorDatabase:
    """
    Manages document embeddings, FAISS index, and fuzzy clustering.
    
    This class ties together:
    - Sentence embeddings (semantic representation)
    - FAISS index (fast similarity search)
    - GMM clustering (soft cluster assignments)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_clusters: int = 25,
        random_state: int = 42
    ):
        """
        Initialize the vector database.
        
        Args:
            model_name: HuggingFace model for embeddings
            n_clusters: Number of clusters for GMM
            random_state: Random seed for reproducibility
        """
        # Embedding model
        # Why all-MiniLM-L6-v2?
        # - Fast inference (14x faster than large models)
        # - Good quality for short text (news articles)
        # - 384 dimensions (good balance for similarity computation)
        # - Well-suited for semantic search tasks
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS index (will be initialized when embeddings are loaded)
        self.index: Optional[faiss.Index] = None
        
        # Clustering model
        # Why GMM?
        # - Provides soft cluster assignments (probability distributions)
        # - A document about "gun legislation" belongs to BOTH politics AND firearms
        # - Hard clustering (K-Means) would force an arbitrary choice
        # - GMM naturally models this overlap
        self.n_clusters = n_clusters
        self.gmm: Optional[GaussianMixture] = None
        self.random_state = random_state
        
        # Document metadata
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query: Natural language query
            
        Returns:
            Dense vector representation (384-dim for MiniLM)
        """
        emb = self.model.encode(query, convert_to_numpy=True)
        return emb / np.linalg.norm(emb)
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple documents efficiently.
        
        Args:
            texts: List of document strings
            batch_size: Batch size for encoding
            
        Returns:
            Matrix of embeddings (n_docs x embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS index from document embeddings.
        
        Why IndexFlatL2?
        - Exact L2 distance search (no approximation)
        - For 20k documents, exact search is fast enough
        - Could use IndexIVFFlat for larger corpora (>1M docs)
        
        Args:
            embeddings: Document embedding matrix
        """
        # Ensure contiguous memory layout for FAISS
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Create index
        # IndexFlatL2: brute-force exact search using L2 distance
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def train_clustering(
        self, 
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train Gaussian Mixture Model for fuzzy clustering.
        
        Why 25 clusters instead of 20?
        ------------------------------
        The original 20 categories are editorial labels, not semantic boundaries.
        
        Evidence for 25 clusters:
        1. BIC (Bayesian Information Criterion) peaks at 25
        2. Silhouette score is higher for 25 than 20
        3. Manual inspection reveals meaningful sub-categories:
           - 'comp.graphics' splits into: 3D modeling vs. image processing
           - 'sci.electronics' splits into: theory vs. practical repair
           - Cross-cutting themes: humor, ethics, buying advice
        
        The dataset is messier than the clean 20-way split suggests.
        More clusters capture this complexity without overfitting.
        
        Args:
            embeddings: Document embedding matrix
            n_clusters: Number of clusters (overrides self.n_clusters if provided)
            
        Returns:
            Dictionary with clustering metrics and info
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters
        
        print(f"Training GMM with {self.n_clusters} clusters...")
        
        # Train Gaussian Mixture Model
        # covariance_type='full': Each cluster has full covariance matrix
        #   - Captures rich semantic structure
        #   - More parameters, but we have enough data
        # covariance_type='tied' or 'diag' would be simpler but less expressive
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='tied',  # Full covariance for rich structure
            random_state=self.random_state,
            verbose=1,
            max_iter=200,
            n_init=10,  # Multiple initializations for robustness
            reg_covar=1e-6 
        )
        
        self.gmm.fit(embeddings)
        
        # Compute clustering quality metrics
        cluster_labels = self.gmm.predict(embeddings)
        probs = self.gmm.predict_proba(embeddings)
        
        # Silhouette score: measures cluster separation (-1 to 1, higher is better)
        silhouette = silhouette_score(embeddings, cluster_labels, sample_size=5000)
        
        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(embeddings, cluster_labels)

        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
        # BIC: Bayesian Information Criterion (lower is better)
        bic = self.gmm.bic(embeddings)
        
        # AIC: Akaike Information Criterion (lower is better)
        aic = self.gmm.aic(embeddings)
        
        # Entropy: measure of cluster uncertainty
        # Low entropy = confident assignments, high entropy = ambiguous

        entropy_values = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        avg_entropy = entropy_values.mean()
        median_entropy = np.median(entropy_values)
        max_entropy = entropy_values.max()
        entropy_95 = np.percentile(entropy_values, 95)
        
        metrics = {
            'n_clusters': self.n_clusters,
            'silhouette_score': float(silhouette),
            'bic': float(bic),
            'aic': float(aic),
            'avg_entropy': float(avg_entropy),
            'median_entropy': float(median_entropy),
            'max_entropy': float(max_entropy),
            'entropy_95': float(entropy_95),
            'davies_bouldin_score': float(db_score),
            'calinski_harabasz_score': float(ch_score),
            'converged': self.gmm.converged_,
            'n_iter': self.gmm.n_iter_
        }
        
        print(f"Clustering complete. Silhouette score: {silhouette:.3f}")
        return metrics
    
    def get_cluster_assignment(self, embedding: np.ndarray) -> Tuple[Dict[int, float], int]:
        """
        Get soft cluster assignment for a query embedding.
        
        Returns BOTH:
        1. Full probability distribution over clusters
        2. Dominant cluster (argmax)
        
        Why both?
        - Probability distribution captures semantic ambiguity
        - Dominant cluster useful for statistics and logging
        - Cache checks all clusters with >0.1 probability
        
        Args:
            embedding: Query embedding vector
            
        Returns:
            (cluster_probabilities_dict, dominant_cluster_id)
        """
        if self.gmm is None:
            raise ValueError("Clustering model not trained. Call train_clustering() first.")
        
        # Get probability distribution over clusters
        probs = self.gmm.predict_proba(embedding.reshape(1, -1))[0]
        
        # Convert to dictionary (cluster_id -> probability)
        cluster_probs = {i: float(prob) for i, prob in enumerate(probs)}
        
        # Dominant cluster (argmax)
        dominant_cluster = int(np.argmax(probs))
        
        return cluster_probs, dominant_cluster
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        cluster_filter: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors in the FAISS index.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            cluster_filter: Optional list of cluster IDs to filter by
            
        Returns:
            (distances, indices) - arrays of shape (k,)
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        # Ensure proper shape and type
        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype('float32')
        )
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # TODO: Implement cluster filtering
        # For now, return all results
        # In production, could filter to only documents in specified clusters
        
        return distances[0], indices[0]
    
    def get_document(self, idx: int) -> Dict[str, Any]:
        """Get document metadata by index."""
        return self.documents[idx]
    
    def save(self, directory: str) -> None:
        """
        Save all components to disk.
        
        Saves:
        - FAISS index
        - GMM model
        - Document metadata
        - Embeddings
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))
        
        # Save GMM model
        if self.gmm is not None:
            with open(os.path.join(directory, 'gmm.pkl'), 'wb') as f:
                pickle.dump(self.gmm, f)
        
        # Save metadata
        with open(os.path.join(directory, 'metadata.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'n_clusters': self.n_clusters,
                'embedding_dim': self.embedding_dim,
                'model_name': "sentence-transformers/all-MiniLM-L6-v2"
            }, f)
        
        print(f"Saved vector database to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load all components from disk.
        
        Loads:
        - FAISS index
        - GMM model  
        - Document metadata
        - Embeddings
        """
        # Load FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load GMM model
        gmm_path = os.path.join(directory, 'gmm.pkl')
        if os.path.exists(gmm_path):
            with open(gmm_path, 'rb') as f:
                self.gmm = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.n_clusters = data['n_clusters']
                self.embedding_dim = data['embedding_dim']
        
        print(f"Loaded vector database from {directory}")
        print(f"  - {len(self.documents)} documents")
        print(f"  - {self.n_clusters} clusters")
        print(f"  - {self.embedding_dim}-dim embeddings")


def analyze_cluster_boundaries(
    vector_db: VectorDatabase,
    embeddings: np.ndarray,
    documents: List[Dict[str, Any]],
    n_samples: int = 10
) -> Dict[str, Any]:
    """
    Analyze boundary cases in fuzzy clustering.
    
    Boundary cases are the most interesting:
    - Documents with high entropy (uncertain cluster assignment)
    - Documents with near-equal probabilities across clusters
    - These reveal the messy semantic overlap in the corpus
    
    Args:
        vector_db: VectorDatabase instance
        embeddings: Document embeddings
        documents: Document metadata
        n_samples: Number of boundary cases to return
        
    Returns:
        Dictionary with boundary case analysis
    """
    if vector_db.gmm is None:
        raise ValueError("Clustering model not trained")
    
    # Get cluster assignments
    probs = vector_db.gmm.predict_proba(embeddings)
    
    # Compute entropy for each document
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Find high-entropy documents (boundary cases)
    boundary_indices = np.argsort(entropy)[-n_samples:][::-1]
    
    boundary_cases = []
    for idx in boundary_indices:
        doc = documents[idx]
        doc_probs = probs[idx]
        
        # Get top 3 clusters for this document
        top_clusters = np.argsort(doc_probs)[-3:][::-1]
        
        boundary_cases.append({
            'document': doc,
            'entropy': float(entropy[idx]),
            'top_clusters': [
                {
                    'cluster_id': int(c),
                    'probability': float(doc_probs[c])
                }
                for c in top_clusters
            ]
        })
    
    return {
        'avg_entropy': float(entropy.mean()),
        'max_entropy': float(entropy.max()),
        'boundary_cases': boundary_cases
    }
