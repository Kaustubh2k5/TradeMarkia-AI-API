"""
Data Preparation Pipeline

This script:
1. Downloads the 20 Newsgroups dataset
2. Cleans and preprocesses the text
3. Generates embeddings using sentence-transformers
4. Builds FAISS index
5. Trains fuzzy clustering model
6. Analyzes clustering quality

Design Decisions:
-----------------
1. Text Preprocessing:
   - Remove email headers (From:, Subject:, etc.)
   - Remove quoted text (lines starting with >)
   - Remove URLs and email addresses
   - Keep original case (better for sentence transformers)
   - Minimum length: 50 characters (filters out junk)
   
2. Category Selection:
   - Use all 20 categories
   - Some have significant overlap (expected and desired)
   
3. Embedding Model:
   - all-MiniLM-L6-v2: Fast, good quality
   - Batch size: 32 (balances speed and memory)
   
4. Clustering:
   - GMM with 25 components (determined via BIC)
   - Full covariance (captures semantic structure)
   - Multiple initializations for stability
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
import pickle
from typing import List, Dict, Any
import sys
from sklearn.preprocessing import normalize

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_db import VectorDatabase, analyze_cluster_boundaries


def clean_text(text: str) -> str:
    """
    Clean a single document.
    
    Removes:
    - Email headers (From:, Subject:, Organization:, etc.)
    - Quoted text (lines starting with >)
    - URLs
    - Email addresses
    - Excessive whitespace
    
    Why these choices?
    -----------------
    - Headers are metadata, not semantic content
    - Quoted text is duplicated across documents
    - URLs and emails are noise for semantic search
    - We KEEP original case (better for sentence transformers)
    
    Args:
        text: Raw document text
        
    Returns:
        Cleaned text
    """
    # Remove email headers
    # Pattern: Lines starting with "Word:" at the beginning of text
    text = re.sub(r'^[A-Za-z\-]+:.*$', '', text, flags=re.MULTILINE)
    
    # Remove quoted text (lines starting with >)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def is_valid_document(text: str, min_length: int = 50) -> bool:
    """
    Check if a document is valid for inclusion.
    
    Criteria:
    - Minimum length (after cleaning)
    - Contains actual words (not just symbols)
    
    Why filter?
    ----------
    The dataset is noisy. Some documents are:
    - Empty after header removal
    - Just signatures or footers
    - Spam or corrupted text
    
    Better to exclude junk than pollute embeddings.
    
    Args:
        text: Cleaned document text
        min_length: Minimum character count
        
    Returns:
        True if document should be included
    """
    if len(text) < min_length:
        return False
    
    # Check for actual words (not just symbols)
    word_count = len(re.findall(r'\b[a-zA-Z]+\b', text))
    if word_count < 10:
        return False
    
    return True


def load_and_clean_dataset() -> List[Dict[str, Any]]:
    """
    Load and clean the 20 Newsgroups dataset.
    
    Returns:
        List of document dictionaries with:
        - text: cleaned document text
        - category: newsgroup category
        - original_index: index in original dataset
    """
    print("=" * 60)
    print("📥 Downloading 20 Newsgroups dataset...")
    print("=" * 60)
    
    # Download dataset
    # remove=('headers', 'footers', 'quotes') is too aggressive
    # We clean manually for more control
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=(),  # We'll clean manually
        shuffle=True,
        random_state=42
    )
    
    print(f"✅ Downloaded {len(newsgroups.data)} documents")
    print(f"📊 Categories: {len(newsgroups.target_names)}")
    
    # Clean and filter documents
    print("\n🧹 Cleaning documents...")
    documents = []
    
    for idx, (text, target) in enumerate(tqdm(
        zip(newsgroups.data, newsgroups.target),
        total=len(newsgroups.data),
        desc="Processing"
    )):
        # Clean text
        cleaned = clean_text(text)
        
        # Validate
        if is_valid_document(cleaned):
            documents.append({
                'text': cleaned,
                'category': newsgroups.target_names[target],
                'original_index': idx
            })
    
    print(f"✅ Kept {len(documents)}/{len(newsgroups.data)} documents after cleaning")
    print(f"   Filtered out {len(newsgroups.data) - len(documents)} low-quality documents")
    
    return documents


def select_optimal_clusters(
    embeddings: np.ndarray,
    min_clusters: int = 15,
    max_clusters: int = 35,
    step: int = 5
) -> int:
    """
    Determine optimal number of clusters using BIC and silhouette analysis.
    
    Why not just use 20 (the number of categories)?
    ------------------------------------------------
    The 20 categories are editorial labels, not semantic boundaries.
    The real structure is messier:
    - Some categories have sub-topics
    - Cross-cutting themes emerge
    - Overlapping content
    
    We use BIC (Bayesian Information Criterion) to find the natural
    number of semantic clusters in the data.
    
    Args:
        embeddings: Document embeddings
        min_clusters: Minimum number to try
        max_clusters: Maximum number to try
        step: Step size
        
    Returns:
        Optimal number of clusters
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    
    print("\n🔍 Finding optimal number of clusters...")
    print("Testing range: {} to {}".format(min_clusters, max_clusters))
    
    results = []
    
    for n in range(min_clusters, max_clusters + 1, step):
        print(f"\n  Testing {n} clusters...")
        
        # Train GMM
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            n_init=3  # Fewer inits for speed during search
        )
        gmm.fit(embeddings)
        
        # Compute metrics
        labels = gmm.predict(embeddings)
        
        # BIC (lower is better)
        bic = gmm.bic(embeddings)
        
        # Silhouette score (higher is better)
        # Sample for speed
        sample_size = min(5000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        silhouette = silhouette_score(
            embeddings[sample_indices],
            labels[sample_indices]
        )
        
        results.append({
            'n_clusters': n,
            'bic': bic,
            'silhouette': silhouette
        })
        
        print(f"    BIC: {bic:,.0f}, Silhouette: {silhouette:.4f}")
    
    # Find optimal based on BIC
    best_bic = min(results, key=lambda x: x['bic'])
    best_silhouette = max(results, key=lambda x: x['silhouette'])
    
    print("\n" + "=" * 60)
    print("📊 Cluster Selection Analysis:")
    print("=" * 60)
    print(f"Best BIC: {best_bic['n_clusters']} clusters (BIC={best_bic['bic']:,.0f})")
    print(f"Best Silhouette: {best_silhouette['n_clusters']} clusters (score={best_silhouette['silhouette']:.4f})")
    
    # Use BIC as primary criterion
    optimal = best_bic['n_clusters']
    print(f"\n✅ Selected: {optimal} clusters (based on BIC)")
    
    return optimal


def main():
    """Main data preparation pipeline."""
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)
    
    # Step 1: Load and clean dataset
    documents = load_and_clean_dataset()
    
    # Save processed documents
    processed_path = 'data/processed/documents.pkl'
    with open(processed_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"\n💾 Saved processed documents to {processed_path}")
    
    # Step 2: Initialize vector database
    print("\n" + "=" * 60)
    print("🧠 Initializing Vector Database")
    print("=" * 60)
    vector_db = VectorDatabase()
    
    # Step 3: Generate embeddings
    print("\n📊 Generating embeddings...")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Documents: {len(documents)}")
    
    texts = [doc['text'] for doc in documents]
    embeddings = vector_db.embed_documents(texts, batch_size=32)
    embeddings = normalize(embeddings)

    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Store documents and embeddings in vector_db
    vector_db.documents = documents
    vector_db.embeddings = embeddings
    
    # Step 4: Build FAISS index
    print("\n🗂️  Building FAISS index...")
    vector_db.build_index(embeddings)
    
    # Step 5: Determine optimal number of clusters
    # Uncomment to run cluster selection (takes time)
    # optimal_clusters = select_optimal_clusters(embeddings)
    # vector_db.n_clusters = optimal_clusters
    
    # For this assignment, we use 25 based on prior analysis
    optimal_clusters = select_optimal_clusters(embeddings)
    vector_db.n_clusters = optimal_clusters
    print(f"\n🎯 Using {vector_db.n_clusters} clusters")
    
    # Step 6: Train clustering model
    print("\n" + "=" * 60)
    print("🔮 Training Fuzzy Clustering Model")
    print("=" * 60)
    
    clustering_metrics = vector_db.train_clustering(embeddings)
    
    print("\n📈 Clustering Metrics:")
    for key, value in clustering_metrics.items():
        print(f"  {key}: {value}")
    
    # Step 7: Analyze cluster boundaries
    print("\n" + "=" * 60)
    print("🔍 Analyzing Cluster Boundaries")
    print("=" * 60)
    
    boundary_analysis = analyze_cluster_boundaries(
        vector_db,
        embeddings,
        documents,
        n_samples=10
    )
    
    print(f"\nAverage Entropy: {boundary_analysis['avg_entropy']:.4f}")
    print(f"Max Entropy: {boundary_analysis['max_entropy']:.4f}")
    print("\nTop 5 Boundary Cases (high entropy = ambiguous assignment):")
    
    for i, case in enumerate(boundary_analysis['boundary_cases'][:5], 1):
        print(f"\n{i}. Entropy: {case['entropy']:.4f}")
        print(f"   Category: {case['document']['category']}")
        print(f"   Text preview: {case['document']['text'][:150]}...")
        print(f"   Cluster distribution:")
        for cluster_info in case['top_clusters']:
            print(f"     Cluster {cluster_info['cluster_id']}: {cluster_info['probability']:.3f}")
    
    # Step 8: Save everything
    print("\n" + "=" * 60)
    print("💾 Saving Vector Database")
    print("=" * 60)
    
    embeddings_dir = 'data/embeddings'
    vector_db.save(embeddings_dir)
    
    # Save cluster analysis
    analysis_path = 'data/embeddings/cluster_analysis.pkl'
    with open(analysis_path, 'wb') as f:
        pickle.dump({
            'metrics': clustering_metrics,
            'boundary_analysis': boundary_analysis
        }, f)
    
    print(f"✅ Saved cluster analysis to {analysis_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✨ Data Preparation Complete!")
    print("=" * 60)
    print(f"📁 Documents: {len(documents)}")
    print(f"🔢 Embedding dim: {embeddings.shape[1]}")
    print(f"🎯 Clusters: {vector_db.n_clusters}")
    print(f"📊 Silhouette score: {clustering_metrics['silhouette_score']:.4f}")
    print(f"\n🚀 Ready to start the API:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
