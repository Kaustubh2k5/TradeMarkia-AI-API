# Quick Start Guide

Get the semantic search API up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM
- ~2GB free disk space

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd trademarkia-semantic-search

# Run setup script
chmod +x setup.sh
./setup.sh
```

The script will:
1. Create a virtual environment
2. Install dependencies
3. Download and prepare the dataset
4. Generate embeddings and train clustering model

**Note:** Data preparation takes ~10-15 minutes.

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py
```

## Running the API

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Server starts at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

## Testing the API

### Using the Interactive Docs

1. Open http://localhost:8000/docs in your browser
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Enter parameters and click "Execute"

### Using cURL

#### 1. Query the system

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence and machine learning"}'
```

**Response:**
```json
{
  "query": "artificial intelligence and machine learning",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": {
    "top_documents": [
      {
        "rank": 1,
        "document_id": 1234,
        "text": "Recent advances in neural networks...",
        "category": "comp.ai",
        "similarity": 0.89
      }
    ],
    "cluster_distribution": {
      "cluster_5": 0.65,
      "cluster_12": 0.25
    }
  },
  "dominant_cluster": 5
}
```

#### 2. Query again (should hit cache)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI and ML"}'
```

**Response:**
```json
{
  "query": "AI and ML",
  "cache_hit": true,
  "matched_query": "artificial intelligence and machine learning",
  "similarity_score": 0.91,
  ...
}
```

#### 3. Get cache statistics

```bash
curl "http://localhost:8000/cache/stats"
```

**Response:**
```json
{
  "total_entries": 1,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5,
  "avg_similarity_on_hit": 0.91,
  "cluster_distribution": {
    "5": 1
  }
}
```

#### 4. Clear the cache

```bash
curl -X DELETE "http://localhost:8000/cache"
```

**Response:**
```json
{
  "message": "Cache cleared successfully",
  "entries_deleted": 1
}
```

### Using Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Query
response = requests.post(
    f"{BASE_URL}/query",
    json={"query": "space exploration and NASA"}
)
print(response.json())

# Cache stats
stats = requests.get(f"{BASE_URL}/cache/stats")
print(stats.json())

# Clear cache
clear = requests.delete(f"{BASE_URL}/cache")
print(clear.json())
```

## Testing Different Scenarios

### 1. Exact Match

```bash
# First query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "gun control legislation"}'

# Exact same query (should be cache hit with similarity ~1.0)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "gun control legislation"}'
```

### 2. Semantic Similarity

```bash
# Original query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "recent developments in machine learning"}'

# Similar query (should be cache hit if similarity > 0.85)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "latest advances in ML research"}'
```

### 3. Different Topics

```bash
# Query 1
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "computer graphics rendering"}'

# Query 2 - completely different (should be cache miss)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "baseball statistics"}'
```

### 4. Cluster Information

```bash
# Get cluster info
curl "http://localhost:8000/clusters/info"
```

## Example Queries to Try

### Technology
- "latest GPU technologies"
- "artificial intelligence ethics"
- "computer security vulnerabilities"

### Science
- "space exploration missions"
- "quantum physics research"
- "medical breakthroughs"

### Politics
- "gun control debate"
- "election fraud claims"
- "foreign policy issues"

### Sports
- "baseball world series"
- "hockey championship"
- "basketball playoffs"

### Religion
- "atheism vs religion"
- "Christian theology"
- "religious freedom"

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t semantic-search:latest .

# Run the container
docker run -p 8000:8000 semantic-search:latest

# Or use docker-compose
docker-compose up
```

### Access the API

```bash
# Same as before, but make sure to use the correct host
curl "http://localhost:8000/health"
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cache.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

## Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Embeddings directory not found"

**Solution:**
```bash
# Run data preparation
python scripts/prepare_data.py
```

### Issue: Port 8000 already in use

**Solution:**
```bash
# Use a different port
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Issue: Out of memory

**Solution:**
```bash
# Reduce batch size in prepare_data.py
# Edit scripts/prepare_data.py:
# embeddings = vector_db.embed_documents(texts, batch_size=16)  # Instead of 32
```

## Performance Benchmarks

On a typical laptop (16GB RAM, i7 processor):

- **Cache hit**: ~2ms
- **Cache miss**: ~50ms (includes embedding + search)
- **Data preparation**: ~10-15 minutes
- **Memory usage**: ~500MB (loaded models + embeddings)

## Next Steps

1. **Explore the clustering analysis:**
   ```bash
   jupyter notebook notebooks/clustering_analysis.ipynb
   ```

2. **Read the design rationale:**
   - See `DESIGN.md` for detailed explanations
   - Understand the "why" behind each decision

3. **Experiment with cache threshold:**
   - Edit `app/main.py` line 45
   - Try different values (0.75, 0.85, 0.95)
   - Observe impact on hit rate and precision

4. **Scale up:**
   - Add more documents
   - Try different clustering models
   - Implement cache eviction policies

## Support

For issues or questions:
- Check the README.md
- Review DESIGN.md for rationale
- Run tests to verify setup
