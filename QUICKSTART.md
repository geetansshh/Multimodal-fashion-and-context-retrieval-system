# Quick Start Guide

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### 2. Install Dependencies

```bash
# Navigate to project directory
cd /Users/geetansh/Desktop/glance

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Note**: The CLIP installation from GitHub may take a few minutes.

### 3. Verify Installation

```python
python -c "import torch; import clip; print('✅ Installation successful!')"
```

---

## Usage

### Option 1: Run Complete Demo

```bash
python demo.py
```

This will:
1. Check if index exists (build if not found)
2. Run all 5 test queries from the assignment
3. Display results for each query

**First run**: Takes ~5 minutes to build index (one-time only)
**Subsequent runs**: Loads index in <10 seconds

### Option 2: Build Index Separately

```bash
# Build index (one-time process)
python indexer.py

# Then run retrieval
python retriever.py
```

### Option 3: Use in Python Code

```python
from retriever import FashionRetriever

# Initialize (loads index)
retriever = FashionRetriever()

# Search
results = retriever.search("red dress in a formal setting", top_k=10)

# Display results
retriever.display_results(results)

# Access raw results
for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Score: {result['score']}")
    print(f"Caption: {result['caption']}")
```

---

## Project Structure

```
glance/
├── config.py                    # Configuration settings
├── utils.py                     # Utility functions
├── embedding_generator.py       # CLIP embedding generation
├── indexer.py                   # Index builder (Part A)
├── retriever.py                 # Search engine (Part B)
├── demo.py                      # Complete demo script
├── requirements.txt             # Dependencies
├── README.md                    # Main documentation
├── APPROACH.md                  # Technical approach document
├── QUICKSTART.md               # This file
│
├── data/
│   └── detailed_captions_100.json  # Image captions (provided)
│
├── test/                        # Fashion images (provided)
│   └── *.jpg
│
└── index/                       # Generated index files (created by indexer)
    ├── image_embeddings.npy
    ├── caption_embeddings.npy
    ├── image_paths.json
    └── metadata.json
```

---

## Configuration

Edit `config.py` to customize:

```python
# Model settings
CLIP_MODEL_NAME = "ViT-L/14"  # Large model for accuracy
# Or use "ViT-B/32" for faster but less accurate

# Weights for multi-aspect scoring
ATTIRE_WEIGHT = 0.6           # 60% weight on clothing
ENVIRONMENT_WEIGHT = 0.4      # 40% weight on location

# Retrieval settings
TOP_K_INITIAL = 100           # Initial candidate pool
TOP_K_FINAL = 10              # Final results returned
RERANK_TOP_K = 50             # Candidates for reranking

# Reranker model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## Example Queries

### Attribute Specific
```python
results = retriever.search("A person in a bright yellow raincoat")
```

### Contextual/Place
```python
results = retriever.search("Professional business attire inside a modern office")
```

### Complex Semantic
```python
results = retriever.search("Someone wearing a blue shirt sitting on a park bench")
```

### Style Inference
```python
results = retriever.search("Casual weekend outfit for a city walk")
```

### Compositional
```python
results = retriever.search("A red tie and a white shirt in a formal setting")
```

---

## Troubleshooting

### Issue: "Import clip could not be resolved"
**Solution**: Make sure CLIP is installed from GitHub:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Issue: "CUDA out of memory"
**Solution**: Set device to CPU in `config.py`:
```python
DEVICE = "cpu"
```

### Issue: "Index files not found"
**Solution**: Run the indexer first:
```bash
python indexer.py
```

### Issue: Slow performance
**Solutions**:
1. Use smaller CLIP model: `CLIP_MODEL_NAME = "ViT-B/32"`
2. Reduce `RERANK_TOP_K` in config
3. Disable reranking: `retriever.search(query, use_reranking=False)`
4. Use GPU if available: `DEVICE = "cuda"`

---

## Performance Expectations

### Indexing (One-time)
- **Time**: ~5 minutes for 1,000 images (CPU)
- **Storage**: ~50MB for embeddings
- **Memory**: ~2GB RAM

### Retrieval (Per query)
- **Time**: ~2 seconds with reranking
- **Time**: ~0.5 seconds without reranking
- **Memory**: ~1GB RAM

### With GPU
- **Indexing**: ~1-2 minutes
- **Retrieval**: ~0.3 seconds with reranking

---

## Testing the System

### Test Query Success Criteria
A good result should:
1. Match the specified attributes (colors, items)
2. Match the environment/context
3. Have high relevance scores (>0.7)
4. Appear in top 10 results

### Expected Results for Test Queries

**Query 1**: "A person in a bright yellow raincoat"
- Expected: Images with yellow/bright colored coats
- Key attributes: color:yellow, item:coat/jacket

**Query 2**: "Professional business attire inside a modern office"
- Expected: Formal wear (suits, blazers) in indoor settings
- Key attributes: formal, professional, office, indoor

**Query 3**: "Someone wearing a blue shirt sitting on a park bench"
- Expected: Casual clothing with outdoor context
- Key attributes: color:blue, item:shirt, outdoor, park

**Query 4**: "Casual weekend outfit for a city walk"
- Expected: Relaxed, comfortable clothing
- Key attributes: casual, outdoor, urban

**Query 5**: "A red tie and a white shirt in a formal setting"
- Expected: Formal attire with specified colors
- Key attributes: color:red, color:white, item:tie, item:shirt, formal

---

## Next Steps

1. **Explore Results**: Run demo and examine retrieved images
2. **Try Custom Queries**: Experiment with your own searches
3. **Tune Weights**: Adjust `ATTIRE_WEIGHT` and `ENVIRONMENT_WEIGHT` in config
4. **Read Documentation**: 
   - `README.md` for overview
   - `APPROACH.md` for technical details

---

## Support

For issues or questions:
1. Check `APPROACH.md` for detailed technical explanations
2. Review configuration in `config.py`
3. Examine code comments in source files

---

**Happy Searching! 🔍👗**
