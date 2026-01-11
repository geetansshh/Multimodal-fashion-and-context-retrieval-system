# Fashion & Context Retrieval System

An intelligent multimodal search engine for fashion images that understands both **what** someone is wearing and **where** they are, using advanced CLIP embeddings and cross-encoder reranking.

## 🎯 Project Overview

This system solves the problem of finding specific fashion images based on natural language descriptions. Unlike vanilla CLIP applications, it:

1. **Multi-Aspect Understanding**: Separates and weighs attire vs. environment aspects
2. **Dual Embedding Strategy**: Combines visual (CLIP image) and textual (caption) embeddings
3. **Smart Reranking**: Uses cross-encoder to refine top results for better accuracy
4. **Compositional Queries**: Handles complex queries like "red tie and white shirt in office"

## 🏗️ Architecture

### Part A: Indexer (`indexer.py`)

Creates a searchable index from fashion images:

```
Images + Captions → CLIP Embeddings → Vector Storage
```

**Features:**
- Generates visual embeddings using CLIP ViT-L/14 (large model for accuracy)
- Generates textual embeddings from detailed captions
- Stores embeddings as numpy arrays for fast retrieval
- One-time indexing process (no need to re-run unless data changes)

### Part B: Retriever (`retriever.py`)

Searches the index using natural language queries:

```
Query → Aspect Parsing → Multi-Score Fusion → Reranking → Results
```

**Features:**
- **Query Parsing**: Automatically splits queries into attire and environment
- **Multi-Aspect Scoring**: 
  - Attire: 60% weight (colors, clothing items)
  - Environment: 40% weight (location, context)
- **Dual Embedding Matching**: Compares against both visual and caption embeddings
- **Cross-Encoder Reranking**: Refines top-100 results using a more powerful model

## 📊 Why This Approach?

### Problem with Vanilla CLIP
- Struggles with compositional understanding ("red shirt, blue pants")
- Can't weight different aspects (attire vs. environment)
- Single-stage retrieval may miss nuanced matches

### Our Solution
1. **Detailed Captions**: Rich descriptions capture every detail
2. **Aspect-Based Weighting**: Attire gets more weight than environment (configurable)
3. **Hybrid Matching**: Visual + textual embeddings for robustness
4. **Reranking**: Cross-encoder provides fine-grained relevance scoring

## 🚀 Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Build Index (One-Time)

```bash
python indexer.py
```

This will:
- Load captions from `data/detailed_captions_100.json`
- Generate embeddings for all images
- Save index to `index/` directory

**Output Files:**
- `index/image_embeddings.npy` - Visual embeddings
- `index/caption_embeddings.npy` - Textual embeddings
- `index/image_paths.json` - Image path mappings
- `index/metadata.json` - Index metadata

## 🔍 Usage

### Interactive Search

```bash
python search.py
```

This will:
- Check if index exists (build if needed)
- Start interactive search interface
- Prompt for search queries and number of results
- Display results with images and scores

### Programmatic Search

```python
from retriever import FashionRetriever

# Initialize retriever (loads index)
retriever = FashionRetriever()

# Search
results = retriever.search("A person in a bright yellow raincoat", top_k=10)

# Display results
retriever.display_results(results)
```

### Example Queries

```python
# Attribute-specific
results = retriever.search("A person in a bright yellow raincoat")

# Contextual/Place
results = retriever.search("Professional business attire inside a modern office")

# Complex semantic
results = retriever.search("Someone wearing a blue shirt sitting on a park bench")

# Style inference
results = retriever.search("Casual weekend outfit for a city walk")

# Compositional
results = retriever.search("A red tie and a white shirt in a formal setting")
```

## 📁 Project Structure

```
glance/
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── embedding_generator.py    # CLIP embedding generation
├── indexer.py               # Index creation
├── retriever.py             # Search/retrieval engine
├── search.py                # Interactive search interface
├── requirements.txt         # Python dependencies
├── data/
│   └── detailed_captions_final.json  # Image captions
├── test/                    # Fashion images
│   └── *.jpg
└── index/                   # Generated index files
    ├── image_embeddings.npy
    ├── caption_embeddings.npy
    └── image_paths.json
```

## 🎓 Design Principles

### SOLID
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Easy to extend with new models or rerankers
- **Dependency Inversion**: Uses configuration abstractions

### DRY (Don't Repeat Yourself)
- Shared utilities in `utils.py`
- Reusable embedding methods
- Centralized configuration

### KISS (Keep It Simple, Stupid)
- Clear module separation
- Straightforward data flow
- Minimal complexity in each component

### YAGNI (You Aren't Gonna Need It)
- No unnecessary features
- Focused on core retrieval functionality
- Extensions planned but not implemented prematurely

### SOC (Separation of Concerns)
- Indexing separate from retrieval
- Embedding generation isolated
- Configuration separate from logic

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model settings
CLIP_MODEL_NAME = "ViT-L/14"  # Or "ViT-B/32" for faster but less accurate

# Retrieval weights
ATTIRE_WEIGHT = 0.6      # Weight for clothing/attire
ENVIRONMENT_WEIGHT = 0.4  # Weight for location/context

# Top-K settings
TOP_K_INITIAL = 100      # Initial retrieval pool
TOP_K_FINAL = 10         # Final results
RERANK_TOP_K = 50        # How many to rerank

# Reranker model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## 🔬 Evaluation Queries (Assignment Requirements)

The system is evaluated on these query types:

1. **Attribute Specific**: "A person in a bright yellow raincoat"
   - Focus: Color and clothing type
   - Weight: 100% attire

2. **Contextual/Place**: "Professional business attire inside a modern office"
   - Focus: Setting and style
   - Weight: 60% attire, 40% environment

3. **Complex Semantic**: "Someone wearing a blue shirt sitting on a park bench"
   - Focus: Clothing + action + location
   - Weight: 60% attire, 40% environment

4. **Style Inference**: "Casual weekend outfit for a city walk"
   - Focus: Style understanding
   - Weight: 70% attire, 30% environment

5. **Compositional**: "A red tie and a white shirt in a formal setting"
   - Focus: Multiple items + context
   - Weight: 65% attire, 35% environment

## 🚀 Future Work

### 1. Location & Weather Enhancement

**Approach**:
- Add location/weather metadata to captions
- Create separate embeddings for location features
- Use geolocation APIs to enrich queries

**Implementation**:
```python
# Add to config.py
LOCATION_WEIGHT = 0.2
WEATHER_WEIGHT = 0.1

# Modify retriever to handle location queries
aspects = parse_query_aspects(query)  
# Returns: {'attire': ..., 'environment': ..., 'location': ..., 'weather': ...}
```

### 2. Improving Precision

**Strategy 1: Fine-tuning CLIP**
- Fine-tune CLIP on fashion-specific dataset
- Use contrastive learning with hard negatives
- Focus on compositional understanding

**Strategy 2: Attribute Extraction**
- Use dedicated attribute classifier
- Extract: color, pattern, style, fabric
- Create attribute-specific embeddings

**Strategy 3: Multi-Modal Fusion**
- Add fashion-specific models (FashionCLIP, FashionBERT)
- Ensemble different embedding models
- Late fusion of scores

**Strategy 4: Query Understanding**
- Use LLM (GPT-3.5) to parse complex queries
- Extract structured attributes
- Generate multiple query variations

**Strategy 5: User Feedback**
- Implement relevance feedback
- Learn user preferences
- Personalized reranking

### 3. Scalability to 1M Images

**Vector Database**: Replace numpy arrays with specialized DB
```python
# Use FAISS, Milvus, or Weaviate
import faiss

index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
index.add(embeddings)  # Add embeddings
D, I = index.search(query_embedding, k)  # Fast search
```

**Approximate Nearest Neighbors**:
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- Trade-off: Speed vs. recall

**Batch Processing**:
- Process images in parallel
- Distributed indexing
- Incremental index updates

## 📝 License

MIT License - Feel free to use and modify!

## 👤 Author

Built for Glance ML Internship Assignment

---

**Note**: This system prioritizes **accuracy** over engineering complexity, using state-of-the-art models (CLIP ViT-L/14, cross-encoder reranking) while maintaining clean, maintainable code.
