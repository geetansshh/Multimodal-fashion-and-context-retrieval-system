# Fashion Image Retrieval System

A multimodal search system for fashion images using CLIP embeddings, detailed captions, and cross-encoder reranking.

## Overview

Retrieves fashion images based on natural language queries by combining visual and textual signals. Handles compositional queries like "red tie and white shirt in office".

**Key features:**
- Dual embeddings (visual + caption)
- Query parsing (attire + environment)
- Cross-encoder reranking
- Adaptive similarity thresholding

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# Clone repository
git clone https://github.com/geetansshh/Multimodal-fashion-and-context-retrieval-system.git
cd glance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import clip; print('Installation successful')"
```

### Build Index

```bash
# One-time indexing (~5 minutes)
python indexer.py
```

### Run Search

```bash
# Interactive search
python search.py
```

## Usage

### Interactive Search
```bash
python search.py
```

### Python API
```python
from retriever import FashionRetriever

retriever = FashionRetriever()
results = retriever.search("bright yellow raincoat", top_k=10)
retriever.display_results(results)
```

## Architecture

**Indexing:**
1. Generate captions (BLIP-base + BLIP-large)
2. Refine with Groq LLM
3. Create CLIP embeddings (ViT-L/14)
4. Save visual and caption embeddings

**Retrieval:**
1. Parse query into aspects
2. Compute visual + textual similarity
3. Late fusion with adaptive weights
4. Adaptive thresholding
5. Cross-encoder reranking
6. Return top results

## Project Structure

```
glance/
├── config.py                 # Configuration
├── utils.py                  # Helper functions
├── embedding_generator.py    # CLIP embeddings
├── indexer.py               # Index builder
├── retriever.py             # Search engine
├── search.py                # Interactive interface
├── data/
│   └── detailed_captions_final.json
├── test/                    # Images
└── index/                   # Generated embeddings
```

## Configuration

Key settings in `config.py`:

```python
CLIP_MODEL_NAME = "ViT-L/14"
ATTIRE_WEIGHT = 0.6
ENVIRONMENT_WEIGHT = 0.4
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Example Queries

- "A person in a bright yellow raincoat"
- "Professional business attire inside a modern office"
- "Someone wearing a blue shirt sitting on a park bench"
- "Casual weekend outfit for a city walk"
- "A red tie and a white shirt in a formal setting"

## Performance

- Index build: ~5 minutes (3,200 images)
- Query time: ~200ms
- Top-5 accuracy: ~84% (vs 55% vanilla CLIP)

## Technical Details

**Why dual embeddings?**
- Visual embeddings capture appearance
- Caption embeddings capture semantic details
- Combined scores improve compositional query accuracy

**Why reranking?**
- First stage: CLIP retrieves ~100 candidates (fast)
- Second stage: Cross-encoder reranks top-50 (accurate)
- Applied only to top candidates, keeping latency bounded

**Why late fusion?**
- Separate visual and textual scores
- Adaptive weights based on query type
- Can tune without re-indexing

## License

MIT

---

Built for Glance ML Internship Assignment
