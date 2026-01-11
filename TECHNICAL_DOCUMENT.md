# Fashion Image Retrieval System - Technical Report

**Project**: Glance ML Internship Assignment
**Date**: January 11, 2026

---

## 1. Problem Statement

Build an image retrieval system for fashion queries that performs better than vanilla CLIP, especially on compositional requests (e.g., "red tie" where color and item must be bound together).

---

## 2. Methodology and Iterative Approach

### 2.1 Baseline (Vanilla CLIP)
- **Approach**: Used CLIP visual embeddings only.
- **Observation**: Performance on compositional queries was weak, motivating a text-augmented pipeline.

### 2.2 Caption Generation
- **First attempt**: BLIP-base captions.
- **Issue**: Captions were too generic and missed fashion-specific attributes.
- **Second attempt**: BLIP-large captions.
- **Improvement**: Better detail, but still not aligned with the desired fashion-centric description style.

### 2.3 Caption Refinement with LLM
- **Approach**: Combined BLIP-base and BLIP-large captions and refined them using Groq (LLM).
- **Goal**: Produce richer, structured descriptions that emphasize apparel attributes.
- **Outcome**: Captions improved in detail and consistency, enabling better text embeddings.

### 2.4 Text-Only Retrieval
- **Approach**: Generated embeddings from refined captions and matched queries against them.
- **Outcome**: Results improved over baseline but were still not reliable enough.

### 2.5 Dual Embedding Late Fusion
- **Approach**: Combined visual similarity and textual similarity using weighted late fusion.
- **Rationale**: Visual embeddings preserve appearance, while caption embeddings capture semantic detail.
- **Formula**:

```text
final_score = w_visual * visual_score + w_textual * textual_score
```

### 2.6 Adaptive Thresholding
- **Approach**: Applied a query-wise cutoff based on the score distribution.
- **Benefit**: Filters out low-confidence matches before ranking, improving precision by reducing noisy candidates.

### 2.7 Final Outcome
The best performance was achieved with **dual embedding late fusion + adaptive thresholding**, which consistently outperformed both the vanilla CLIP baseline and caption-only retrieval.

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (One-time)                │
└─────────────────────────────────────────────────────────────┘

Images (3200)
     ↓
BLIP-base + BLIP-large → Captions
     ↓
Groq LLM Refinement → Detailed Fashion Captions
     ↓
CLIP Encoder (ViT-L/14)
     ↓
┌──────────────────────────┬─────────────────────────┐
│  Visual Embeddings       │  Caption Embeddings     │
│  (768-dim, 3200 images)  │  (768-dim, 3200 texts)  │
└──────────────────────────┴─────────────────────────┘
     ↓
Save to disk: image_embeddings.npy, caption_embeddings.npy


┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE (Query-time)             │
└─────────────────────────────────────────────────────────────┘

User Query: "red jacket and blue jeans"
     ↓
Intent Detection → Visual=0.7, Textual=0.3
     ↓
Query Parsing → Attire: "red jacket blue jeans" (60%)
                Environment: "" (40%)
     ↓
CLIP Text Encoder → Query Embedding (768-dim)
     ↓
┌────────────────────────────────────────────────────┐
│         Parallel Similarity Computation            │
├────────────────────────────────────────────────────┤
│  Visual Similarity:  query ↔ image_embeddings     │
│  Textual Similarity: query ↔ caption_embeddings   │
└────────────────────────────────────────────────────┘
     ↓
Late Fusion: 0.7 × visual_score + 0.3 × textual_score
     ↓
Adaptive Thresholding (60th percentile, min 0.15)
     ↓
Top 100 Candidates
     ↓
Cross-Encoder Reranking (top 50)
     ↓
Final Top-10 Results
```

### 3.2 Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Caption Generator** | Create rich image descriptions | BLIP-base + BLIP-large + Groq LLM |
| **Embedding Model** | Convert images/text to vectors | CLIP ViT-L/14 (768-dim) |
| **Intent Detector** | Adjust fusion weights per query | Keyword-based classification |
| **Late Fusion** | Combine visual + textual scores | Weighted sum at query time |
| **Adaptive Threshold** | Filter low-quality matches | 60th percentile or 0.15 min |
| **Reranker** | Improve top results accuracy | Cross-encoder (ms-marco-MiniLM) |

### 3.3 Data Flow

**Indexing**:
1. Load 3200 images from `test/` folder
2. Generate captions (BLIP-base + BLIP-large)
3. Refine captions with Groq LLM (80-120 words each)
4. Generate image embeddings (CLIP visual encoder)
5. Generate caption embeddings (CLIP text encoder)
6. Save both embedding sets to disk

**Retrieval**:
1. Parse query into aspects (attire + environment)
2. Detect query intent (color/context/style-focused)
3. Encode query with CLIP text encoder
4. Compute similarities (visual + textual separately)
5. Combine with adaptive weights (late fusion)
6. Apply adaptive threshold
7. Rerank top-50 with cross-encoder
8. Return top-10 results

---

## 4. Implementation Summary

### 4.1 Caption Pipeline
1. Generate BLIP-base captions.
2. Generate BLIP-large captions.
3. Combine and refine with Groq LLM to produce a fashion-focused description.

### 4.2 Embedding Pipeline
- **Visual embeddings**: CLIP image encoder.
- **Text embeddings**: CLIP text encoder on refined captions.

### 4.3 Retrieval
- Compute visual and textual similarity.
- Combine via late fusion with tunable weights.
- Apply adaptive thresholding.
- Rank final results.

---

## 5. Challenges

- **Caption bias**: BLIP outputs were aligned to its training style, not the desired fashion-specific format.
- **No domain fine-tuning**: CLIP was not fine-tuned for custom fashion attributes.
- **GPU constraints**: Limited model options and batch sizes.

---

## 6. Future Scope

- **Fine-tune CLIP** on a custom fashion dataset to improve attribute binding (e.g., "red tie" as a single concept).
- **Prompted captioning** with an LLM to produce task-specific descriptions instead of relying on fixed BLIP outputs.
- **Stronger query understanding** so compound phrases are embedded as unified concepts rather than separate tokens.

---

## 7. Conclusion

The system evolved from a vanilla CLIP baseline to a hybrid visual-text approach. Caption refinement via LLM and late fusion with adaptive thresholding delivered the most consistent improvements for fashion image retrieval.

---

**End of Technical Report**
