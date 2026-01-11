"""Fashion image retrieval with multi-aspect scoring and reranking"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sentence_transformers import CrossEncoder

from config import (
    IMAGE_EMBEDDINGS_FILE, CAPTION_EMBEDDINGS_FILE, IMAGE_PATHS_FILE,
    CAPTIONS_FILE, TOP_K_INITIAL, TOP_K_FINAL, ATTIRE_WEIGHT,
    ENVIRONMENT_WEIGHT, RERANK_MODEL, RERANK_TOP_K,
    VISUAL_WEIGHT, TEXTUAL_WEIGHT, USE_ADAPTIVE_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD, FusionConfig
)
from embedding_generator import EmbeddingGenerator
from utils import (
    load_json, parse_query_aspects, calculate_weighted_score,
    format_results
)


class FashionRetriever:
    """Retrieves fashion images based on natural language queries."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.reranker = None
        
        print("Loading search index...")
        self._load_index()
        print("Retriever initialized successfully!")
    
    def _load_index(self) -> None:
        """Load precomputed embeddings and metadata from disk."""
        # Load embeddings
        self.image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
        self.caption_embeddings = np.load(CAPTION_EMBEDDINGS_FILE)
        
        # Load image paths
        image_paths_list = load_json(IMAGE_PATHS_FILE)
        self.image_paths = [Path(p) for p in image_paths_list]
        
        # Load caption data
        self.captions_data = load_json(CAPTIONS_FILE)
        
        print(f"Loaded index with {len(self.image_paths)} images")
    
    def _load_reranker(self) -> None:
        """Lazy load the cross-encoder reranker."""
        if self.reranker is None:
            print(f"Loading reranker model: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)
            print("Reranker loaded!")
    
    def _detect_query_intent(self, query: str) -> Dict[str, float]:
        """
        Detect query intent and return fusion weights.
        
        Args:
            query: Search query string
            
        Returns:
            Dict with 'visual' and 'textual' weights
        """
        query_lower = query.lower()
        
        # Color-focused queries (more visual)
        color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                         'pink', 'purple', 'orange', 'brown', 'gray', 'grey',
                         'bright', 'dark', 'light', 'colorful', 'vibrant']
        color_count = sum(1 for kw in color_keywords if kw in query_lower)
        
        # Context/setting-focused queries (more textual)
        context_keywords = ['office', 'park', 'street', 'indoor', 'outdoor',
                           'professional', 'casual', 'formal', 'business',
                           'setting', 'scene', 'background', 'environment']
        context_count = sum(1 for kw in context_keywords if kw in query_lower)
        
        # Style-focused queries (balanced)
        style_keywords = ['style', 'fashion', 'trendy', 'elegant', 'chic',
                         'modern', 'classic', 'vintage', 'sporty']
        style_count = sum(1 for kw in style_keywords if kw in query_lower)
        
        # Determine intent and set weights
        if color_count >= 2 or (color_count == 1 and context_count == 0):
            # Color-focused: prefer visual
            return {
                'visual': FusionConfig.COLOR_FOCUSED_VISUAL_WEIGHT,
                'textual': 1.0 - FusionConfig.COLOR_FOCUSED_VISUAL_WEIGHT
            }
        elif context_count >= 1:
            # Context-focused: prefer textual
            return {
                'visual': FusionConfig.CONTEXT_FOCUSED_VISUAL_WEIGHT,
                'textual': 1.0 - FusionConfig.CONTEXT_FOCUSED_VISUAL_WEIGHT
            }
        elif style_count >= 1:
            # Style-focused: slightly prefer visual
            return {
                'visual': FusionConfig.STYLE_FOCUSED_VISUAL_WEIGHT,
                'textual': 1.0 - FusionConfig.STYLE_FOCUSED_VISUAL_WEIGHT
            }
        else:
            # Default: balanced
            return {
                'visual': VISUAL_WEIGHT,
                'textual': TEXTUAL_WEIGHT
            }
    
    def search(self, query: str, top_k: int = TOP_K_FINAL, 
               use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Search for images matching the query using late fusion and multi-aspect scoring.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            List of result dictionaries with image paths and scores
        """
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        # Step 1: Detect query intent for adaptive fusion
        fusion_weights = self._detect_query_intent(query)
        print(f"Query intent: Visual={fusion_weights['visual']:.2f}, Textual={fusion_weights['textual']:.2f}")
        
        # Step 2: Parse query into aspects
        aspects = parse_query_aspects(query)
        print(f"Attire aspect: '{aspects['attire']}'")
        print(f"Environment aspect: '{aspects['environment']}'")
        
        # Step 3: Generate embeddings for query aspects
        attire_embedding = self.embedding_generator.generate_text_embedding(
            aspects['attire']
        )
        
        # Only compute environment embedding if we have environment info
        if aspects['environment']:
            env_embedding = self.embedding_generator.generate_text_embedding(
                aspects['environment']
            )
            has_env = True
        else:
            env_embedding = None
            has_env = False
        
        # Step 4: Compute similarities with LATE FUSION (separate visual & textual)
        print("\nComputing similarities with late fusion...")
        
        # Attire similarities - keep visual and textual separate
        attire_visual_sim = self.embedding_generator.compute_similarities_batch(
            attire_embedding, self.image_embeddings
        )
        attire_textual_sim = self.embedding_generator.compute_similarities_batch(
            attire_embedding, self.caption_embeddings
        )
        
        # Late fusion: combine with intent-based weights
        attire_scores = (fusion_weights['visual'] * attire_visual_sim + 
                        fusion_weights['textual'] * attire_textual_sim)
        
        # Environment similarities - keep visual and textual separate
        if has_env:
            env_visual_sim = self.embedding_generator.compute_similarities_batch(
                env_embedding, self.image_embeddings
            )
            env_textual_sim = self.embedding_generator.compute_similarities_batch(
                env_embedding, self.caption_embeddings
            )
            # Late fusion for environment
            env_scores = (fusion_weights['visual'] * env_visual_sim + 
                         fusion_weights['textual'] * env_textual_sim)
        else:
            # If no environment specified, give neutral score
            env_scores = np.ones(len(attire_scores)) * 0.5
        
        # Step 5: Calculate weighted combined scores (attire + environment)
        weights = self._determine_weights(has_env)
        combined_scores = np.array([
            calculate_weighted_score(
                attire_scores[i], env_scores[i],
                weights['attire'], weights['environment']
            )
            for i in range(len(self.image_paths))
        ])
        
        # Keep original scores for reranking (before filtering)
        original_scores = combined_scores.copy()
        
        # Step 6: Apply adaptive thresholding
        if USE_ADAPTIVE_THRESHOLD:
            threshold = max(
                MIN_SIMILARITY_THRESHOLD,
                np.percentile(combined_scores, FusionConfig.ADAPTIVE_PERCENTILE * 100)
            )
            print(f"Adaptive threshold: {threshold:.4f}")
            # Filter low-quality matches
            valid_mask = combined_scores >= threshold
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                filtered_scores = combined_scores[valid_indices]
                candidate_pool = valid_indices
            else:
                # Fallback if threshold too strict
                filtered_scores = combined_scores
                candidate_pool = np.arange(len(combined_scores))
        else:
            filtered_scores = combined_scores
            candidate_pool = np.arange(len(combined_scores))
        
        # Step 7: Get top-k candidates for reranking
        top_in_pool = np.argsort(filtered_scores)[::-1][:min(TOP_K_INITIAL, len(filtered_scores))]
        top_indices = candidate_pool[top_in_pool]
        
        # Step 8: Apply reranking if enabled
        if use_reranking:
            print(f"\nApplying cross-encoder reranking to top {min(RERANK_TOP_K, len(top_indices))} candidates...")
            top_indices = self._rerank(
                query, top_indices[:RERANK_TOP_K], original_scores
            )
        
        # Step 9: Prepare results
        results = []
        for idx in top_indices[:top_k]:
            image_path = self.image_paths[idx]
            rel_path = self._get_relative_path(image_path)
            metadata = self.captions_data.get(rel_path, {})
            
            results.append((
                image_path,
                float(original_scores[idx]),
                metadata
            ))
        
        return format_results(results, top_k)
    
    def _determine_weights(self, has_env: bool) -> Dict[str, float]:
        """
        Determine weights for attire and environment based on query.
        
        Args:
            has_env: Whether environment information is present in query
            
        Returns:
            Dictionary with 'attire' and 'environment' weights
        """
        if has_env:
            # Use configured weights
            return {
                'attire': ATTIRE_WEIGHT,
                'environment': ENVIRONMENT_WEIGHT
            }
        else:
            # No environment specified, focus entirely on attire
            return {
                'attire': 1.0,
                'environment': 0.0
            }
    
    def _rerank(self, query: str, candidate_indices: np.ndarray, 
                initial_scores: np.ndarray) -> np.ndarray:
        """
        Rerank candidates using cross-encoder for better accuracy.
        
        Args:
            query: Search query
            candidate_indices: Indices of candidate images
            initial_scores: Initial similarity scores
            
        Returns:
            Reranked indices
        """
        self._load_reranker()
        
        # Prepare query-caption pairs for reranking
        pairs = []
        for idx in candidate_indices:
            rel_path = self._get_relative_path(self.image_paths[idx])
            caption = self.captions_data.get(rel_path, {}).get('detailed_caption', '')
            pairs.append([query, caption])
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Extract initial scores for candidates
        # Handle case where initial_scores may be filtered (from adaptive thresholding)
        candidate_initial_scores = np.array([initial_scores[idx] for idx in candidate_indices])
        
        # Combine with initial scores (weighted average)
        # Give more weight to reranker (70%) since it's more accurate
        combined = 0.3 * candidate_initial_scores + 0.7 * rerank_scores
        
        # Sort by combined score
        reranked_order = np.argsort(combined)[::-1]
        
        return candidate_indices[reranked_order]
    
    def _get_relative_path(self, abs_path: Path) -> str:
        """Convert absolute path to relative path format."""
        try:
            from config import TEST_DIR
            return str(abs_path.relative_to(TEST_DIR.parent))
        except ValueError:
            return str(abs_path)
    
    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Display search results in a readable format.
        
        Args:
            results: List of result dictionaries
        """
        print("\n" + "=" * 60)
        print("SEARCH RESULTS")
        print("=" * 60)
        
        for result in results:
            print(f"\nRank {result['rank']}: Score = {result['score']:.4f}")
            print(f"Image: {result['image_path']}")
            print(f"Caption: {result['caption'][:150]}...")
            if result['attributes']:
                print(f"Attributes: {', '.join(result['attributes'][:5])}")
            print("-" * 60)


def main():
    """Main function for testing the retriever."""
    retriever = FashionRetriever()
    
    # Example queries matching the assignment requirements
    test_queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=5)
        retriever.display_results(results)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
