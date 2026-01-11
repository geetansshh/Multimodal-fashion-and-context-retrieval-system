"""Configuration settings"""
import torch
from pathlib import Path
from typing import Final


class Paths:
    """File system paths for the project."""
    
    PROJECT_ROOT: Final[Path] = Path(__file__).parent
    DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
    TEST_DIR: Final[Path] = PROJECT_ROOT / "test"
    INDEX_DIR: Final[Path] = PROJECT_ROOT / "index"
    
    # File paths
    CAPTIONS_FILE: Final[Path] = DATA_DIR / "detailed_captions_final.json"
    IMAGE_EMBEDDINGS_FILE: Final[Path] = INDEX_DIR / "image_embeddings.npy"
    CAPTION_EMBEDDINGS_FILE: Final[Path] = INDEX_DIR / "caption_embeddings.npy"
    IMAGE_PATHS_FILE: Final[Path] = INDEX_DIR / "image_paths.json"
    METADATA_FILE: Final[Path] = INDEX_DIR / "metadata.json"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.INDEX_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)


class ModelConfig:
    """Model-related configuration."""
    
    # CLIP model for embeddings
    CLIP_MODEL_NAME: Final[str] = "ViT-L/14"
    
    # Caption generation models
    BLIP_BASE: Final[str] = "Salesforce/blip-image-captioning-base"
    BLIP_LARGE: Final[str] = "Salesforce/blip-image-captioning-large"
    
    # Reranking model
    RERANK_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Device selection
    @staticmethod
    def get_device() -> str:
        """Automatically select best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


class RetrievalConfig:
    """Retrieval and search configuration."""
    
    TOP_K_INITIAL: Final[int] = 100  # Initial retrieval pool
    TOP_K_FINAL: Final[int] = 10  # Final results after reranking
    RERANK_TOP_K: Final[int] = 50  # Number of candidates to rerank
    
    # Multi-aspect weights
    ATTIRE_WEIGHT: Final[float] = 0.6  # Weight for attire matching
    ENVIRONMENT_WEIGHT: Final[float] = 0.4  # Weight for environment matching


class FusionConfig:
    """Late fusion configuration for combining visual and textual signals."""
    
    # Late fusion weights (visual vs textual)
    VISUAL_WEIGHT: Final[float] = 0.6  # Weight for visual (image) embeddings
    TEXTUAL_WEIGHT: Final[float] = 0.4  # Weight for textual (caption) embeddings
    
    # Intent-based weights (can be adjusted dynamically)
    COLOR_FOCUSED_VISUAL_WEIGHT: Final[float] = 0.7  # More visual for color queries
    STYLE_FOCUSED_VISUAL_WEIGHT: Final[float] = 0.6  # Slightly more visual for style
    CONTEXT_FOCUSED_VISUAL_WEIGHT: Final[float] = 0.3  # More textual for context
    
    # Adaptive threshold settings
    USE_ADAPTIVE_THRESHOLD: Final[bool] = True
    MIN_SIMILARITY_THRESHOLD: Final[float] = 0.15  # Minimum score to consider
    ADAPTIVE_PERCENTILE: Final[float] = 0.6  # Use top 60% of scores


class ProcessingConfig:
    """Processing and performance configuration."""
    
    BATCH_SIZE: Final[int] = 32  # Batch size for embedding generation
    MAX_CAPTION_LENGTH: Final[int] = 250  # Max tokens for caption generation
    NUM_BEAMS: Final[int] = 4  # Beam search size for caption generation


# Initialize directories on import
Paths.ensure_directories()

# Convenience exports for backward compatibility
PROJECT_ROOT = Paths.PROJECT_ROOT
DATA_DIR = Paths.DATA_DIR
TEST_DIR = Paths.TEST_DIR
INDEX_DIR = Paths.INDEX_DIR
CAPTIONS_FILE = Paths.CAPTIONS_FILE
IMAGE_EMBEDDINGS_FILE = Paths.IMAGE_EMBEDDINGS_FILE
CAPTION_EMBEDDINGS_FILE = Paths.CAPTION_EMBEDDINGS_FILE
IMAGE_PATHS_FILE = Paths.IMAGE_PATHS_FILE
METADATA_FILE = Paths.METADATA_FILE
CLIP_MODEL_NAME = ModelConfig.CLIP_MODEL_NAME
DEVICE = ModelConfig.get_device()
TOP_K_INITIAL = RetrievalConfig.TOP_K_INITIAL
TOP_K_FINAL = RetrievalConfig.TOP_K_FINAL
ATTIRE_WEIGHT = RetrievalConfig.ATTIRE_WEIGHT
ENVIRONMENT_WEIGHT = RetrievalConfig.ENVIRONMENT_WEIGHT
RERANK_MODEL = ModelConfig.RERANK_MODEL
RERANK_TOP_K = RetrievalConfig.RERANK_TOP_K
BATCH_SIZE = ProcessingConfig.BATCH_SIZE
VISUAL_WEIGHT = FusionConfig.VISUAL_WEIGHT
TEXTUAL_WEIGHT = FusionConfig.TEXTUAL_WEIGHT
USE_ADAPTIVE_THRESHOLD = FusionConfig.USE_ADAPTIVE_THRESHOLD
MIN_SIMILARITY_THRESHOLD = FusionConfig.MIN_SIMILARITY_THRESHOLD
