"""Create and store search index from images and captions"""
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm

from config import (
    CAPTIONS_FILE, IMAGE_EMBEDDINGS_FILE, CAPTION_EMBEDDINGS_FILE,
    IMAGE_PATHS_FILE, METADATA_FILE, TEST_DIR
)
from embedding_generator import EmbeddingGenerator
from utils import load_json, save_json


class FashionIndexer:
    """Creates and manages the fashion image search index."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.image_paths: List[Path] = []
        self.captions_data: Dict = {}
        self.image_embeddings: np.ndarray = None
        self.caption_embeddings: np.ndarray = None
        
    def load_captions(self) -> None:
        """Load caption data from JSON file."""
        print(f"Loading captions from {CAPTIONS_FILE}...")
        self.captions_data = load_json(CAPTIONS_FILE)
        print(f"Loaded {len(self.captions_data)} image captions")
        
    def build_image_list(self) -> None:
        """Build list of image paths from caption data."""
        print("Building image path list...")
        self.image_paths = []
        
        for image_rel_path in self.captions_data.keys():
            # Convert relative path to absolute
            image_path = Path(image_rel_path)
            if not image_path.is_absolute():
                image_path = TEST_DIR.parent / image_path
            
            if image_path.exists():
                self.image_paths.append(image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        print(f"Found {len(self.image_paths)} valid images")
    
    def generate_image_embeddings(self) -> None:
        """Generate CLIP embeddings for all images."""
        print("Generating image embeddings...")
        self.image_embeddings = self.embedding_generator.generate_image_embeddings_batch(
            self.image_paths
        )
        print(f"Generated embeddings shape: {self.image_embeddings.shape}")
    
    def generate_caption_embeddings(self) -> None:
        """Generate CLIP embeddings for all detailed captions."""
        print("Generating caption embeddings...")
        
        # Extract detailed captions in the same order as image_paths
        captions = []
        for image_path in self.image_paths:
            # Convert to relative path format used in captions_data
            rel_path = self._get_relative_path(image_path)
            caption_data = self.captions_data.get(rel_path, {})
            
            # Use detailed_caption which contains all the information
            caption = caption_data.get('detailed_caption', '')
            if not caption:
                # Fallback to base_caption
                caption = caption_data.get('base_caption', '')
            
            captions.append(caption)
        
        self.caption_embeddings = self.embedding_generator.generate_text_embeddings_batch(
            captions
        )
        print(f"Generated caption embeddings shape: {self.caption_embeddings.shape}")
    
    def save_index(self) -> None:
        """Save all index data to disk."""
        print("Saving index to disk...")
        
        # Save embeddings as numpy arrays
        np.save(IMAGE_EMBEDDINGS_FILE, self.image_embeddings)
        np.save(CAPTION_EMBEDDINGS_FILE, self.caption_embeddings)
        
        # Save image paths as JSON (convert Path to string)
        image_paths_str = [self._get_relative_path(p) for p in self.image_paths]
        save_json(image_paths_str, IMAGE_PATHS_FILE)
        
        # Save metadata
        metadata = {
            'num_images': len(self.image_paths),
            'embedding_dim': int(self.image_embeddings.shape[1]),
            'model': self.embedding_generator.model.__class__.__name__,
        }
        save_json(metadata, METADATA_FILE)
        
        print(f"Index saved successfully!")
        print(f"  - Image embeddings: {IMAGE_EMBEDDINGS_FILE}")
        print(f"  - Caption embeddings: {CAPTION_EMBEDDINGS_FILE}")
        print(f"  - Image paths: {IMAGE_PATHS_FILE}")
        print(f"  - Metadata: {METADATA_FILE}")
    
    def build_index(self) -> None:
        """
        Build the complete search index.
        This is the main method that orchestrates the indexing process.
        """
        print("=" * 60)
        print("Building Fashion Retrieval Index")
        print("=" * 60)
        
        # Step 1: Load caption data
        self.load_captions()
        
        # Step 2: Build image path list
        self.build_image_list()
        
        # Step 3: Generate image embeddings
        self.generate_image_embeddings()
        
        # Step 4: Generate caption embeddings
        self.generate_caption_embeddings()
        
        # Step 5: Save everything
        self.save_index()
        
        print("=" * 60)
        print("Indexing Complete!")
        print("=" * 60)
    
    def _get_relative_path(self, abs_path: Path) -> str:
        """Convert absolute path to relative path format used in captions."""
        try:
            # Try to get relative path from project root
            return str(abs_path.relative_to(TEST_DIR.parent))
        except ValueError:
            # If not relative to project root, use as is
            return str(abs_path)


def main():
    """Main function to run the indexer."""
    indexer = FashionIndexer()
    indexer.build_index()


if __name__ == "__main__":
    main()
