"""Generate visual and textual embeddings using CLIP"""
import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm

from config import CLIP_MODEL_NAME, DEVICE, BATCH_SIZE


class EmbeddingGenerator:
    """Generates embeddings using CLIP model for images and text."""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
    def generate_image_embedding(self, image_path: Path) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(768)  # CLIP ViT-L/14 embedding dimension
    
    def generate_image_embeddings_batch(self, image_paths: List[Path], 
                                       batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process in each batch
            
        Returns:
            Array of normalized embeddings (N x embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating image embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    batch_images.append(self.preprocess(image))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # Add a blank image as placeholder
                    batch_images.append(torch.zeros(3, 224, 224))
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            text_input = clip.tokenize([text], truncate=True).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_text(text_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return np.zeros(768)
    
    def generate_text_embeddings_batch(self, texts: List[str], 
                                      batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for multiple text strings in batches.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process in each batch
            
        Returns:
            Array of normalized embeddings (N x embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating text embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                text_inputs = clip.tokenize(batch_texts, truncate=True).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode_text(text_inputs)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add zero embeddings for this batch
                embeddings.append(np.zeros((len(batch_texts), 768)))
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        # Clip to [0, 1] range
        return np.clip(similarity, 0, 1)
    
    def compute_similarities_batch(self, query_embedding: np.ndarray, 
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between a query and multiple embeddings.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            embeddings: Array of embeddings (2D array: N x embedding_dim)
            
        Returns:
            Array of similarity scores
        """
        # Embeddings are normalized, so matrix multiplication gives cosine similarity
        similarities = np.dot(embeddings, query_embedding)
        # Clip to [0, 1] range
        return np.clip(similarities, 0, 1)
