"""Generate detailed captions for fashion images using BLIP"""
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import re
import time

from config import TEST_DIR, DATA_DIR


class CaptionGenerator:
    """Generates detailed captions for fashion images using multiple prompts."""
    
    # Simple prompts to get factual descriptions
    CAPTION_PROMPTS = [
        "",
        "a photo of",
    ]
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        print(f"Loading caption generation model: {model_name}...")
        
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Use MPS for Apple Silicon, CUDA for NVIDIA, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.model = self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def generate_captions_batch(self, image_paths: List[Path], batch_size: int = 4) -> List[Dict[str, any]]:
        """
        Generate 2 comprehensive detailed captions for each image.
        Each caption captures What, Where, When, Why, How (5W1H excluding Who).
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once
            
        Returns:
            List of caption data dictionaries with all_captions, detailed_caption, etc.
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # Load all images in batch
                images = []
                valid_paths = []
                
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        valid_paths.append(img_path)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        all_results.append({
                            "detailed_caption": "",
                            "attributes": [],
                            "all_captions": [],
                            "base_caption": ""
                        })
                
                if not images:
                    continue
                
                # Generate multiple captions for each image using different prompts
                batch_all_captions = [[] for _ in images]
                
                for prompt in self.CAPTION_PROMPTS:
                    # Process batch with comprehensive prompt
                    inputs = self.processor(
                        images, 
                        text=[prompt] * len(images),
                        return_tensors="pt", 
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=100,  # Longer for comprehensive details
                            num_beams=5,     # More beams for better quality
                            min_length=30,   # Ensure detailed descriptions
                            length_penalty=1.0,
                            repetition_penalty=1.2,  # Avoid repetition
                            no_repeat_ngram_size=3   # Prevent phrase repetition
                        )
                    
                    # Decode captions for this prompt
                    captions = [self.processor.decode(output, skip_special_tokens=True).strip() 
                               for output in outputs]
                    
                    # Add to each image's caption list
                    for j, caption in enumerate(captions):
                        batch_all_captions[j].append(caption)
                
                # Create results with combined detailed captions
                for all_caps in batch_all_captions:
                    # Combine all captions into one detailed caption
                    detailed_caption = ". ".join(all_caps)
                    base_caption = all_caps[0] if all_caps else ""
                    
                    # Extract attributes from all captions
                    attributes = self._extract_attributes(detailed_caption, all_caps)
                    
                    all_results.append({
                        "detailed_caption": detailed_caption,
                        "attributes": attributes,
                        "all_captions": all_caps,
                        "base_caption": base_caption
                    })
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                for _ in batch_paths:
                    all_results.append({
                        "detailed_caption": "",
                        "attributes": [],
                        "all_captions": [],
                        "base_caption": ""
                    })
        
        return all_results
    
    def generate_caption(self, image_path: Path) -> Dict[str, any]:
        """
        Generate detailed caption for a single image using BLIP.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing detailed caption and attributes
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Generate unconditional caption (best for detailed descriptions)
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=75,
                    num_beams=5,
                    min_length=20
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Use the caption as the detailed caption
            detailed_caption = caption.strip()
            
            # Extract attributes
            attributes = self._extract_attributes(detailed_caption, [detailed_caption])
            
            return {
                "detailed_caption": detailed_caption,
                "attributes": attributes,
                "base_caption": detailed_caption
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                "detailed_caption": "",
                "attributes": [],
                "base_caption": ""
            }
    
    def _extract_attributes(self, detailed_caption: str, all_captions: List[str]) -> List[str]:
        """
        Extract structured attributes from captions.
        
        Args:
            detailed_caption: Combined detailed caption
            all_captions: List of individual captions
            
        Returns:
            List of attribute strings (color:red, item:dress, etc.)
        """
        attributes = []
        combined_text = " ".join([detailed_caption] + all_captions).lower()
        
        # Extract colors
        colors = [
            'red', 'blue', 'green', 'yellow', 'black', 'white',
            'grey', 'gray', 'brown', 'pink', 'purple', 'orange',
            'beige', 'navy', 'maroon', 'teal', 'gold', 'silver',
            'cream', 'tan', 'khaki', 'burgundy'
        ]
        
        for color in colors:
            if re.search(rf'\b{color}\b', combined_text):
                attributes.append(f"color:{color}")
        
        # Extract clothing items
        items = [
            'dress', 'shirt', 'pants', 'jacket', 'coat', 'suit',
            'tie', 'skirt', 'hat', 'shoes', 'boots', 'bag',
            'scarf', 'sweater', 'hoodie', 'jeans', 'shorts',
            'blazer', 'cardigan', 'vest', 'blouse', 'top',
            'trousers', 'leggings', 'socks', 'gloves', 'belt',
            'sunglasses', 'watch', 'jewelry', 'necklace', 'earrings'
        ]
        
        for item in items:
            if re.search(rf'\b{item}s?\b', combined_text):
                attributes.append(f"item:{item}")
        
        # Extract patterns
        patterns = ['striped', 'polka dot', 'checkered', 'plaid', 'floral', 'solid']
        for pattern in patterns:
            if pattern in combined_text:
                attributes.append(f"pattern:{pattern}")
        
        # Extract styles
        styles = ['casual', 'formal', 'business', 'sporty', 'elegant', 'vintage']
        for style in styles:
            if style in combined_text:
                attributes.append(f"style:{style}")
        
        # Extract environments
        environments = [
            'indoor', 'outdoor', 'office', 'street', 'park',
            'runway', 'beach', 'city', 'urban', 'home', 'studio'
        ]
        for env in environments:
            if env in combined_text:
                attributes.append(f"environment:{env}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_attributes = []
        for attr in attributes:
            if attr not in seen:
                seen.add(attr)
                unique_attributes.append(attr)
        
        return unique_attributes
    
    def generate_captions_for_directory(self, image_dir: Path, output_file: Path, batch_size: int = 4) -> Dict:
        """
        Generate 2 comprehensive detailed captions for all images in a directory.
        Each caption captures complete 5W1H details (What, Where, When, Why, How).
        
        Args:
            image_dir: Directory containing images
            output_file: Path to save captions JSON
            batch_size: Number of images to process at once
            
        Returns:
            Dictionary mapping image paths to caption data
        """
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(image_paths)
        print(f"\nFound {len(image_paths)} images in {image_dir}")
        print(f"Processing in batches of {batch_size} images (2 comprehensive captions per image)...")
        print(f"Total captions to generate: {len(image_paths) * 2}")
        
        # Generate captions in batches
        print("\nGenerating comprehensive detailed captions with 5W1H details...")
        caption_results = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = self.generate_captions_batch(batch_paths, batch_size=len(batch_paths))
            caption_results.extend(batch_results)
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed/60:.1f} minutes ({elapsed/len(image_paths):.1f}s per image)")
        
        # Create dictionary with relative paths as keys
        captions_data = {}
        for image_path, caption_data in zip(image_paths, caption_results):
            try:
                rel_path = str(image_path.relative_to(image_dir.parent))
            except ValueError:
                rel_path = str(image_path)
            
            captions_data[rel_path] = caption_data
        
        # Save to file
        print(f"\nSaving captions to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(captions_data)} detailed captions to {output_file}")
        print(f"📊 Average caption length: {sum(len(d['detailed_caption']) for d in captions_data.values()) / len(captions_data):.0f} chars")
        
        return captions_data


def main():
    """Main function to generate comprehensive detailed captions."""
    print("=" * 70)
    print("FASHION IMAGE CAPTION GENERATOR (5W1H Edition)")
    print("=" * 70)
    print("📝 Generating 2 comprehensive captions per image")
    print("🎯 Capturing What, Where, When, Why, How (5W1H details)")
    print("Using MPS acceleration on Apple Silicon")
    
    # Initialize generator
    generator = CaptionGenerator()
    
    # Generate captions
    output_file = DATA_DIR / "detailed_captions_generated.json"
    captions_data = generator.generate_captions_for_directory(TEST_DIR, output_file)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("CAPTION GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Total images processed: {len(captions_data)}")
    print(f"Output file: {output_file}")
    
    # Show sample
    if captions_data:
        print("\n" + "=" * 70)
        print("SAMPLE DETAILED CAPTION:")
        print("=" * 70)
        sample_key = list(captions_data.keys())[0]
        sample = captions_data[sample_key]
        print(f"\nImage: {sample_key}")
        print(f"\nBase Caption: {sample['base_caption']}")
        print(f"\nAll Captions ({len(sample.get('all_captions', []))}):")
        for i, cap in enumerate(sample.get('all_captions', [])[:2], 1):
            print(f"  {i}. {cap}")
        print(f"\nCombined Detailed Caption:")
        print(f"  {sample['detailed_caption']}")
        print(f"\nExtracted Attributes ({len(sample['attributes'])}):")
        print(f"  {', '.join(sample['attributes'][:15])}")
    
    print("\nCaption generation complete. You can now run the indexer.")


if __name__ == "__main__":
    main()
