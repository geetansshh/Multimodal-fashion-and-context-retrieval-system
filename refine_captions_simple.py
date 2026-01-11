"""
SIMPLE CAPTION REFINER - Groq Version
======================================

Refines fashion image captions by combining outputs from two BLIP models
and using Groq LLaMA for enhancement.

SETUP:
1. Install dependencies: pip install groq python-dotenv tqdm
2. Create .env file with: GROQ_API_KEY=your-key-here
3. Place these files in same folder:
   - detailed_captions_generated.json (from BLIP-base)
   - detailed_captions_generated_2.json (from BLIP-large)
   - This script

RUN:
   python refine_captions_simple.py

OUTPUT:
   detailed_captions_final.json (refined captions with metadata)

Author: Fashion Retrieval System
License: MIT
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Configuration Constants
MODEL_NAME: str = "llama-3.3-70b-versatile"
INPUT_FILE_1: str = "detailed_captions_generated.json"
INPUT_FILE_2: str = "detailed_captions_generated_2.json"
OUTPUT_FILE: str = "detailed_captions_final.json"
SAVE_INTERVAL: int = 50
MAX_TOKENS: int = 250
TEMPERATURE: float = 0.1


def load_api_key() -> Optional[str]:
    """
    Load Groq API key from environment.
    
    Returns:
        API key if found, None otherwise
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    return os.getenv("GROQ_API_KEY")


def initialize_groq_client(api_key: str):
    """
    Initialize Groq client with error handling.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Initialized Groq client
        
    Raises:
        ImportError: If groq package not installed
    """
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        print("Error: Groq package not installed")
        print("Install with: pip install groq")
        sys.exit(1)


# Initialize client
api_key = load_api_key()
if not api_key:
    print("Error: GROQ_API_KEY not found")
    print("Create a .env file with: GROQ_API_KEY=your-key-here")
    sys.exit(1)

client = initialize_groq_client(api_key)


def create_refinement_prompt(captions_1: List[str], captions_2: List[str], 
                           attributes: List[str]) -> str:
    """
    Create a prompt for caption refinement.
    
    Args:
        captions_1: Captions from first model
        captions_2: Captions from second model
        attributes: Combined attributes from both models
        
    Returns:
        Formatted prompt string
    """
    return f"""Based on these captions from two models, create ONE factual, simple description.

Model 1 Captions:
{chr(10).join(f"- {cap}" for cap in captions_1 if cap)}

Model 2 Captions:
{chr(10).join(f"- {cap}" for cap in captions_2 if cap)}

Attributes: {', '.join(attributes) if attributes else 'None'}

Create a clear caption (80-120 words) covering:
- WHAT: clothing items, colors, patterns, materials, accessories
- WHERE: setting/environment (runway, street, indoor, outdoor, etc.)
- WHEN: context if visible (fashion show, casual, event, etc.)
- WHY: purpose/occasion of the outfit
- HOW: how items are styled together

Rules:
- Use ONLY information from above
- Combine best details from both models
- Keep language simple and factual
- No creative/flowery writing
- Focus on searchable facts

Output ONLY the caption."""


def extract_captions_and_attributes(captions_dict: Dict) -> tuple[List[str], List[str]]:
    """
    Extract captions and attributes from caption dictionary.
    
    Args:
        captions_dict: Dictionary containing caption data
        
    Returns:
        Tuple of (captions list, attributes list)
    """
    captions = captions_dict.get('all_captions', [captions_dict.get('detailed_caption', '')])
    attributes = captions_dict.get('attributes', [])
    return captions, attributes


def refine_caption(captions_1: Dict, captions_2: Dict) -> Optional[Dict]:
    """
    Combine and refine captions using OpenAI.
    
    Args:
        captions_1: Caption data from first model
        captions_2: Caption data from second model
        
    Returns:
        Dictionary with refined caption and metadata, or None on error
    """
    # Extract data from both sources
    all_caps_1, attrs_1 = extract_captions_and_attributes(captions_1)
    all_caps_2, attrs_2 = extract_captions_and_attributes(captions_2)
    
    # Combine attributes (remove duplicates)
    combined_attributes = list(set(attrs_1 + attrs_2))
    
    # Create prompt
    prompt = create_refinement_prompt(all_caps_1, all_caps_2, combined_attributes)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a factual fashion caption writer. You combine information into clear descriptions without adding new details."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        refined = response.choices[0].message.content.strip()
        
        return {
            "detailed_caption": refined,
            "attributes": combined_attributes,
            "base_caption": refined,
            "all_captions": [refined],
            "source_models": ["blip-base", "blip-large"],
            "refined_by": MODEL_NAME
        }
        
    except Exception as e:
        print(f"\nAPI Error: {e}")
        return None


def load_caption_file(filepath: str) -> Optional[Dict]:
    """
    Load a caption JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary or None if error
    """
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_progress(data: Dict, filepath: str) -> bool:
    """
    Save progress to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")
        return False


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 70)
    print("SIMPLE CAPTION REFINER - OpenAI")
    print("=" * 70)
    
    # Load caption files
    print(f"\nLoading {INPUT_FILE_1}...")
    captions_1 = load_caption_file(INPUT_FILE_1)
    if captions_1 is None:
        return 1
    
    print(f"Loading {INPUT_FILE_2}...")
    captions_2 = load_caption_file(INPUT_FILE_2)
    if captions_2 is None:
        return 1
    
    # Get common images
    common_keys = sorted(list(set(captions_1.keys()) & set(captions_2.keys())))
    print(f"\nTotal images to refine: {len(common_keys)}")
    print(f"Using model: {MODEL_NAME}")
    
    # Load existing progress
    refined = {}
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        print(f"Loading {OUTPUT_FILE}...")
        refined = load_caption_file(OUTPUT_FILE) or {}
        print(f"Resuming: {len(refined)} already done")
    
    # Process captions
    print(f"\nSaving progress every {SAVE_INTERVAL} captions")
    print("=" * 70)
    
    errors = 0
    processed = 0
    
    try:
        for i, img_key in enumerate(tqdm(common_keys, desc="Refining"), 1):
            # Skip if already done
            if img_key in refined:
                continue
            
            # Refine caption
            result = refine_caption(captions_1[img_key], captions_2[img_key])
            
            if result:
                refined[img_key] = result
                processed += 1
            else:
                errors += 1
                # Keep original on error
                refined[img_key] = captions_1[img_key]
            
            # Save progress periodically
            if len(refined) % SAVE_INTERVAL == 0:
                if not save_progress(refined, OUTPUT_FILE):
                    print(f"Warning: Failed to save progress at {len(refined)} captions")
        
        # Final save
        if not save_progress(refined, OUTPUT_FILE):
            print("Error: Failed to save final output!")
            return 1
        
        # Print results
        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Total refined: {len(refined)}")
        print(f"Newly processed: {processed}")
        print(f"Errors: {errors}")
        print(f"Output: {OUTPUT_FILE}")
        
        # Show sample
        if refined:
            sample_key = list(refined.keys())[0]
            sample = refined[sample_key]
            print(f"\nSample caption:")
            print(f"   {sample['detailed_caption'][:200]}...")
            print(f"\nAttributes: {', '.join(sample['attributes'][:5])}...")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        save_progress(refined, OUTPUT_FILE)
        print(f"Progress saved: {len(refined)} captions")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        save_progress(refined, OUTPUT_FILE)
        return 1


if __name__ == "__main__":
    sys.exit(main())
