"""Utility functions"""
import json
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path


def load_json(file_path: Path) -> Dict:
    """Load JSON file safely"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: Path) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_query_aspects(query: str) -> Dict[str, str]:
    """
    Parse user query into attire and environment aspects.
    
    Args:
        query: Natural language search query
        
    Returns:
        Dictionary with 'attire' and 'environment' keys
    """
    # Keywords for environment detection
    environment_keywords = {
        'office', 'street', 'park', 'home', 'indoor', 'outdoor',
        'runway', 'fashion show', 'catwalk', 'beach', 'city',
        'urban', 'building', 'room', 'garden', 'cafe', 'restaurant',
        'modern', 'professional', 'casual setting', 'formal setting'
    }
    
    # Keywords for attire/clothing
    attire_keywords = {
        'wearing', 'outfit', 'dress', 'shirt', 'pants', 'jacket',
        'coat', 'suit', 'tie', 'skirt', 'hoodie', 't-shirt',
        'blazer', 'jeans', 'formal', 'casual', 'professional',
        'business', 'attire', 'clothing', 'raincoat', 'clothes'
    }
    
    # Color keywords
    color_keywords = {
        'red', 'blue', 'green', 'yellow', 'black', 'white',
        'grey', 'gray', 'brown', 'pink', 'purple', 'orange',
        'bright', 'dark', 'light', 'colorful'
    }
    
    query_lower = query.lower()
    words = set(query_lower.split())
    
    # Extract environment-related terms
    environment_terms = []
    for keyword in environment_keywords:
        if keyword in query_lower:
            environment_terms.append(keyword)
    
    # Extract attire-related terms (including colors)
    attire_terms = []
    for keyword in attire_keywords | color_keywords:
        if keyword in query_lower:
            attire_terms.append(keyword)
    
    # If no clear separation, use simple heuristic
    # Split on prepositions that often separate attire from environment
    env_separators = [' in ', ' at ', ' inside ', ' on ', ' near ']
    
    attire_text = query
    environment_text = ""
    
    for separator in env_separators:
        if separator in query_lower:
            parts = query.lower().split(separator, 1)
            # Check if the part after separator is environment or attire
            if any(env_kw in parts[1] for env_kw in environment_keywords):
                # Standard case: attire IN/AT environment
                attire_text = query.split(separator, 1)[0].strip()
                environment_text = query.split(separator, 1)[1].strip() if len(parts) > 1 else ""
            else:
                # Inverted case: "in a red shirt" means shirt is attire
                # Keep full query as attire
                attire_text = query
                environment_text = ""
            break
    
    # If we found environment terms but no split, try to extract them
    if not environment_text and environment_terms:
        # Try to extract the environment part
        for term in environment_terms:
            pattern = rf'\b{re.escape(term)}\b.*'
            match = re.search(pattern, query_lower)
            if match:
                environment_text = query[match.start():].strip()
                attire_text = query[:match.start()].strip()
                break
    
    # Fallback: if we have both types of keywords, split appropriately
    if not environment_text and environment_terms and attire_terms:
        # Keep query as is for attire, extract environment keywords
        attire_text = query
        environment_text = " ".join(environment_terms)
    
    return {
        'attire': attire_text if attire_text else query,
        'environment': environment_text if environment_text else ""
    }


def extract_attributes_from_caption(caption_data: Dict) -> List[str]:
    """
    Extract structured attributes from caption data.
    
    Args:
        caption_data: Dictionary containing caption information
        
    Returns:
        List of attribute strings
    """
    attributes = []
    
    # Get predefined attributes if available
    if 'attributes' in caption_data:
        attributes.extend(caption_data['attributes'])
    
    # Extract from detailed caption
    if 'detailed_caption' in caption_data:
        caption = caption_data['detailed_caption'].lower()
        
        # Extract colors
        colors = re.findall(r'\b(red|blue|green|yellow|black|white|grey|gray|brown|pink|purple|orange)\b', caption)
        for color in set(colors):
            attr = f"color:{color}"
            if attr not in attributes:
                attributes.append(attr)
        
        # Extract clothing items
        items = re.findall(r'\b(dress|shirt|pants|jacket|coat|suit|tie|skirt|hat|shoes|boots|bag|scarf)\b', caption)
        for item in set(items):
            attr = f"item:{item}"
            if attr not in attributes:
                attributes.append(attr)
    
    return attributes


def format_results(results: List[Tuple[str, float, Dict]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Format retrieval results for display.
    
    Args:
        results: List of (image_path, score, metadata) tuples
        top_k: Number of top results to return
        
    Returns:
        List of formatted result dictionaries
    """
    formatted = []
    for idx, (image_path, score, metadata) in enumerate(results[:top_k], 1):
        formatted.append({
            'rank': idx,
            'image_path': str(image_path),
            'score': float(score),
            'caption': metadata.get('detailed_caption', ''),
            'attributes': metadata.get('attributes', [])
        })
    return formatted


def calculate_weighted_score(attire_score: float, env_score: float, 
                            attire_weight: float, env_weight: float) -> float:
    """
    Calculate weighted score from attire and environment scores.
    
    Args:
        attire_score: Score for attire matching
        env_score: Score for environment matching
        attire_weight: Weight for attire (0-1)
        env_weight: Weight for environment (0-1)
        
    Returns:
        Weighted combined score
    """
    # Normalize weights to ensure they sum to 1
    total_weight = attire_weight + env_weight
    if total_weight == 0:
        total_weight = 1.0
    
    norm_attire_weight = attire_weight / total_weight
    norm_env_weight = env_weight / total_weight
    
    return (norm_attire_weight * attire_score) + (norm_env_weight * env_score)
