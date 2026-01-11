"""Interactive search interface for fashion retrieval system"""
from pathlib import Path
from config import IMAGE_EMBEDDINGS_FILE, CAPTION_EMBEDDINGS_FILE
from indexer import FashionIndexer
from retriever import FashionRetriever


def check_index_exists() -> bool:
    return (
        IMAGE_EMBEDDINGS_FILE.exists() and 
        CAPTION_EMBEDDINGS_FILE.exists()
    )


def main():
    print("=" * 70)
    print("FASHION IMAGE RETRIEVAL SYSTEM")
    print("=" * 70)
    
    # Check/build index
    if not check_index_exists():
        print("\nIndex not found. Building index...")
        print("This will take a few minutes (one-time only).\n")
        indexer = FashionIndexer()
        indexer.build_index()
        print("\nIndex built successfully\n")
    
    # Initialize retriever
    print("\nLoading retrieval system...")
    retriever = FashionRetriever()
    print(f"Ready! Index contains {len(retriever.image_paths)} images.\n")
    
    # Interactive search loop
    while True:
        print("\n" + "=" * 70)
        query = input("Enter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting. Thank you!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        # Get number of results
        try:
            top_k = input("How many results? (default 10): ").strip()
            top_k = int(top_k) if top_k else 10
            top_k = max(1, min(top_k, 50))  # Limit between 1-50
        except ValueError:
            top_k = 10
            print("Invalid number, using default: 10")
        
        # Search
        print("\nSearching...")
        results = retriever.search(query, top_k=top_k, use_reranking=True)
        
        # Display results
        retriever.display_results(results)
        
        # Ask if user wants to search again
        continue_search = input("\nSearch again? (y/n): ").strip().lower()
        if continue_search not in ['y', 'yes', '']:
            print("\nExiting. Thank you!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting. Thank you!")
    except Exception as e:
        print(f"\nError: {e}")
