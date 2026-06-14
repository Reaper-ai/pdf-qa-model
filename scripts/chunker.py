from typing import List, Dict, Any

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        :param chunk_size: Target maximum number of words per chunk.
        :param chunk_overlap: Word overlap size between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits text blocks into overlapping chunks at clean word boundaries 
        while propagating parent file metadata down to every single chunk.
        """
        final_chunks = []
        chunk_id_counter = 0

        for page in parsed_pages:
            text = page.get("text", "")
            base_metadata = page.get("metadata", {})
            
            if not text.strip():
                continue

            # Split on whitespace to treat words as atomic tokens (avoids breaking data mid-word)
            words = text.split(" ")
            
            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + self.chunk_size, len(words))
                
                # Reconstruct chunk text safely
                chunk_text = " ".join(words[start_idx:end_idx]).strip()
                
                if chunk_text:
                    # Propagate and extend metadata dictionary
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": chunk_id_counter,
                        "word_count": len(words[start_idx:end_idx])
                    })
                    
                    final_chunks.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata
                    })
                    chunk_id_counter += 1
                
                # Handle sliding window progress
                if end_idx >= len(words):
                    break
                    
                start_idx += (self.chunk_size - self.chunk_overlap)
                
        return final_chunks