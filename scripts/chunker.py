from typing import List, Dict, Any

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits normalized text into overlapping chunks while propagating metadata.
        """
        final_chunks = []
        chunk_id_counter = 0

        for page in parsed_pages:
            text = page["text"]
            base_metadata = page["metadata"]
            
            # Simple character-based windowing (upgrade to token-based if using precise limits)
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # Build a unique metadata packet per chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id_counter,
                    "char_start": start,
                    "char_end": min(end, len(text))
                })
                
                final_chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_id_counter += 1
                # Move forward by chunk size minus the overlap
                start += (self.chunk_size - self.chunk_overlap)
                
                # Break if we reached the end of the text
                if end >= len(text):
                    break
                    
        return final_chunks