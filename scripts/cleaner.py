import re
import unicodedata

class TextNormalizer:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        
        # 1. Normalize Unicode characters (fixes ligatures and weird accents)
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Fix words split by hyphens at line breaks (e.g., "en- \n vironment")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 3. Replace multiple newlines or tabs with a single space (or single newline)
        text = re.sub(r'\s+', ' ', text)
        
        # 4. Strip leading/trailing whitespace
        return text.strip()