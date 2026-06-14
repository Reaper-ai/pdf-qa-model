import re
import unicodedata

class TextNormalizer:
    @staticmethod
    def clean(page_data: dict) -> None:
        """
        Cleans the text within a page/block dictionary in-place, removing layout artifacts,
        ligatures, and useless spacing control characters.
        """
        if not page_data or "text" not in page_data:
            return

        text = page_data["text"]

        # 1. Standardize unicode encodings (collapses ligatures like 'fi', standardizes accents)
        text = unicodedata.normalize("NFKC", text)

        # 2. Strip control characters / non-printable characters (except normal tabs/newlines for now)
        # This eliminates form feeds (\x0c / \f), zero-width spaces, and obscure parsing noise
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 3. Stitch words split by hyphens across line breaks (e.g., "sys-\ntem" -> "system")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 4. Clean up inline line breaks (converts standard wrapping newlines within a block into single spaces)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # 5. Collapse all multi-spaces, tabs, and duplicate consecutive lines into single spaces/breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        # Update dictionary in-place with clean boundaries
        page_data["text"] = text.strip()