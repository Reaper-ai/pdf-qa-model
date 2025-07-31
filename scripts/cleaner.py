import re

def clean_text(text: str) -> str:
    """
    perfroms some artifact cleaning on text

    :param text: content to be cleaned
    :return: cleaned content
    """

    text = text.strip()

    # Remove common artifacts
    text = re.sub(r'\f', '', text)  # Form feed (page break markers)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)  # Multiple spaces
    text = re.sub(r'\n+', '\n', text)  # Multiple newlines
    text = re.sub(r'-\n', '', text)  # Join hyphenated line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Line breaks within paragraphs → space
    return text
