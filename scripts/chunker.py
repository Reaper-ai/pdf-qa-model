from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def split_text(text: str , chunk_size: int =500, chunk_overlap: int =50) -> List[str]:
    """
    split large text into smaller, manageable pieces (chunks)

    :param text: cleaned text
    :param chunk_size: chunk size
    :param chunk_overlap: chunk overlap
    :return: chunked text
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
