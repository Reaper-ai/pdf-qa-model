from sentence_transformers import SentenceTransformer
from numpy import ndarray
from typing import List

def get_embedder() -> SentenceTransformer:
    """
    initialize sentence embedder
    :return: sentence embedder
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(embedder: SentenceTransformer, text:List[str]) -> ndarray:
    """
    return ndarray of embedded text
    :param embedder: embeding model
    :param text: pdf content
    :return: emebedded text
    """
    return embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
