import numpy as np
import faiss
from typing import List


def build_index(embeddings : np.ndarray) -> faiss.IndexFlatL2:
    """
    build vector space on the data
    :param embeddings: embedded data
    :return: index
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def get_top_k_chunks(query_embedding: np.ndarray, index: faiss.IndexFlatL2, k: int =3) -> List[int]:
    """
    :param query_embedding: embedded question
    :param index: vector index
    :param k: how many chunks to return
    :return: indices of top k matching chunks relevant to the question
    """

    _, indices = index.search(np.array([query_embedding]), k)
    return list(indices[0])

