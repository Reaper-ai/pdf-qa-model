import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple

class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Initializes Dense (FAISS), Sparse (BM25) indexes, and a Cross-Encoder Reranker.
        """
        self.chunks = chunks
        
        # 1. Setup Dense Retriever (FAISS)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        # 2. Setup Sparse Retriever (BM25)
        tokenized_corpus = [chunk["content"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Setup Two-Stage Cross-Encoder Reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

    def dense_search(self, query_embedding: np.ndarray, k: int) -> List[int]:
        """Semantic search using FAISS"""
        _, indices = self.index.search(np.array([query_embedding]), k)
        return list(indices[0])

    def sparse_search(self, query: str, k: int) -> List[int]:
        """Exact keyword search using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices.tolist()

    def search(
        self, 
        query: str, 
        query_embedding: np.ndarray, 
        top_k: int = 5, 
        rrf_k: int = 60,
        confidence_threshold: float = -2.5
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Two-Stage Pipeline with Source Confidence Filtering:
        Stage 1: Retrieve candidate pool using Hybrid RRF.
        Stage 2: Re-score candidates using Cross-Encoder.
        Filter: Drop chunks below confidence_threshold.
        
        :return: A tuple of (filtered_chunks, is_confident)
        """
        # Step 1: Broad pool extraction
        pool_size = max(top_k * 5, 25)
        dense_results = self.dense_search(query_embedding, k=pool_size)
        sparse_results = self.sparse_search(query, k=pool_size)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rrf_k + rank + 1))
            
        for rank, doc_id in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (rrf_k + rank + 1))

        # Sort based on RRF and extract candidates
        sorted_candidate_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:pool_size]
        candidates = [self.chunks[doc_id] for doc_id in sorted_candidate_ids]

        if not candidates:
            return [], False

        # Step 2: Cross-Encoder Reranking
        pairs = [[query, chunk["content"]] for chunk in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Attach scores to metadata
        for idx, score in enumerate(rerank_scores):
            candidates[idx]["metadata"]["rerank_score"] = float(score)

        # Sort candidates strictly by their cross-encoder score
        ranked_candidates = sorted(candidates, key=lambda x: x["metadata"]["rerank_score"], reverse=True)

        # Step 3: Confidence Filtering
        # Filter out chunks that do not cross our minimal context threshold
        confident_chunks = [
            chunk for chunk in ranked_candidates 
            if chunk["metadata"]["rerank_score"] >= confidence_threshold
        ]

        # Flag indicating whether the document corpus actually holds the answer
        is_confident = len(confident_chunks) > 0

        # Return top_k matching slices of validated text
        return confident_chunks[:top_k], is_confident