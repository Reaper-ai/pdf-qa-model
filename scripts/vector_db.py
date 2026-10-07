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
        return [int(i) for i in indices[0] if i != -1]

    def sparse_search(self, query: str, k: int) -> List[int]:
        """Exact keyword search using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices.tolist()

    def rank(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        mode: str = "hybrid",
        use_reranker: bool = True,
        rrf_k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Returns (chunk_id, score) pairs ordered best-first (higher = better),
        WITHOUT the confidence filter. This is the raw ranking used for Recall@k.

        :param mode: "dense" (FAISS only), "sparse" (BM25 only), or "hybrid" (RRF).
        :param use_reranker: re-score the candidate pool with the cross-encoder.
        """
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        pool_size = max(top_k * 5, 25)

        if mode == "dense":
            distances, indices = self.index.search(
                np.array([query_embedding]), pool_size
            )
            # L2 distance -> negated so that higher score = better match.
            return [
                (int(idx), float(-dist))
                for dist, idx in zip(distances[0], indices[0])
                if idx != -1
            ]

        if mode == "sparse":
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            order = np.argsort(scores)[::-1][:pool_size]
            return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

        # mode == "hybrid": Reciprocal Rank Fusion over both retrievers
        rrf_scores: Dict[int, float] = {}
        for rank_, doc_id in enumerate(self.dense_search(query_embedding, k=pool_size)):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1 / (rrf_k + rank_ + 1))
        for rank_, doc_id in enumerate(self.sparse_search(query, k=pool_size)):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1 / (rrf_k + rank_ + 1))

        sorted_candidate_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:pool_size]
        candidates = [(doc_id, rrf_scores[doc_id]) for doc_id in sorted_candidate_ids]

        if not candidates or not use_reranker:
            return candidates

        # Cross-Encoder reranking over the fused candidate pool
        candidate_chunks = [self.chunks[doc_id] for doc_id, _ in candidates]
        pairs = [[query, chunk["content"]] for chunk in candidate_chunks]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip([doc_id for doc_id, _ in candidates], rerank_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(doc_id, float(score)) for doc_id, score in reranked]

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        rrf_k: int = 60,
        confidence_threshold: float = -2.5,
        mode: str = "hybrid",
        use_reranker: bool = True,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Two-Stage Pipeline with Source Confidence Filtering:
        Stage 1: Retrieve candidate pool (dense / sparse / hybrid RRF).
        Stage 2: Re-score candidates using Cross-Encoder.
        Filter: Drop chunks below confidence_threshold.

        :return: A tuple of (filtered_chunks, is_confident)
        """
        ranked = self.rank(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            mode=mode,
            use_reranker=use_reranker,
            rrf_k=rrf_k,
        )

        if not ranked:
            return [], False

        candidates = [self.chunks[doc_id] for doc_id, score in ranked]
        for (doc_id, score), chunk in zip(ranked, candidates):
            chunk["metadata"]["retrieval_score"] = float(score)
            if use_reranker and mode == "hybrid":
                chunk["metadata"]["rerank_score"] = float(score)

        # Confidence filtering only applies to cross-encoder logits; RRF/BM25
        # scores use a different scale, so abstention gating stays reranker-only.
        if use_reranker and mode == "hybrid":
            confident_chunks = [
                chunk for chunk in candidates
                if chunk["metadata"].get("rerank_score", 0.0) >= confidence_threshold
            ]
        else:
            confident_chunks = candidates

        is_confident = len(confident_chunks) > 0
        return confident_chunks[:top_k], is_confident