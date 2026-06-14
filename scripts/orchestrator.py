import time
import uuid
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

# Custom filter to prevent external libraries from crashing on missing trace fields
class TraceIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "trace_id"):
            record.trace_id = "N/A"
        return True

# Configure root logging safely
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] TraceID: %(trace_id)s | %(message)s')
handler.setFormatter(formatter)
handler.addFilter(TraceIDFilter())

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear any old handlers to prevent duplicate output traps
if root_logger.handlers:
    root_logger.handlers.clear()
root_logger.addHandler(handler)

class SemanticCache:
    def __init__(self, embedding_dim: int, similarity_threshold: float = 0.95):
        """
        A production-grade semantic cache that stores past queries and answers.
        Uses cosmic cosine similarity/L2 distance to match incoming questions against old ones.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.cached_queries: List[np.ndarray] = []
        self.cached_responses: List[Dict[str, Any]] = []
        self.threshold = similarity_threshold  # Lower distance means closer match

    def lookup(self, query_embedding: np.ndarray) -> Tuple[Dict[str, Any], bool]:
        """Checks if a structurally similar query has been executed before."""
        if self.index.ntotal == 0:
            return {}, False

        # Reshape for FAISS search
        vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(vector, 1)
        
        best_distance = distances[0][0]
        best_index = indices[0][0]

        # In L2 distance, values very close to 0 denote near-identical semantic statements
        if best_index != -1 and best_distance <= (1.0 - self.threshold):
            return self.cached_responses[best_index], True

        return {}, False

    def update(self, query_embedding: np.ndarray, response: Dict[str, Any]):
        """Saves a verified generation pipeline output into the cache space."""
        vector = np.array([query_embedding]).astype('float32')
        self.index.add(vector)
        self.cached_responses.append(response)


class RAGSessionMemory:
    def __init__(self, max_turns: int = 5):
        """Manages a sliding context window of short-term chat logs."""
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_formatted_context(self) -> str:
        """Formats conversational memory history directly into a prompt-safe string."""
        if not self.history:
            return "No previous conversation history."
        return "\n".join([f"User: {turn['question']}\nBot: {turn['answer']}" for turn in self.history])


class ProductionRAGOrchestrator:
    def __init__(self, retriever: Any, llm: Any, embedding_dim: int = 384):
        self.retriever = retriever
        self.llm = llm
        self.cache = SemanticCache(embedding_dim=embedding_dim)
        self.memory = RAGSessionMemory()

    def execute_pipeline(self, question: str, query_embedding: np.ndarray, embedder: Any) -> Dict[str, Any]:
        """
        The central execution controller running full caching, retrieval, 
        reranking, verification, memory tracking, and telemetry recording.
        """
        # STEP 10: Generate a distinct transaction Trace ID for end-to-end observability tracing
        trace_id = str(uuid.uuid4())[:8]
        extra_log = {"trace_id": trace_id}
        
        logging.info(f"Initiating pipeline for query: '{question}'", extra=extra_log)
        start_time = time.time()

        # STEP 9A: Semantic Cache Lookup
        cache_hit, found = self.cache.lookup(query_embedding)
        if found:
            latency = (time.time() - start_time) * 1000
            logging.info(f"CACHE HIT! Served response semantically in {latency:.2f}ms", extra=extra_log)
            # Inject history update even on cache hits
            self.memory.add_turn(question, cache_hit["answer"])
            return cache_hit

        logging.info("Cache miss. Progressing to dense/sparse retrieval index...", extra=extra_log)
        
        # STEP 9B: Fetch Chat Memory context to augment the query context if necessary
        chat_history_str = self.memory.get_formatted_context()

        # STEP 2, 3 & 4: Retrieval + Reranking + Confidence Filtering
        retrieval_start = time.time()
        retrieved_chunks, is_confident = self.retriever.search(
            query=question, 
            query_embedding=query_embedding,
            top_k=5
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000
        logging.info(f"Retrieval complete. Found {len(retrieved_chunks)} valid chunks. Confident={is_confident} ({retrieval_latency:.2f}ms)", extra=extra_log)

       # STEP 5, 6 & 7: Generation + Citation formatting + NLI Fallback Checking
        generation_start = time.time()
        # Fix: Absolute package path resolution relative to project root
        from scripts import qa_model
        
        # Append sliding conversation context into the QA parameters
        augmented_question = f"Conversation History:\n{chat_history_str}\n\nCurrent Question: {question}"
        
        # FIX: Changed 'llm=' keyword to 'client=' to map correctly to InferenceClient
        final_output = qa_model.answer_question(
            client=self.llm, 
            question=augmented_question, 
            retrieved_chunks=retrieved_chunks, 
            is_confident=is_confident
        )
        generation_latency = (time.time() - generation_start) * 1000
        generation_latency = (time.time() - generation_start) * 1000
        logging.info(f"Generation layer finalized execution ({generation_latency:.2f}ms)", extra=extra_log)

        # Update Session Memory state with the verified answer string
        self.memory.add_turn(question, final_output["answer"])

        # STEP 9C: Update Semantic Cache with fresh, validated output data
        if is_confident and final_output.get("has_answer", True):
            self.cache.update(query_embedding, final_output)

        total_latency = (time.time() - start_time) * 1000
        logging.info(f"Pipeline transaction complete. Total pipeline latency: {total_latency:.2f}ms", extra=extra_log)

        return final_output