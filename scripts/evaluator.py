import json
import logging
import re
from typing import List, Dict, Any

class RAGEvaluator:
    def __init__(self, client: Any):
        """
        Initializes the production evaluation suite using an LLM-as-a-Judge matrix.
        :param client: The active Hugging Face InferenceClient instance
        """
        self.client = client
        self.logger = logging.getLogger("RAG_Evaluator")

    def _get_llm_score(self, evaluation_prompt: str) -> float:
        """Helper to extract a clean float score from an LLM judge prompt evaluation pass."""
        try:
            messages = [
                {"role": "system", "content": "You are a strict, objective quality control auditor. Respond with exactly a decimal score between 0.0 and 1.0, and absolutely nothing else."},
                {"role": "user", "content": evaluation_prompt}
            ]
            response = self.client.chat_completion(messages=messages, max_tokens=10, temperature=0.1)
            raw_text = response.choices[0].message.content.strip()
            
            # Use regex to extract the first decimal or integer float representation found
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_text)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score)) # Bound tightly between 0.0 and 1.0
            return 0.5
        except Exception:
            return 0.5

    def evaluate_turn(self, query: str, pipeline_output: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates a live pipeline turn using conceptual LLM auditing metrics.
        """
        answer = pipeline_output.get("answer", "")
        has_answer = pipeline_output.get("has_answer", True)
        combined_context = " ".join([chunk["content"] for chunk in retrieved_chunks])

        if not has_answer or not answer or not retrieved_chunks:
            return {
                "context_relevance": 1.0 if not retrieved_chunks else 0.0,
                "faithfulness": 1.0,
                "answer_completeness": 0.0
            }

        # 1. Evaluate Context Relevance (Is the retriever fetching noisy filler or helpful targets?)
        cr_prompt = (
            f"Question: {query}\n\n"
            f"Retrieved Context Chunks: {combined_context}\n\n"
            "Task: Rate how relevant and useful the retrieved context chunks are to answering the question. "
            "Output a decimal score strictly between 0.0 (completely useless) and 1.0 (perfect context matching)."
        )
        context_relevance = self._get_llm_score(cr_prompt)

        # 2. Evaluate Faithfulness / Groundedness (Did the model hallucinate?)
        f_prompt = (
            f"Reference Context: {combined_context}\n\n"
            f"Generated Answer: {answer}\n\n"
            "Task: Check if every fact stated in the Generated Answer is directly supported by the Reference Context. "
            "If the answer includes outside data or unmentioned specs, penalize heavily. "
            "Output a decimal score strictly between 0.0 (completely hallucinated) and 1.0 (100% faithful and grounded)."
        )
        faithfulness = self._get_llm_score(f_prompt)

        # 3. Evaluate Answer Completeness (Did it actually satisfy the user's prompt request?)
        ac_prompt = (
            f"User Question: {query}\n\n"
            f"Generated Answer: {answer}\n\n"
            "Task: Rate whether the Generated Answer fully addresses and satisfies the implicit requirements of the User Question. "
            "Output a decimal score strictly between 0.0 (ignored the prompt) and 1.0 (perfectly comprehensive answer)."
        )
        answer_completeness = self._get_llm_score(ac_prompt)

        return {
            "context_relevance": round(context_relevance, 2),
            "faithfulness": round(faithfulness, 2),
            "answer_completeness": round(answer_completeness, 2)
        }

    def run_benchmark_suite(self, orchestrator: Any, embedder: Any, test_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Runs the loop over the test case validation dataset to output global metrics summaries."""
        total_cr, total_f, total_ac = 0.0, 0.0, 0.0
        count = len(test_dataset)
        
        for index, test_case in enumerate(test_dataset, start=1):
            question = test_case["question"]
            query_embedding = embedder.encode(question)
            
            # Extract live retrieval hits
            chunks, is_confident = orchestrator.retriever.search(question, query_embedding, top_k=5)
            # Execute full production pipeline run
            output = orchestrator.execute_pipeline(question, query_embedding, embedder)
            
            scores = self.evaluate_turn(question, output, chunks)
            
            total_cr += scores["context_relevance"]
            total_f += scores["faithfulness"]
            total_ac += scores["answer_completeness"]
            
            print(f"Test case #{index} audited -> Context Relevance: {scores['context_relevance']} | Faithfulness: {scores['faithfulness']} | Answer Completeness: {scores['answer_completeness']}")

        metrics_summary = {
            "avg_context_relevance": round(total_cr / count, 2) if count > 0 else 0.0,
            "avg_faithfulness": round(total_f / count, 2) if count > 0 else 0.0,
            "avg_answer_completeness": round(total_ac / count, 2) if count > 0 else 0.0,
            "total_tests_executed": count
        }

        print(f"\n================ BENCHMARK FINISHED ================\n{json.dumps(metrics_summary, indent=2)}")
        return metrics_summary