# PDF-QA Benchmark Report

Generated: 2026-10-07 13:14:57

## 1. Retrieval quality (Recall@k)

Corpus: 414 chunks from 5 arXiv PDFs · Gold set: 40 questions · k for Recall: 1 / 5 / 10

| Config | R@1 | R@5 | R@10 | MRR@10 | retrieval ms (med / p95) |
|---|---|---|---|---|---|
| FAISS (dense only) | 0.225 | 0.425 | 0.600 | 0.334 | 0.1 / 0.1 |
| BM25 (sparse only) | 0.400 | 0.700 | 0.825 | 0.559 | 1.0 / 1.6 |
| Hybrid RRF (no rerank) | 0.250 | 0.675 | 0.875 | 0.440 | 1.1 / 1.6 |
| Hybrid RRF + cross-encoder | 0.500 | 0.850 | 0.975 | 0.654 | 922.1 / 1093.3 |

## 2. Faithfulness / hallucination rate

| Config | answered | abstentions | supported | unsupported | hallucination rate |
|---|---|---|---|---|---|
| Reflection OFF | 39/40 | 1 | 33 | 6 | **15.4%** |
| Reflection ON | 37/40 | 3 | 34 | 3 | **8.1%** |

Hallucination rate **15.4% -> 8.1%** (delta -7.3%) with the self-reflection layer.

## 3. Latency (end-to-end pipeline, median / p95)

| Scenario | median ms | p95 ms |
|---|---|---|
| Cold cache (first pass) | 2106 | 3930 |
| Cache ON (pass 2, repeats+paraphrases) | 2741 | 3576 |
| Cache OFF (pass 3, same queries) | 2823 | 4927 |

Retrieval stage only (no LLM):

| Config | median ms | p95 ms |
|---|---|---|
| FAISS (dense only) | 0.1 | 0.1 |
| BM25 (sparse only) | 0.9 | 1.5 |
| Hybrid RRF (no rerank) | 0.8 | 1.1 |
| Hybrid RRF + cross-encoder | 359.8 | 452.3 |

## 4. Semantic cache

- Shipped threshold: `0.95` -> max L2 distance `0.05`
- Hit rate overall: **0.0%** (0/40)
- Exact repeats: **0.0%** · Paraphrases: **0.0%**
- Distance stats: exact max `1.6377`, paraphrase median `1.1085`, min `0.6039`, separable=False

