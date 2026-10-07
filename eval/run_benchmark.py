"""Benchmark harness for the PDF-QA RAG pipeline.

Sections:
  --retrieval     Recall@1/@5/@10 + MRR for FAISS, BM25, hybrid RRF, hybrid+rerank
                  (plus per-config retrieval latency). No LLM calls.
  --faithfulness  Full generation with the self-reflection layer ON vs OFF,
                  each answer judged supported/unsupported against its context.
  --latency       End-to-end median/p95 with and without the semantic cache,
                  cache hit rate (exact repeats vs paraphrases), distance stats.
  --all           All three sections.
  --report        Rebuild eval/REPORT.md from existing result files.

Results land in eval/results/*.json and eval/REPORT.md is regenerated after
every section, so partial runs still produce a readable report.
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.join(EVAL_DIR, "results")

import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

from corpus import load_corpus, load_gold, resolve_gold_chunks  # noqa: E402

RETRIEVER_CONFIGS = [
    # (label, mode, use_reranker)
    ("FAISS (dense only)", "dense", False),
    ("BM25 (sparse only)", "sparse", False),
    ("Hybrid RRF (no rerank)", "hybrid", False),
    ("Hybrid RRF + cross-encoder", "hybrid", True),
]

PRODUCTION_MODE = ("hybrid", True)  # mode, use_reranker used by the app


def silence_logs():
    logging.getLogger().setLevel(logging.WARNING)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), p))


def median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def setup():
    """Build the shared world: corpus, gold set, embedder, retriever, client."""
    from scripts.embedding import get_embedder
    from scripts.vector_db import HybridRetriever
    from scripts import qa_model

    chunks, embeddings = load_corpus()
    records = load_gold()
    gold_map, problems = resolve_gold_chunks(chunks, records)
    if problems:
        raise SystemExit("Gold set no longer validates:\n  " + "\n  ".join(problems))

    embedder = get_embedder()
    retriever = HybridRetriever(chunks=chunks, embeddings=embeddings)
    client = qa_model.load_model()
    return records, chunks, embeddings, gold_map, embedder, retriever, client


# --------------------------------------------------------------------------
# Section A: retrieval quality (no LLM)
# --------------------------------------------------------------------------
def run_retrieval() -> Dict:
    records, chunks, embeddings, gold_map, embedder, retriever, _ = setup()
    print(f"[retrieval] {len(records)} questions, {len(chunks)} chunks")

    query_embeddings = {r["id"]: embedder.encode(r["question"]) for r in records}

    results = {}
    for label, mode, use_reranker in RETRIEVER_CONFIGS:
        latencies, per_q = [], {}
        hits = {1: 0, 5: 0, 10: 0}
        rr_sum = 0.0

        for rec in records:
            qid = rec["id"]
            gold = set(gold_map[qid])
            t0 = time.perf_counter()
            ranked = retriever.rank(
                query=rec["question"],
                query_embedding=query_embeddings[qid],
                top_k=10,
                mode=mode,
                use_reranker=use_reranker,
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            ranked_ids = [doc_id for doc_id, _ in ranked]
            for k in hits:
                if any(cid in gold for cid in ranked_ids[:k]):
                    hits[k] += 1
            first_gold_rank = next(
                (i + 1 for i, cid in enumerate(ranked_ids) if cid in gold), None
            )
            rr_sum += (1.0 / first_gold_rank) if first_gold_rank else 0.0
            per_q[qid] = {
                "rank_of_first_gold": first_gold_rank,
                "top5": ranked_ids[:5],
                "gold_chunks": sorted(gold),
            }

        n = len(records)
        results[label] = {
            "mode": mode,
            "use_reranker": use_reranker,
            "recall@1": round(hits[1] / n, 4),
            "recall@5": round(hits[5] / n, 4),
            "recall@10": round(hits[10] / n, 4),
            "mrr@10": round(rr_sum / n, 4),
            "retrieval_ms_median": round(median(latencies), 2),
            "retrieval_ms_p95": round(percentile(latencies, 95), 2),
            "per_question": per_q,
        }
        print(
            f"  {label:28s} R@1={results[label]['recall@1']:.3f} "
            f"R@5={results[label]['recall@5']:.3f} "
            f"R@10={results[label]['recall@10']:.3f} "
            f"MRR={results[label]['mrr@10']:.3f} "
            f"({results[label]['retrieval_ms_median']:.1f}ms)"
        )

    out = {
        "n_questions": len(records),
        "n_chunks": len(chunks),
        "configs": results,
    }
    _save("retrieval.json", out)
    return out


# --------------------------------------------------------------------------
# Section B: faithfulness / hallucination rate (reflection ON vs OFF)
# --------------------------------------------------------------------------
def run_faithfulness() -> Dict:
    from scripts import qa_model
    from scripts.evaluator import RAGEvaluator

    records, chunks, embeddings, gold_map, embedder, retriever, client = setup()
    judge = RAGEvaluator(client=client)
    out_path = os.path.join(RESULTS_DIR, "faithfulness.jsonl")

    # Resume support: keep only SUCCESSFUL (qid, reflection) pairs so errored
    # rows (e.g. rate limits, payment errors) are retried on the next run.
    done: Dict[tuple, Dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        good = [r for r in rows if not r.get("error")]
        if len(good) != len(rows):
            with open(out_path, "w", encoding="utf-8") as f:
                for r in good:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[faithfulness] dropped {len(rows) - len(good)} errored rows")
        for row in good:
            done[(row["qid"], row["reflection"])] = row
        if good:
            print(f"[faithfulness] resuming, {len(good)} rows already done")

    modes = [False, True]
    n_total = len(records) * len(modes)
    processed = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        for rec in records:
            qid = rec["id"]
            query_vec = embedder.encode(rec["question"])

            # Identical retrieval for both reflection settings (deterministic)
            t_ret = time.perf_counter()
            retrieved, is_confident = retriever.search(
                query=rec["question"], query_embedding=query_vec, top_k=5
            )
            retrieval_ms = (time.perf_counter() - t_ret) * 1000

            for reflection in modes:
                key = (qid, reflection)
                if key in done:
                    processed += 1
                    continue

                t0 = time.perf_counter()
                try:
                    output = qa_model.answer_question(
                        client=client,
                        question=rec["question"],
                        retrieved_chunks=retrieved,
                        is_confident=is_confident,
                        enable_reflection=reflection,
                    )
                    gen_error = None
                except Exception as exc:  # surface API failures, allow resume
                    output, gen_error = {"answer": "", "has_answer": False}, str(exc)
                gen_ms = (time.perf_counter() - t0) * 1000

                abstained = (not output.get("has_answer", True)) or not output.get("answer")
                supported: Optional[bool] = None
                if not abstained and gen_error is None:
                    t_j = time.perf_counter()
                    supported = judge.judge_supported(
                        rec["question"], output["answer"], retrieved
                    )
                    judge_ms = (time.perf_counter() - t_j) * 1000
                else:
                    judge_ms = 0.0

                row = {
                    "qid": qid,
                    "reflection": reflection,
                    "answer": output.get("answer", ""),
                    "has_answer": output.get("has_answer", True),
                    "abstained": bool(abstained),
                    "supported": supported,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(gen_ms, 2),
                    "judge_ms": round(judge_ms, 2),
                    "retrieved_chunk_ids": [c["metadata"]["chunk_id"] for c in retrieved],
                    "gold_chunks": sorted(gold_map[qid]),
                    "error": gen_error,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done[key] = row
                processed += 1
                status = "ABS" if abstained else ("OK " if supported else "UNS")
                print(
                    f"  [{processed}/{n_total}] {qid} reflection={'on' if reflection else 'off'} "
                    f"-> {status} ({gen_ms:.0f}ms)"
                )

    summary = _summarize_faithfulness(done)
    _save("faithfulness.json", summary)
    return summary


def _summarize_faithfulness(done: Dict[tuple, Dict]) -> Dict:
    summary = {"configs": {}}
    for reflection in (False, True):
        rows = [r for (q, rf), r in done.items() if rf == reflection and not r["error"]]
        answered = [r for r in rows if not r["abstained"]]
        supported = [r for r in answered if r["supported"]]
        unsupported = [r for r in answered if not r["supported"]]
        cfg = {
            "n": len(rows),
            "answered": len(answered),
            "abstentions": sum(1 for r in rows if r["abstained"]),
            "abstention_rate": round(sum(1 for r in rows if r["abstained"]) / len(rows), 4) if rows else 0.0,
            "supported": len(supported),
            "unsupported": len(unsupported),
            "hallucination_rate": round(len(unsupported) / len(answered), 4) if answered else 0.0,
            "supported_rate_overall": round(len(supported) / len(rows), 4) if rows else 0.0,
            "generation_ms_median": median([r["generation_ms"] for r in rows]),
            "errors": sum(1 for r in rows if r["error"]),
        }
        summary["configs"]["reflection_" + ("on" if reflection else "off")] = cfg

    a = summary["configs"]["reflection_off"]["hallucination_rate"]
    b = summary["configs"]["reflection_on"]["hallucination_rate"]
    summary["hallucination_rate_change"] = {"without": a, "with": b, "delta": round(b - a, 4)}
    return summary


# --------------------------------------------------------------------------
# Section C+D: latency and semantic cache
# --------------------------------------------------------------------------
def run_latency() -> Dict:
    from scripts.orchestrator import ProductionRAGOrchestrator
    from scripts import qa_model

    records, chunks, embeddings, gold_map, embedder, retriever, client = setup()
    n = len(records)

    # ---- C1: retrieval-only latency per retriever config ----
    retrieval_lat: Dict[str, List[float]] = {label: [] for label, _, _ in RETRIEVER_CONFIGS}
    for rec in records:
        qvec = embedder.encode(rec["question"])
        for label, mode, use_reranker in RETRIEVER_CONFIGS:
            t0 = time.perf_counter()
            retriever.rank(rec["question"], qvec, top_k=5, mode=mode, use_reranker=use_reranker)
            retrieval_lat[label].append((time.perf_counter() - t0) * 1000)
    retrieval_latency = {
        label: {
            "median_ms": round(median(v), 2),
            "p95_ms": round(percentile(v, 95), 2),
        }
        for label, v in retrieval_lat.items()
    }
    print("[latency] retrieval-only:", json.dumps(retrieval_latency))

    def new_orchestrator() -> ProductionRAGOrchestrator:
        orch = ProductionRAGOrchestrator(
            retriever=retriever, llm=client, embedding_dim=embeddings.shape[1]
        )
        orch.memory.history = []
        return orch

    def run_pass(queries, use_cache: bool, label: str, orch=None):
        """Run queries sequentially through the production pipeline.

        Pass an existing orchestrator to keep the cache warm across passes
        (pass 1 -> pass 2 must share one cache, as a real session would).
        """
        if orch is None:
            orch = new_orchestrator()
        rows = []
        for idx, (qid, text) in enumerate(queries):
            qvec = embedder.encode(text)
            try:
                out = orch.execute_pipeline(
                    question=text,
                    query_embedding=qvec,
                    embedder=embedder,
                    use_cache=use_cache,
                    enable_reflection=True,
                )
                err = None
            except Exception as exc:
                out, err = {"answer": "", "has_answer": False, "_meta": {}}, str(exc)
            meta = out.get("_meta", {})
            rows.append({
                "qid": qid,
                "query": text,
                "answer": out.get("answer", ""),
                "total_ms": meta.get("total_ms", 0.0),
                "retrieval_ms": meta.get("retrieval_ms", 0.0),
                "generation_ms": meta.get("generation_ms", 0.0),
                "cache_hit": meta.get("cache_hit", False),
                "cache_distance": meta.get("cache_distance"),
                "error": err,
            })
            print(
                f"  [{label} {idx + 1}/{len(queries)}] {qid} "
                f"{'HIT' if rows[-1]['cache_hit'] else 'miss'} {rows[-1]['total_ms']:.0f}ms"
            )
        return orch, rows

    # ---- D1: pass 1 — cold cache, all 40 originals ----
    # One orchestrator serves pass 1 + pass 2 so the cache warmed by pass 1
    # is visible to pass 2 (mirrors a real multi-turn session).
    original_queries = [(r["id"], r["question"]) for r in records]
    session_orch = new_orchestrator()
    _, pass1 = run_pass(original_queries, use_cache=True, label="cold", orch=session_orch)
    pass1_answer_by_qid = {r["qid"]: r["answer"] for r in pass1}

    # Pass 2 mix: first 20 exact repeats, last 20 paraphrases
    exact = [(r["id"], r["question"]) for r in records[:20]]
    paraphrases = [(r["id"], r["paraphrase"]) for r in records[20:]]
    pass2_queries = exact + paraphrases

    # ---- D2: pass 2 — repeats+paraphrases against the WARM cache ----
    _, pass2 = run_pass(pass2_queries, use_cache=True, label="cache-on", orch=session_orch)
    # ---- D3: pass 3 — identical mix, cache OFF (fresh orchestrator) ----
    _, pass3 = run_pass(pass2_queries, use_cache=False, label="cache-off")

    # ---- Cache analysis ----
    exact_rows = pass2[:20]
    para_rows = pass2[20:]

    # A cache hit is VALID when the served answer matches the answer that was
    # generated for the paired original question in pass 1 (i.e. the cache
    # returned the semantically right response, not a near-miss from an
    # unrelated question).
    def validity(rows):
        hits = [r for r in rows if r["cache_hit"]]
        if not hits:
            return 0, 0
        valid = sum(
            1 for r in hits
            if r["answer"].strip() == pass1_answer_by_qid.get(r["qid"], "").strip()
        )
        return valid, len(hits)

    exact_valid, exact_hits = validity(exact_rows)
    para_valid, para_hits = validity(para_rows)

    def hit_rate(rows):
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["cache_hit"]) / len(rows)

    exact_distances = [r["cache_distance"] for r in exact_rows if r["cache_distance"] is not None]
    para_distances = [r["cache_distance"] for r in para_rows if r["cache_distance"] is not None]
    miss_distances = [r["cache_distance"] for r in pass2 if not r["cache_hit"] and r["cache_distance"] is not None]

    # Calibration: what limit would be needed to capture exact repeats, and
    # would it also fire on paraphrases (true) or would paraphrases stay out?
    calib = {}
    if exact_distances:
        calib["max_exact_distance"] = round(max(exact_distances), 4)
    if para_distances:
        calib["min_paraphrase_distance"] = round(min(para_distances), 4)
        calib["median_paraphrase_distance"] = round(median(para_distances), 4)
    if exact_distances and para_distances:
        calib["separable"] = max(exact_distances) < min(para_distances)

    lat_on = [r["total_ms"] for r in pass2]
    lat_off = [r["total_ms"] for r in pass3]
    lat_cold = [r["total_ms"] for r in pass1]
    hits_on = sum(1 for r in pass2 if r["cache_hit"])

    out = {
        "retrieval_latency": retrieval_latency,
        "pass1_cold": {
            "n": len(pass1),
            "median_ms": round(median(lat_cold), 2),
            "p95_ms": round(percentile(lat_cold, 95), 2),
        },
        "cache": {
            "shipped_threshold": 0.95,
            "shipped_l2_limit": round(1.0 - 0.95, 4),
            "pass2_hit_rate": round(hit_rate(pass2), 4),
            "pass2_exact_hit_rate": round(hit_rate(exact_rows), 4),
            "pass2_paraphrase_hit_rate": round(hit_rate(para_rows), 4),
            "hits": hits_on,
            "queries": len(pass2),
            "exact_hits": exact_hits,
            "exact_valid_hits": exact_valid,
            "paraphrase_hits": para_hits,
            "paraphrase_valid_hits": para_valid,
            "exact_distances": [round(d, 4) for d in exact_distances],
            "paraphrase_distances": [round(d, 4) for d in para_distances],
            "miss_distances": [round(d, 4) for d in miss_distances[:60]],
            "calibration": calib,
        },
        "latency_with_cache": {
            "median_ms": round(median(lat_on), 2),
            "p95_ms": round(percentile(lat_on, 95), 2),
        },
        "latency_without_cache": {
            "median_ms": round(median(lat_off), 2),
            "p95_ms": round(percentile(lat_off, 95), 2),
        },
        "pass1": pass1,
        "pass2": pass2,
        "pass3": pass3,
    }
    _save("latency.json", out)
    print(
        f"[latency] cache hit rate {out['cache']['pass2_hit_rate']:.1%} | "
        f"median {out['latency_with_cache']['median_ms']:.0f}ms (cache on) vs "
        f"{out['latency_without_cache']['median_ms']:.0f}ms (off)"
    )
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _load(name: str) -> Optional[Dict]:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] skipping unreadable {path}: {exc}")
        return None


def _save(name: str, data: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[saved] {path}")


def build_report() -> str:
    retrieval = _load("retrieval.json")
    faithfulness = _load("faithfulness.json")
    latency = _load("latency.json")

    lines = ["# PDF-QA Benchmark Report", ""]
    lines.append("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")

    if retrieval:
        lines += [
            "## 1. Retrieval quality (Recall@k)",
            "",
            f"Corpus: {retrieval['n_chunks']} chunks from 5 arXiv PDFs · "
            f"Gold set: {retrieval['n_questions']} questions · k for Recall: 1 / 5 / 10",
            "",
            "| Config | R@1 | R@5 | R@10 | MRR@10 | retrieval ms (med / p95) |",
            "|---|---|---|---|---|---|",
        ]
        for label, cfg in retrieval["configs"].items():
            lines.append(
                f"| {label} | {cfg['recall@1']:.3f} | {cfg['recall@5']:.3f} | "
                f"{cfg['recall@10']:.3f} | {cfg['mrr@10']:.3f} | "
                f"{cfg['retrieval_ms_median']:.1f} / {cfg['retrieval_ms_p95']:.1f} |"
            )
        lines.append("")

    if faithfulness:
        lines += ["## 2. Faithfulness / hallucination rate", ""]
        lines.append("| Config | answered | abstentions | supported | unsupported | hallucination rate |")
        lines.append("|---|---|---|---|---|---|")
        for name, cfg in faithfulness["configs"].items():
            label = "Reflection ON" if name.endswith("on") else "Reflection OFF"
            lines.append(
                f"| {label} | {cfg['answered']}/{cfg['n']} | {cfg['abstentions']} "
                f"| {cfg['supported']} | {cfg['unsupported']} | **{cfg['hallucination_rate']:.1%}** |"
            )
        ch = faithfulness["hallucination_rate_change"]
        lines += [
            "",
            f"Hallucination rate **{ch['without']:.1%} -> {ch['with']:.1%}** "
            f"(delta {ch['delta']:+.1%}) with the self-reflection layer.",
            "",
        ]

    if latency:
        lines += ["## 3. Latency (end-to-end pipeline, median / p95)", ""]
        lines.append("| Scenario | median ms | p95 ms |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Cold cache (first pass) | {latency['pass1_cold']['median_ms']:.0f} | "
            f"{latency['pass1_cold']['p95_ms']:.0f} |"
        )
        lines.append(
            f"| Cache ON (pass 2, repeats+paraphrases) | {latency['latency_with_cache']['median_ms']:.0f} | "
            f"{latency['latency_with_cache']['p95_ms']:.0f} |"
        )
        lines.append(
            f"| Cache OFF (pass 3, same queries) | {latency['latency_without_cache']['median_ms']:.0f} | "
            f"{latency['latency_without_cache']['p95_ms']:.0f} |"
        )
        lines.append("")
        lines.append("Retrieval stage only (no LLM):")
        lines.append("")
        lines.append("| Config | median ms | p95 ms |")
        lines.append("|---|---|---|")
        for label, v in latency["retrieval_latency"].items():
            lines.append(f"| {label} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} |")
        lines.append("")

        c = latency["cache"]
        lines += [
            "## 4. Semantic cache",
            "",
            f"- Shipped threshold: `{c['shipped_threshold']}` -> max L2 distance "
            f"`{c['shipped_l2_limit']}`",
            f"- Hit rate overall: **{c['pass2_hit_rate']:.1%}** "
            f"({c['hits']}/{c['queries']})",
            f"- Exact repeats: **{c['pass2_exact_hit_rate']:.1%}** · "
            f"Paraphrases: **{c['pass2_paraphrase_hit_rate']:.1%}**",
        ]
        if c.get("calibration"):
            cal = c["calibration"]
            lines.append(
                f"- Distance stats: exact max `{cal.get('max_exact_distance')}`, "
                f"paraphrase median `{cal.get('median_paraphrase_distance')}`, "
                f"min `{cal.get('min_paraphrase_distance')}`, "
                f"separable={cal.get('separable')}"
            )
        lines.append("")

    report = "\n".join(lines) + "\n"
    path = os.path.join(EVAL_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[saved] {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    silence_logs()

    if not (args.retrieval or args.faithfulness or args.latency or args.all or args.report):
        args.all = True

    try:
        if args.all or args.retrieval:
            run_retrieval()
        if args.all or args.faithfulness:
            run_faithfulness()
        if args.all or args.latency:
            run_latency()
    finally:
        # Never let report generation mask the original section error.
        try:
            build_report()
        except Exception as exc:
            print(f"[warn] report generation failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
