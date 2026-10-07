"""Shared corpus loading for the benchmark.

Runs the production ingestion path (UniversalParser -> TextNormalizer ->
SemanticChunker -> embedder) over eval/corpus/*.pdf and caches the result so
validation, retrieval runs, and latency runs all index the exact same chunks.
"""
import glob
import json
import os
import sys

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, ROOT_DIR)

CORPUS_DIR = os.path.join(EVAL_DIR, "corpus")
CACHE_DIR = os.path.join(EVAL_DIR, "cache")
CHUNKS_CACHE = os.path.join(CACHE_DIR, "chunks.json")
EMB_CACHE = os.path.join(CACHE_DIR, "embeddings.npy")

# Mirrors app.py ingestion settings exactly.
CHUNK_SIZE = 150
CHUNK_OVERLAP = 25


def load_corpus(force: bool = False):
    """Return (chunks, embeddings). Uses on-disk cache unless force=True."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.pdf")))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {CORPUS_DIR}. Run: uv run python eval/download_corpus.py"
        )

    if not force and os.path.exists(CHUNKS_CACHE) and os.path.exists(EMB_CACHE):
        with open(CHUNKS_CACHE, encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(EMB_CACHE)
        if len(chunks) == embeddings.shape[0]:
            return chunks, embeddings

    from scripts.parser import UniversalParser
    from scripts.cleaner import TextNormalizer
    from scripts.chunker import SemanticChunker
    from scripts.embedding import get_embedder, embed

    parser = UniversalParser()
    chunker = SemanticChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_pages = []
    for path in pdfs:
        for page in parser.parse(path):
            TextNormalizer.clean(page)
            all_pages.append(page)

    chunks = chunker.split_pages(all_pages)

    embedder = get_embedder()
    embeddings = embed(embedder, [c["content"] for c in chunks])

    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.save(EMB_CACHE, embeddings)
    return chunks, embeddings


def load_gold():
    """Return the parsed gold question list."""
    path = os.path.join(EVAL_DIR, "gold_set.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _norm(text: str) -> str:
    return " ".join(text.split())


def resolve_gold_chunks(chunks, records):
    """Map each gold question to the chunk ids whose text contains its snippet.

    Returns (mapping, problems) where mapping is {qid: [chunk_idx, ...]}.
    """
    normalized = [_norm(c["content"]) for c in chunks]
    mapping, problems = {}, []

    for rec in records:
        snippet = _norm(rec["snippet"])
        hits = [
            i for i, c in enumerate(chunks)
            if c["metadata"].get("source") == rec["source"]
            and c["metadata"].get("page_number") == rec["page"]
            and snippet in normalized[i]
        ]
        if not hits:
            # Fallback: chunk boundary may split the snippet -> accept chunks
            # that contain the majority of the snippet's words on the gold page.
            snip_words = set(snippet.lower().split())
            best, best_cov = [], 0.0
            for i, c in enumerate(chunks):
                if c["metadata"].get("source") != rec["source"] or \
                   c["metadata"].get("page_number") != rec["page"]:
                    continue
                cw = set(normalized[i].lower().split())
                cov = len(snip_words & cw) / len(snip_words) if snip_words else 0.0
                if cov > best_cov:
                    best, best_cov = [i], cov
                elif cov == best_cov and cov > 0:
                    best.append(i)
            if best_cov >= 0.6:
                hits = best
            else:
                problems.append(
                    f"{rec['id']}: snippet not found in any chunk of "
                    f"{rec['source']} p{rec['page']} (best coverage {best_cov:.2f})"
                )
        mapping[rec["id"]] = hits

    return mapping, problems
