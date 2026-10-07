"""Validate the gold set against the real ingested chunk index.

Checks per question:
  1. The gold snippet exists verbatim in the claimed source page.
  2. At least one chunk (the real indexed unit) contains the snippet, and the
     resolved gold chunk set is recorded for Recall@k scoring.

Exit code 0 = gold set is trustworthy; anything else = do not run the benchmark.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import load_corpus, load_gold, resolve_gold_chunks, _norm, EVAL_DIR


def main() -> int:
    records = load_gold()
    print(f"Loaded {len(records)} gold questions")

    chunks, embeddings = load_corpus()
    print(f"Indexed {len(chunks)} chunks / {embeddings.shape[0]} vectors "
          f"(dim {embeddings.shape[1]})")

    # 1. Snippet-vs-page check against cleaned pages
    from corpus import CORPUS_DIR
    import glob
    import pymupdf
    from scripts.cleaner import TextNormalizer

    page_texts = {}
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.pdf"))):
        name = os.path.basename(path)
        doc = pymupdf.open(path)
        for pno, page in enumerate(doc, start=1):
            data = {"text": page.get_text("text")}
            TextNormalizer.clean(data)
            page_texts[(name, pno)] = data["text"]

    page_failures = []
    for rec in records:
        text = page_texts.get((rec["source"], rec["page"]), "")
        if _norm(rec["snippet"]) not in _norm(text):
            page_failures.append(
                f"{rec['id']}: snippet missing from {rec['source']} p{rec['page']}"
            )

    # 2. Chunk resolution
    mapping, chunk_failures = resolve_gold_chunks(chunks, records)

    if page_failures or chunk_failures:
        print("\nGOLD SET VALIDATION FAILED")
        for msg in page_failures + chunk_failures:
            print("  -", msg)
        return 1

    multi = {q: ids for q, ids in mapping.items() if len(ids) > 1}
    stats = {
        "questions": len(records),
        "chunks_indexed": len(chunks),
        "all_snippets_found_in_pages": True,
        "all_snippets_resolved_to_chunks": True,
        "questions_with_multiple_gold_chunks": len(multi),
        "avg_gold_chunks_per_question": round(
            sum(len(v) for v in mapping.values()) / len(mapping), 2
        ),
    }

    out = os.path.join(EVAL_DIR, "results", "gold_chunks.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "stats": stats}, f, indent=2)

    print("\nGOLD SET VALIDATED")
    print(json.dumps(stats, indent=2))
    if multi:
        print("Note: snippet spans a chunk boundary for:", ", ".join(sorted(multi)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
