"""Fetch the fixed benchmark corpus (5 public arXiv PDFs) into eval/corpus/.

PDFs are gitignored by design; this script is the reproducible source of truth.
"""
import os
import sys
import urllib.request

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

SOURCES = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "retrieval_augmented_generation.pdf": "https://arxiv.org/pdf/2005.11401",
    "lora.pdf": "https://arxiv.org/pdf/2106.09685",
    "deep_residual_learning.pdf": "https://arxiv.org/pdf/1512.03385",
}


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (benchmark)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def main() -> int:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    for filename, url in SOURCES.items():
        dest = os.path.join(CORPUS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
            print(f"[skip] {filename} ({os.path.getsize(dest)} bytes)")
            continue
        print(f"[get ] {filename} <- {url}")
        try:
            download(url, dest)
        except Exception as exc:
            print(f"[fail] {filename}: {exc}", file=sys.stderr)
            return 1
        print(f"[ok  ] {filename} ({os.path.getsize(dest)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
