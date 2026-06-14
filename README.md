Here is a clean, straightforward update for your `README.md` that accurately reflects everything we just built—without any corporate buzzwords or unnecessary jargon.

---

# PDF & Document Question Answering App

This is a Streamlit-based application that allows users to upload various document types and ask natural language questions about their content. The app extracts text, normalizes it, handles search using a hybrid retrieval setup, and answers questions using an LLM with precise source citations.

---

## Features

* **Multi-Format Ingestion**: Supports uploading PDF (scanned/digital), DOCX, PPTX, TXT, MD, and JSON files.
* **Advanced Text Processing**: Normalizes text by removing layout noise, fixing line-break hyphenations, and clearing hidden control characters.
* **Hybrid Retrieval Pipeline**: Combines dense semantic search (FAISS) with exact keyword matching (BM25) using Reciprocal Rank Fusion (RRF) for highly accurate document retrieval.
* **Two-Stage Reranking**: Filters retrieved text using a Cross-Encoder to prioritize the most contextually relevant chunks.
* **Source Confidence Filter**: Blocks irrelevant queries from reaching the LLM if the document corpus does not contain the answer.
* **Citation-Backed Answers**: Formats responses with interactive, superscript inline footnotes that link directly to specific page numbers, slides, or text blocks.
* **Hallucination Safeguards**: Cross-checks generated answers against raw source documents using a self-reflective validation layer.
* **Semantic Caching & Memory**: Intercepts repeating queries instantly using a vector cache (saving API calls) and maintains a sliding conversation history for multi-turn chat.

---

## Tech Stack

* **Frontend**: Streamlit
* **Core Languages**: Python
* **Parsing & OCR**: PyMuPDF, python-docx, python-pptx, EasyOCR, Pillow
* **Vector Search & Indexing**: FAISS (Dense) & Rank-BM25 (Sparse)
* **NLP & Reranking**: Sentence Transformers (Embedding & Cross-Encoder)
* **Inference Interface**: HuggingFace Hub Inference Client (Qwen/Qwen2.5-7B-Instruct)

---

## Project Structure

```text
├── app.py                  # Streamlit user interface & state coordinator
├── requirements.txt        # Project dependencies
└── scripts/
    ├── __init__.py         # Package initialization marker
    ├── document_parser.py  # Path-aware multimodal document reader
    ├── cleaner.py          # In-place unicode and spacing normalizer
    ├── chunker.py          # Whitespace-safe sliding window text splitter
    ├── embedding.py        # Sentence Transformer vector generation wrapper
    ├── vector_db.py        # Hybrid retriever, RRF, reranker, and confidence filter
    ├── qa_model.py         # Structured JSON response generator and self-audit layer
    ├── orchestrator.py     # Session memory, semantic cache, and logging tracker
    └── evaluator.py        # Programmatic LLM-as-a-Judge benchmarking suite

```

---

## Getting Started

1. **Install Dependencies**:
```bash
pip install uv
uv sync

```


2. **Configure Environment Variables**:
Create a `.env` file in the root directory and add your Hugging Face API token:
```env
HF_TOKEN=your_huggingface_token_here

```


3. **Run the Application**:
```bash
streamlit run app.py

```