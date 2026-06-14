import numpy as np
from parser import  UniversalParser
from cleaner import TextNormalizer
from chunker import SemanticChunker
from embedding import get_embedder, embed
from vector_db import HybridRetriever
import qa_model
from orchestrator import ProductionRAGOrchestrator
from evaluator import RAGEvaluator

# 1. Ingest + Normalize Docs (Step 1)
parser = UniversalParser()
raw_pages = parser.parse("sample.pdf") # Works for any supported path format

for page in raw_pages:
    TextNormalizer.clean(page)

chunker = SemanticChunker(chunk_size=150, chunk_overlap=25)
chunks = chunker.split_pages(raw_pages)

# 2. Extract Dense Array Base
embedder = get_embedder()
text_pool = [c["content"] for c in chunks]
embeddings_array = embed(embedder, text_pool)

# 3. Spin up Step 2, 3, & 4 Indexing
retriever = HybridRetriever(chunks=chunks, embeddings=embeddings_array)

# 4. Spin up Step 5, 6, 7, 9, & 10 Core Routing Systems
llm_endpoint = qa_model.load_model()
orchestrator = ProductionRAGOrchestrator(retriever=retriever, llm=llm_endpoint)

# --- LIVE TEST TURN ---
test_query = "What is the a k complex?"
test_vector = embedder.encode(test_query)

# Execute integrated query pass
response = orchestrator.execute_pipeline(test_query, test_vector, embedder)
print("\nFinal Output Object Recieved:\n", response)

# 5. Run Step 8 Benchmark Validation Layer
eval_suite = RAGEvaluator(client=llm_endpoint)
mock_dataset = [
    {"question": "What is the a k complex?"},
    {"question": "Who authored the report"}
]
eval_suite.run_benchmark_suite(orchestrator, embedder, mock_dataset)