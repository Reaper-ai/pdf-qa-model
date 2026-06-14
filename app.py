import os
import tempfile
import streamlit as st
import numpy as np
from scripts.parser import UniversalParser
from scripts.cleaner import TextNormalizer
from scripts.chunker import SemanticChunker
from scripts.embedding import get_embedder, embed
from scripts.vector_db import HybridRetriever
from scripts import qa_model    
from scripts.orchestrator import ProductionRAGOrchestrator

# 1. Page Configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="⚡")
st.title("Multimodal RAG")
st.markdown("---")

# 2. Lazy Initialization of Infrastructure Models
@st.cache_resource
def initialize_core_models():
    """Initializes heavy embedding and generation layers once and caches them globally."""
    embedder = get_embedder()
    llm_client = qa_model.load_model()
    doc_parser = UniversalParser()
    chunker_instance = SemanticChunker(chunk_size=150, chunk_overlap=25)
    return embedder, llm_client, doc_parser, chunker_instance

embedder, llm_client, doc_parser, chunker_instance = initialize_core_models()

# 3. Session State Persistent Setup
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'cached_file_name' not in st.session_state:
    st.session_state.cached_file_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 4. Sidebar File Ingestion Controls
with st.sidebar:
    st.header("📂 Document Control Center")
    uploaded_file = st.file_uploader(
        "Upload Source File", 
        type=["pdf", "docx", "doc", "pptx", "ppt", "jpg", "jpeg", "png", "json", "txt", "md"]
    )
    
    if uploaded_file:
        st.info(f"Active File: `{uploaded_file.name}`")
        
        # Trigger parsing if a completely new file is dropped in
        if st.session_state.cached_file_name != uploaded_file.name:
            with st.spinner("Executing Production Ingestion Engine..."):
                
                # Streamlit gives us an in-memory buffer. Write it to a safe temp file path for our parser.
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_filepath = temp_file.name
                
                try:
                    # Stage 1: Multimodal Parse
                    raw_pages = doc_parser.parse(temp_filepath)
                    # Override the temporary random name in metadata back to the real uploaded name
                    for page in raw_pages:
                        page["metadata"]["source"] = uploaded_file.name
                    
                    # Stage 2: Normalization Loop
                    for page in raw_pages:
                        TextNormalizer.clean(page)
                        
                    # Stage 3: Sliding Word Chunks
                    chunks = chunker_instance.split_pages(raw_pages)
                    
                    # Stage 4: Matrix Embedding Vectors
                    text_pool = [c["content"] for c in chunks]
                    embeddings_array = embed(embedder, text_pool)
                    
                    # Stage 5: Hybrid Dual-Indexed Retriever Initialization
                    retriever = HybridRetriever(chunks=chunks, embeddings=embeddings_array)
                    
                    # Stage 6: Build the Central Production Orchestration Layer
                    st.session_state.orchestrator = ProductionRAGOrchestrator(
                        retriever=retriever, 
                        llm=llm_client,
                        embedding_dim=embeddings_array.shape[1]
                    )
                    
                    # Sync State flags
                    st.session_state.cached_file_name = uploaded_file.name
                    st.session_state.chat_history = []  # Clear previous chat session boundaries
                    st.success("File context successfully compiled into hybrid vector spaces!")
                
                finally:
                    # Safely remove the temporary file from the disk storage
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)

# 5. Main Chat Interface Execution Block
if st.session_state.orchestrator is None:
    st.warning("Please upload a supported document in the sidebar to initialize the vector intelligence databases.")
else:
    # Render chat input bar
    user_query = st.chat_input("Ask a question about your uploaded document...")
    
    if user_query:
        with st.spinner("Running deep semantic retrieval + verification pipelines..."):
            query_vector = embedder.encode(user_query)
            
            # Fire transaction through entire caching and validation orchestration layer
            output_response = st.session_state.orchestrator.execute_pipeline(
                question=user_query,
                query_embedding=query_vector,
                embedder=embedder
            )
            
            # Extract fields from our strict production JSON schema format
            raw_answer = output_response.get("answer", "")
            citations = output_response.get("citations", [])
            
            # --- FORMAT INTERACTIVE SMALL IN-TEXT CITATIONS ---
            formatted_answer = raw_answer
            citation_footnotes = []
            
            if citations and output_response.get("has_answer", True):
                for idx, cite in enumerate(citations, start=1):
                    # Build superscript markdown index tags (e.g., <sup>[1]</sup>)
                    citation_tag = f" <sup>[{idx}]</sup>"
                    formatted_answer += citation_tag
                    
                    # Format human-readable metadata details for footer rendering
                    doc_type = cite.get("type", "unknown").upper()
                    source_name = cite.get("source", "Document")
                    
                    if "page_number" in cite:
                        loc = f"Page {cite['page_number']}"
                    elif "slide_number" in cite:
                        loc = f"Slide {cite['slide_number']}"
                    elif "block" in cite:
                        loc = f"Paragraph Block {cite['block']}"
                    elif "row" in cite:
                        loc = f"Data Row {cite['row']}"
                    else:
                        loc = "Global Context"
                        
                    author_str = f" by {cite['author']}" if cite.get("author") and cite["author"] != "Unknown" else ""
                    citation_footnotes.append(f"**[{idx}]** {source_name} ({doc_type}) — {loc}{author_str}")
            
            # Append transaction into history state
            st.session_state.chat_history.append({
                "query": user_query,
                "answer": formatted_answer,
                "footnotes": citation_footnotes
            })

    # Render Historical Turn Feeds (Reversed to keep latest conversation turns at top)
    if st.session_state.chat_history:
        st.subheader("💬 Chat Timeline")
        for turn in reversed(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**User:** {turn['query']}")
                # unsafe_allow_html=True parses our superscript <sup> tags beautifully!
                st.markdown(f"**Bot:** {turn['answer']}", unsafe_allow_html=True)
                
                # If footnotes exist, render them cleanly inside a micro-expander block
                if turn["footnotes"]:
                    with st.expander("🔍 Verified Source Citations", expanded=False):
                        for footnote in turn["footnotes"]:
                            st.caption(footnote)
                st.markdown("---")