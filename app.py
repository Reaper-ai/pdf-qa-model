import streamlit as st
from scripts import pdf_parser, cleaner, chunker, embedding, vector_db, qa_model

st.set_page_config(page_title="PDF QA", layout="wide")
model = qa_model.load_model()

st.title("PDF Q/A")

# Initialize session state for Q&A and caching
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

if 'cached_file_name' not in st.session_state:
    st.session_state.cached_file_name = None

file = st.file_uploader("Upload a PDF file", type=["pdf"])
is_scanned_file = st.checkbox("Is the PDF made of scanned images?")
question = st.text_input("Ask a question")


# Process PDF only when new file is uploaded
if file and st.session_state.cached_file_name != file.name:
    with st.spinner("Processing PDF..."):
        content = pdf_parser.parse_pdf(file, is_scanned_file)
        text = content

        cleaned_text = cleaner.clean_text(text)
        chunked_text = chunker.split_text(cleaned_text)
        embedder = embedding.get_embedder()
        embedded_text = embedding.embed(embedder, chunked_text)
        index = vector_db.build_index(embedded_text)

        # Cache for reuse
        st.session_state.chunked_text = chunked_text
        st.session_state.embedded_text = embedded_text
        st.session_state.index = index
        st.session_state.embedder = embedder
        st.session_state.cached_file_name = file.name
        st.session_state.qa_history = []

    st.success("PDF processed successfully.")

# Handle user query
if file and question:
    with st.spinner("Generating answer..."):
        query_embedding = st.session_state.embedder.encode(question)
        context_indices = vector_db.get_top_k_chunks(query_embedding, st.session_state.index)
        context_chunks = [st.session_state.chunked_text[i] for i in context_indices]
        context = ' '.join(context_chunks)

        answer = qa_model.answer_question(model, question, context)
        st.session_state.qa_history.insert(0, (question, answer))

    st.success("Answer generated.")

# Display chat-style Q&A history
if st.session_state.qa_history:
    st.markdown("### Answers: ")
    for q, a in st.session_state.qa_history:
        with st.container():
            st.markdown(f"**Q: {q}**")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
