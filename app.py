import tempfile
import time
from pathlib import Path

import streamlit as st

from services.rag import RAGPipeline



# Page Configuration
st.set_page_config(
    page_title="AI Legal Document Assistant",
    page_icon="📄",
    layout="wide"
)



# Load RAG Pipeline
@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()



# Session State
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Title
st.title("📄 AI Legal Document Assistant")

st.write(
    "Upload a PDF or DOCX file and ask questions about its contents."
)



# Upload Document
uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx"]
)

if uploaded_file is not None:

    # Build index only if a new file is uploaded
    if st.session_state.document_name != uploaded_file.name:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp:

            tmp.write(uploaded_file.read())
            temp_path = Path(tmp.name)

        with st.spinner("Processing document..."):
            rag.ingest_document(temp_path)

        st.session_state.indexed = True
        st.session_state.document_name = uploaded_file.name

    st.success(f"✅ {uploaded_file.name} is ready!")
    st.caption(f"Indexed {len(rag.vector_store.chunks)} chunks.")



# Clear Conversation
col1, col2 = st.columns([4, 1])

with col2:
    if st.button(" Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()



# Conversation History
if st.session_state.chat_history:

    st.subheader(" Conversation History")

    for chat in st.session_state.chat_history:

        with st.container():

            st.markdown(f"** You:** {chat['question']}")
            st.markdown(f"** AI:** {chat['answer']}")

            st.divider()



# Question
question = st.text_input(
    "Ask a question about the document",
    placeholder="type your question here",
    disabled=not st.session_state.indexed
)



# Answer
if question:

    with st.spinner("thinking..."):

        response = rag.ask(question)

    st.subheader(" AI Answer")

    placeholder = st.empty()

    full_answer = ""

    # Fake streaming effect
    for word in response["answer"].split():

        full_answer += word + " "

        placeholder.markdown(full_answer)

        time.sleep(0.03)

    # Save conversation
    st.session_state.chat_history.append({
        "question": question,
        "answer": full_answer
    })

    
    # Retrieved Sources
    with st.expander(" Retrieved Sources"):

        for source in response["sources"]:

            st.markdown(
                f"**Chunk {source['id']}** | Similarity: {source['score']:.4f}"
            )

            st.write(source["chunk"])

            st.divider()