# 📄 AI Legal Document Assistant

[![tests](https://github.com/ze2322/AI-Legal-Document-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ze2322/AI-Legal-Document-Assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

An AI-powered **Retrieval-Augmented Generation (RAG)** application that lets you upload **PDF** or **DOCX** legal documents and ask natural language questions about their contents, using a locally hosted **Llama 3.2** model through **Ollama**.

Everything runs locally — the document never leaves the machine.

---

# Preview

![Application Preview](screenshots/Screenshot%202026-06-27%20225131.png)

---

# Overview

The AI Legal Document Assistant automatically:

- Extracts text from PDF and DOCX documents.
- Splits documents into semantic chunks.
- Generates embeddings using Sentence Transformers.
- Stores embeddings in a FAISS vector database.
- Retrieves the most relevant chunks based on the user's question.
- Uses Llama 3.2 (via Ollama) to generate context-aware answers.
- Displays both the generated answer and the retrieved source chunks.

---

# Features

- Upload PDF and DOCX documents (DOCX tables are extracted too, not just paragraphs)
- Automatic document parsing and indexing
- Semantic search using FAISS (cosine similarity on normalised vectors)
- Retrieval-Augmented Generation (RAG)
- Local inference using Ollama (Llama 3.2)
- Token-by-token streaming answers, straight from Ollama
- Conversation history
- Retrieved source chunks with similarity scores
- Index persisted to disk and restored on restart
- Fully configurable through environment variables
- Unit-tested service layer: 73 tests, of which 71 run with no GPU, no model download and no Ollama process
- Streamlit web interface
- Docker support

---

# Technologies Used

- Python 3.11+
- Streamlit
- Ollama
- Llama 3.2
- FAISS
- Sentence Transformers
- LangChain Text Splitters
- PyMuPDF
- python-docx
- NumPy

---

# Architecture

```text
Upload  ->  DocumentParser  ->  TextChunker  ->  EmbeddingService  ->  VectorStore
                                                                          |
Question  ->  EmbeddingService  ->  VectorStore.search  ->  top-k chunks  -+
                                                               |
                                                        LLMService  ->  answer
```

Each stage is a small, independently testable class. `RAGPipeline` wires them
together and accepts injected collaborators, which is what lets the test suite
swap in fakes instead of loading real models.

---

# Project Structure

```text
AI-Legal-Document-Assistant/
│
├── app.py                     # Streamlit UI
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Test and lint dependencies
├── pyproject.toml             # pytest and ruff configuration
├── Dockerfile
├── .dockerignore
├── .env.example               # Documented configuration template
├── README.md
│
├── services/
│   ├── parser.py              # PDF / DOCX text extraction
│   ├── chunker.py             # Overlapping text splitting
│   ├── embeddings.py          # Sentence-Transformers embeddings
│   ├── vector_store.py        # FAISS index, search and persistence
│   ├── llm.py                 # Ollama chat client
│   ├── rag.py                 # End-to-end pipeline
│   └── exceptions.py          # Application error types
│
├── utils/
│   ├── config.py              # Environment-driven settings
│   └── logger.py              # Shared logging setup
│
├── tests/                     # pytest suite
│
├── data/
│   ├── documents/             # Sample documents
│   └── faiss_index/           # Generated index (git-ignored)
│
└── screenshots/
    ├── Screenshot 2026-06-27 224909.png
    ├── Screenshot 2026-06-27 224923.png
    ├── Screenshot 2026-06-27 225118.png
    ├── Screenshot 2026-06-27 225131.png
    └── Screenshot 2026-06-27 225141.png
```

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/ze2322/AI-Legal-Document-Assistant.git
cd AI-Legal-Document-Assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Install Ollama

Download Ollama from:

https://ollama.com/download

---

## 6. Download the Model

```bash
ollama pull llama3.2
```

---

## 7. Start Ollama

```bash
ollama serve
```

---

## 8. Run the Application

Open your browser at:

```bash
streamlit run app.py
```

```
http://localhost:8501
```

---

# Configuration

All settings are optional and read from environment variables, or from a `.env`
file in the project root. Copy `.env.example` to get started:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model used for embeddings |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model used to generate answers |
| `OLLAMA_HOST` | *(client default)* | Ollama server URL, e.g. `http://host.docker.internal:11434` |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared between neighbouring chunks |
| `TOP_K` | `3` | Number of chunks retrieved per question |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_TOKENS` | `512` | Maximum tokens generated per answer |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout for Ollama |
| `INDEX_DIR` | `data/faiss_index` | Where the FAISS index is persisted |
| `MAX_UPLOAD_MB` | `25` | Maximum accepted upload size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

> Changing `MAX_UPLOAD_MB` also means updating `maxUploadSize` in
> `.streamlit/config.toml`, which is Streamlit's own upload cap.

---

# Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The default run uses fake embedding and LLM services, so it needs neither a GPU
nor a running Ollama instance and finishes in a few seconds.

Tests that exercise the real embedding model are marked `integration` and are
deselected by default. Run them explicitly (the first run downloads model
weights):

```bash
pytest -m integration
```

Lint checks:

```bash
ruff check .
```

---

# Docker

## Build the Image

```bash
docker build -t legal-rag .
```

## Run the Container

```bash
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 legal-rag
```

> **Note:** Ollama must already be installed and running on the host machine.

Ollama runs on the **host**, not in the container. `host.docker.internal`
resolves on Docker Desktop for Windows and macOS. On Linux, either run the
container with `--add-host=host.docker.internal:host-gateway` or point it at the
host directly:

```bash
docker run -p 8501:8501 -e OLLAMA_HOST=http://172.17.0.1:11434 legal-rag
```

---

# How It Works

1. **Parsing** — PyMuPDF extracts text page by page from PDFs; `python-docx`
   reads paragraphs *and* table cells from DOCX files, because legal documents
   often put obligations in tables. A document with no extractable text (a
   scanned image, for example) is rejected with a clear message rather than
   producing an empty index.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits on paragraph, line and
   sentence boundaries first, falling back to characters. Overlap keeps
   sentences that straddle a boundary retrievable.
3. **Embedding** — `all-MiniLM-L6-v2` produces 384-dimensional unit-norm
   vectors. The model loads lazily on first use, so start-up stays fast.
4. **Indexing** — `IndexFlatIP` over normalised vectors makes the inner product
   equal to cosine similarity. Chunks are persisted as JSON rather than pickle,
   so loading an index never executes arbitrary code.
5. **Retrieval** — the question is embedded with the same model, and the top-k
   most similar chunks are returned with their scores.
6. **Generation** — the retrieved chunks are formatted into a labelled context
   block and sent to Llama 3.2 with a system prompt that forbids answering from
   outside that context. Tokens are streamed straight to the UI.

---

# Application Walkthrough

## 1. Home Screen

Launch the application and upload a legal document.

![Home](screenshots/Screenshot%202026-06-27%20224909.png)

---

## 2. Document Uploaded

After selecting a document, it is automatically parsed and indexed.

![Upload](screenshots/Screenshot%202026-06-27%20224923.png)

---

## 3. Ask a Question

Ask any question related to the uploaded document.

![Question](screenshots/Screenshot%202026-06-27%20225118.png)

---

## 4. AI Generated Answer

The RAG pipeline retrieves the relevant chunks and Llama 3.2 generates a context-aware response.

![Answer](screenshots/Screenshot%202026-06-27%20225131.png)

---

## 5. Retrieved Source Chunks

The retrieved chunks and similarity scores are displayed for transparency.

![Sources](screenshots/Screenshot%202026-06-27%20225141.png)

---

# Sample Questions

- What are the three specific entities or units responsible for assessing the technical merits of submitted proposals during a competitive selection procedure?
- Summarize this document.
- Who can become an implementing partner?
- What are the eligibility requirements?
- Explain the grant award procedure.
- What records should be retained, and for how long?
- What are the key obligations mentioned in the agreement?

---

# Sample Output

### Question

What are the responsibilities of implementing partners?

### Answer

Implementing Partners (IPs) assume full responsibility and accountability for the effective use of resources transferred by UNITAR. They are responsible for delivering project outputs defined in the Grant-out Agreement while supporting UNITAR's strategic and programmatic objectives.

The application also displays the retrieved source chunks and similarity scores used to generate the response.

---

# Limitations

- Scanned PDFs are not supported; they would need an OCR step.
- One document is indexed at a time — uploading a new file replaces the previous
  index, so answers never mix sources.
- Conversation history is not fed back into the prompt, so follow-up questions
  need to be self-contained.
- Chat history lives in the Streamlit session and is lost on refresh.

---

# Future Improvements

- OCR fallback for scanned documents
- Support for multiple uploaded documents, with per-document filtering
- Multi-turn conversational memory (query rewriting for follow-up questions)
- Re-ranking retrieved chunks before generation
- Persistent chat history
- Citation highlighting inside generated answers
- Cloud deployment
- User authentication

---

# License

Released under the [MIT License](LICENSE).
