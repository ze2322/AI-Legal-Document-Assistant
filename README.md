# AI Legal Document Assistant

## Overview

AI Legal Document Assistant is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or DOCX legal documents and ask natural language questions about their contents.

The application automatically parses the uploaded document, generates vector embeddings, stores them in a FAISS vector database, retrieves the most relevant document chunks, and uses a locally hosted Llama 3.2 model through Ollama to generate accurate, context-aware answers.

---

## Features

* Upload PDF and DOCX documents
* Automatic document indexing
* Retrieval-Augmented Generation (RAG)
* Semantic search using FAISS
* Local inference using Ollama (Llama 3.2)
* Conversation history
* Streaming typing effect
* Retrieved source chunks with similarity scores
* Streamlit web interface
* Docker support

---

## Technologies Used

* Python 3.11+
* Streamlit
* Ollama
* Llama 3.2
* FAISS
* Sentence Transformers
* LangChain Text Splitters
* PyMuPDF
* python-docx
* NumPy

---

# Project Structure

```text
AI-Legal-Document-Assistant/
│
├── app.py
├── config.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
│
├── services/
│   ├── parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   └── rag.py
│
├── data/
│   ├── documents/
│   └── faiss_index/
│
├── tests/
│
└── screenshots/
```

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Legal-Document-Assistant.git

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

## 6. Download the Llama 3.2 Model

```bash
ollama pull llama3.2
```

---

## 7. Start the Ollama Server

```bash
ollama serve
```

---

## 8. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Docker

## Build the Docker Image

```bash
docker build -t legal-rag .
```

## Run the Docker Container

```bash
docker run -p 8501:8501 legal-rag
```

> **Note:** Ollama must be installed and running on the host machine before starting the application.

---

# Example Workflow

1. Upload a PDF or DOCX document.
2. The document is automatically indexed.
3. Ask questions in natural language.
4. The system retrieves the most relevant document chunks.
5. Llama 3.2 generates an answer based on the retrieved context.
6. The application displays both the answer and the retrieved source chunks.

---

# Sample Questions

* What are the responsibilities of implementing partners?
* Summarize this document.
* Who can become an implementing partner?
* What records should be retained?
* What are the key obligations mentioned in the agreement?

---

# Future Improvements

* True token streaming from Ollama
* Multi-document retrieval
* Persistent chat history
* Cloud deployment
* Authentication and user management

