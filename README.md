# 📄 AI Legal Document Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload **PDF** or **DOCX** legal documents and ask natural language questions about their contents using a locally hosted **Llama 3.2** model through **Ollama**.

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

-  Upload PDF and DOCX documents
-  Automatic document parsing and indexing
-  Semantic search using FAISS
-  Retrieval-Augmented Generation (RAG)
-  Local inference using Ollama (Llama 3.2)
-  Conversation history
-  Streaming typing effect
-  Retrieved source chunks with similarity scores
-  Streamlit web interface
-  Docker support

---

# Technologies Used

- Python 3.11+
- Streamlit
- Ollama
- Llama 3.2
- FAISS
- Sentence Transformers
- LangChain
- PyMuPDF
- python-docx
- NumPy

---

# Project Structure

```text
AI-Legal-Document-Assistant/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .dockerignore
│
├── services/
│   ├── parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   └── rag.py
│
├── utils/
│   └── config.py
│
├── data/
│   ├── documents/
│   └── faiss_index/
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

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
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

---

# How It Works

1. Upload a PDF or DOCX document.
2. The document is automatically parsed.
3. The text is split into chunks.
4. Sentence Transformers creates embeddings.
5. FAISS builds a vector index.
6. Ask a question.
7. The most relevant chunks are retrieved.
8. Llama 3.2 generates an answer.
9. The answer and retrieved source chunks are displayed.

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

- What are the responsibilities of implementing partners?
- Summarize this document.
- Who can become an implementing partner?
- What are the eligibility requirements?
- Explain the grant award procedure.
- What records should be retained?

---

# Sample Output

### Question

> What are the responsibilities of implementing partners?

### Answer

Implementing Partners (IPs) assume full responsibility and accountability for the effective use of resources transferred by UNITAR. They are responsible for delivering project outputs defined in the Grant-out Agreement while supporting UNITAR's strategic and programmatic objectives.

The application also displays the retrieved source chunks and similarity scores used to generate the response.

---

# Future Improvements

- True streaming responses directly from Ollama
- Support for multiple uploaded documents
- Persistent vector database
- Multi-turn conversational memory
- Cloud deployment
- User authentication
- Citation highlighting inside generated answers

