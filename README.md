# 📄 AI Legal Document Assistant

## Overview

The **AI Legal Document Assistant** is a Retrieval-Augmented Generation (RAG) application that allows users to upload **PDF** or **DOCX** legal documents and ask natural language questions about their contents.

The application automatically extracts the document text, splits it into semantic chunks, generates vector embeddings using Sentence Transformers, stores them in a FAISS vector database, retrieves the most relevant chunks, and uses a locally hosted **Llama 3.2** model through **Ollama** to generate accurate, context-aware answers.

---

# Features

* 📄 Upload PDF and DOCX legal documents
* ⚡ Automatic document parsing and indexing
* 🔍 Semantic search using FAISS
* 🤖 Retrieval-Augmented Generation (RAG)
* 🦙 Local inference using Llama 3.2 via Ollama
* 💬 Conversation history
* ⌨️ Streaming typing effect
* 📚 Retrieved source chunks with similarity scores
* 🌐 Streamlit web interface
* 🐳 Docker support

---

# Technologies Used

* Python 3.11+
* Streamlit
* Ollama
* Llama 3.2
* FAISS
* Sentence Transformers
* LangChain
* PyMuPDF
* python-docx
* NumPy

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
├── screenshots/
│   ├── Screenshot 2026-06-27 224909.png
│   ├── Screenshot 2026-06-27 224923.png
│   ├── Screenshot 2026-06-27 225118.png
│   ├── Screenshot 2026-06-27 225131.png
│   └── Screenshot 2026-06-27 225141.png
│
└── requirements.txt
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
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 legal-rag
```

> **Note:** Ollama must already be installed on the host machine, the `llama3.2` model must be downloaded, and the Ollama server must be running before starting the Docker container.

---

# How the Application Works

1. Upload a PDF or DOCX document.
2. The document is automatically parsed.
3. The text is split into semantic chunks.
4. Sentence Transformers generates embeddings for each chunk.
5. FAISS builds a searchable vector index.
6. Ask a question in natural language.
7. The system retrieves the most relevant chunks.
8. Llama 3.2 generates an answer using the retrieved context.
9. The answer and supporting source chunks are displayed.

---

# Application Walkthrough

## Step 1 – Launch the Application

Run the application using:

```bash
streamlit run app.py
```

The home page allows users to upload a PDF or DOCX legal document.

![Home Screen](screenshots/Screenshot%202026-06-27%20224909.png)

---

## Step 2 – Upload a Document

Click **Upload Document** and choose a PDF or DOCX file.

The application automatically:

* Parses the document
* Splits it into semantic chunks
* Generates embeddings
* Builds a FAISS vector index

The document is then ready for querying.

![Upload Document](screenshots/Screenshot%202026-06-27%20224923.png)

---

## Step 3 – Ask a Question

Type any natural language question related to the uploaded document.

Example:

> What are the three specific entities or units responsible for assessing the technical merits of submitted proposals during a competitive selection procedure?

![Ask Question](screenshots/Screenshot%202026-06-27%20225118.png)

---

## Step 4 – AI Generates an Answer

The RAG pipeline retrieves the most relevant chunks and sends them to **Llama 3.2** through **Ollama**.

The generated answer is displayed to the user.

![Generated Answer](screenshots/Screenshot%202026-06-27%20225131.png)

---

## Step 5 – Retrieved Sources

The application also displays the retrieved document chunks together with their similarity scores, providing transparency about the information used to generate the response.

![Retrieved Sources](screenshots/Screenshot%202026-06-27%20225141.png)

---

# Sample Questions

* What are the responsibilities of implementing partners?
* Summarize this document.
* Who can become an implementing partner?
* What records should be retained?
* Explain the grant award procedure.
* What are the eligibility requirements?
* What are the main obligations mentioned in the agreement?

---

# Sample Output

### Question

> What are the responsibilities of implementing partners?

### Answer

Implementing Partners (IPs) assume full responsibility and accountability for the effective use of resources transferred by UNITAR. They are responsible for delivering project outputs specified in the Grant-out Agreement while supporting UNITAR's strategic and programmatic objectives.

The application also displays the retrieved document chunks and similarity scores used to generate the answer.

---

# Future Improvements

* True token streaming from Ollama
* Support for multiple uploaded documents
* Persistent vector database
* Cloud deployment
* User authentication
* Citation highlighting inside generated answers

---

# Author

**Ziad Ayman**

AI Engineer

---

# License

This project was developed as part of a technical assessment for an AI Software Engineer position.
