# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions based on local documents (PDFs or text). Built with LangChain, HuggingFace Transformers, FAISS, and served via FastAPI.

## Features

- Load and parse documents from a local folder (`/data/documents`)
- Create and store embeddings using FAISS
- Use pre-trained HuggingFace models to generate answers
- REST API (`/ask`) powered by FastAPI
- Handles text and PDF documents

## Technologies

- Python
- LangChain
- HuggingFace Transformers
- FAISS
- PyMuPDF / pdfminer / tiktoken
- FastAPI


## ⚙️ Installation

```bash
git clone https://github.com/islemfatnassi/chatbot-RAG.git
cd chatbot-RAG
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt



## Build Embeddings
python build_embeddings.py



## Run the Chatbot API
python main.py


## Demo

Voici un aperçu de l'interface FastAPI :

![FastAPI UI](images\Screenshot 2025-07-31 153651)
