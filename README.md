# RAG Chatbot 📚🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions based on local documents (PDFs or text). Built with LangChain, HuggingFace Transformers, FAISS, and served via FastAPI.

## 🧠 Features

- Load and parse documents from a local folder (`/data/documents`)
- Create and store embeddings using FAISS
- Use pre-trained HuggingFace models to generate answers
- REST API (`/ask`) powered by FastAPI
- Handles text and PDF documents

## 🚀 Technologies

- Python
- LangChain
- HuggingFace Transformers
- FAISS
- PyMuPDF / pdfminer / tiktoken
- FastAPI

## 📁 Project Structure

chatbot-RAG/
│
├── app/ # FastAPI app
├── data/ # Folder containing PDF/text documents
├── training/ # Embedding and preprocessing scripts
├── build_embeddings.py # Script to build FAISS index
├── main.py # FastAPI main entry point
├── requirements.txt # Project dependencies
└── README.md