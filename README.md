# chatbot-RAG
Retrieval-Augmented Generation chatbot for document question answering using LangChain and HuggingFace models.
# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on documents stored locally.  
It uses LangChain, HuggingFace models, and FastAPI to serve the chatbot API.

## Features

- Load documents (text and PDF) from local directory
- Use embeddings and vector stores (FAISS) for document retrieval
- Use HuggingFace language models for answer generation
- FastAPI REST API with `/ask` endpoint for question answering

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/rag-chatbot.git
cd rag-chatbot


2.Create a Python virtual environment and activate it:

```python 
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
