from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # ✅ ici
from pathlib import Path
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_PATH = Path("C:/Users/testo/Documents/chatbot/data/rag_embeddings")

def load_and_split_documents(directory):
    texts = []
    directory = Path(directory)
    for file_path in directory.iterdir():
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(texts)

def create_vectorstore(docs):
    return FAISS.from_documents(docs, embedding_model)

def save_vectorstore(store, path=EMBEDDING_PATH):
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

def load_vectorstore(path=EMBEDDING_PATH):
    return FAISS.load_local(str(path), embedding_model, allow_dangerous_deserialization=True)
