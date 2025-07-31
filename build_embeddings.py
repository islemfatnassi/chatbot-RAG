# build_embeddings.py

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Définition des chemins
data_path = Path("C:/Users/testo/Documents/chatbot/data/documents")  # 📂 Ton dossier de PDF
save_path = Path("C:/Users/testo/Documents/chatbot/data/rag_embeddings")  # 📂 Où enregistrer FAISS

# 2. Chargement des documents PDF
loader = DirectoryLoader(
    str(data_path),
    glob="**/*.pdf",          # 🔍 Tous les fichiers PDF dans tous les sous-dossiers
    loader_cls=PyPDFLoader    # 🧠 Utilise PyPDFLoader pour les PDFs
)
docs = loader.load()
print(f"✅ {len(docs)} documents chargés.")

# 3. Découpage en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✂️ {len(chunks)} chunks créés.")

# Vérification : s'assurer qu'on a bien des chunks
if not chunks:
    raise ValueError("❌ Aucun chunk généré. Vérifie que les fichiers PDF sont valides et lisibles.")

# 4. Création des embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Construction de la base FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 6. Sauvegarde des vecteurs FAISS
vectorstore.save_local(str(save_path))
print("✅ Embeddings FAISS créés et sauvegardés avec succès.")
