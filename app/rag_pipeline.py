# app/rag_pipeline.py
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from .utils import load_vectorstore

# Charger la base vectorielle
retriever = load_vectorstore().as_retriever()

# Créer un pipeline avec un modèle léger
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.7,
)

# LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Construire la chaîne RAG
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Fonction pour obtenir la réponse
def get_answer(query):
    return rag_chain({"query": query})
