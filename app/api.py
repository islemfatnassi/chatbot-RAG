# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from .rag_pipeline import get_answer

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(question: Question):
    result = get_answer(question.query)
    return {"response": result["result"]}
