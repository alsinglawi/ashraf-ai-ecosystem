"""
app.py
-----------------------------------
FastAPI app for interacting with the Ashraf AI Ecosystem.
"""

from fastapi import FastAPI
from src.ai_core.assistant_api import AssistantAPI
from src.ai_core.data_pipeline import DataPipeline

app = FastAPI(title="Ashraf AI Ecosystem API", version="1.0")
assistant = AssistantAPI()
pipeline = DataPipeline()


@app.get("/")
def home():
    return {"message": "Welcome to the Ashraf AI Ecosystem!"}


@app.post("/ask")
def ask_assistant(question: str):
    response = assistant.respond(question)
    return {"assistant_response": response}


@app.get("/data/load")
def load_data():
    df = pipeline.load_data()
    return {"rows": len(df), "columns": list(df.columns)}


# Run locally: uvicorn src.web_app.app:app --reload
