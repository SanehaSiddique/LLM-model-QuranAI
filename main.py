import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from config import load_config
from data_processing import load_index
from deen_buddy import deen_buddy

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="QuranAI API", description="A compassionate Islamic companion API")

# Load configuration
config = load_config()

# Load FAISS index and related data
index, metadata, docs = load_index(config["INDEX_FILE"], config["EMBEDDING_MODEL"])

class QueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 6

@app.post("/deen_buddy")
async def deen_buddy_endpoint(
    request: QueryRequest
):
    try:
        # Use request.top_k if provided, otherwise default to 6
        top_k = request.top_k or 6
        response = deen_buddy(
            user_input=request.query,
            top_k=top_k,
            index=index,
            metadata=metadata,
            docs=docs,
            embedding_model_name=config["EMBEDDING_MODEL"],
            llm_config=config["LLM_CONFIG"]
        )
        # Log or forward the interaction to Node backend
        interaction = {
            "user_id": request.user_id,
            "query": request.query,
            "response": response
        }
        # Optionally: forward to Node.js backend using requests.post()
        # import requests
        # requests.post("http://localhost:3000/api/saveChat", json=interaction)

        return {"response": response, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)