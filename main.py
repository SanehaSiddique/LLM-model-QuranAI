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
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="QuranAI API", description="A compassionate Islamic companion API")

# Allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # or ["POST"] if you want to restrict
    allow_headers=["*"],
)

# Load configuration
config = load_config()

# Load FAISS index and related data
index, metadata, docs = load_index(config["INDEX_FILE"], config["EMBEDDING_MODEL"])

class QueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 6

@app.post("/quran-ai")
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
            llm_config=config["LLM_CONFIG"],
            user_id=request.user_id
        )

        return {"response": response, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)