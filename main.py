import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
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
    query: str
    top_k: int = 6

@app.post("/deen_buddy")
async def deen_buddy_endpoint(request: QueryRequest):
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
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)