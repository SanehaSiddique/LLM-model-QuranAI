from dotenv import load_dotenv
import os

def load_config():
    load_dotenv()
    return {
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "INDEX_FILE": os.getenv("INDEX_FILE", "fiass_indexer.pkl"),
        "LLM_CONFIG": {
            "model": os.getenv("LLM_MODEL", "deepseek/deepseek-r1-0528:free"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        },
        "EMOTION_DIR": os.getenv("EMOTION_DIR", "emotions"),
        "IBADAH_DIR": os.getenv("IBADAH_DIR", "ibadah")
    }