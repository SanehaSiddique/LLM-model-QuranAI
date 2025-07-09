import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_and_index_data(emotion_dir, ibadah_dir, embedding_model_name):
    embedding_model = SentenceTransformer(embedding_model_name)
    docs = []
    metadata = []

    # Handle Emotions
    for filename in os.listdir(emotion_dir):
        if filename.endswith(".json"):
            with open(os.path.join(emotion_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                emotion = data.get("emotion", "unknown")

                for item in data.get("ayahs", {}).get("primary", []):
                    docs.append(f"Ayah ({item['reference']}): {item['text_en']}")
                    metadata.append({"type": "ayah", "category": "emotion", "tag": emotion})

                for item in data.get("hadiths", {}).get("primary", []):
                    docs.append(f"Hadith: {item['text_en']}")
                    metadata.append({"type": "hadith", "category": "emotion", "tag": emotion})

                for item in data.get("duas", []):
                    docs.append(f"Dua: {item['text_en']}")
                    metadata.append({"type": "dua", "category": "emotion", "tag": emotion})

    # Handle Ibadah
    for filename in os.listdir(ibadah_dir):
        if filename.endswith(".json"):
            with open(os.path.join(ibadah_dir, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid or empty file: {filename}")
                    continue

                topic = data.get("topic", "unknown")

                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("text_en")
                                if text:
                                    docs.append(f"{key.capitalize()}: {text}")
                                    metadata.append({"type": key, "category": "ibadah", "topic": topic})
                            elif isinstance(item, str):
                                docs.append(f"{key.capitalize()}: {item}")
                                metadata.append({"type": key, "category": "ibadah", "topic": topic})
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, list):
                                for subitem in subvalue:
                                    if isinstance(subitem, dict):
                                        text = subitem.get("text") or subitem.get("text_en")
                                        if text:
                                            docs.append(f"{key.capitalize()} - {subkey}: {text}")
                                            metadata.append({"type": key, "category": "ibadah", "topic": topic})
                                    elif isinstance(subitem, str):
                                        docs.append(f"{key.capitalize()} - {subkey}: {subitem}")
                                        metadata.append({"type": key, "category": "ibadah", "topic": topic})

    print(f"Encoding {len(docs)} total docs...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, metadata, docs

def save_index(index, metadata, docs, index_file):
    with open(index_file, "wb") as f:
        pickle.dump({"index": index, "metadata": metadata, "docs": docs}, f)
    print("Indexing completed.")

def load_index(index_file, embedding_model_name):
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["metadata"], data["docs"]