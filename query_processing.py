def format_context(docs, metadata, indices):
    context_blocks = []
    for idx in indices[0]:
        meta = metadata[idx]
        text = docs[idx]
        block_type = meta.get("type", "").lower()

        if block_type == "ayah":
            block = f"📖 Ayah ({meta.get('reference', 'unknown')}):\n{text}"
        elif block_type == "hadith":
            block = f"🗣 Hadith ({meta.get('source', 'unknown')}):\n{text}"
        elif block_type == "dua":
            block = f"🤲 Dua:\n{text}"
        else:
            block = f"{text}"

        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def classify_query(user_input: str) -> str:
    query = user_input.lower()
    if any(w in query for w in ["sad", "depressed", "alone", "hopeless", "anxious", "scared", "heart", "crying"]):
        return "emotional"
    elif any(w in query for w in ["how to pray", "how to fast", "explain wudu", "zakat", "hajj", "umrah", "ibadah", "tahajjud"]):
        return "ibadah"
    elif any(w in query for w in ["fiqh", "is it haram", "halal", "fatwa", "allowed in islam", "permissible"]):
        return "fiqh"
    elif any(w in query for w in ["tafsir", "ayah", "verse", "surah", "explain this ayah", "what does this mean in quran"]):
        return "tafsir"
    elif any(w in query for w in ["prophet", "story of", "life of", "nabi", "messenger", "who was"]):
        return "story"
    else:
        return "general"