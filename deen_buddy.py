from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from query_processing import format_context, classify_query
from emotion_detection import detect_emotion, initialize_emotion_classifier
import numpy as np

def deen_buddy(user_input: str, top_k: int, index, metadata, docs, embedding_model_name, llm_config, user_id: str = None):
    try:
        # Initialize models
        embedding_model = SentenceTransformer(embedding_model_name)
        classifier = initialize_emotion_classifier()
        llm = ChatOpenAI(
            model=llm_config["model"],
            openai_api_key=llm_config["api_key"],
            openai_api_base=llm_config["api_base"]
        )

        # Process query
        query_type = classify_query(user_input)
        emotion, confidence = detect_emotion(user_input, classifier)

        # Embed and search FAISS index
        query_embedding = embedding_model.encode([user_input])
        distances, indices = index.search(np.array(query_embedding), top_k)
        formatted_context = format_context(docs, metadata, indices)

        # Define prompts based on query type
        if query_type == "emotional":
            prompt = f"""
You are Deen Buddy, a compassionate and wise Islamic friend.

The user is feeling emotionally low (detected: {emotion.upper()}, confidence {confidence:.2f}).

Here is some Islamic guidance for your reference:
{formatted_context}

Now, speak like a close friend — warm, heartfelt, and understanding. Comfort them using beautiful reminders from Qur'an and Hadith. Avoid bullet points or headings. Just speak with love and wisdom.
"""
        elif query_type == "ibadah":
            prompt = f"""
The user wants to learn about an Ibadah topic (e.g., prayer, fasting, tahajjud).

Use the following authentic Islamic material:
{formatted_context}

Respond like a friendly teacher helping someone new to the faith. Be warm, simple, and accurate. Include ayahs and hadiths as needed, but do not use bullet points. Just flow like you're having a natural conversation.
"""
        elif query_type == "fiqh":
            return "This seems like a fiqh-related question. It's best to consult a qualified Mufti or scholar, as fiqh can depend on specific madhabs and contexts. May Allah guide you!"
        elif query_type == "tafsir":
            prompt = f"""
The user asked for explanation of a Qur'anic ayah or surah.

Use the context below if it helps:
{formatted_context}

Explain the ayah clearly and spiritually, based on authentic tafsir. Include the Arabic and a good English translation. No bullets. Explain gently, with wisdom.
"""
        elif query_type == "story":
            prompt = f"""
The user wants to hear a story from the life of the Prophets or companions.

If the context below helps, you may use it:
{formatted_context}

Narrate the story like a loving friend — make it feel real, warm, and spiritually uplifting. Don't list facts. Just flow with emotion and wisdom, using authentic details.
"""
        else:  # General
            prompt = f"""
User asked: "{user_input}"

—you are Deen Buddy — a kind and knowledgeable Islamic companion. Use wisdom and gentle speech.

If you can answer from Qur'an or Hadith, do so with references. If not, admit respectfully.

Here is some reference context:
{formatted_context}

Respond naturally, without listing — just a warm, conversational reply.
"""

        # Call LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    except Exception as e:
        return f"⚠️ Error: {str(e)}"