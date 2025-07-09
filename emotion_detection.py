from transformers import pipeline

def initialize_emotion_classifier():
    return pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

def detect_emotion(text, classifier):
    try:
        result = classifier(text)[0]
        label = result['label']
        score = round(result['score'] * 100, 2)
        return label, score
    except Exception as e:
        return "unknown", 0.0, f"Emotion detection error: {str(e)}"