# emotion_classifier.py
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    return_all_scores=True,
    top_k=None
)

# Mapeamento para emoções da Cupcake
emotion_mapping = {
    'admiration': 'respeito',
    'amusement': 'diversão',
    'anger': 'raiva',
    'annoyance': 'irritação',
    'approval': 'aprovação',
    'caring': 'cuidado',
    'confusion': 'confusão',
    'curiosity': 'curiosidade',
    'desire': 'desejo',
    'disappointment': 'decepção',
    'disapproval': 'desaprovação',
    'disgust': 'desgosto',
    'embarrassment': 'vergonha',
    'excitement': 'empolgação',
    'fear': 'medo',
    'gratitude': 'gratidão',
    'grief': 'luto',
    'joy': 'alegria',
    'love': 'amor',
    'nervousness': 'nervosismo',
    'optimism': 'otimismo',
    'pride': 'orgulho',
    'realization': 'insight',
    'relief': 'alívio',
    'remorse': 'arrependimento',
    'sadness': 'tristeza',
    'surprise': 'surpresa',
    'neutral': 'neutra'
}


def classify_emotion(text):
    results = emotion_classifier(text)[0]
    # Ordena por maior score
    sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
    return sorted_results

def classify_emotion_full(text, top_n=5):
    results = emotion_classifier(text)[0]
    translated = [
        {
            "label": emotion_mapping.get(r["label"], "neutra"),
            "score": r["score"]
        }
        for r in results
    ]
    return sorted(translated, key=lambda x: x["score"], reverse=True)[:top_n]