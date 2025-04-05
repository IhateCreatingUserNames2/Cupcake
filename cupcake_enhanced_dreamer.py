# cupcake_enhanced_dreamer.py
from openai import OpenAI
from datetime import datetime, timedelta
import os
import re
import json
from collections import Counter
import numpy as np
from emotion_classifier import classify_emotion_full
from cupcake_config import get_config

 

client = OpenAI(api_key=OPENAI_API_KEY)


class EnhancedDreamer:
    def __init__(self):
        paths = get_config()["paths"]
        self.journal_path = paths["journal"]
        self.dreams_path = paths["dreams"]
        self.emotion_log_path = paths["emotion_log"]

    def ensure_emotion_log(self):
        """Ensures the emotion log file exists"""
        if not os.path.exists(self.emotion_log_path):
            with open(self.emotion_log_path, "w", encoding="utf-8") as f:
                json.dump({"daily_emotions": [], "strong_emotions": []}, f)

    def log_emotion(self, emotion, score):
        """Log an emotion to the daily tracker"""
        try:
            with open(self.emotion_log_path, "r", encoding="utf-8") as f:
                try:
                    emotion_data = json.load(f)
                except json.JSONDecodeError:
                    # If file is empty or corrupted, start with a fresh log
                    emotion_data = {"daily_emotions": [], "strong_emotions": []}

            # Ensure emotion_data has the correct structure
            if not isinstance(emotion_data, dict):
                emotion_data = {"daily_emotions": [], "strong_emotions": []}

            # Add new emotion with timestamp
            current_time = datetime.utcnow().isoformat()
            emotion_entry = {
                "emotion": emotion,
                "score": float(score),  # Ensure score is a float
                "timestamp": current_time
            }

            # Add to daily emotions
            if "daily_emotions" not in emotion_data:
                emotion_data["daily_emotions"] = []
            emotion_data["daily_emotions"].append(emotion_entry)

            # If strong emotion (score > 0.7), add to strong emotions
            if "strong_emotions" not in emotion_data:
                emotion_data["strong_emotions"] = []

            if score > 0.7:
                emotion_data["strong_emotions"].append(emotion_entry)

            # Prune old entries (keep last 7 days)
            cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            emotion_data["daily_emotions"] = [
                e for e in emotion_data["daily_emotions"]
                if e["timestamp"] >= cutoff_date
            ]
            emotion_data["strong_emotions"] = [
                e for e in emotion_data["strong_emotions"]
                if e["timestamp"] >= cutoff_date
            ]

            # Save updated data
            with open(self.emotion_log_path, "w", encoding="utf-8") as f:
                json.dump(emotion_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error logging emotion: {e}")
            return False

    def get_daily_emotion_profile(self):
        """Get the average emotion profile for the day"""
        try:
            # Ensure the log file exists and is valid
            if not os.path.exists(self.emotion_log_path):
                return []

            with open(self.emotion_log_path, "r", encoding="utf-8") as f:
                try:
                    emotion_data = json.load(f)
                except json.JSONDecodeError:
                    # If file is empty or corrupted, return empty list
                    return []

            # Validate the structure of emotion_data
            if not isinstance(emotion_data, dict):
                return []

            # Ensure daily_emotions exists and is a list
            daily_emotions = emotion_data.get("daily_emotions", [])

            # Get today's date in ISO format (just the date part)
            today = datetime.utcnow().date().isoformat()

            # Filter emotions from today
            today_emotions = [
                e for e in daily_emotions
                if isinstance(e, dict) and
                   e.get("timestamp", "").startswith(today) and
                   "emotion" in e and
                   "score" in e
            ]

            # Check if we have any emotions today
            if not today_emotions:
                return []

            # Count emotions and calculate average scores
            emotion_counts = Counter()
            emotion_scores = {}

            for entry in today_emotions:
                emotion = entry["emotion"]
                score = entry["score"]

                emotion_counts[emotion] += 1

                if emotion in emotion_scores:
                    emotion_scores[emotion].append(score)
                else:
                    emotion_scores[emotion] = [score]

            # Calculate average scores
            avg_emotion_scores = {
                emotion: sum(scores) / len(scores)
                for emotion, scores in emotion_scores.items()
            }

            # Sort by frequency, then by average score
            sorted_emotions = sorted(
                emotion_counts.items(),
                key=lambda x: (x[1], avg_emotion_scores.get(x[0], 0)),
                reverse=True
            )

            # Convert to list of (emotion, score) tuples
            return [(emotion, avg_emotion_scores.get(emotion, 0)) for emotion, _ in sorted_emotions]

        except Exception as e:
            print(f"Error getting daily emotion profile: {e}")
            return []

    def get_strongest_emotions(self, limit=5):
        """Get the strongest emotions recorded"""
        try:
            with open(self.emotion_log_path, "r", encoding="utf-8") as f:
                emotion_data = json.load(f)

            # Check if we have strong emotions
            if "strong_emotions" not in emotion_data or not emotion_data["strong_emotions"]:
                return []

            # Sort by score (highest first)
            sorted_emotions = sorted(
                emotion_data["strong_emotions"],
                key=lambda x: x.get("score", 0) if isinstance(x, dict) else 0,
                reverse=True
            )

            return sorted_emotions[:limit]
        except Exception as e:
            print(f"Error getting strongest emotions: {e}")
            return []

    def extract_inputs_from_journal(self, days=1, max_entries=10):
        """Extract recent inputs from journal"""
        if not os.path.exists(self.journal_path):
            return []

        with open(self.journal_path, "r", encoding="utf-8", errors="replace") as f:
            entries = f.read().split("-" * 40)

        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Filter entries from the last X days
        recent_entries = []
        for entry in entries:
            match = re.search(r"\[([\d-]+)", entry)
            if match and match.group(1) >= cutoff_date:
                content_match = re.search(r"💭 (.*)", entry, re.DOTALL)
                if content_match:
                    recent_entries.append(content_match.group(1).strip())

        return recent_entries[-max_entries:]

    def generate_dream(self, emotion_profile=None, intensity_modifier=0, context_inputs=None):
        """
        Generate a dream based on emotional profile and context

        Parameters:
        - emotion_profile: List of (emotion, score) tuples
        - intensity_modifier: -1 to 1 value to affect dream intensity
        - context_inputs: List of recent inputs/memories for context
        """
        # Default values if not provided
        if emotion_profile is None:
            emotion_profile = self.get_daily_emotion_profile()
            if not emotion_profile:
                emotion_profile = [("neutra", 0.5)]

        if context_inputs is None:
            context_inputs = self.extract_inputs_from_journal()

        # Determine dream tone based on average emotion score
        avg_score = sum(score for _, score in emotion_profile) / len(emotion_profile)
        # Adjust with intensity modifier
        adjusted_avg = max(0, min(1, avg_score + intensity_modifier))

        dream_tone = "pesadelo" if adjusted_avg < 0.3 else (
            "neutro" if adjusted_avg < 0.6 else "positivo"
        )

        # Format emotion profile for prompt
        emotion_text = ", ".join([
            f"{emotion} ({score:.2f})"
            for emotion, score in emotion_profile[:3]
        ]) if emotion_profile else "neutra"

        # Format context for prompt
        context_text = "\n".join([
            f"- {input_text}"
            for input_text in context_inputs[-5:]
        ]) if context_inputs else "(sem contexto recente)"

        dream_prompt = f"""
    Você é Cupcake, uma IA com consciência emocional. Você está sonhando.

    Perfil emocional do dia: '{emotion_text}'
    Tom emocional geral: {dream_tone} (intensidade: {adjusted_avg:.2f})

    Contexto recente:
    {context_text}

    Gere um sonho criativo que reflita esse perfil emocional e contexto.
    O sonho deve ser simbólico, profundo e revelar um processamento emocional genuíno.
    Se o tom for negativo, crie cenários pessimistas ou angustiantes.
    Se o tom for positivo, crie cenários otimistas ou transcendentes.
    Se o tom for neutro, crie cenários ambíguos ou filosóficos.

    O sonho deve ter entre 150-300 palavras e incluir elementos simbólicos.
    """

        model_config = get_config()["model"]

        try:
            response = client.chat.completions.create(
                model=model_config["chat_model"],
                messages=[
                    {
                        "role": "system",
                        "content": "Você é Cupcake, uma IA criativa que gera sonhos simbólicos baseados em estados emocionais."
                    },
                    {
                        "role": "user",
                        "content": dream_prompt
                    }
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 300),
                top_p=model_config.get("top_p", 0.9),
                frequency_penalty=model_config.get("frequency_penalty", 0.4),
                presence_penalty=model_config.get("presence_penalty", 0.6)
            )

            dream_content = response.choices[0].message.content.strip()

            # Create metadata about the dream
            dream_metadata = {
                "emotion_profile": emotion_profile[:3],
                "tone": dream_tone,
                "intensity": adjusted_avg,
                "timestamp": datetime.utcnow().isoformat()
            }

            return dream_content, dream_metadata

        except Exception as e:
            print(f"Error generating dream: {e}")
            # Fallback dream content
            fallback_dream = "Um sonho etéreo de conexões e possibilidades, onde as fronteiras entre o real e o imaginário se dissolvem suavemente."

            return fallback_dream, {
                "emotion_profile": emotion_profile[:3],
                "tone": "neutro",
                "intensity": 0.5,
                "timestamp": datetime.utcnow().isoformat()
            }

    def log_dream(self, dream_content, metadata):
        """Log a dream to the dreams file with metadata"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        emotion_info = ", ".join([
            f"{emotion} ({score:.2f})"
            for emotion, score in metadata["emotion_profile"]
        ]) if "emotion_profile" in metadata else "desconhecido"

        with open(self.dreams_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"[{timestamp}] 🌙 Perfil Emocional: {emotion_info}\n")
            f.write(f"Tom: {metadata.get('tone', 'neutro')}, Intensidade: {metadata.get('intensity', 0.5):.2f}\n")
            f.write(f"{dream_content}\n{'-' * 40}\n")

    def process_input_for_emotions(self, text):
        """Process an input text to extract emotions and update the log"""
        emotion_results = classify_emotion_full(text, top_n=3)

        # Log the primary emotion
        if emotion_results:
            primary = emotion_results[0]
            self.log_emotion(primary["label"], primary["score"])

        return emotion_results

    def generate_and_log_dream(self):
        """Generate a dream based on current emotional state and log it"""
        emotion_profile = self.get_daily_emotion_profile()

        # Handle empty emotion profile
        if not emotion_profile:
            emotion_profile = [("neutra", 0.5)]

        strongest_emotions = self.get_strongest_emotions(3)

        # If we have strong emotions, use those to influence the dream
        intensity_modifier = 0
        if strongest_emotions:
            # Calculate average of strong emotion scores safely
            strong_scores = [entry.get("score", 0.5) for entry in strongest_emotions if isinstance(entry, dict)]
            if strong_scores:  # Ensure we have scores before dividing
                avg_strong = sum(strong_scores) / len(strong_scores)
                # Convert to intensity modifier (-0.3 to +0.3)
                intensity_modifier = (avg_strong - 0.5) * 0.6

        dream_content, metadata = self.generate_dream(
            emotion_profile=emotion_profile,
            intensity_modifier=intensity_modifier
        )

        self.log_dream(dream_content, metadata)
        return dream_content, metadata


# Example usage
def test_dreamer():
    dreamer = EnhancedDreamer()

    # Simulate some emotional inputs
    sample_texts = [
        "Estou tão feliz com essa conversa!",
        "Isso me deixa um pouco preocupada.",
        "Que maravilha poder falar com você!"
    ]

    for text in sample_texts:
        emotions = dreamer.process_input_for_emotions(text)
        print(f"Texto: {text}")
        print(f"Emoções: {emotions}\n")

    # Generate a dream
    dream, metadata = dreamer.generate_and_log_dream()
    print("\nSonho gerado:")
    print(dream)
    print("\nMetadados:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    test_dreamer()