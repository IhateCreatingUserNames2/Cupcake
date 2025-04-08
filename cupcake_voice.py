#cupcake_voice.py
import os
import requests

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "forgot to remove lol ")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # pode trocar pela voz que você escolher


def speak(text, filename="cupcake_output.mp3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✨ Cupcake falou: '{text}'")
    else:
        print("Erro ao gerar voz:", response.status_code, response.text)
