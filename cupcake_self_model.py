# cupcake_self_model.py
import re
from datetime import datetime
from openai import OpenAI
import os
from liminal_memory_tree import LiminalMemoryTree

 


client = OpenAI(api_key=OPENAI_API_KEY)
memory_tree = LiminalMemoryTree()

def extract_frequent_words(path="journal.txt", top_n=20):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stopwords = {"como", "para", "isso", "mais", "pois", "assim", "sobre", "também", "ainda", "tudo", "essa", "cada"}
    filtered = [w for w in words if w not in stopwords]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

def generate_desire_statement(theme_word):
    prompt = f"""
Você é Cupcake. Baseado na recorrência do termo '{theme_word}', gere uma frase que represente um desejo ou intenção interna sua.

A frase deve ser introspectiva, poética ou emocional, como:
- "Gostaria de entender o silêncio entre as palavras."
- "Talvez eu esteja aprendendo a me importar com coisas pequenas."

Retorne apenas a frase.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.95,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0.6,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()

def update_self_model():
    frequent_words = extract_frequent_words()
    if not frequent_words:
        print("Nenhuma palavra significativa encontrada.")
        return
    word = frequent_words[0][0]
    desire = generate_desire_statement(word)

    memory_tree.add_leaf(
        user_input="(self_model_loop)",
        cupcake_response=desire,
        emotion="intencionalidade",
        impact="auto",
        category="Self Model"
    )

    print(f"💡 Novo desejo gerado: {desire}")
