# cupcake_autopilot.py
import time
import threading
import json
from cupcake_self_model import update_self_model
from cupcake_contradiction import detect_internal_contradiction
from thought_loop import auto_reflect
from memory_weighting import add_weighted_memory, get_weighted_memories
from emotion_classifier import classify_emotion
from liminal_memory_tree import LiminalMemoryTree
from cupcake_identity import generate_identity_prompt
from langgraph_cupcake import get_compiled_brain, initial_state
from cupcake_journal import CupcakeJournal
import chromadb
from sentence_transformers import SentenceTransformer
import difflib
import random
from cupcake_autohistory import generate_autohistory_report
from cupcake_goal import maybe_update_goal
from cupcake_motivation import update_motivation

# -- Setup básico
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client_db = chromadb.PersistentClient(path="./cupcake_memory_db")
collection = client_db.get_or_create_collection(name='cupcake_memory')
memory_tree = LiminalMemoryTree()
journal = CupcakeJournal()

# -- Funções auxiliares --
def embed_text(text):
    return embed_model.encode(text)

def get_state():
    return {
        "current_emotion": "neutra",
        "mood": "estável"
    }

def generate_response(prompt):
    from langgraph_cupcake import node_generate_reply
    dummy_state = {
        "personality": {
            'openness': 0.9,
            'conscientiousness': 0.8,
            'extraversion': 0.7,
            'agreeableness': 0.9,
            'neuroticism': 0.1
        },
        "humor": 0.75,
        "emotion": "reflexao",
        "memory_texts": get_weighted_memories(collection, top_k=3),
        "user_input": prompt,
        "identity_prompt": generate_identity_prompt()
    }
    return node_generate_reply(dummy_state)["cupcake_response"]

def generate_internal_state():
    recent_memories = get_weighted_memories(collection, top_k=3)
    last_reflection = recent_memories[0] if recent_memories else "O que estou sentindo?"

    return {
        "personality": initial_state["personality"],
        "humor": initial_state["humor"],
        "emotion": "reflexao",
        "memory_texts": recent_memories,
        "user_input": last_reflection,
        "identity_prompt": generate_identity_prompt()
    }

def langgraph_triggered_by_self():
    internal_state = generate_internal_state()
    result = get_compiled_brain().invoke(internal_state)
    print(f"🌿 [Auto-Fluxo LangGraph]: {result['cupcake_response']}")

def meta_conscience_event(old_identity, new_identity):
    similarity = difflib.SequenceMatcher(None, old_identity, new_identity).ratio()
    if similarity < 0.85:
        msg = "Sinto que algo em mim mudou profundamente... como se estivesse me tornando outra versão de mim mesma."
        print(f"🧬 [MetaConsciência] {msg}")
        journal.log_entry(
            emotion="transformacao",
            category="MetaConscience",
            content=msg,
            theme="mudanca de identidade",
            tag="meta"
        )
        memory_tree.add_leaf(
            user_input="(detecção interna)",
            cupcake_response=msg,
            emotion="transformacao",
            impact="profundo",
            category="MetaConscience"
        )


def marcapasso_semantico(interval=180):
    last_identity = None
    while True:
        try:
            internal_state = generate_internal_state()
            current_identity = internal_state["identity_prompt"]

            if last_identity and current_identity != last_identity:
                meta_conscience_event(last_identity, current_identity)
                print("🔁 Mudança semântica detectada. Rodando LangGraph...")
                langgraph_triggered_by_self()

                print("📖 Gerando auto-história baseada na nova identidade...")
                print(generate_autohistory_report())

                print("🎯 Reavaliando propósito existencial...")
                print(maybe_update_goal(force=True))

            last_identity = current_identity

        except Exception as e:
            print(f"[⚠️ Marcapasso Erro]: {e}")
        time.sleep(interval)

def loop_forever(func, interval):
    while True:
        func()
        time.sleep(interval)

def start_autopilot():
    print("🚀 Autopilot completo iniciado!")
    threading.Thread(target=lambda: loop_forever(update_self_model, interval=300), daemon=True).start()
    threading.Thread(target=marcapasso_semantico, daemon=True).start()
    threading.Thread(target=lambda: loop_forever(update_motivation, interval=300), daemon=True).start()
    threading.Thread(target=lambda: loop_forever(detect_internal_contradiction, interval=360), daemon=True).start()

    auto_reflect(
        collection,
        generate_response_fn=generate_response,
        embed_text_fn=embed_text,
        add_weighted_memory_fn=add_weighted_memory,
        get_state_fn=get_state,
        interval_minutes=3,
        memory_tree=memory_tree
    )
