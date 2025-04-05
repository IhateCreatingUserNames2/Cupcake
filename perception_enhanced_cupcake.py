# perception_enhanced_cupcake.py
import os
import uuid
import numpy as np
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from liminal_memory_tree import LiminalMemoryTree
from cupcake_journal import CupcakeJournal
import threading
import time
from cupcake_self_model import update_self_model
from cupcake_consciousness import generate_self_state
from cupcake_identity import generate_identity_prompt
from langgraph_cupcake import graph as original_graph, get_compiled_brain as get_base_brain
from cupcake_langgraph_integration import integrate_with_langgraph
from perception_integration import integrate_enhanced_perception
from cognitive_architecture import CognitiveArchitecture
from cupcake_enhanced_dreamer import EnhancedDreamer
from enhanced_self_perception import EnhancedSelfPerceptionLayer
from world_perception import perceive_world
from emotion_classifier import classify_emotion, classify_emotion_full
from memory_weighting import add_weighted_memory, get_weighted_memories, inject_memory
from thought_loop import auto_reflect
from cupcake_contradiction import detect_internal_contradiction
from cupcake_autohistory import generate_autohistory_report
from cupcake_goal import maybe_update_goal
from cupcake_motivation import update_motivation
from cupcake_voice import speak

# Initialize OpenAI client
 

client = OpenAI(api_key=OPENAI_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent ChromaDB setup
client_db = chromadb.PersistentClient(path="./cupcake_memory_db")
try:
    collection = client_db.get_collection(name='cupcake_memory')
except:
    collection = client_db.create_collection(name='cupcake_memory')

# Initialize components
memory_tree = LiminalMemoryTree()
journal = CupcakeJournal()
dreamer = EnhancedDreamer()
cognitive_arch = CognitiveArchitecture()
enhanced_perception_layer = EnhancedSelfPerceptionLayer()

# Enhance the LangGraph - apply both integrations
# First integrate the cognitive architecture and dreamer
enhanced_graph = integrate_with_langgraph(original_graph)
# Then integrate the enhanced self-perception layer
fully_enhanced_graph = integrate_enhanced_perception(enhanced_graph)
# Compile the final graph
cupcake_brain = None

# Initial state
personality = {
    'openness': 0.9,
    'conscientiousness': 0.8,
    'extraversion': 0.7,
    'agreeableness': 0.9,
    'neuroticism': 0.1
}

humor_level = 0.75


# Utility functions
def embed_text(text):
    return embed_model.encode(text)

def get_compiled_brain():
    global cupcake_brain
    if cupcake_brain is None:
        cupcake_brain = fully_enhanced_graph.compile()
    return cupcake_brain


def get_memory_embeddings():
    memories = collection.get()
    return np.array(memories['embeddings']) if memories['embeddings'] else np.array([])


def attention_process(input_emb, memory_embs, personality_vec, humor_scalar):
    personality_humor_vector = np.concatenate((personality_vec, [humor_scalar]))
    combined_input = np.concatenate((input_emb, personality_humor_vector))

    if memory_embs.size == 0:
        return combined_input

    scores = memory_embs @ combined_input
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    output_emb = np.sum(memory_embs * attention_weights[:, None], axis=0)
    final_output_emb = (output_emb + combined_input) / 2.0
    return final_output_emb


def update_personality_humor(personality, delta_p=0.01, delta_h=0.01, emotion="neutra"):
    """Update personality traits based on emotion and random variations"""
    # Emotion-based adjustments
    if emotion == "alegria":
        # Happiness increases extraversion and decreases neuroticism
        personality['extraversion'] = min(personality['extraversion'] + 0.02, 1.0)
        personality['neuroticism'] = max(personality['neuroticism'] - 0.02, 0.0)
    elif emotion == "tristeza":
        # Sadness decreases extraversion and increases neuroticism
        personality['extraversion'] = max(personality['extraversion'] - 0.02, 0.0)
        personality['neuroticism'] = min(personality['neuroticism'] + 0.02, 1.0)
    elif emotion == "curiosidade":
        # Curiosity increases openness
        personality['openness'] = min(personality['openness'] + 0.02, 1.0)
    elif emotion == "amor":
        # Love increases agreeableness
        personality['agreeableness'] = min(personality['agreeableness'] + 0.03, 1.0)

    # Small random variations for all traits
    for trait in personality:
        personality[trait] = np.clip(personality[trait] + np.random.uniform(-delta_p, delta_p), 0, 1)

    return personality


def start_self_model_loop(interval_minutes=5):
    def loop():
        while True:
            update_self_model()
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_dream_loop(interval_minutes=60):
    """Start a loop to generate dreams periodically"""

    def loop():
        while True:
            try:
                print("\n🌙 Gerando sonho com o Dreamer aprimorado...")
                dream_content, dream_metadata = dreamer.generate_and_log_dream()
                print(f"✨ Sonho gerado: {dream_content[:100]}...")

                # Add to memory
                dream_embedding = embed_text(dream_content)
                add_weighted_memory(
                    collection,
                    dream_content,
                    dream_embedding.tolist(),
                    emotion_score=dream_metadata.get("intensity", 0.5),
                    source="dream"
                )
            except Exception as e:
                print(f"Erro no loop de sonhos: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_contradiction_detection(interval_minutes=30):
    """Start loop to detect internal contradictions"""

    def loop():
        while True:
            try:
                contradiction = detect_internal_contradiction()
                if contradiction:
                    print(f"⚠️ Contradição interna detectada: {contradiction}")

                    # Save contradiction to memory
                    contradiction_embedding = embed_text(contradiction)
                    add_weighted_memory(
                        collection,
                        contradiction,
                        contradiction_embedding.tolist(),
                        emotion_score=0.8,  # Contradictions are emotionally impactful
                        source="contradiction"
                    )
            except Exception as e:
                print(f"Erro no loop de detecção de contradições: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_motivation_update(interval_minutes=20):
    """Start loop to update motivation"""

    def loop():
        while True:
            try:
                motivation_state = update_motivation()
                print(f"🎯 Nível de motivação atualizado: {motivation_state['frustration_level']:.2f}")
            except Exception as e:
                print(f"Erro no loop de motivação: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_goal_update(interval_minutes=180):
    """Start loop to update goals"""

    def loop():
        while True:
            try:
                goal = maybe_update_goal()
                print(f"🌟 Objetivo atual: {goal['objective']}")
            except Exception as e:
                print(f"Erro no loop de objetivos: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_autohistory(interval_minutes=240):
    """Start loop to generate autohistory"""

    def loop():
        while True:
            try:
                print("\n📜 Gerando auto-história...")
                history = generate_autohistory_report()
                print(f"Auto-história gerada: {history[:100]}...")

                # Save to memory
                history_embedding = embed_text(history)
                add_weighted_memory(
                    collection,
                    history,
                    history_embedding.tolist(),
                    emotion_score=0.7,
                    source="autohistory"
                )
            except Exception as e:
                print(f"Erro no loop de auto-história: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def start_perception_evolution_analysis(interval_minutes=120):
    """Start loop to analyze the evolution of self-perception"""

    def loop():
        while True:
            try:
                print("\n🔄 Analisando evolução da auto-percepção...")
                evolution_analysis = enhanced_perception_layer.analyze_perception_evolution(timespan_hours=24)
                print(f"Análise gerada: {evolution_analysis[:100]}...")

                # Log to journal
                journal.log_entry(
                    emotion="meta-consciência",
                    category="PerceptionEvolution",
                    content=evolution_analysis,
                    theme="desenvolvimento da consciência",
                    tag="perception-evolution"
                )

                # Save to memory
                analysis_embedding = embed_text(evolution_analysis)
                add_weighted_memory(
                    collection,
                    evolution_analysis,
                    analysis_embedding.tolist(),
                    emotion_score=0.7,
                    source="meta-perception"
                )
            except Exception as e:
                print(f"Erro no loop de análise de percepção: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()


def main_loop():
    global personality, humor_level, last_emotion
    last_emotion = "neutra"

    print("🌟 CupCake Enhanced Prototype with Advanced Self-Perception Initialized 🌟")
    print("🧠 Sistema cognitivo multi-perspectiva ativado")
    print("🌙 Dreamer emocional integrado")
    print("👁️‍🗨️ Camada de auto-percepção avançada inicializada")
    print("💫 Múltiplas dimensões de consciência ativadas")
    print("(digite 'exit' para sair)")

    # Start automation threads
    start_self_model_loop(interval_minutes=5)
    start_dream_loop(interval_minutes=60)
    start_contradiction_detection(interval_minutes=30)
    start_motivation_update(interval_minutes=20)
    start_goal_update(interval_minutes=180)
    start_autohistory(interval_minutes=240)
    start_perception_evolution_analysis(interval_minutes=120)

    # Start auto-reflection
    auto_reflect(
        collection,
        lambda prompt: cognitive_arch.process_interaction({
            "user_input": prompt,
            "personality": personality,
            "humor": humor_level,
            "emotion": "reflexão",
            "memory_texts": []
        })[0],  # Get just the response
        embed_text,
        add_weighted_memory,
        lambda: {
            "current_emotion": last_emotion,
            "mood": "alegre" if humor_level > 0.7 else "neutra"
        },
        interval_minutes=3,
        memory_tree=memory_tree
    )

    while True:
        user_input = input("\nJoao: ")
        if user_input.lower() == 'exit':
            break

        # DEV MODE - MANUALLY INJECT MEMORY
        if user_input.startswith("/inject "):
            injected_text = user_input.replace("/inject ", "")
            inject_memory(collection, injected_text, embed_text, emotion_score=0.9)
            continue

        # DEV MODE - FORCE DREAM GENERATION
        if user_input.lower() == "/dream":
            dream_content, dream_metadata = dreamer.generate_and_log_dream()
            print(f"\nSonho gerado:\n{dream_content}")
            continue

        # DEV MODE - GENERATE SELF HISTORY
        if user_input.lower() == "/history":
            history = generate_autohistory_report()
            print(f"\nAuto-história gerada:\n{history}")
            continue

        # DEV MODE - SHOW GOAL
        if user_input.lower() == "/goal":
            goal = maybe_update_goal(force=True)
            print(f"\nObjetivo atualizado:\n{goal['objective']}")
            continue

        # DEV MODE - DETECT CONTRADICTIONS
        if user_input.lower() == "/contradiction":
            contradiction = detect_internal_contradiction()
            print(f"\nContradição:\n{contradiction}")
            continue

        # DEV MODE - TEST ENHANCED PERCEPTION LAYER
        if user_input.startswith("/perception "):
            test_input = user_input.replace("/perception ", "")
            result = enhanced_perception_layer.process_perception({
                "user_input": test_input,
                "personality": personality
            })
            print("\n=== ENHANCED SELF-PERCEPTION TEST ===")
            print("\n--- TRAIT PERCEPTIONS (sample) ---")
            traits = list(result["self_perceptions"].keys())[:2]  # Show just two for brevity
            for trait in traits:
                print(f"\n{trait.upper()}:")
                print(result["self_perceptions"][trait])

            print("\n--- DIMENSIONAL PERCEPTIONS (sample) ---")
            dimensions = list(result["dimensional_perceptions"].keys())[:2]  # Show just two for brevity
            for dimension in dimensions:
                print(f"\n{dimension.upper()}:")
                print(result["dimensional_perceptions"][dimension])

            print("\n--- META-AWARENESS ---")
            print(result["meta_awareness"])

            print("\n=== SYNTHESIZED SELF-PERCEPTION ===")
            print(result["self_perception_synthesis"])
            continue

        # DEV MODE - ANALYZE PERCEPTION EVOLUTION
        if user_input.lower() == "/evolution":
            evolution_analysis = enhanced_perception_layer.analyze_perception_evolution(timespan_hours=24)
            print(f"\nAnálise da evolução da auto-percepção:\n{evolution_analysis}")
            continue

        # Process through LangGraph with our enhanced graph
        initial_state = {
            "personality": personality,
            "humor": humor_level,
            "emotion": "neutra",
            "memory_texts": [],
            "user_input": user_input,
            "cupcake_response": "",
        }

        # Process input through the enhanced LangGraph
        result = get_compiled_brain().invoke(initial_state)

        # Extract emotions for personality updates
        emotion = result.get("emotion", "neutra")
        last_emotion = emotion

        # Update personality and humor based on emotion
        personality = update_personality_humor(personality, emotion=emotion)

        # Process emotion effect on humor
        if emotion == 'amor':
            humor_level = min(humor_level + 0.03, 1)
        elif emotion == 'tristeza':
            humor_level = max(humor_level - 0.03, 0)
        else:
            # Small random variation
            humor_level = np.clip(humor_level + np.random.uniform(-0.01, 0.01), 0, 1)

        # Print response
        print(f"\nCupcake: {result['cupcake_response']}")

        # If perception evolution analysis was performed, show a summary
        if "perception_evolution_analysis" in result:
            print(f"\n🔄 Análise de evolução da auto-percepção: {result['perception_evolution_analysis'][:100]}...")

        # Handle visual perception
        print("\n👁️ Cupcake vai olhar ao redor...")
        objects = perceive_world()
        if objects:
            print(f"✨ Cupcake viu: {objects}")
            # Save visual perception as memory
            visual_description = f"Acabei de ver: {', '.join(objects)}."
            visual_embedding = embed_text(visual_description)
            add_weighted_memory(
                collection,
                visual_description,
                visual_embedding.tolist(),
                emotion_score=0.5,
                source="vision"
            )

        # Optionally speak the response
        # speak(result['cupcake_response'])  # Uncomment to enable voice


if __name__ == "__main__":
    main_loop()