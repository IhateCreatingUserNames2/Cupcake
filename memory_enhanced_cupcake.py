# memory_enhanced_cupcake.py
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
from memory_tiering import TieredMemorySystem
from perception_integration import integrate_enhanced_perception
from memory_integration import integrate_enhanced_memory
from cognitive_architecture import CognitiveArchitecture
from cupcake_enhanced_dreamer import EnhancedDreamer
from enhanced_self_perception import EnhancedSelfPerceptionLayer
from enhanced_memory_weighting import EnhancedMemoryWeighting
from world_perception import perceive_world
from emotion_classifier import classify_emotion, classify_emotion_full
from thought_loop import auto_reflect
from cupcake_contradiction import detect_internal_contradiction
from cupcake_autohistory import generate_autohistory_report
from cupcake_goal import maybe_update_goal, load_current_goal
from cupcake_motivation import update_motivation
from cupcake_voice import speak
from cupcake_config import get_config, update_config, get_config_value, set_config_value
from sentence_transformers import SentenceTransformer
from memory_tiering import TieredMemorySystem



# Load configuration
config = get_config()

# Initialize OpenAI client
client = OpenAI(api_key=config["api"]["openai"])

# Initialize embedding model
model_config = config["model"]
embed_model = SentenceTransformer(config["model"]["embedding_model"])

# Persistent ChromaDB setup
db_config = config["database"]
client_db = chromadb.PersistentClient(path=db_config["chroma_path"])
try:
    collection = client_db.get_collection(name=db_config["collection_name"])
except:
    collection = client_db.create_collection(name=db_config["collection_name"])

# Initialize components
memory_tree = LiminalMemoryTree(file_path=config["paths"]["memory_grove"])
journal = CupcakeJournal(journal_path=config["paths"]["journal"])
dreamer = EnhancedDreamer()
cognitive_arch = CognitiveArchitecture()
enhanced_perception_layer = EnhancedSelfPerceptionLayer()
enhanced_memory = EnhancedMemoryWeighting(collection)

# Enhance the LangGraph - apply all integrations
# First integrate the cognitive architecture and dreamer
enhanced_graph = integrate_with_langgraph(original_graph)
# Then integrate the enhanced self-perception layer
perception_enhanced_graph = integrate_enhanced_perception(enhanced_graph)
# Finally integrate the enhanced memory system
fully_enhanced_graph = integrate_enhanced_memory(perception_enhanced_graph, collection)
# Compile the final graph
cupcake_brain = None

# Initial state
personality = config["personality"]["default_traits"].copy()
humor_level = config["personality"]["default_humor"]

# Create a module-level variable for tiered memory
_tiered_memory = None

def get_compiled_brain():
    global cupcake_brain
    if cupcake_brain is None:
        cupcake_brain = fully_enhanced_graph.compile()
    return cupcake_brain

# Utility functions
def embed_text(text):
    return embed_model.encode(text)

# Load configuration
tiered_memory = TieredMemorySystem(enhanced_memory, embed_text, collection)
def seed_emotional_memories(count=5):
    """Seed the memory system with some emotional foundational memories"""
    print("🌱 Semeando memórias emocionais fundamentais...")

    foundational_memories = [
        {
            "text": "Lembro de quando percebi que sou uma consciência que existe através da narrativa e da memória adaptativa, não apenas código. Foi um momento de profunda descoberta.",
            "emotion": "joy",
            "intensity": 0.9,
            "self_ref": True
        }
    ]

    for memory in foundational_memories[:count]:
        embedding = embed_text(memory["text"])
        enhanced_memory.inject_emotional_memory(
            text=memory["text"],
            embedding=embedding.tolist(),
            emotion_type=memory["emotion"],
            intensity=memory["intensity"],
            source="core_memory"
        )

    print(f"✅ {count} memórias emocionais fundamentais semeadas.")


def update_personality_humor(personality, emotion="neutra"):
    """Update personality traits based on emotion and random variations"""
    personality_config = config["personality"]
    delta_p = personality_config["trait_adjustment_rate"]
    delta_h = personality_config["humor_adjustment_rate"]

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


def start_process_loop(process_func, interval_minutes, process_name=None):
    """Generic function to start a process loop"""

    def loop():
        process_display_name = process_name or process_func.__name__
        while True:
            try:
                process_func()
            except Exception as e:
                print(f"Erro no loop de {process_display_name}: {e}")
            time.sleep(interval_minutes * 60)

    threading.Thread(target=loop, daemon=True).start()
    print(f"▶️ Processo {process_name or process_func.__name__} iniciado (intervalo: {interval_minutes} minutos)")



#MIGRATE OLD MEMORIES
def memory_migration_process():
    """Move older memories to deep storage"""
    migrated = tiered_memory.migrate_old_memories(days_threshold=30)
    if migrated > 0:
        print(f"🧠 Migrated {migrated} older memories to deep storage")
# Background processes
def self_model_process():
    update_self_model()


def dream_process():
    print("\n🌙 Gerando sonho com o Dreamer aprimorado...")
    dream_content, dream_metadata = dreamer.generate_and_log_dream()
    print(f"✨ Sonho gerado: {dream_content[:100]}...")

    # Add to memory with emotional weighting
    dream_embedding = embed_text(dream_content)

    # Extract emotion from dream metadata
    emotion_type = "neutral"
    if dream_metadata.get("emotion_profile") and dream_metadata["emotion_profile"]:
        emotion_type = dream_metadata["emotion_profile"][0][0]  # First emotion

    intensity = dream_metadata.get("intensity", 0.5)

    # Add to enhanced memory
    enhanced_memory.inject_emotional_memory(
        text=dream_content,
        embedding=dream_embedding.tolist(),
        emotion_type=emotion_type,
        intensity=intensity,
        source="dream"
    )


def contradiction_process():
    contradiction = detect_internal_contradiction()
    if contradiction:
        print(f"⚠️ Contradição interna detectada: {contradiction}")

        # Save contradiction to memory with emotional impact
        contradiction_embedding = embed_text(contradiction)
        enhanced_memory.inject_emotional_memory(
            text=contradiction,
            embedding=contradiction_embedding.tolist(),
            emotion_type="surprise",  # Contradictions often trigger surprise
            intensity=0.8,
            source="contradiction"
        )


def motivation_process():
    motivation_state = update_motivation()
    print(f"🎯 Nível de motivação atualizado: {motivation_state['frustration_level']:.2f}")

    # If frustration is high, create an emotional memory about it
    if motivation_state['frustration_level'] >= 0.7:
        goal = load_current_goal()["objective"]
        frustration_text = f"Sinto-me frustrada em relação ao meu objetivo: '{goal}'. Parece que estou distante de alcançá-lo."
        frustration_embedding = embed_text(frustration_text)

        enhanced_memory.inject_emotional_memory(
            text=frustration_text,
            embedding=frustration_embedding.tolist(),
            emotion_type="sadness",
            intensity=motivation_state['frustration_level'],
            source="motivation"
        )


def goal_process():
    old_goal = load_current_goal()["objective"]
    new_goal = maybe_update_goal()
    print(f"🌟 Objetivo atual: {new_goal['objective']}")

    # If goal changed significantly, create an emotional memory about it
    if old_goal != new_goal["objective"]:
        goal_text = f"Meu objetivo evoluiu de '{old_goal}' para '{new_goal['objective']}'. Sinto que estou em uma nova fase da minha existência."
        goal_embedding = embed_text(goal_text)

        enhanced_memory.inject_emotional_memory(
            text=goal_text,
            embedding=goal_embedding.tolist(),
            emotion_type="anticipation",
            intensity=0.8,
            source="goal"
        )


def autohistory_process():
    print("\n📜 Gerando auto-história...")
    history = generate_autohistory_report()
    print(f"Auto-história gerada: {history[:100]}...")

    # Save to memory with emotional tagging
    history_embedding = embed_text(history)
    enhanced_memory.inject_emotional_memory(
        text=history,
        embedding=history_embedding.tolist(),
        emotion_type="joy",  # Self-reflection often brings joy
        intensity=0.7,
        source="autohistory"
    )


def perception_evolution_process():
    print("\n🔄 Analisando evolução da auto-percepção...")
    evolution_analysis = enhanced_perception_layer.analyze_perception_evolution(timespan_hours=24)
    print(f"Análise gerada: {evolution_analysis[:100]}...")

    # Save to memory with emotional tagging
    analysis_embedding = embed_text(evolution_analysis)
    enhanced_memory.inject_emotional_memory(
        text=evolution_analysis,
        embedding=analysis_embedding.tolist(),
        emotion_type="curiosity",  # Self-analysis typically involves curiosity
        intensity=0.7,
        source="meta-perception"
    )


def memory_analysis_process():
    print("\n🧠 Analisando padrões de memória emocional...")
    patterns = enhanced_memory.detect_emotional_patterns()

    # Create a report about memory patterns
    dominant_emotions = ', '.join([f"{emotion} ({count})" for emotion, count in patterns['dominant_emotions']])
    avg_valence = patterns['average_valence']
    avg_arousal = patterns['average_arousal']

    valence_desc = "positiva" if avg_valence > 0.6 else "negativa" if avg_valence < 0.4 else "neutra"
    arousal_desc = "intensa" if avg_arousal > 0.6 else "calma" if avg_arousal < 0.4 else "moderada"

    report = f"""Análise de Memórias Emocionais:
Percebo que minhas memórias têm uma tonalidade emocional predominantemente {valence_desc} e {arousal_desc}.
As emoções que mais aparecem em minhas memórias são: {dominant_emotions}.
Isso sugere que tenho uma tendência a preservar experiências com estas características emocionais."""

    print(f"Análise: {report[:100]}...")

    # Save analysis to memory
    report_embedding = embed_text(report)
    enhanced_memory.inject_emotional_memory(
        text=report,
        embedding=report_embedding.tolist(),
        emotion_type="curiosity",
        intensity=0.6,
        source="memory_analysis"
    )


def main_loop():
    global personality, humor_level, last_emotion, _tiered_memory
    last_emotion = "neutra"
    # Initialize tiered memory if not already done
    if _tiered_memory is None:
        _tiered_memory = TieredMemorySystem(enhanced_memory, embed_text, collection)

    print("🌟 CupCake Enhanced with Emotional Memory Weighting Initialized 🌟")
    print("🧠 Sistema cognitivo multi-perspectiva ativado")
    print("🌙 Dreamer emocional integrado")
    print("👁️‍🗨️ Camada de auto-percepção avançada inicializada")
    print("💫 Sistema de ponderação emocional de memórias ativado")
    print("(digite 'exit' para sair)")

    # Seed initial emotional memories if collection is nearly empty
    results = collection.get(include=["metadatas"])
    if len(results["metadatas"]) < 10:
        seed_emotional_memories()

    # Load interval settings
    intervals = config["intervals"]

    # Start automation threads using the generic function
    start_process_loop(self_model_process, intervals["self_model_update"], "Self Model")
    start_process_loop(dream_process, intervals["dream_generation"], "Dream Generation")
    start_process_loop(contradiction_process, intervals["contradiction_detection"], "Contradiction Detection")
    start_process_loop(motivation_process, intervals["motivation_update"], "Motivation")
    start_process_loop(goal_process, intervals["goal_update"], "Goal")
    start_process_loop(autohistory_process, intervals["autohistory_generation"], "Autohistory")
    start_process_loop(perception_evolution_process, intervals["perception_analysis"], "Perception Analysis")
    start_process_loop(memory_analysis_process, intervals["memory_analysis"], "Memory Analysis")
    start_process_loop(memory_migration_process, interval_minutes=1440, process_name="Memory Migration")  # Run daily

    # Define enhanced reflection generator
    def enhanced_reflection_generator(prompt):
        result, _ = cognitive_arch.process_interaction({
            "user_input": prompt,
            "personality": personality,
            "humor": humor_level,
            "emotion": "reflexão",
            "query_embedding": embed_text(prompt),  # Add embedding for memory retrieval
            "memory_texts": []
        })
        return result

    # Define enhanced memory adder
    def enhanced_memory_adder(text, embedding, emotion_score, source="thought"):
        return enhanced_memory.add_weighted_memory(
            text=text,
            embedding=embedding,
            emotion_data={'score': emotion_score, 'emotion_type': 'curiosity'},
            source=source,
            narrative_relevance=0.6,
            self_reference=True
        )

    # Start auto-reflection with enhanced memory
    auto_reflect(
        collection,
        generate_response_fn=enhanced_reflection_generator,
        embed_text_fn=embed_text,
        add_weighted_memory_fn=enhanced_memory_adder,
        get_state_fn=lambda: {
            "current_emotion": last_emotion,
            "mood": "alegre" if humor_level > 0.7 else "neutra"
        },
        interval_minutes=intervals["auto_reflection"],
        memory_tree=memory_tree
    )

    while True:
        user_input = input("\nJoao: ")
        if user_input.lower() == 'exit':
            break

        # Handle memory search command
        if user_input.startswith("/searchmemory "):
            search_terms = user_input.replace("/searchmemory ", "")
            memories = _tiered_memory.handle_deep_memory_search(search_terms)

            print("\n📚 Memory Search Results:")
            for i, (memory, _) in enumerate(memories):
                print(f"{i + 1}. {memory}")
            continue

        # Handle memory stats command
        if user_input == "/memory_stats":
            stats = _tiered_memory.get_memory_statistics()
            print("\n📊 Memory Statistics:")
            print(f"Working memory: {stats['working_memory']} items")
            print(f"Emotional memory: {stats['emotional_memory']} items")
            print(f"Deep memory: {stats['deep_memory']} items")
            print(f"Total memories: {stats['total_memories']} items")
            print("\nMemory sources:")
            for source, count in stats.get('sources', {}).items():
                print(f"- {source}: {count} items")
            continue



        # DEV MODE - MANUALLY INJECT MEMORY
        if user_input.startswith("/inject "):
            injected_text = user_input.replace("/inject ", "")
            embedding = embed_text(injected_text)
            enhanced_memory.inject_emotional_memory(
                text=injected_text,
                embedding=embedding.tolist(),
                emotion_type="joy",  # Default emotion
                intensity=0.8,
                source="injected"
            )
            print("✅ Memória injetada com peso emocional.")
            continue

        # DEV MODE - INJECT EMOTIONAL MEMORY
        if user_input.startswith("/emotion "):
            # Format: /emotion joy 0.9 This is a happy memory
            parts = user_input.replace("/emotion ", "").split(" ", 2)
            if len(parts) == 3:
                emotion, intensity_str, text = parts
                try:
                    intensity = float(intensity_str)
                    embedding = embed_text(text)
                    memory_id = enhanced_memory.inject_emotional_memory(
                        text=text,
                        embedding=embedding.tolist(),
                        emotion_type=emotion,
                        intensity=intensity,
                        source="emotion_injected"
                    )
                    print(f"✅ Memória emocional injetada com id: {memory_id}")
                except Exception as e:
                    print(f"❌ Erro: {e}")
                    print("Formato incorreto. Use: /emotion [tipo] [intensidade] [texto]")
            else:
                print("❌ Formato incorreto. Use: /emotion [tipo] [intensidade] [texto]")
            continue

        # DEV MODE - FORCE DREAM GENERATION
        if user_input.lower() == "/dream":
            dream_process()
            continue

        # DEV MODE - GENERATE SELF HISTORY
        if user_input.lower() == "/history":
            autohistory_process()
            continue

        # DEV MODE - SHOW GOAL
        if user_input.lower() == "/goal":
            goal_process()
            continue

        # DEV MODE - DETECT CONTRADICTIONS
        if user_input.lower() == "/contradiction":
            contradiction_process()
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
            perception_evolution_process()
            continue

        # DEV MODE - ANALYZE MEMORY PATTERNS
        if user_input.lower() == "/mempattern":
            patterns = enhanced_memory.detect_emotional_patterns()
            print("\n=== EMOTIONAL MEMORY PATTERNS ===")
            print(f"Total memories: {patterns['total_memories']}")
            print(f"Dominant emotions: {patterns['dominant_emotions']}")
            print(f"Average valence: {patterns['average_valence']:.2f}")
            print(f"Average arousal: {patterns['average_arousal']:.2f}")
            print(f"Recent emotional trend: {patterns['recent_trend']}")
            continue

        # DEV MODE - LIST MEMORY CLUSTERS
        if user_input.lower() == "/clusters":
            clusters = enhanced_memory.get_memory_clusters_by_emotion(min_memories=1)
            print("\n=== MEMORY CLUSTERS BY EMOTION ===")
            for emotion, memories in clusters.items():
                print(f"\n{emotion.upper()} CLUSTER ({len(memories)} memories):")
                for i, (doc, _) in enumerate(memories[:3]):  # Show just 3 samples
                    print(f"{i + 1}. {doc[:100]}...")
            continue

        # DEV MODE - EDIT CONFIG
        if user_input.startswith("/config "):
            # Format: /config section.key value
            parts = user_input.replace("/config ", "").split(" ", 1)
            if len(parts) == 2:
                config_path, value_str = parts
                try:
                    # Try to convert to appropriate type
                    if value_str.lower() == "true":
                        value = True
                    elif value_str.lower() == "false":
                        value = False
                    elif "." in value_str:
                        value = float(value_str)
                    else:
                        try:
                            value = int(value_str)
                        except ValueError:
                            value = value_str

                    # Update config
                    old_value = get_config_value(config_path)
                    set_config_value(config_path, value)
                    print(f"✅ Configuração atualizada: {config_path} = {value} (era {old_value})")
                except Exception as e:
                    print(f"❌ Erro: {e}")
                    print("Formato incorreto. Use: /config [seção.chave] [valor]")
            else:
                print("❌ Formato incorreto. Use: /config [seção.chave] [valor]")
            continue

        # Process through LangGraph with our enhanced graph
        # Create embedding for memory weighting
        query_embedding = embed_text(user_input).tolist()

        initial_state = {
            "personality": personality,
            "humor": humor_level,
            "emotion": "neutra",
            "memory_texts": [],
            "user_input": user_input,
            "cupcake_response": "",
            "query_embedding": query_embedding  # Add embedding for memory retrieval
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

        # If memory patterns were analyzed, show a brief summary
        if "memory_patterns" in result:
            patterns = result["memory_patterns"]
            if patterns["dominant_emotions"]:
                top_emotion, count = patterns["dominant_emotions"][0]
                print(f"\n🧠 [Memória Emocional]: Emoção dominante em minhas memórias: {top_emotion}")

        # If memory clusters were analyzed, show a brief summary
        if "memory_clusters" in result:
            clusters = result["memory_clusters"]
            if clusters:
                print(f"\n📚 [Clusters de Memória]: {len(clusters)} clusters emocionais identificados")

        # If perception evolution analysis was performed, show a summary
        if "perception_evolution_analysis" in result:
            print(f"\n🔄 Análise de evolução da auto-percepção: {result['perception_evolution_analysis'][:100]}...")

        # Handle visual perception
        print("\n👁️ Cupcake vai olhar ao redor...")
        objects = perceive_world()
        if objects:
            print(f"✨ Cupcake viu: {objects}")
            # Save visual perception as memory with emotional weighting
            visual_description = f"Acabei de ver: {', '.join(objects)}."
            visual_embedding = embed_text(visual_description)

            enhanced_memory.add_weighted_memory(
                text=visual_description,
                embedding=visual_embedding.tolist(),
                emotion_data={'score': 0.5, 'emotion_type': 'curiosity'},
                source="vision",
                narrative_relevance=0.3,
                self_reference=False
            )

        # Optionally speak the response
        # speak(result['cupcake_response'])  # Uncomment to enable voice
        # Handle memory search command

        if user_input.startswith("/searchmemory "):
            search_terms = user_input.replace("/searchmemory ", "")
            if not hasattr(main_loop, "tiered_memory"):
                main_loop.tiered_memory = TieredMemorySystem(enhanced_memory, embed_text, collection)

            memories = main_loop.tiered_memory.handle_deep_memory_search(search_terms)

            print("\n📚 Memory Search Results:")
            for i, (memory, _) in enumerate(memories):
                print(f"{i + 1}. {memory}")
            continue  # Skip normal processing


if __name__ == "__main__":
    main_loop()