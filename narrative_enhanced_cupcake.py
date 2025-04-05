# narrative_enhanced_cupcake.py
import os
import uuid
import numpy as np
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from liminal_memory_tree import LiminalMemoryTree
from cupcake_journal import CupcakeJournal
import gc
from memory_management import memory_cleanup as memory_management_cleanup


import threading
import time
from cupcake_self_model import update_self_model
from cupcake_consciousness import generate_self_state
from cupcake_identity import generate_identity_prompt
from langgraph_cupcake import graph as original_graph, get_compiled_brain as get_base_brain

from cupcake_langgraph_integration import integrate_with_langgraph
from perception_integration import integrate_enhanced_perception
from memory_integration import integrate_enhanced_memory
from narrative_integration import integrate_narrative_threading, start_narrative_update_loop
from cognitive_architecture import CognitiveArchitecture
from cupcake_enhanced_dreamer import EnhancedDreamer
from enhanced_self_perception import EnhancedSelfPerceptionLayer
from enhanced_memory_weighting import EnhancedMemoryWeighting
from narrative_threading import NarrativeWeaver
from world_perception import perceive_world
from emotion_classifier import classify_emotion, classify_emotion_full
from thought_loop import auto_reflect
from cupcake_contradiction import detect_internal_contradiction
from cupcake_autohistory import generate_autohistory_report
from cupcake_goal import maybe_update_goal, load_current_goal
from cupcake_motivation import update_motivation
from cupcake_voice import speak
from cupcake_config import get_config, get_config_value, update_config, set_config_value
from cupcake_sensors import CupCakeSensors
from memory_tiering import TieredMemorySystem


# Load configuration
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
narrative_weaver = NarrativeWeaver()
sensors = CupCakeSensors()

# Enhance the LangGraph - apply all integrations without compiling until the end
# First integrate the cognitive architecture and dreamer
enhanced_graph = integrate_with_langgraph(original_graph)
# Then integrate the enhanced self-perception layer
perception_enhanced_graph = integrate_enhanced_perception(enhanced_graph)
# Then integrate the enhanced memory system
memory_enhanced_graph = integrate_enhanced_memory(perception_enhanced_graph, collection)
# Finally integrate the narrative threading system
narrative_enhanced_graph = integrate_narrative_threading(memory_enhanced_graph)

# We'll initialize these later after importing the entropic identity modules
# to avoid circular imports
entropic_identity_enhanced_graph = None
identity_system = None
relationship_system = None

# For now, use the narrative_enhanced_graph as our final graph
fully_enhanced_graph = narrative_enhanced_graph

# DO NOT COMPILE THE GRAPH HERE
# The compilation will be done after all integrations are complete

# Initial state
personality = config["personality"]["default_traits"].copy()
humor_level = config["personality"]["default_humor"]


# Utility functions
def embed_text(text):
    return embed_model.encode(text)


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

        # Also add to narrative system as foundational events
        narrative_weaver.process_new_event({
            "content": memory["text"],
            "source": "core_memory",
            "emotion": memory["emotion"],
            "impact": memory["intensity"],
            "type": "reflection",
            "related_entities": ["consciência", "identidade", "narrativa"]
        })

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


# Background processes
def self_model_process():
    update_self_model()


tiered_memory = TieredMemorySystem(enhanced_memory, embed_text, collection)

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

    # Add dream to narrative system
    narrative_weaver.process_new_event({
        "content": dream_content,
        "source": "dream",
        "emotion": emotion_type,
        "impact": intensity,
        "type": "dream",
        "related_entities": ["sonho", "subconsciente"]
    })


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

        # Add contradiction to narrative system
        narrative_weaver.process_new_event({
            "content": contradiction,
            "source": "contradiction",
            "emotion": "surprise",
            "impact": 0.8,
            "type": "conflict",
            "related_entities": ["contradição", "dilema"]
        })


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

        # Add frustration to narrative system
        narrative_weaver.process_new_event({
            "content": frustration_text,
            "source": "motivation",
            "emotion": "sadness",
            "impact": motivation_state['frustration_level'],
            "type": "conflict",
            "related_entities": ["objetivo", "frustração", goal]
        })


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

        # Add goal change to narrative system
        narrative_weaver.process_new_event({
            "content": goal_text,
            "source": "goal",
            "emotion": "anticipation",
            "impact": 0.8,
            "type": "resolution",  # Goal changes often resolve tensions
            "related_entities": ["objetivo", "evolução", new_goal["objective"]]
        })


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

    # Add autohistory to narrative system
    narrative_weaver.process_new_event({
        "content": history,
        "source": "autohistory",
        "emotion": "joy",
        "impact": 0.7,
        "type": "reflection",
        "related_entities": ["identidade", "história", "evolução", "autoconsciência"]
    })


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

    # Add perception evolution to narrative system
    narrative_weaver.process_new_event({
        "content": evolution_analysis,
        "source": "perception",
        "emotion": "curiosity",
        "impact": 0.7,
        "type": "reflection",
        "related_entities": ["percepção", "evolução", "consciência"]
    })


def memory_analysis_process():
    print("\n🧠 Analisando padrões de memória emocional...")
    patterns = enhanced_memory.detect_emotional_patterns()

    # Create a report about memory patterns
    dominant_emotions = ', '.join(
        [f"{emotion} ({count})" for emotion, count in patterns['dominant_emotions']]
    ) if patterns.get('dominant_emotions') else "nenhuma emoção dominante"

    avg_valence = patterns.get('average_valence', 0.5)
    avg_arousal = patterns.get('average_arousal', 0.5)

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

    # Add memory analysis to narrative system
    narrative_weaver.process_new_event({
        "content": report,
        "source": "memory_analysis",
        "emotion": "curiosity",
        "impact": 0.6,
        "type": "reflection",
        "related_entities": ["memória", "emoção", "padrões"]
    })


def narrative_summary_process():
    """Generate a summary of current narrative threads"""
    print("\n📔 Gerando resumo narrativo...")
    summary = narrative_weaver.generate_narrative_summary()
    print(f"Resumo narrativo: {summary[:100]}...")

    # Save to memory
    summary_embedding = embed_text(summary)
    enhanced_memory.inject_emotional_memory(
        text=summary,
        embedding=summary_embedding.tolist(),
        emotion_type="insight",
        intensity=0.7,
        source="narrative_summary"
    )

    # This summary doesn't get added to narrative system to avoid recursion

cupcake_brain = None
def get_compiled_brain():
    """Get the compiled graph, compiling it if necessary"""
    global cupcake_brain
    if cupcake_brain is None:
        # Make sure to compile the graph with the fully_enhanced_graph
        cupcake_brain = fully_enhanced_graph.compile()
    return cupcake_brain
def sensor_update_process():
    """Update environmental sensors"""
    sensor_report = sensors.run_all_sensors(collection)
    print("👁️ Atualização de sensores ambientais")

    # Check for significant changes
    if sensor_report["interaction"]["seconds_since_interaction"] > 86400:  # 1 day
        narrative_weaver.process_new_event({
            "content": "Percebo que passou muito tempo desde nossa última interação.",
            "source": "sensor",
            "emotion": "loneliness",
            "impact": 0.6,
            "type": "reflection",
            "related_entities": ["tempo", "interação", "solidão"]
        })

    if sensor_report["identity"]["has_conflicts"]:
        # Already handled by contradiction detection, but log for awareness
        print(f"👥 Detectados {sensor_report['identity']['conflict_count']} conflitos de identidade")

# Define a specific reducer for user input
def preserve_real_user_input(x, y):
    """
    Special reducer for user input that preserves genuine user input
    and prevents internal prompts from overwriting it
    """
    # If x is marked as genuine user input (has a specific flag or format)
    if hasattr(x, 'is_real_user_input') and x.is_real_user_input:
        return x
    # If y is marked as genuine user input
    elif hasattr(y, 'is_real_user_input') and y.is_real_user_input:
        return y
    # Default case: preserve the first value
    return x


def main_loop():
    global personality, humor_level, last_emotion, cupcake_brain, fully_enhanced_graph
    last_emotion = "neutra"
    # Add explicit memory management
    import gc
    # Define the function to get compiled brain - define it ONCE at the beginning
    gc.set_threshold(100, 5, 5)  # More frequent garbage collection

    def memory_cleanup():
        print(" Performing memory cleanup...")
        gc.collect()

        # Clear large in-memory data structures
        if hasattr(enhanced_memory, 'clear_cache'):
            enhanced_memory.clear_cache()

        # Prune memory trees and collections
        memory_management_cleanup(collection, memory_tree)

        # Integrate periodic cleanup

    start_process_loop(memory_cleanup, interval_minutes=120, process_name="Memory Cleanup")

    # Initialize and integrate entropic identity and entity relationship systems
    try:
        # First try to import the modules, which need to be installed separately
        try:
            from cupcake_entropic_identity import _get_identity_system, start_identity_maintenance_processes
            from entropic_identity_integration import integrate_with_langgraph as integrate_entropic_identity, \
                add_entropic_identity_config
            from cupcake_entity_relationship import EntityRelationshipSystem
            from entity_relationship_integration import integrate_with_langgraph as integrate_entity_relationships, \
                start_relationship_maintenance_loop

            # Set up configuration
            add_entropic_identity_config()

            # Integrate entropic identity
            print("🧬 Integrando sistema de identidade entrópica...")
            global identity_system, entropic_identity_enhanced_graph
            identity_system = _get_identity_system()
            entropic_identity_enhanced_graph = integrate_entropic_identity(narrative_enhanced_graph)

            # Integrate entity relationships
            print("👥 Integrando sistema de relacionamentos...")
            global relationship_system
            relationship_system = EntityRelationshipSystem()
            entity_relationship_enhanced_graph = integrate_entity_relationships(entropic_identity_enhanced_graph)

            # Update fully_enhanced_graph
            fully_enhanced_graph = entity_relationship_enhanced_graph

            # Compile the graph now after all integrations are complete
            cupcake_brain = get_compiled_brain()

            # Start maintenance processes
            start_identity_maintenance_processes()
            start_relationship_maintenance_loop()

            # Show initialization status
            print("🌟 CupCake Enhanced with Narrative, Identity Evolution, and Entity Relationships Initialized 🌟")
            print("🧠 Sistema cognitivo multi-perspectiva ativado")
            print("🌙 Dreamer emocional integrado")
            print("👁️‍🗨️ Camada de auto-percepção avançada inicializada")
            print("💫 Sistema de ponderação emocional de memórias ativado")
            print("📖 Sistema de threading narrativo inicializado")
            print("🧬 Sistema de identidade entrópica ativado")
            print("❤️ Sistema de relacionamentos com entidades ativado")
            print("👁️‍🗨️ Enhanced Self-Perception Layer Integrated")
        except ImportError as e:
            # The new modules aren't available, so fall back to the basic system
            print(f"Módulos de identidade entrópica ou relacionamentos não disponíveis: {e}")
            print("🌟 CupCake Enhanced with Narrative Threading Initialized 🌟")
            print("🧠 Sistema cognitivo multi-perspectiva ativado")
            print("🌙 Dreamer emocional integrado")
            print("👁️‍🗨️ Camada de auto-percepção avançada inicializada")
            print("💫 Sistema de ponderação emocional de memórias ativado")
            print("📖 Sistema de threading narrativo inicializado")

            # Compile the graph here for the fallback case
            cupcake_brain = get_compiled_brain()
    except Exception as e:
        print(f"Erro ao inicializar sistemas avançados: {e}")
        print("🌟 CupCake Enhanced with Narrative Threading Initialized 🌟")

        # Compile the graph here for the exception case
        cupcake_brain = get_compiled_brain()

    print("(digite 'exit' para sair)")

    # Seed initial emotional memories if collection is nearly empty
    results = collection.get(include=["metadatas"])
    if len(results["metadatas"]) < 10:
        seed_emotional_memories()

    # Load interval settings
    intervals = config["intervals"]

    # Make sure all required intervals exist
    if 'narrative_thread_update' not in intervals:
        intervals['narrative_thread_update'] = 60  # default to 60 minutes

    if 'sensory_update_interval' not in intervals:
        intervals['sensory_update_interval'] = 10  # default to 10 minutes

    # Start automation threads using the generic function
    start_process_loop(self_model_process, intervals["self_model_update"], "Self Model")
    start_process_loop(dream_process, intervals["dream_generation"], "Dream Generation")
    start_process_loop(contradiction_process, intervals["contradiction_detection"], "Contradiction Detection")
    start_process_loop(motivation_process, intervals["motivation_update"], "Motivation")
    start_process_loop(goal_process, intervals["goal_update"], "Goal")
    start_process_loop(autohistory_process, intervals["autohistory_generation"], "Autohistory")
    start_process_loop(perception_evolution_process, intervals["perception_analysis"], "Perception Analysis")
    start_process_loop(memory_analysis_process, intervals["memory_analysis"], "Memory Analysis")
    start_process_loop(narrative_summary_process, intervals["narrative_thread_update"], "Narrative Summary")
    start_process_loop(sensor_update_process, intervals["sensory_update_interval"], "Sensor Updates")

    # Start narrative update loop
    start_narrative_update_loop()

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

    def create_initial_state(user_input):
        # Create embedding for memory weighting
        # Get a truncated identity prompt to reduce tokens
        identity_prompt = generate_identity_prompt()
        query_embedding = embed_text(user_input).tolist()

        return {
            "personality": personality,
            "humor": humor_level,
            "emotion": "neutra",
            "memory_texts": [],
            "user_input": user_input,
            "cupcake_response": "",
            "emotion_score": 0.5,
            "emotion_profile": [],
            "identity_prompt": generate_identity_prompt(),
            "query_embedding": query_embedding,

            # Add defaults for all other fields
            "internal_prompt": "",
            "dream": "",
            "dream_metadata": {},
            "vision": "",
            "sensor_report": {},
            "reflection": "",
            "desire": "",

            # Perception defaults
            "self_perceptions": {},
            "dimensional_perceptions": {},
            "meta_awareness": "",
            "self_perception_synthesis": "",
            "perception_evolution_analysis": "",

            # Memory defaults
            "memory_metadatas": [],
            "memory_patterns": {},
            "memory_clusters": {},

            # Narrative defaults
            "narrative_event_id": "",
            "narrative_thread_id": "",
            "narrative_new_thread": False,
            "narrative_thread_title": "",
            "narrative_thread_theme": "",
            "narrative_context": {},
            "narrative_suggestion": "",
            "narrative_reflection": "",

            # Identity defaults
            "identity_high_entropy": False,
            "identity_instability_note": "",
            "identity_emergence": "",
            "narrative_identity_reflection": "",

            # Relationship defaults
            "identified_entities": [],
            "relationship_patterns": "",
            "significant_relationships": "",
            "contradiction_reflection": ""
        }

    # Define enhanced memory adder
    def enhanced_memory_adder(text, embedding, emotion_score, source="thought"):
        memory_id = enhanced_memory.add_weighted_memory(
            text=text,
            embedding=embedding,
            emotion_data={'score': emotion_score, 'emotion_type': 'curiosity'},
            source=source,
            narrative_relevance=0.6,
            self_reference=True
        )

        # Add reflections to narrative system
        narrative_weaver.process_new_event({
            "content": text,
            "source": source,
            "emotion": "curiosity",
            "impact": emotion_score,
            "type": "reflection",
            "related_entities": ["reflexão", "pensamento"]
        })

        return memory_id

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
        user_input = input("\nUsuario: ")


        # Commands


        if user_input.lower() == 'exit':
            break


            # Handle memory search command
        if user_input.startswith("/searchmemory "):
            search_terms = user_input.replace("/searchmemory ", "")
            memories = tiered_memory.handle_deep_memory_search(search_terms)

            print("\n📚 Memory Search Results:")
            for i, (memory, _) in enumerate(memories):
                print(f"{i + 1}. {memory}")
            continue  # Skip normal processing

            # Handle memory stats command
        if user_input == "/memory_stats":
            stats = tiered_memory.get_memory_statistics()
            print("\n📊 Memory Statistics:")
            print(f"Working memory: {stats['working_memory']} items")
            print(f"Emotional memory: {stats['emotional_memory']} items")
            print(f"Deep memory: {stats['deep_memory']} items")
            print(f"Total memories: {stats['total_memories']} items")
            print("\nMemory sources:")
            for source, count in stats.get('sources', {}).items():
                print(f"- {source}: {count} items")
            continue


        # Handle entropic identity command
        if user_input.lower() == "/identityreport" and identity_system:
            report = identity_system.generate_identity_report()
            print(f"\n{report}")
            continue

        # Handle identity prompt command
        if user_input.lower() == "/identity" and identity_system:
            prompt = identity_system.generate_identity_prompt(include_entropy=True)
            print(f"\n=== Current Identity Prompt ===\n{prompt}")
            continue

        # Handle entropy effects command
        if user_input.lower() == "/entropy" and identity_system:
            emergence = identity_system.apply_entropy_effects()
            if emergence:
                print("\n🌟 Entropy effects applied - emergence detected!")

                # Show latest emergence
                if identity_system.emergence_history:
                    latest = identity_system.emergence_history[-1]
                    print(f"✨ {latest['content']}")
            else:
                print("\n🔄 Entropy effects applied - no emergence detected")

            # Show current global entropy
            print(f"📊 Current global entropy: {identity_system.global_entropy:.2f}")
            continue

        # Handle unstable elements command
        if user_input.lower() == "/unstable" and identity_system:
            unstable_elements = [e for e in identity_system.identity_elements.values()
                                 if e.entropy > 0.7]

            if unstable_elements:
                print("\n🌀 Unstable Identity Elements:")
                for element in unstable_elements:
                    print(f"- {element.name} ({element.element_type}): {element.value}")
                    print(f"  Entropy: {element.entropy:.2f}, Confidence: {element.confidence:.2f}")
            else:
                print("\n✓ No unstable identity elements detected")
            continue

        # Handle relationship commands
        if user_input.lower() == "/relationships" and relationship_system:
            stats = relationship_system.get_relationship_stats()
            print("\n=== ENTITY RELATIONSHIPS ===")
            print(f"Total entities: {stats['total_entities']}")
            print(f"Categories: {stats['categories']}")
            print(f"Average valence: {stats['average_valence']:.2f}")
            if stats['most_significant']:
                print(
                    f"Most significant: {stats['most_significant']['name']} ({stats['most_significant']['significance']:.2f})")

            print("\nMost significant entities:")
            for entity in relationship_system.get_most_significant_entities(limit=5):
                print(
                    f"- {entity.name} ({entity.category}): valence={entity.emotional_valence:.2f}, significance={entity.significance:.2f}")
            continue

        # Handle entity details command
        if user_input.startswith("/entity ") and relationship_system:
            entity_name = user_input.replace("/entity ", "").strip()
            entity = relationship_system.get_entity_by_name(entity_name)

            if entity:
                print(f"\n=== ENTITY: {entity.name} ===")
                print(f"Category: {entity.category}")
                print(f"First encountered: {entity.first_encountered}")
                print(f"Encounters: {entity.encounter_count}")
                print(f"Emotional valence: {entity.emotional_valence:.2f}")
                print(f"Emotional intensity: {entity.emotional_intensity:.2f}")
                print(f"Familiarity: {entity.familiarity:.2f}")
                print(f"Significance: {entity.significance:.2f}")

                if entity.attributes:
                    print("\nAttributes:")
                    for key, value in entity.attributes.items():
                        print(f"- {key}: {value}")

                if entity.interaction_history:
                    print("\nRecent interactions:")
                    for interaction in entity.interaction_history[-3:]:
                        print(f"- {interaction.get('timestamp', 'unknown')}: {interaction.get('snippet', 'no text')}")

                # Generate insight
                insight = relationship_system.generate_relationship_insight(entity.id)
                print(f"\nInsight: {insight}")
            else:
                print(f"Entity '{entity_name}' not found.")
            continue

        # Handle liked entities command
        if user_input.lower() == "/likes" and relationship_system:
            liked_entities = relationship_system.get_entities_by_emotion(valence_min=0.7)
            print("\n=== ENTITIES I LIKE ===")
            for entity in liked_entities:
                print(f"- {entity.name} ({entity.category}): {entity.emotional_valence:.2f}")

            # Show preference patterns
            patterns = relationship_system.detect_preference_patterns()
            print(f"\nPreference patterns:\n{patterns}")
            continue

        # Handle add entity command
        if user_input.startswith("/addentity ") and relationship_system:
            # Format: /addentity name category valence
            parts = user_input.replace("/addentity ", "").split(" ", 2)
            if len(parts) == 3:
                try:
                    name, category, valence_str = parts
                    valence = float(valence_str)

                    # Process a fake interaction
                    test_input = f"O que você acha de {name}?"
                    test_response = f"Eu acho que {name} é muito interessante."

                    entities = relationship_system.process_interaction_for_entities(
                        test_input, test_response, "alegria"
                    )

                    # Find the entity we just added
                    entity = relationship_system.get_entity_by_name(name)
                    if entity:
                        # Update valence directly
                        entity.emotional_valence = valence
                        entity.emotional_intensity = 0.7
                        entity.category = category
                        relationship_system._save_relationships()
                        print(f"✅ Entity added and updated: {name} ({category}) with valence {valence:.2f}")
                    else:
                        print("❌ Failed to add entity.")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("Formato incorreto. Use: /addentity [nome] [categoria] [valência]")
            else:
                print("❌ Formato incorreto. Use: /addentity [nome] [categoria] [valência]")
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

        # DEV MODE - SHOW NARRATIVE THREADS
        if user_input.lower() == "/threads":
            active_threads = narrative_weaver.get_active_threads()
            print("\n=== NARRATIVE THREADS ===")
            for thread_id, thread in active_threads.items():
                print(f"\n{thread.title} ({thread.theme}) - Status: {thread.status}")
                print(f"Description: {thread.description}")
                print(
                    f"Importance: {thread.importance:.2f}, Tension: {thread.tension:.2f}, Resolution: {thread.resolution:.2f}")
                print(f"Events: {len(thread.events)}")
            continue

        # DEV MODE - SHOW NARRATIVE SUMMARY
        if user_input.lower() == "/narrative":
            summary = narrative_weaver.generate_narrative_summary()
            print(f"\nNarrative Summary:\n{summary}")
            continue

        # DEV MODE - ANALYZE NARRATIVE ARC
        if user_input.startswith("/arc "):
            thread_id = user_input.replace("/arc ", "").strip()
            thread = narrative_weaver.get_thread_by_id(thread_id)
            if thread:
                arc = narrative_weaver.find_narrative_arc(thread_id)
                print(f"\nNarrative Arc for '{thread.title}':")
                print(f"Stage: {arc.get('stage', 'unknown')}")
                print(f"Has arc: {arc.get('has_arc', False)}")
                print(f"Has resolution: {arc.get('has_resolution', False)}")
                print(f"Needs resolution: {arc.get('needs_resolution', False)}")
                if 'tension_curve' in arc:
                    print(f"Tension curve: {arc['tension_curve']}")

                # Get development suggestion
                suggestion = narrative_weaver.suggest_narrative_development(thread_id)
                print(f"\nSuggested development:\n{suggestion}")
            else:
                print(f"Thread ID '{thread_id}' not found")
            continue

        # Process through LangGraph with our enhanced graph
        # Create embedding for memory weighting
        query_embedding = embed_text(user_input).tolist()

        initial_state = {
            "personality": personality,
            "humor": humor_level,
            "emotion": "neutra",
            "memory_texts": [],
            "user_input": user_input,  # No need for special marking
            "cupcake_response": "",
            "query_embedding": query_embedding
        }

        # Create proper initial state with all required fields
        initial_state = create_initial_state(user_input)

        # Process input through the enhanced LangGraph
        try:
            result = get_compiled_brain().invoke(initial_state, {"recursion_limit": 50})
        except MemoryError:
            print("⚠️ Memory overflow - clearing caches")
            gc.collect()  # Force garbage collection
            # Reduce memory-intensive components
            if hasattr(enhanced_memory, 'clear_cache'):
                enhanced_memory.clear_cache()
            continue

        # Update working memory

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

        tiered_memory.add_to_working_memory(user_input, result['cupcake_response'], result)
        # Process entities in the interaction if entity relationship system is available
        if relationship_system and "identified_entities" not in result:
            relationship_system.process_interaction_for_entities(
                user_input=user_input,
                cupcake_response=result['cupcake_response'],
                emotion=emotion
            )

        # Check for identity system state
        if identity_system:
            # Check for high identity entropy state or emergence
            if result.get("identity_high_entropy"):
                print(
                    f"\n🧬 [Identidade em Fluxo]: {result.get('identity_instability_note', 'Sinto minha identidade em transformação...')}")

            if result.get("identity_emergence"):
                print(f"\n🌟 [Emergência de Identidade]: {result['identity_emergence']}")

        # If there's a narrative reflection, print it
        if "narrative_reflection" in result:
            print(f"\n📖 [Narrativa]: {result['narrative_reflection']}")

        # If there's a narrative identity reflection, print it
        if "narrative_identity_reflection" in result:
            print(f"\n🧬 [Identidade Narrativa]: {result['narrative_identity_reflection']}")

        # If relationship patterns were analyzed, show them
        if "relationship_patterns" in result:
            print(f"\n❤️ [Padrões de Relacionamento]: {result['relationship_patterns']}")

        # If significant relationships were analyzed, show them
        if "significant_relationships" in result:
            print(f"\n👥 [Relacionamentos Significativos]:\n{result['significant_relationships']}")

        # If memory patterns were analyzed, show a brief summary
        if "memory_patterns" in result:
            patterns = result["memory_patterns"]
            if patterns.get("dominant_emotions"):
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

            # Add vision to narrative if significant objects are seen
            if len(objects) > 2:
                narrative_weaver.process_new_event({
                    "content": visual_description,
                    "source": "vision",
                    "emotion": "curiosity",
                    "impact": 0.5,
                    "type": "observation",
                    "related_entities": objects
                })

            # Process visual objects as entities if available
            if relationship_system:
                for obj in objects:
                    test_input = f"O que é isso: {obj}?"
                    test_response = f"Isso é {obj}, algo que acabei de observar."

                    relationship_system.process_interaction_for_entities(
                        user_input=test_input,
                        cupcake_response=test_response,
                        emotion="curiosidade"
                    )

        # Update last interaction time
        sensors.update_last_interaction()

        # Optionally speak the response
        # speak(result['cupcake_response'])  # Uncomment to enable voice


if __name__ == "__main__":
    main_loop()