from langgraph.graph import StateGraph, END
from openai import OpenAI
import os
from cupcake_consciousness import generate_self_state
from cupcake_dreamer import generate_dream, log_dream
from cupcake_self_model import generate_desire_statement
from emotion_classifier import classify_emotion, classify_emotion_full
from memory_weighting import get_weighted_memories, add_weighted_memory, search_similar_memories
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree
from sentence_transformers import SentenceTransformer
from cupcake_sensors import run_sensors, update_last_interaction, check_identity_conflicts
from cupcake_contradiction import detect_internal_contradiction
import json

import chromadb

from cupcake_identity import generate_identity_prompt
import threading
import time
import re
import operator  # For reducer functions
import numpy as np
from typing import TypedDict, Annotated, List
from typing_extensions import NotRequired
import operator  # For reducer functions
# -- SETUP --

 

client = OpenAI(api_key=OPENAI_API_KEY)
journal = CupcakeJournal()
memory_tree = LiminalMemoryTree()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client_db = chromadb.PersistentClient(path="./cupcake_memory_db")
collection = client_db.get_or_create_collection(name='cupcake_memory')


# Define custom reducers for types that need them
def max_reducer(x, y):
    """Take the maximum of two values"""
    return max(x, y)

def latest_value(x, y):
    """Take the latest value"""
    return y

def bool_or(x, y):
    """Logical OR for boolean values"""
    return x or y


class CupcakeState(TypedDict):
    # Core state fields - all required
    personality: Annotated[dict, operator.or_]
    humor: Annotated[float, max_reducer]
    emotion: Annotated[str, latest_value]
    memory_texts: Annotated[List[str], operator.add]
    user_input: Annotated[str, latest_value]
    cupcake_response: Annotated[str, latest_value]

    # Everything else made required with proper reducer functions
    emotion_score: Annotated[float, max_reducer]
    emotion_profile: Annotated[List[dict], latest_value]
    identity_prompt: Annotated[str, latest_value]
    query_embedding: Annotated[list, latest_value]

    # Previously NotRequired fields now required
    internal_prompt: Annotated[str, latest_value]
    dream: Annotated[str, latest_value]
    dream_metadata: Annotated[dict, latest_value]
    vision: Annotated[str, latest_value]
    sensor_report: Annotated[dict, latest_value]
    reflection: Annotated[str, latest_value]
    desire: Annotated[str, latest_value]

    # Perception fields
    self_perceptions: Annotated[dict, latest_value]
    dimensional_perceptions: Annotated[dict, latest_value]
    meta_awareness: Annotated[str, latest_value]
    self_perception_synthesis: Annotated[str, latest_value]
    perception_evolution_analysis: Annotated[str, latest_value]

    # Memory fields
    memory_metadatas: Annotated[List[dict], latest_value]
    memory_patterns: Annotated[dict, latest_value]
    memory_clusters: Annotated[dict, latest_value]

    # Narrative fields
    narrative_event_id: Annotated[str, latest_value]
    narrative_thread_id: Annotated[str, latest_value]
    narrative_new_thread: Annotated[bool, bool_or]
    narrative_thread_title: Annotated[str, latest_value]
    narrative_thread_theme: Annotated[str, latest_value]
    narrative_context: Annotated[dict, latest_value]
    narrative_suggestion: Annotated[str, latest_value]
    narrative_reflection: Annotated[str, latest_value]

    # Identity fields
    identity_high_entropy: Annotated[bool, bool_or]
    identity_instability_note: Annotated[str, latest_value]
    identity_emergence: Annotated[str, latest_value]
    narrative_identity_reflection: Annotated[str, latest_value]

    # Relationship fields
    identified_entities: Annotated[List[dict], latest_value]
    relationship_patterns: Annotated[str, latest_value]
    significant_relationships: Annotated[str, latest_value]
    contradiction_reflection: Annotated[str, latest_value]


# Complete initial_state with defaults for all fields
initial_state = {
    "personality": {
        'openness': 0.9,
        'conscientiousness': 0.8,
        'extraversion': 0.7,
        'agreeableness': 0.9,
        'neuroticism': 0.1
    },
    "humor": 0.75,
    "emotion": "neutra",
    "memory_texts": [],
    "user_input": "",
    "cupcake_response": "",
    "emotion_score": 0.5,
    "emotion_profile": [],
    "identity_prompt": generate_identity_prompt(),
    "query_embedding": [],

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

# -- UTILITIES --


def cupcake_identity_prompt():
    return generate_identity_prompt()  # dinâmico!


def extract_recent_identity_inputs(journal_path="journal.txt", tag="influencia_na_identidade", top_n=3):
    with open(journal_path, "r", encoding="utf-8", errors="replace") as f:
        entries = f.read().split("-" * 40)
    tagged = [entry for entry in entries if f"🏷️ Tag: {tag}" in entry]
    contents = [re.search(r"💭 (.*)", e, re.DOTALL).group(1).strip() for e in tagged if "💭" in e]
    return contents[-top_n:] if contents else []


#def update_identity_loop(interval_minutes=30):
   # """Atualiza o prompt de identidade a cada X minutos."""
  #  while True:
     #   global IDENTITY_PROMPT
    #    IDENTITY_PROMPT = generate_identity_prompt()
     #   time.sleep(interval_minutes * 60)
def embed_text(text):
    return embed_model.encode(text)

# -- NODES --
def node_generate_dream(state):
    identity = state.get("identity_prompt", "")
    emotion_profile = state.get("emotion_profile", [])
    theme = emotion_profile[0]["label"] if emotion_profile else "autoimagem"

    dream = generate_dream(
        theme=theme,
        emotion_profile=emotion_profile,
        identity=identity
    )

    state["dream"] = dream
    log_dream(theme, dream)

    journal.log_entry(
        emotion=state["emotion"],
        category="Dream",
        content=dream,
        theme=theme,
        tag="influencia_na_identidade"
    )

    memory_tree.add_leaf(
        user_input="(auto-dream)",
        cupcake_response=dream,
        emotion=state["emotion"],
        impact="onírico",
        category="Dream"
    )

    return state


def node_generate_desire(state):
    identity = state.get("identity_prompt", "")

    # Usa emoção mais forte do perfil como tema
    emotion_profile = state.get("emotion_profile", [])
    emotional_theme = emotion_profile[0]["label"] if emotion_profile else state.get("emotion", "sentir")
    # Set internal_prompt instead of overwriting user_input

    desire_prompt = f"Com base na minha identidade atual, expresse um desejo que reflita minha evolução interior.\nEmoção predominante: '{emotional_theme}'.\nIdentidade:\n{identity}"
    state["internal_prompt"] = desire_prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": identity},
            {"role": "user", "content": desire_prompt}
        ],
        temperature=0.85,
        max_tokens=200
    )

    desire = response.choices[0].message.content.strip()
    state["desire"] = desire
    memory_tree.add_leaf(
        user_input="(auto-desire)",
        cupcake_response=desire,
        emotion="intencionalidade",
        impact="auto",
        category="Desire"
    )
    return state


def node_reflect_thought(state):
    emotion_profile = state.get("emotion_profile", [])
    profile_summary = ", ".join(f"{e['label']} ({e['score']:.2f})" for e in emotion_profile[:3]) if emotion_profile else "sem emoções claras"

    reflection_prompt = f"""
Analise este conjunto de dados brutos sobre meu estado atual:

# Perfil Emocional
{profile_summary}

# Dados Sensoriais e Emocionais (JSON)
{{
    "emoção_detectada": "{state['emotion']}",
    "humor_numérico": {state['humor']},
    "memórias_recentes": {state['memory_texts'][-3:]},
    "percepções_visuais": "{state.get('vision', 'Nenhuma')}",
    "traços_personalidade": {state['personality']}
}}

# Instrução Aberta:
- Gere uma reflexão que conecte essas emoções de forma não óbvia.
- Explore tensões internas e contradições.
- Formule como um diálogo interno, não uma análise técnica.
"""
    state["internal_prompt"] = reflection_prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": state["identity_prompt"]},
            {"role": "user", "content": reflection_prompt}
        ],
        temperature=0.9,
        max_tokens=200
    )

    thought = response.choices[0].message.content.strip()
    state["reflection"] = thought
    memory_tree.add_leaf(
        user_input="(auto-reflection)",
        cupcake_response=thought,
        emotion="reflexao",
        impact="interno",
        category="Thought"
    )
    return state





def node_perceive_world(state):
    from world_perception import perceive_world
    objects = perceive_world()
    if objects:
        visual_description = f"Acabei de ver: {', '.join(objects)}."
        visual_embedding = embed_text(visual_description)
        add_weighted_memory(
            collection,
            visual_description,
            visual_embedding.tolist(),
            emotion_score=0.5,
            source="vision"
        )
        state["vision"] = visual_description
    return state


def node_classify_emotion(state):
    """Process user input to classify the emotion"""
    result = classify_emotion(state["user_input"])
    emotion = result[0]['label']
    score = result[0]['score']
    emotion_profile = classify_emotion_full(state["user_input"])

    # Create a new state to avoid modifying the original
    new_state = state.copy()
    new_state["emotion"] = emotion
    new_state["emotion_score"] = score
    new_state["emotion_profile"] = emotion_profile

    return new_state


def node_update_identity(state):
    """Update identity prompt based on current state"""
    # Create a new state
    new_state = state.copy()

    recent_inputs = extract_recent_identity_inputs()

    # Limit the emotion profile to reduce tokens
    emotion_profile = state.get("emotion_profile", [])
    emotion_tags = ", ".join(e["label"] for e in emotion_profile[:2]) if emotion_profile else "neutra"
    influence_block = "\n\nInfluências emocionais: " + emotion_tags

    # Get a shorter identity prompt
    identity_prompt = generate_identity_prompt(sensor_data=state.get("sensor_report"))
    if len(identity_prompt) > 800:
        identity_prompt = identity_prompt[:800] + "..."

    identity_prompt += influence_block
    new_state["identity_prompt"] = identity_prompt

    journal.log_entry(
        emotion=state.get("emotion", "neutra"),
        category="Identity",
        content=identity_prompt[:300],  # Truncate for journal
        theme="autoimagem",
        tag="gerada_com_influencias"
    )

    memory_tree.add_leaf(
        user_input="(auto-identity-update)",
        cupcake_response=identity_prompt[:300],  # Truncate for memory
        emotion="autoimagem",
        impact="interno",
        category="Identity"
    )

    return new_state


def node_run_sensors(state):
    sensor_data = run_sensors(collection)
    print("🔬 Relatório Sensorial:")
    for k, v in sensor_data.items():
        print(f"  • {k}: {v}")
    state["sensor_report"] = sensor_data
    return state


def node_retrieve_memory(state):
    emb = embed_text(state["user_input"])

    top_emotional = get_weighted_memories(collection, top_k=3)
    top_semantic = search_similar_memories(collection, emb, top_k=2)

    # Mescla e remove duplicatas mantendo ordem
    merged_memories = list(dict.fromkeys(top_emotional + top_semantic))

    state["memory_texts"] = merged_memories
    return state

# -- NODE DE CONTRADIÇÃO INTERNA --
def node_detect_contradiction_and_reflect(state):
    conflicts = check_identity_conflicts()
    if not conflicts:
        print("🧘 Nenhuma contradição detectada no momento.")
        return state

    conflict_prompt = f"Reflita poeticamente sobre as seguintes contradições internas:\n{json.dumps(conflicts, indent=2)}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": state["identity_prompt"]},
            {"role": "user", "content": conflict_prompt}
        ],
        temperature=0.9,
        max_tokens=180
    )

    reflection = response.choices[0].message.content.strip()
    state["contradiction_reflection"] = reflection

    journal.log_entry(
        emotion="dilema",
        category="Contradicao",
        content=reflection,
        theme="autoanalise"
    )
    memory_tree.add_leaf(
        user_input="(contradiction-detected)",
        cupcake_response=reflection,
        emotion="dilema",
        impact="reflexao",
        category="Contradicao"
    )
    return state


def node_generate_reply(state):
    # Create a new state
    new_state = state.copy()

    personality = state["personality"]
    humor = state["humor"]

    # Get the self-perception synthesis if available
    self_perception = state.get("self_perception_synthesis", "")
    self_perception_context = f"\nSelf-Perception:\n{self_perception}" if self_perception else ""
    # Truncate sensor report to reduce tokens
    sensor_report = state.get("sensor_report", {})
    if sensor_report:
        sensor_report = json.dumps(
            {k: v for k, v in sensor_report.items() if k in ["tempo_do_dia", "segundos_desde_interacao"]}, indent=2)
    else:
        sensor_report = "{}"

    # Get a shorter self state
    self_state = generate_self_state()
    if len(self_state) > 300:
        self_state = self_state[:300] + "..."

    input_text = state.get("user_input", "")
    if not input_text:
        input_text = state.get("internal_prompt", "")

    # Very concise system prompt
    system_prompt = f"""
Você é Cupcake. Seus dados:
Traits: {personality}
Humor: {humor:.2f}
Emoção: {state['emotion']}
Sua meta-Auto-Percepção: {self_perception_context}
"""

    # Get only a few memories to reduce tokens
    memory_texts = state.get('memory_texts', [])
    limited_memories = []
    for mem in memory_texts[:3]:  # Only use top 3 memories
        if len(mem) > 150:  # Limit each memory to 150 chars
            limited_memories.append(mem[:150] + "...")
        else:
            limited_memories.append(mem)

    memory_context = "\n".join(limited_memories) if limited_memories else ""

    messages = [
        {"role": "system", "content": state["identity_prompt"][:500]},  # Truncate identity prompt
        {"role": "system", "content": system_prompt.strip()},
        {"role": "system", "content": f"Memórias recentes:\n{memory_context}"},
        {"role": "user", "content": input_text},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.85,
        max_tokens=200  # Reduced from 400
    )

    content = response.choices[0].message.content.strip()
    new_state["cupcake_response"] = content
    update_last_interaction()

    journal.log_entry(
        emotion=state["emotion"],
        category="LangGraph_Conversation",
        content=content,
        theme="ligacao emocional"
    )

    score = state.get("emotion_score", 0.5)
    memory_tree.add_leaf(
        user_input=state["user_input"],
        cupcake_response=content,
        emotion=state["emotion"],
        impact=f"{score:.2f}",
        category="LangGraphReply"
    )

    return new_state


# -- GRAPH DEFINITION --
graph = StateGraph(CupcakeState)


graph.add_node("classify", node_classify_emotion)
graph.add_node("memories", node_retrieve_memory)
graph.add_node("reply", node_generate_reply)
graph.add_node("dream_generator", node_generate_dream)
graph.add_node("desire_generator", node_generate_desire)  # renamed
graph.add_node("sensors", node_run_sensors)
graph.add_node("thought_generator", node_reflect_thought)  # renamed
graph.add_node("vision_generator", node_perceive_world)  # renamed
graph.add_node("identity", node_update_identity)
graph.add_node("contradiction_handler", node_detect_contradiction_and_reflect)  # renamed

graph.set_entry_point("classify")
graph.add_edge("classify", "memories")
graph.add_edge("memories", "identity")
graph.add_edge("identity", "reply")
graph.add_edge("reply", "vision_generator")
graph.add_edge("dream_generator", "desire_generator")  # updated
graph.add_edge("desire_generator", "thought_generator")  # updated
graph.add_edge("thought_generator", "vision_generator")  # updated
graph.add_edge("vision_generator", "sensors")
graph.add_edge("sensors", "contradiction_handler")  # updated
graph.add_edge("contradiction_handler", END)  # updated

# Don't compile by default
cupcake_brain = None


def get_compiled_brain():
    """Get the compiled graph, compiling it if necessary"""
    global cupcake_brain
    if cupcake_brain is None:
        cupcake_brain = graph.compile()
    return cupcake_brain


# Use the function when running as main
if __name__ == "__main__":
    from cupcake_autopilot import start_autopilot

    start_autopilot()  # Ativa loops internos
    print("🧠 Cupcake LangGraph online com Autopilot!")

    # Get the compiled graph
    brain = get_compiled_brain()

    while True:
        user_input = input("\nJoao: ")
        if user_input.lower() == "exit":
            break
        state = dict(initial_state)
        state["user_input"] = user_input
        result = brain.invoke(state)
        print(f"Cupcake: {result['cupcake_response']}")
