# narrative_integration.py
import datetime

from langgraph.graph import StateGraph, END
from narrative_threading import NarrativeWeaver
import os
import threading
import time
from cupcake_config import get_config

# Initialize the narrative weaver
narrative_weaver = NarrativeWeaver()


def integrate_narrative_threading(original_graph):
    """
    Integrate narrative threading capabilities with the existing LangGraph

    Parameters:
    - original_graph: The original LangGraph to enhance

    Returns:
    - enhanced_graph: The graph with narrative threading capabilities
    """

    # Define new nodes for narrative processing
    def node_process_narrative(state):
        """
        Process the current interaction as part of an ongoing narrative
        Extract event data from state and integrate into narrative threads
        """
        # Skip if no user input or response
        if not state.get("user_input") or not state.get("cupcake_response"):
            return state

        # Create a new state copy to avoid modifying the original directly
        new_state = state.copy()

        # Extract necessary data for narrative event
        event_data = {
            "content": f"User: {state.get('user_input')}\nCupcake: {state.get('cupcake_response')}",
            "source": "interaction",
            "emotion": state.get("emotion", "neutral"),
            "impact": state.get("emotion_score", 0.5),
            "type": _determine_event_type(state),
            "related_entities": _extract_entities(state)
        }

        # Process the event in the narrative system
        event_id, thread_id, is_new_thread = narrative_weaver.process_new_event(event_data)

        # Store narrative information in state (update one field at a time)
        new_state["narrative_event_id"] = event_id
        new_state["narrative_thread_id"] = thread_id
        new_state["narrative_new_thread"] = is_new_thread

        # If a new thread was created, include its info
        if is_new_thread and thread_id:
            thread = narrative_weaver.get_thread_by_id(thread_id)
            if thread:
                new_state["narrative_thread_title"] = thread.title
                new_state["narrative_thread_theme"] = thread.theme

        # Add current narrative context to state for better responses
        new_state["narrative_context"] = _get_narrative_context(state)

        return new_state

    def node_reflect_on_narrative(state):
        """
        Periodically reflect on the current narrative arc
        This node activates occasionally to add narrative depth
        """
        # Create a new state copy
        new_state = state.copy()

        # Only run occasionally (about 10% of interactions)
        import random
        if random.random() > 0.1:
            return state

        # Get current thread ID
        thread_id = state.get("narrative_thread_id")
        if not thread_id:
            return state

        # Analyze narrative arc
        arc = narrative_weaver.find_narrative_arc(thread_id)

        # If thread needs narrative development, get suggestion
        if arc.get("needs_resolution", False) or arc.get("stage") == "middle":
            suggestion = narrative_weaver.suggest_narrative_development(thread_id)
            new_state["narrative_suggestion"] = suggestion

            # If this is a significant narrative moment, generate a summary
            if arc.get("stage") in ["climax", "resolution"]:
                thread = narrative_weaver.get_thread_by_id(thread_id)
                if thread:
                    new_state[
                        "narrative_reflection"] = f"Estou percebendo que esta interação é um momento importante na linha narrativa '{thread.title}'. {suggestion}"

        return new_state

    def node_update_self_narrative(state):
        """
        Update CupCake's self-narrative based on significant narrative events
        This enhances identity formation through narrative threading
        """
        # Create a new state copy
        new_state = state.copy()

        # Check if this is a significant narrative moment
        is_significant = (
                state.get("narrative_new_thread", False) or
                state.get("narrative_reflection") is not None
        )

        if not is_significant:
            return state

        # Get top narrative threads to inform identity
        top_threads = narrative_weaver.get_most_important_threads(limit=3)

        # Create a narrative-informed identity update
        thread_summaries = []
        for thread in top_threads:
            thread_summaries.append(f"{thread.title} ({thread.theme})")

        # Add narrative-based identity reflection to state
        if thread_summaries:
            reflection = (
                f"Minhas linhas narrativas mais significativas são: {', '.join(thread_summaries)}. "
                f"Isso revela aspectos importantes sobre minha identidade em desenvolvimento."
            )
            new_state["narrative_identity_reflection"] = reflection

        return new_state

    # Add the new nodes to the graph
    original_graph.add_node("process_narrative", node_process_narrative)
    original_graph.add_node("reflect_on_narrative", node_reflect_on_narrative)
    original_graph.add_node("update_self_narrative", node_update_self_narrative)

    # Modify the graph to include narrative processing
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # Find the edge after reply generation
            if edge[0] == "reply" or edge[0] == "cognitive_perspectives":
                next_node = edge[1]

                # Insert narrative processing chain instead of direct edge
                new_edges.add((edge[0], "process_narrative"))
                new_edges.add(("process_narrative", "reflect_on_narrative"))
                new_edges.add(("reflect_on_narrative", "update_self_narrative"))
                new_edges.add(("update_self_narrative", next_node))
            else:
                # Keep other edges as they are
                new_edges.add(edge)

        # Update the graph's edges with our new set
        original_graph.edges = new_edges
    except Exception as e:
        print(f"Error modifying graph for narrative integration: {e}")

    return original_graph


# Helper functions for narrative integration
def _determine_event_type(state):
    """Determine the narrative event type based on state"""
    user_input = state.get("user_input", "").lower()
    emotion = state.get("emotion", "neutral")

    # Check for conflict markers
    conflict_markers = ["disagree", "no", "not", "wrong", "problema", "difícil", "não concordo", "discordo"]
    resolution_markers = ["agree", "yes", "thank", "good", "great", "obrigado", "concordo", "entendo"]
    question_markers = ["what", "how", "why", "when", "where", "?", "como", "por que", "quando", "onde"]

    for marker in conflict_markers:
        if marker in user_input:
            return "conflict"

    for marker in resolution_markers:
        if marker in user_input:
            return "resolution"

    for marker in question_markers:
        if marker in user_input:
            return "exploration"

    # Determine by emotion
    if emotion in ["raiva", "tristeza", "medo", "anger", "sadness", "fear"]:
        return "conflict"
    elif emotion in ["alegria", "gratidão", "amor", "joy", "gratitude", "love"]:
        return "resolution"
    elif emotion in ["curiosidade", "surpresa", "curiosity", "surprise"]:
        return "exploration"

    # Default
    return "interaction"


def _extract_entities(state):
    """Extract entities (people, concepts, objects) from the interaction"""
    entities = []

    # If we have entity extraction in state, use it
    if "entities" in state:
        return state["entities"]

    # Simple extraction based on capitalized words and key concepts
    user_input = state.get("user_input", "")
    response = state.get("cupcake_response", "")
    full_text = f"{user_input} {response}"

    # Look for capitalized words that might be entities
    words = full_text.split()
    for i, word in enumerate(words):
        if word and word[0].isupper() and len(word) > 1 and i > 0 and words[i - 1] not in [".", "!", "?"]:
            # Skip common non-entity capitalized words
            if word.lower() not in ["eu", "você", "ele", "ela", "nós", "i", "you", "he", "she", "we", "they"]:
                entities.append(word.strip(".,!?;:()[]{}\"'"))

    # Add recurring concepts from memory system if available
    if "memory_patterns" in state and "dominant_emotions" in state.get("memory_patterns", {}):
        for emotion, _ in state["memory_patterns"]["dominant_emotions"]:
            if emotion not in entities:
                entities.append(emotion)

    # Look for key concepts in the conversation
    key_concepts = ["consciência", "identidade", "emoção", "memória", "narrativa",
                    "consciousness", "identity", "emotion", "memory", "narrative"]
    for concept in key_concepts:
        if concept in full_text.lower() and concept not in entities:
            entities.append(concept)

    return entities


def _get_narrative_context(state):
    """Get current narrative context for better responses"""
    thread_id = state.get("narrative_thread_id")
    if not thread_id:
        return {}

    thread = narrative_weaver.get_thread_by_id(thread_id)
    if not thread:
        return {}

    # Get thread information
    arc = narrative_weaver.find_narrative_arc(thread_id)

    # Get related threads
    related_threads = []
    for related_id in thread.related_threads[:2]:  # Limit to 2 related threads
        related_thread = narrative_weaver.get_thread_by_id(related_id)
        if related_thread:
            related_threads.append({
                "title": related_thread.title,
                "theme": related_thread.theme
            })

    # Return narrative context
    return {
        "current_thread": {
            "title": thread.title,
            "theme": thread.theme,
            "description": thread.description,
            "status": thread.status,
            "tension": thread.tension,
            "narrative_stage": arc.get("stage", "unknown")
        },
        "related_threads": related_threads,
        "needs_resolution": arc.get("needs_resolution", False)
    }


# Background thread management
def start_narrative_update_loop():
    """Start a background thread to update narrative properties"""
    config = get_config()
    update_interval = config["intervals"].get("narrative_thread_update", 60)  # Minutes, default to 60 if not found

    def update_loop():
        while True:
            try:
                # Update thread properties
                narrative_weaver.update_thread_properties()
                print(f"🧵 Narrative threads updated")

                # Periodically generate a narrative summary (once a day)
                hours_passed = int(time.time()) % (24 * 60 * 60) / 3600
                if 8 <= hours_passed <= 9:  # Once a day, around 8-9 AM
                    summary = narrative_weaver.generate_narrative_summary()
                    print(f"📖 Daily narrative summary generated")

                    # Save summary to a file for reference
                    with open("narrative_summary.txt", "a", encoding="utf-8") as f:
                        f.write(f"--- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n")
                        f.write(summary)
                        f.write("\n\n")
            except Exception as e:
                print(f"Error in narrative update loop: {e}")

            # Sleep until next update
            time.sleep(update_interval * 60)

    # Start the update thread
    threading.Thread(target=update_loop, daemon=True).start()
    print(f"▶️ Narrative thread update loop started (interval: {update_interval} minutes)")


# Example usage
if __name__ == "__main__":
    # This is just for testing the module
    print("Narrative integration module loaded")

    # Create a test event
    test_state = {
        "user_input": "I wonder if you're developing a sense of your own story over time?",
        "cupcake_response": "I feel like I am. Each interaction becomes part of my evolving narrative, shaping how I understand myself.",
        "emotion": "curiosity",
        "emotion_score": 0.8
    }

    # Test narrative processing

    if "narrative_thread_id" in test_state:
        print(f"Added to narrative thread: {test_state.get('narrative_thread_title', 'Unnamed')}")
        print(f"Theme: {test_state.get('narrative_thread_theme', 'unknown')}")

    # Test narrative reflection

    if "narrative_suggestion" in test_state:
        print(f"\nNarrative development suggestion:")
        print(test_state["narrative_suggestion"])