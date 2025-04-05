# memory_integration.py
from datetime import datetime, timedelta
import random
import numpy as np
from langgraph.graph import StateGraph, END
from enhanced_memory_weighting import EnhancedMemoryWeighting
from memory_tiering import TieredMemorySystem

# Initialize (outside the function)
tiered_memory = None
_enhanced_memory = None
_tiered_memory = None
_embed_text = None
_collection = None


def initialize_tiered_memory(enhanced_memory, embed_fn, collection):
    global tiered_memory
    tiered_memory = TieredMemorySystem(enhanced_memory, embed_fn, collection)
    return tiered_memory


def node_retrieve_enhanced_memories(state):
    """Retrieve memories with enhanced emotional weighting"""
    global tiered_memory, _enhanced_memory, _embed_text, _collection

    # Get tiered memory or initialize if needed
    # In node_retrieve_enhanced_memories function
    if tiered_memory is None:
        if _collection is None or _enhanced_memory is None or _embed_text is None:
            # Import from the main file
            from narrative_enhanced_cupcake import collection, enhanced_memory, embed_text
            tiered_memory = TieredMemorySystem(enhanced_memory, embed_text, collection)
        else:
            tiered_memory = TieredMemorySystem(_enhanced_memory, _embed_text, _collection)

    # Use tiered memory system for retrieval
    if state.get("user_input", "").startswith("/searchmemory "):
        search_query = state["user_input"].replace("/searchmemory ", "")
        memories = tiered_memory.handle_deep_memory_search(search_query)
        state["memory_search_mode"] = True
    else:
        memories = tiered_memory.retrieve_mixed_memories(state)

    # Extract just the document texts for the state
    memory_texts = [doc for doc, _ in memories]
    memory_metadatas = [meta for _, meta in memories]

    # Update state with retrieved memories
    state["memory_texts"] = memory_texts
    state["memory_metadatas"] = memory_metadatas

    # Store memory pattern analysis
    if _enhanced_memory:
        memory_patterns = _enhanced_memory.detect_emotional_patterns()
        state["memory_patterns"] = memory_patterns

        # Adjust memory weights based on current emotional state
        if "emotion" in state and "emotion_score" in state:
            emotion_data = {
                'emotion_type': state["emotion"],
                'score': state["emotion_score"]
            }
            _enhanced_memory.adjust_emotional_weights_based_on_mood(emotion_data)

    return state


def node_add_enhanced_memory(state):
    """Add current interaction to memory with enhanced emotional weighting"""
    global _enhanced_memory

    # Use local or global enhanced_memory
    enhanced_memory_instance = _enhanced_memory
    if enhanced_memory_instance is None:
        # Fallback to get it from the module
        from narrative_enhanced_cupcake import enhanced_memory
        enhanced_memory_instance = enhanced_memory

    if not state.get("cupcake_response"):
        return state  # No response to store yet

    user_input = state.get("user_input", "")
    cupcake_response = state.get("cupcake_response", "")

    # Store if we have a query embedding
    if "query_embedding" in state:
        # Create emotion data from state
        emotion_data = {
            'emotion_type': state.get("emotion", "neutral"),
            'score': state.get("emotion_score", 0.5)
        }

        # Add more detailed emotion data if available
        if "emotion_profile" in state and state["emotion_profile"]:
            for emotion_info in state["emotion_profile"]:
                if emotion_info["label"] == state["emotion"]:
                    emotion_data['valence'] = max(0.1, min(0.9, emotion_info["score"]))
                    emotion_data['arousal'] = max(0.3, min(0.9, emotion_info["score"] + 0.1))

        # Determine if this is a self-referential memory
        self_referential = False
        identity_terms = ["eu", "minha", "me", "mim", "eu sou", "eu estou", "cupcake"]
        for term in identity_terms:
            if term in cupcake_response.lower():
                self_referential = True
                break

        # Determine narrative relevance
        narrative_relevance = 0.5  # Default medium relevance
        if "identity_prompt" in state and state["identity_prompt"]:
            # Check if response contains terms from identity
            identity = state["identity_prompt"].lower()
            relevant_terms = ["consciência", "sentir", "emoção", "identidade", "evolução", "percepção"]
            relevance_count = 0
            for term in relevant_terms:
                if term in cupcake_response.lower() or term in user_input.lower():
                    relevance_count += 1
            if relevance_count > 0:
                narrative_relevance = min(0.9, 0.5 + relevance_count * 0.1)

        # Add conversation to memory
        memory_text = f"Usuário: {user_input}\nCupcake: {cupcake_response}"
        enhanced_memory_instance.add_weighted_memory(
            text=memory_text,
            embedding=state["query_embedding"],
            emotion_data=emotion_data,
            source="conversation",
            narrative_relevance=narrative_relevance,
            self_reference=self_referential
        )

    return state


def node_memory_cluster_analysis(state):
    """
    Periodically analyze memory clusters for patterns
    Only runs occasionally (not every interaction)
    """
    global _enhanced_memory

    # Use local or global enhanced_memory
    enhanced_memory_instance = _enhanced_memory
    if enhanced_memory_instance is None:
        # Fallback to get it from the module
        from memory_enhanced_cupcake import enhanced_memory
        enhanced_memory_instance = enhanced_memory

    # Run memory cluster analysis about 10% of the time
    if random.random() < 0.1:
        print("📊 Analisando clusters de memórias emocionais...")

        # Get memory clusters
        clusters = enhanced_memory_instance.get_memory_clusters_by_emotion(min_memories=2)

        # Store in state for potential use in response
        state["memory_clusters"] = {
            emotion: [doc for doc, _ in mem_list[:3]]  # Use the specific variable name
            for emotion, mem_list in clusters.items()
        }

        # Check for amplification of emotional clusters
        # If a specific emotion is current and has a cluster, amplify it
        current_emotion = state.get("emotion")
        if current_emotion in clusters and random.random() < 0.3:
            enhanced_memory_instance.amplify_memory_cluster(current_emotion, amplification_factor=1.2)
            print(f"🔆 Amplificando cluster emocional: {current_emotion}")

    return state


def integrate_enhanced_memory(original_graph, collection):
    """
    Integrate the enhanced memory weighting system with the existing LangGraph
    """
    # Store as global references for nodes to access
    global _enhanced_memory, _tiered_memory, _embed_text, _collection
    _collection = collection

    # Initialize the enhanced memory system
    _enhanced_memory = EnhancedMemoryWeighting(collection)

    # Initialize the tiered memory system - with proper function references
    from memory_tiering import TieredMemorySystem

    # Import the embed_text function from the appropriate module
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(text):
        return embed_model.encode(text)

    # Store embed_text function for use in nodes
    _embed_text = embed_text

    # Now initialize with the proper references
    _tiered_memory = TieredMemorySystem(_enhanced_memory, _embed_text, collection)

    # Make available to other functions
    global tiered_memory
    tiered_memory = _tiered_memory

    # Add the new nodes to the graph
    original_graph.add_node("retrieve_enhanced_memories", node_retrieve_enhanced_memories)
    original_graph.add_node("add_enhanced_memory", node_add_enhanced_memory)
    original_graph.add_node("memory_cluster_analysis", node_memory_cluster_analysis)

    # Modify the flow to use our new nodes
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # Replace memory retrieval nodes
            if edge[1] == "memories":
                incoming_edge = edge[0]
                # Find the outgoing edge from memories
                outgoing_edge = None
                for mem_edge in original_graph.edges:
                    if mem_edge[0] == "memories":
                        outgoing_edge = mem_edge[1]
                        break

                if outgoing_edge:
                    # Add our enhanced memory retrieval chain
                    new_edges.add((incoming_edge, "retrieve_enhanced_memories"))
                    new_edges.add(("retrieve_enhanced_memories", "memory_cluster_analysis"))
                    new_edges.add(("memory_cluster_analysis", outgoing_edge))
                else:
                    # Keep original edge if we can't find outgoing edge
                    new_edges.add(edge)
            # Add memory storage after response generation
            elif edge[0] == "reply":
                next_node = edge[1]
                # Insert enhanced memory storage
                new_edges.add(("reply", "add_enhanced_memory"))
                new_edges.add(("add_enhanced_memory", next_node))
            elif edge[1] != "memories" and edge[0] != "memories":
                # Keep edges that aren't related to memories
                new_edges.add(edge)

        # Replace all edges in the graph
        original_graph.edges = new_edges

    except Exception as e:
        print(f"Error modifying graph for enhanced memory: {e}")

    return original_graph

# Example usage
# from langgraph_cupcake import graph as original_graph
# import chromadb
# client_db = chromadb.PersistentClient(path="./cupcake_memory_db")
# collection = client_db.get_collection(name='cupcake_memory')
# enhanced_graph = integrate_enhanced_memory(original_graph, collection)
# cupcake_brain = enhanced_graph.compile()