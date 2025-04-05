# cupcake_langgraph_integration.py
from langgraph.graph import StateGraph, END
from cognitive_architecture import CognitiveArchitecture
from cupcake_enhanced_dreamer import EnhancedDreamer
from self_perception import SelfPerceptionLayer
import json

# Initialize components
cognitive_arch = CognitiveArchitecture()
dreamer = EnhancedDreamer()
self_perception = SelfPerceptionLayer()


def integrate_with_langgraph(original_graph):
    """
        Integrate the cognitive architecture, enhanced dreamer, and self-perception layer
        with the existing LangGraph
        """

    # Define new nodes for the graph
    def node_self_perception(state):
        """Process self-perception in relation to user input"""
        return self_perception.process_perception(state)

    def node_cognitive_perspectives(state):
        """Generate cognitive perspectives from each personality dimension"""
        cupcake_response, perspectives = cognitive_arch.process_interaction(state)
        state["cupcake_response"] = cupcake_response
        state["cognitive_perspectives"] = perspectives
        return state

    def node_process_emotions_for_dreamer(state):
        """Process user input for emotional content and log it for the dreamer"""
        emotions = dreamer.process_input_for_emotions(state["user_input"])
        state["dreamer_emotions"] = emotions
        return state

    def node_enhanced_dream(state):
        """Generate a dream using the enhanced dreamer"""
        dream_content, dream_metadata = dreamer.generate_and_log_dream()
        state["dream"] = dream_content
        state["dream_metadata"] = dream_metadata
        return state

    # Add the new nodes to the graph
    original_graph.add_node("self_perception", node_self_perception)
    original_graph.add_node("cognitive_perspectives", node_cognitive_perspectives)
    original_graph.add_node("process_emotions_for_dreamer", node_process_emotions_for_dreamer)
    original_graph.add_node("enhanced_dream", node_enhanced_dream)

    # Modify the flow to use our new nodes
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # 1. Modify classify -> next connection
            if edge[0] == "classify":
                next_node = edge[1]
                # Replace with emotion processing chain
                new_edges.add(("classify", "process_emotions_for_dreamer"))
                new_edges.add(("process_emotions_for_dreamer", next_node))
            # 2. Modify reply -> next connection
            elif edge[0] == "reply":
                next_node = edge[1]
                # Replace with cognitive perspectives chain
                new_edges.add(("reply", "cognitive_perspectives"))
                new_edges.add(("cognitive_perspectives", next_node))
            # 3. Modify dream -> next connection
            elif edge[0] == "dream":
                next_node = edge[1]
                # Replace with enhanced dream chain
                new_edges.add(("dream", "enhanced_dream"))
                new_edges.add(("enhanced_dream", next_node))
            else:
                # Keep other edges unchanged
                new_edges.add(edge)

        # Replace all edges in the graph
        original_graph.edges = new_edges

    except Exception as e:
        print(f"Error modifying graph for LangGraph integration: {e}")

    return original_graph

# Usage example
# from langgraph_cupcake import graph as original_graph
# enhanced_graph = integrate_with_langgraph(original_graph)
# cupcake_brain = enhanced_graph.compile()