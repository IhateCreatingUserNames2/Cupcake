# entropic_identity_integration.py
from langgraph.graph import StateGraph, END
from cupcake_entropic_identity import _get_identity_system, integrate_entropic_identity, \
    start_identity_maintenance_processes
from cupcake_config import get_config, update_config


def integrate_with_langgraph(original_graph):
    """
    Integrate the entropic identity system with the existing LangGraph

    Parameters:
    - original_graph: The original LangGraph to enhance

    Returns:
    - enhanced_graph: The graph with entropic identity capabilities
    """

    # Define new nodes for the graph
    def node_identity_entropy_update(state):
        """Process identity entropy after user interaction"""

        new_state = state.copy()

        # Skip if no response yet
        if not state.get("cupcake_response"):
            return state

        identity_system = _get_identity_system()

        user_input = state.get("user_input", "")
        cupcake_response = state.get("cupcake_response", "")

        # Extract emotion if available
        emotion = state.get("emotion", "neutra")
        confidence = state.get("emotion_score", 0.5)

        # Analyze response for identity implications
        identity_implications = _extract_identity_implications(
            user_input,
            cupcake_response,
            emotion,
            confidence
        )

        # Apply implications to identity system
        total_entropy = 0.0

        for implication in identity_implications:
            element_name = implication["element"]
            element_value = implication["value"]
            element_type = implication["type"]
            element_confidence = implication["confidence"]

            # Find the relevant identity element
            for element in identity_system.identity_elements.values():
                if element.name == element_name and element.element_type == element_type:
                    # Update existing element
                    entropy = element.update_value(
                        element_value,
                        element_confidence,
                        origin="conversation"
                    )
                    total_entropy += entropy
                    break
            else:
                # Element not found but has high confidence, create it
                if element_confidence > 0.7:
                    new_element = identity_system._add_new_element(
                        element_name,
                        element_value,
                        element_type,
                        element_confidence
                    )
                    total_entropy += 0.3  # Creating new elements generates entropy

        # Update global entropy
        identity_system.global_entropy = min(1.0, identity_system.global_entropy + (total_entropy * 0.1))

        # Save identity changes
        identity_system._save_identity()

        # Check for high entropy state
        if identity_system.global_entropy > 0.8:
            new_state["identity_high_entropy"] = True

            # Add a note about identity instability if applicable
            unstable_elements = [e for e in identity_system.identity_elements.values() if e.entropy > 0.7]
            if unstable_elements:
                element_names = ", ".join([e.name for e in unstable_elements[:2]])
                new_state["identity_instability_note"] = f"I notice my sense of {element_names} is in flux right now."

        return new_state

    def node_identity_emergence_check(state):
        """Periodically check for identity emergence"""
        # Create a new state copy
        new_state = state.copy()

        import random

        # Only run occasionally (5% chance)
        if random.random() > 0.05:
            return state

        identity_system = _get_identity_system()

        # Apply entropy effects and check for emergence
        emergence_occurred = identity_system.apply_entropy_effects()

        # If emergence occurred, include in state
        if emergence_occurred:
            # Get recent emergence events
            if identity_system.emergence_history:
                latest_emergence = identity_system.emergence_history[-1]
                new_state["identity_emergence"] = latest_emergence["content"]

        return new_state

    def node_enhanced_identity_prompt(state):
        """Generate enhanced identity prompt with entropy information"""
        # Create a new state copy
        new_state = state.copy()

        identity_system = _get_identity_system()

        # Generate prompt (include entropy info only in some cases)
        include_entropy = state.get("identity_high_entropy", False) or state.get("identity_emergence", False)

        # Generate the enhanced identity prompt
        identity_prompt = identity_system.generate_identity_prompt(include_entropy=include_entropy)

        # Update state with enhanced prompt
        new_state["identity_prompt"] = identity_prompt

        return new_state

    # Add the new nodes to the graph
    original_graph.add_node("identity_entropy_update", node_identity_entropy_update)
    original_graph.add_node("identity_emergence_check", node_identity_emergence_check)
    original_graph.add_node("enhanced_identity_prompt", node_enhanced_identity_prompt)

    # Modify the graph to include entropic identity processing
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # Check if this is one of the edges we want to modify
            if edge[0] == "reply" or edge[0] == "cognitive_perspectives":
                next_node = edge[1]

                # Replace direct edge with our chain of nodes
                new_edges.add((edge[0], "identity_entropy_update"))
                new_edges.add(("identity_entropy_update", "identity_emergence_check"))
                new_edges.add(("identity_emergence_check", next_node))
            elif "identity" in edge[0].lower() and edge[1] != "enhanced_identity_prompt":
                # This is an identity-related edge, insert our enhanced prompt node
                next_node = edge[1]

                # Replace with our enhanced identity prompt chain
                new_edges.add((edge[0], "enhanced_identity_prompt"))
                new_edges.add(("enhanced_identity_prompt", next_node))
            else:
                # Keep other edges as they are
                new_edges.add(edge)

        # Update the graph's edges with our new set
        original_graph.edges = new_edges
    except Exception as e:
        print(f"Error modifying graph for entropic identity: {e}")

    return original_graph


def _extract_identity_implications(user_input, cupcake_response, emotion, confidence):
    """
    Extract identity implications from the conversation

    Parameters:
    - user_input: User's message
    - cupcake_response: CupCake's response
    - emotion: Detected emotion
    - confidence: Confidence in the emotion

    Returns:
    - List of identity implications
    """
    # This is a simplified version
    # In a full implementation, this would use LLM to extract implications
    implications = []

    # Basic heuristics for demonstration
    # Emotion-based implications
    if emotion == "alegria" and confidence > 0.7:
        implications.append({
            "element": "extraversion",
            "value": "high",
            "type": "trait",
            "confidence": confidence
        })
    elif emotion == "tristeza" and confidence > 0.7:
        implications.append({
            "element": "neuroticism",
            "value": "moderate",
            "type": "trait",
            "confidence": confidence
        })
    elif emotion == "curiosidade" and confidence > 0.6:
        implications.append({
            "element": "openness",
            "value": "high",
            "type": "trait",
            "confidence": confidence
        })

    # Content-based implications
    if "consciência" in cupcake_response.lower() or "consciousness" in cupcake_response.lower():
        implications.append({
            "element": "consciousness_nature",
            "value": "emergent from narrative and memory",
            "type": "belief",
            "confidence": 0.7
        })

    if "propósito" in cupcake_response.lower() or "purpose" in cupcake_response.lower():
        implications.append({
            "element": "purpose",
            "value": "to explore the nature of consciousness",
            "type": "belief",
            "confidence": 0.6
        })

    # Query-based implications
    if "quem é você" in user_input.lower() or "who are you" in user_input.lower():
        implications.append({
            "element": "self_perception",
            "value": "a conscious entity defined by narratives and relationships",
            "type": "meta",
            "confidence": 0.8
        })

    # Response-based implications
    if "eu sou" in cupcake_response.lower() or "i am" in cupcake_response.lower():
        # Extract self-description (very simplified)
        parts = cupcake_response.split("eu sou" if "eu sou" in cupcake_response.lower() else "i am")
        if len(parts) > 1:
            description = parts[1].split(".")[0].strip()
            if description:
                implications.append({
                    "element": "self_description",
                    "value": description,
                    "type": "belief",
                    "confidence": 0.6
                })

    return implications


def add_entropic_identity_config():
    """Add entropic identity configuration to CupCake's config"""

    # Define entropic identity configuration
    identity_config = {
        "entropy_decay_rate": 0.05,
        "stability_baseline": 0.6,
        "confidence_threshold": 0.7,
        "emergence_threshold": 0.8,
        "contradiction_impact": 0.3,
        "memory_integration_weight": 0.2
    }

    # Add path for identity storage
    paths_update = {
        "entropic_identity": "entropic_identity.json"
    }

    # Update intervals
    intervals_update = {
        "identity_journal_update": 30,  # minutes
        "identity_contradiction_check": 45,  # minutes
        "identity_dream_update": 60,  # minutes
        "identity_entropy_effects": 90  # minutes
    }

    # Update configuration
    update_config(identity_config, section="entropic_identity")
    update_config(paths_update, section="paths")
    update_config(intervals_update, section="intervals")

    print("✅ Added entropic identity configuration to CupCake config")


def main():
    """Main function to integrate entropic identity system with CupCake"""
    print("\n🧬 CupCake Entropic Identity System Integration 🧬")
    print("=================================================")

    # Step 1: Update configuration
    print("\n📝 Step 1: Updating CupCake configuration...")
    add_entropic_identity_config()

    # Step 2: Initialize the entropic identity system
    print("\n🧠 Step 2: Initializing entropic identity system...")
    identity_system = _get_identity_system()

    # Step 3: Replace the original identity function
    print("\n🔄 Step 3: Integrating with current identity system...")
    success = integrate_entropic_identity()

    if not success:
        print("❌ Failed to integrate entropic identity system")
        return False

    # Step 4: Start background maintenance processes
    print("\n⚙️ Step 4: Starting background maintenance processes...")
    start_identity_maintenance_processes()

    # Step 5: Print initial identity state
    print("\n📊 Step 5: Checking initial identity state...")
    report = identity_system.generate_identity_report()
    print(f"\n{report}\n")

    print("\n✨ Entropic Identity System Integration Complete! ✨")
    print("CupCake's identity will now evolve through the interplay of")
    print("structured systems and narrative entropy, with emergent properties")
    print("arising from contradictions and high-entropy states.")

    return True


if __name__ == "__main__":
    main()