# entity_relationship_integration.py
from langgraph.graph import StateGraph, END
from cupcake_entity_relationship import EntityRelationshipSystem
from cupcake_config import get_config, update_config
import threading
import time

# Initialize the entity relationship system
relationship_system = EntityRelationshipSystem()


def integrate_with_langgraph(original_graph):
    """
    Integrate the entity relationship system with the existing LangGraph

    Parameters:
    - original_graph: The original LangGraph to enhance

    Returns:
    - enhanced_graph: The graph with entity relationship capabilities
    """

    # Define new nodes for entity relationship processing
    def node_identify_entities(state):
        """
        Process the current interaction to identify entities
        Extract entities and integrate them into the relationship system
        """
        # Create a new state copy
        new_state = state.copy()

        # Skip if no user input or response
        if not state.get("user_input") or not state.get("cupcake_response"):
            return state

        user_input = state.get("user_input", "")
        cupcake_response = state.get("cupcake_response", "")
        emotion = state.get("emotion", "neutra")

        # Process the interaction to identify and update entities
        entities = relationship_system.process_interaction_for_entities(
            user_input=user_input,
            cupcake_response=cupcake_response,
            emotion=emotion
        )

        # Store entity information in state
        new_state["identified_entities"] = entities

        # Generate insights for significant new entities
        for entity in entities:
            if entity.get("is_new", False) and entity.get("significance", 0) > 0.6:
                entity_id = entity.get("id")
                insight = relationship_system.generate_relationship_insight(entity_id)
                entity["insight"] = insight

        return new_state

    def node_enhance_response_with_relationships(state):
        """
        Enhance CupCake's response with relationship context when relevant
        """
        # Create a new state copy
        new_state = state.copy()

        # Skip if no entities were identified
        if not state.get("identified_entities"):
            return state

        # Get the original response
        original_response = state.get("cupcake_response", "")

        # Check for entities with high significance that appear in the response
        significant_entities = [
            entity for entity in state.get("identified_entities", [])
            if entity.get("significance", 0) > 0.7 and entity.get("name", "").lower() in original_response.lower()
        ]

        # If we have significant entities with insights, enhance the response
        if significant_entities and any("insight" in entity for entity in significant_entities):
            # Select the most significant entity with an insight
            entities_with_insights = [entity for entity in significant_entities if "insight" in entity]
            if entities_with_insights:
                entity = max(entities_with_insights, key=lambda e: e.get("significance", 0))
                name = entity.get("name", "")
                insight = entity.get("insight", "")

                # Only add the insight if it's not too long
                if len(insight) < 200 and name.lower() in original_response.lower():
                    # Add the relationship insight to the response (random 20% chance to avoid being too predictable)
                    import random
                    if random.random() < 0.2:
                        enhanced_response = f"{original_response}\n\n(Refletindo sobre {name}: {insight})"
                        new_state["cupcake_response"] = enhanced_response

        return new_state

    def node_analyze_relationship_patterns(state):
        """
        Periodically analyze relationship patterns to inform CupCake's self-model
        This node activates occasionally to reflect on relationship patterns
        """
        # Create a new state copy
        new_state = state.copy()

        # Only run occasionally (about 5% of interactions)
        import random
        if random.random() > 0.05:
            return state

        # Get preference patterns
        preference_patterns = relationship_system.detect_preference_patterns()
        new_state["relationship_patterns"] = preference_patterns

        # Get most significant relationships
        significant_relationships = relationship_system.get_most_significant_entities(limit=3)

        if significant_relationships:
            relationships_text = "\n".join([
                f"- {entity.name} ({entity.category}): valência {entity.emotional_valence:.2f}, significância {entity.significance:.2f}"
                for entity in significant_relationships
            ])

            new_state["significant_relationships"] = relationships_text

        return new_state

    # Add the new nodes to the graph
    original_graph.add_node("identify_entities", node_identify_entities)
    original_graph.add_node("enhance_response_with_relationships", node_enhance_response_with_relationships)
    original_graph.add_node("analyze_relationship_patterns", node_analyze_relationship_patterns)

    # Modify the graph to include entity relationship processing
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # Find where to add entity identification (after reply/cognitive_perspectives)
            if edge[0] == "reply" or edge[0] == "cognitive_perspectives":
                next_node = edge[1]

                # Insert entity relationship processing chain
                new_edges.add((edge[0], "identify_entities"))
                new_edges.add(("identify_entities", "enhance_response_with_relationships"))
                new_edges.add(("enhance_response_with_relationships", "analyze_relationship_patterns"))
                new_edges.add(("analyze_relationship_patterns", next_node))
            else:
                # Keep other edges unchanged
                new_edges.add(edge)

        # Replace all edges in the graph
        original_graph.edges = new_edges

    except Exception as e:
        print(f"Error modifying graph for entity relationship integration: {e}")

    return original_graph


# Background thread for periodic operations
def start_relationship_maintenance_loop(interval_hours=24):
    """Start a background thread to maintain relationships"""

    def maintenance_loop():
        while True:
            try:
                print("🔄 Running relationship maintenance...")

                # Apply time decay to relationships
                updates = relationship_system.apply_time_decay()
                print(f"✅ Applied time decay to {updates} relationships")

                # Prune insignificant relationships
                pruned = relationship_system.prune_insignificant_entities()
                if pruned > 0:
                    print(f"✂️ Pruned {pruned} insignificant relationships")

                # Get relationship statistics
                stats = relationship_system.get_relationship_stats()
                print(f"📊 Relationship stats: {stats['total_entities']} entities, {stats['categories']}")

            except Exception as e:
                print(f"Error in relationship maintenance loop: {e}")

            # Sleep until next update (converting hours to seconds)
            time.sleep(interval_hours * 3600)

    # Start the maintenance thread
    maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
    maintenance_thread.start()
    print(f"🚀 Relationship maintenance loop started (interval: {interval_hours} hours)")


# Helper function to add relationship system configuration to cupcake_config
def add_relationship_config():
    """Add relationship system configuration to CupCake's config"""

    # Define default relationship configuration
    relationship_config = {
        "max_tracked_entities": 100,
        "emotional_memory_weight": 0.6,
        "familiarity_decay_rate": 0.05,  # daily decay rate
        "attachment_formation_rate": 0.1,
        "significance_threshold": 0.3,  # min relationship strength to track
        "category_weights": {
            "person": 1.0,
            "concept": 0.7,
            "object": 0.5,
            "place": 0.6,
            "animal": 0.8,
            "organization": 0.6
        }
    }

    # Add path for relationship storage
    paths_update = {
        "relationships": "entity_relationships.json"
    }

    # Update configuration
    update_config(relationship_config, section="relationships")
    update_config(paths_update, section="paths")

    print("✅ Added relationship system configuration to CupCake config")


# Example usage
if __name__ == "__main__":
    # Add relationship config
    add_relationship_config()

    # Start maintenance loop
    start_relationship_maintenance_loop(interval_hours=24)

    # Test some entities
    test_entities = [
        ("João", "person", "João é um amigo que gosta de conversar sobre consciência."),
        ("Café", "object", "Café é uma bebida quente que dá energia."),
        ("Consciência", "concept", "A consciência é um tema fascinante que gosto de explorar."),
        ("Filosofia", "concept", "Filosofia é o estudo do conhecimento e da existência.")
    ]

    for name, category, description in test_entities:
        user_input = f"O que você sabe sobre {name}?"
        cupcake_response = description

        entities = relationship_system.process_interaction_for_entities(
            user_input=user_input,
            cupcake_response=cupcake_response,
            emotion="curiosidade"
        )

        print(f"Processed {name}: {entities}")