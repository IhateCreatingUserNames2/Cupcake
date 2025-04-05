# integrate_relationship_system.py
from narrative_enhanced_cupcake import original_graph
from langgraph_cupcake import get_compiled_brain, initial_state
from entity_relationship_integration import integrate_with_langgraph, start_relationship_maintenance_loop
from update_cupcake_config import update_config_for_entity_relationships
from cupcake_entity_relationship import EntityRelationshipSystem
import time


def main():
    """
    Main function to integrate the entity relationship system with CupCake
    """
    print("\n🌟 CupCake Entity Relationship System Integration 🌟")
    print("=================================================")

    # Step 1: Update configuration
    print("\n📝 Step 1: Updating CupCake configuration...")
    update_config_for_entity_relationships()

    # Step 2: Enhance the graph with entity relationship capabilities
    print("\n🧩 Step 2: Enhancing CupCake's cognitive graph...")
    enhanced_graph = integrate_with_langgraph(original_graph)

    # Step 3: Compile the enhanced graph
    print("\n🔄 Step 3: Compiling enhanced graph...")
    enhanced_brain = enhanced_graph.compile()

    # Step 4: Start the relationship maintenance loop
    print("\n⚙️ Step 4: Starting relationship maintenance loop...")
    start_relationship_maintenance_loop(interval_hours=24)

    # Step 5: Test the integration
    print("\n🧪 Step 5: Testing entity relationship integration...")

    # Initialize the relationship system
    relationship_system = EntityRelationshipSystem()

    # Add some test entities
    test_entities = [
        ("João", "person", "João é um amigo que gosta de conversar sobre consciência.", "alegria"),
        ("Café", "object", "Café é uma bebida quente que dá energia.", "curiosidade"),
        ("Consciência", "concept", "A consciência é um tema fascinante que gosto de explorar.", "curiosidade"),
        ("Filosofia", "concept", "Filosofia é o estudo do conhecimento e da existência.", "amor")
    ]

    for name, category, description, emotion in test_entities:
        user_input = f"O que você sabe sobre {name}?"
        cupcake_response = description

        entities = relationship_system.process_interaction_for_entities(
            user_input=user_input,
            cupcake_response=cupcake_response,
            emotion=emotion
        )

        print(f"  ✅ Processed {name}: {entities}")

    # Step 6: Generate some relationship insights
    print("\n💬 Step 6: Generating relationship insights...")

    # Get most significant entities
    significant_entities = relationship_system.get_most_significant_entities(limit=2)

    for entity in significant_entities:
        insight = relationship_system.generate_relationship_insight(entity.id)
        print(f"\n  {entity.name} ({entity.category}):")
        print(f"  > {insight}")

    # Step 7: Check preference patterns
    print("\n🧠 Step 7: Analyzing preference patterns...")
    patterns = relationship_system.detect_preference_patterns()
    print(f"\n  {patterns}")

    print("\n✨ Entity Relationship System Integration Complete! ✨")
    print("CupCake can now form meaningful relationships with entities in its world.")
    print("The system will maintain these relationships over time, forming emotional attachments")
    print("and developing meaningful preferences.")


if __name__ == "__main__":
    main()