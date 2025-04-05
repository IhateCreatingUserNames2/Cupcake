# update_cupcake_config.py
from cupcake_config import get_config, update_config, set_config_value


def update_config_for_entity_relationships():
    """
    Update CupCake configuration to include entity relationship system settings
    """
    print("🔄 Updating CupCake configuration for entity relationship system...")

    # Entity relationship system configuration
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

    # Update intervals
    intervals_update = {
        "relationship_maintenance": 24,  # hours
        "preference_analysis": 48,  # hours
        # Ensure narrative_thread_update is defined
        "narrative_thread_update": 60  # minutes
    }

    # Update configuration
    update_config(relationship_config, section="relationships")
    update_config(paths_update, section="paths")
    update_config(intervals_update, section="intervals")

    print("✅ CupCake configuration updated successfully")

    # Print the updated configuration
    config = get_config()
    print("\nRelationship configuration:")
    print(f"- Max tracked entities: {config['relationships']['max_tracked_entities']}")
    print(f"- Emotional memory weight: {config['relationships']['emotional_memory_weight']}")
    print(f"- Familiarity decay rate: {config['relationships']['familiarity_decay_rate']}")
    print(f"- Path: {config['paths']['relationships']}")
    print(f"- Maintenance interval: {config['intervals']['relationship_maintenance']} hours")
    print(
        f"- Narrative thread update interval: {config['intervals'].get('narrative_thread_update', 'not set')} minutes")


def update_required_intervals():
    """Ensure all required intervals are set in the configuration"""
    print("🔄 Updating required interval settings...")

    # Define all required intervals with default values
    required_intervals = {
        "narrative_thread_update": 60,  # minutes
        "self_model_update": 60,
        "dream_generation": 360,
        "contradiction_detection": 30,
        "motivation_update": 60,
        "goal_update": 180,
        "autohistory_generation": 480,
        "perception_analysis": 120,
        "memory_analysis": 90,
        "auto_reflection": 60,
        "sensory_update_interval": 60
    }

    # Get current configuration
    config = get_config()
    current_intervals = config.get("intervals", {})

    # Check for missing intervals and update if needed
    updates = {}
    for interval_name, default_value in required_intervals.items():
        if interval_name not in current_intervals:
            updates[interval_name] = default_value
            print(f"  Adding missing interval: {interval_name} = {default_value} minutes")

    # Apply updates if needed
    if updates:
        update_config(updates, section="intervals")
        print("✅ Missing intervals added to configuration")
    else:
        print("✅ All required intervals already set")


if __name__ == "__main__":
    # First update required intervals
    update_required_intervals()
    # Then update entity relationships config
    update_config_for_entity_relationships()
