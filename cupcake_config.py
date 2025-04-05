# cupcake_config.py
import os
import json
from datetime import datetime

# Default configuration values
DEFAULT_CONFIG = {
    # API Keys
    "api": {
        "openai": os.getenv("OPENAI_API_KEY", "")
    },

    # Database Configuration
    "database": {
        "chroma_path": "./cupcake_memory_db",
        "collections": {
            "main_memory": "cupcake_memory",
            "self_perception_memories": "self_perception_memories"
        }
    },

    # Model Settings
    "model": {
        "chat_model": "gpt-3.5-turbo-0125",
        "embedding_model": "all-MiniLM-L6-v2",
        "temperature": 0.85,
        "top_p": 0.9,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.6,
        "max_tokens": 300
    },

    # Personality
    "personality": {
        "default_traits": {
            "openness": 0.9,
            "conscientiousness": 0.8,
            "extraversion": 0.7,
            "agreeableness": 0.9,
            "neuroticism": 0.1
        },
        "default_humor": 0.75,
        "trait_adjustment_rate": 0.01,
        "humor_adjustment_rate": 0.01
    },

    # Memory System Configuration
    "memory": {
        # Decay and weighting parameters
        "emotional_decay_rate": 0.05,
        "narrative_boost": 0.3,
        "repetition_boost": 0.2,
        "contradiction_boost": 0.25,
        "recency_weight": 0.4,
        "emotional_weight": 0.3,
        "semantic_weight": 0.3,
        "self_reference_boost": 0.15,
        "emotional_contagion": 0.1,

        # Self-Perception Memory Specific Configuration
        "self_perception": {
            "memory_categories": {
                "self_reflection": 0.9,  # Highest emotional significance
                "interaction_context": 0.7,  # Moderate emotional relevance
                "general_knowledge": 0.3  # Low emotional weight
            },
            "max_history_entries": 100,
            "similarity_threshold": 0.5,
            "semantic_weight": 0.7,
            "emotional_weight": 0.3
        }
    },

    # Dreamer Configuration
    "dreamer": {
        "dream_intensity_modifier": 0.0,
        "dream_max_length": 400
    },

    # Self-Perception Configuration
    "perception": {
        "meta_awareness_threshold": 0.2,
        "perception_evolution_threshold": 0.4,
        "significant_shift_markers": [
            "percebo uma mudança", "transformação", "evolução", "contraste",
            "diferente", "nova percepção", "realização", "descoberta",
            "pela primeira vez", "nunca antes", "surpreendentemente"
        ],
        "memory_retrieval": {
            "top_k_memories": 3,
            "min_relevance_score": 0.5
        }
    },

    # Narrative Threading
    "narrative": {
        "max_active_threads": 7,
        "new_thread_threshold": 0.7,
        "thread_connection_threshold": 0.6,
        "narrative_update_interval": 120,  # minutes
        "summary_generation_interval": 1440,  # minutes (1 day)
        "thread_importance_decay": 0.01,
        "auto_pruning_threshold": 1000,
        "auto_theme_detection": True,
        "emotional_emphasis": 0.7,
        "identity_emphasis": 0.8
    },

    # Relationship System
    "relationships": {
        "max_tracked_entities": 100,
        "emotional_memory_weight": 0.6,
        "familiarity_decay_rate": 0.05,
        "attachment_formation_rate": 0.1,
        "significance_threshold": 0.3,
        "category_weights": {
            "person": 1.0,
            "concept": 0.7,
            "object": 0.5,
            "place": 0.6
        }
    },

    # Sensors Configuration
    "sensors": {
        "neutral_emotion_threshold": 0.3,
        "conflict_similarity_threshold": 15,
        "environmental_awareness_level": 0.7,
        "sensory_update_interval": 10  # minutes
    },

    # Timing Intervals (minutes)
    "intervals": {
        "self_model_update": 60,
        "dream_generation": 360,
        "contradiction_detection": 60,
        "motivation_update": 180,
        "goal_update": 180,
        "narrative_thread_update": 60,
        "autohistory_generation": 240,
        "perception_analysis": 120,
        "memory_analysis": 180,
        "auto_reflection": 30,
        "sensory_update_interval": 60,
        "self_perception_evolution": 240  # New interval for perception evolution analysis
    },

    # File Paths
    "paths": {
        "journal": "journal.txt",
        "dreams": "dreams.txt",
        "emotion_log": "emotion_log.json",
        "self_perception_history": "self_perception_history.json",
        "memory_grove": "memory_grove.json",
        "emotional_memory_records": "emotional_memory_records.json",
        "perception_memory": "perception_memory_db"
    }
}

# Configuration file path
CONFIG_FILE = "cupcake_config.json"

# Current loaded configuration
_config = None


def _ensure_directory_exists(file_path):
    """Ensure the directory for a file path exists"""
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)


def ensure_config_file():
    """Ensure configuration file exists"""
    _ensure_directory_exists(CONFIG_FILE)

    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG
    return load_config()


def load_config():
    """Load configuration from file"""
    global _config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Merge with default config to ensure all new keys are present
        def deep_merge(default, current):
            for key, value in default.items():
                if isinstance(value, dict):
                    current_value = current.get(key, {})
                    current[key] = deep_merge(value, current_value)
                else:
                    if key not in current:
                        current[key] = value
            return current

        config = deep_merge(DEFAULT_CONFIG, config)
        _config = config
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        _config = DEFAULT_CONFIG
        return DEFAULT_CONFIG


def save_config(config):
    """Save configuration to file"""
    _ensure_directory_exists(CONFIG_FILE)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    global _config
    _config = config


def update_config(updates, section=None):
    """
    Update configuration with new values

    Parameters:
    - updates: Dictionary with updates
    - section: Optional section to update (if None, updates entire config)

    Returns:
    - Updated configuration
    """
    config = load_config()

    def update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = update_recursive(d[k], v)
            else:
                d[k] = v
        return d

    if section:
        if section not in config:
            config[section] = {}
        config[section] = update_recursive(config[section], updates)
    else:
        config = update_recursive(config, updates)

    save_config(config)
    return config


def get_config():
    """Get current configuration"""
    global _config
    if _config is None:
        _config = ensure_config_file()
    return _config


def get_config_value(path, default=None):
    """
    Get a specific configuration value using dot notation

    Example:
    get_config_value("memory.emotional_decay_rate", 0.05)
    """
    config = get_config()
    parts = path.split('.')

    current = config
    for part in parts:
        if part not in current:
            return default
        current = current[part]

    return current


def set_config_value(path, value):
    """
    Set a specific configuration value using dot notation

    Example:
    set_config_value("memory.emotional_decay_rate", 0.07)
    """
    config = get_config()
    parts = path.split('.')

    current = config
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
    save_config(config)
    return config


# Initialize configuration on import
ensure_config_file()