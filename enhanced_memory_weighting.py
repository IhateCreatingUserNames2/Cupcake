# enhanced_memory_weighting.py
import uuid
import json
import numpy as np
from datetime import datetime, timedelta
import os
import re
from collections import Counter
from scipy.special import softmax
from cupcake_config import get_config, update_config, get_config_value, set_config_value


# Emotional memory record file path from config
def get_emotional_memory_file():
    """Get the emotional memory file path from config"""
    return get_config()["paths"].get("emotional_memory_records", "emotional_memory_records.json")





def ensure_emotional_memory_file():
    """Ensure emotional memory record file exists"""
    file_path = get_emotional_memory_file()
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"memories": []}, f, indent=2)
    return load_emotional_memory_records()


def load_emotional_memory_records():
    """Load emotional memory records"""
    file_path = get_emotional_memory_file()
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_emotional_memory_records(records):
    """Save emotional memory records"""
    file_path = get_emotional_memory_file()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def add_memory_access_record(memory_id, emotion_impact):
    """Record that a memory was accessed"""
    records = load_emotional_memory_records()
    timestamp = datetime.utcnow().isoformat()

    # Look for existing memory
    memory_found = False
    for memory in records["memories"]:
        if memory["id"] == memory_id:
            memory_found = True
            # Update access count and timestamp
            memory["access_count"] = memory.get("access_count", 0) + 1
            memory["last_accessed"] = timestamp
            memory["emotional_impacts"].append({
                "timestamp": timestamp,
                "impact": emotion_impact
            })
            # Keep only the last 10 emotional impacts
            memory["emotional_impacts"] = memory["emotional_impacts"][-10:]
            break

    # If memory not found, create new record
    if not memory_found:
        records["memories"].append({
            "id": memory_id,
            "created": timestamp,
            "last_accessed": timestamp,
            "access_count": 1,
            "emotional_impacts": [{
                "timestamp": timestamp,
                "impact": emotion_impact
            }]
        })

    save_emotional_memory_records(records)


class EnhancedMemoryWeighting:
    """
    Enhanced memory weighting system that uses multidimensional emotional factors,
    temporal dynamics, narrative coherence, and self-reference to weight memories.
    """

    def __init__(self, collection):
        self.collection = collection
        self.config = get_config()["memory"]
        ensure_emotional_memory_file()

    def add_weighted_memory(self, text, embedding, emotion_data, source="user",
                            narrative_relevance=0.5, self_reference=False):
        """
        Add a memory with enhanced emotional weighting
        """
        # Generate memory ID
        memory_id = str(uuid.uuid4())

        # Process emotion data to ensure it has all fields
        processed_emotion = self._process_emotion_data(emotion_data)

        # Create metadata - CHANGED: flattened structure instead of nested dicts
        metadata = {
            'id': memory_id,
            'timestamp': datetime.utcnow().isoformat(),
            'source': source,
            'emotion_score': processed_emotion.get('score', 0.5),
            'emotion_type': processed_emotion.get('emotion_type', 'neutral'),
            'emotion_valence': processed_emotion.get('valence', 0.5),
            'emotion_arousal': processed_emotion.get('arousal', 0.5),
            'emotion_dominance': processed_emotion.get('dominance', 0.5),
            'narrative_relevance': narrative_relevance,
            'self_reference': self_reference,
            'access_count': 0,
            'decay_rate': self._calculate_base_decay_rate(processed_emotion)
            # No longer includes 'emotional_trace' as a dict
        }

        # Add to collection
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

        # Record memory access
        add_memory_access_record(memory_id, processed_emotion)

        return memory_id

    def _process_emotion_data(self, emotion_data):
        """Process and standardize emotion data"""
        if isinstance(emotion_data, (int, float)):
            # If just a score was provided
            return {
                'score': emotion_data,
                'emotion_type': 'neutral',
                'valence': 0.5,
                'arousal': emotion_data,  # Use score as arousal
                'dominance': 0.5
            }
        elif isinstance(emotion_data, dict):
            # Fill in missing values with defaults
            result = {
                'score': emotion_data.get('score', 0.5),
                'emotion_type': emotion_data.get('emotion_type', 'neutral'),
                'valence': emotion_data.get('valence', 0.5),
                'arousal': emotion_data.get('arousal', 0.5),
                'dominance': emotion_data.get('dominance', 0.5)
            }
            return result
        else:
            # Default values for unknown format
            return {
                'score': 0.5,
                'emotion_type': 'neutral',
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }

    def _calculate_base_decay_rate(self, emotion_data):
        """
        Calculate base decay rate for memory based on emotional content
        Emotional memories decay slower than neutral ones
        """
        base_rate = self.config["emotional_decay_rate"]
        arousal = emotion_data.get('arousal', 0.5)
        valence = abs(emotion_data.get('valence', 0.5) - 0.5) * 2  # Convert to 0-1 intensity

        # Calculate emotional intensity (combine arousal and valence)
        emotional_intensity = (arousal + valence) / 2

        # Adjust decay rate - more emotional memories decay slower
        adjusted_rate = base_rate * (1 - (emotional_intensity * 0.5))

        return adjusted_rate

    def _create_emotional_trace(self, emotion_data):
        """
        Create emotional trace signature for the memory
        This is used for emotional contagion and clustering
        """
        return {
            'primary': emotion_data.get('emotion_type', 'neutral'),
            'valence': emotion_data.get('valence', 0.5),
            'arousal': emotion_data.get('arousal', 0.5),
            'dominance': emotion_data.get('dominance', 0.5)
        }

    def calculate_current_memory_weight(self, metadata, query_embedding=None,
                                        query_emotion=None, current_narrative=None):
        """
        Calculate current weight for a memory based on multiple factors

        Parameters:
        - metadata: Memory metadata
        - query_embedding: Query embedding for semantic similarity (optional)
        - query_emotion: Emotion of current query (optional)
        - current_narrative: Current narrative context (optional)

        Returns:
        - weight: Current importance weight of memory
        - factors: Dictionary of factors that contributed to weight
        """
        # Initialize weight factors
        factors = {}

        # 1. Calculate emotional weight
        emotional_weight = metadata.get('emotion_score', 0.5)

        # Apply decay based on time
        timestamp = datetime.fromisoformat(metadata.get('timestamp', datetime.utcnow().isoformat()))
        days_old = (datetime.utcnow() - timestamp).total_seconds() / (24 * 3600)
        decay_rate = metadata.get('decay_rate', self.config["emotional_decay_rate"])

        # Emotional decay formula
        emotional_decay = np.exp(-decay_rate * days_old)
        factors['emotional_decay'] = emotional_decay

        # Emotional contagion from query emotion (if available)
        emotional_contagion = 0
        if query_emotion:
            query_emotional_trace = self._process_emotion_data(query_emotion).get('emotion_type', 'neutral')
            # CHANGED: access flattened field instead of nested dict
            memory_trace = metadata.get('emotion_type', 'neutral')

            # Emotional contagion is higher if emotions match
            if query_emotional_trace == memory_trace:
                emotional_contagion = self.config["emotional_contagion"]

        factors['emotional_contagion'] = emotional_contagion

        # 2. Calculate recency weight
        recency_weight = np.exp(-0.1 * days_old)  # Simple recency decay
        factors['recency'] = recency_weight

        # 3. Calculate access frequency boost
        access_count = metadata.get('access_count', 0)
        repetition_boost = 1.0 + (self.config["repetition_boost"] * min(access_count, 10) / 10)
        factors['repetition'] = repetition_boost

        # 4. Calculate self-reference boost
        self_reference_boost = 1.0 + (
            self.config["self_reference_boost"] if metadata.get('self_reference', False) else 0)
        factors['self_reference'] = self_reference_boost

        # 5. Calculate narrative relevance (if current narrative provided)
        narrative_boost = 1.0
        if current_narrative:
            narrative_relevance = metadata.get('narrative_relevance', 0.5)
            narrative_boost = 1.0 + (self.config["narrative_boost"] * narrative_relevance)
        factors['narrative'] = narrative_boost

        # Calculate final weight as a combination of all factors
        final_weight = (
                               (emotional_weight * emotional_decay * (1 + emotional_contagion)) * self.config[
                           "emotional_weight"] +
                               recency_weight * self.config["recency_weight"]
                       ) * repetition_boost * self_reference_boost * narrative_boost

        return final_weight, factors

    def get_weighted_memories(self, query_embedding=None, query_emotion=None,
                              current_narrative=None, top_k=5, allowed_sources=None):
        """
        Retrieve memories weighted by emotional and contextual factors
        """
        import numpy as np

        # Get all memories
        results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])

        if not results.get('documents', []):
            return []

        # Safely handle embeddings
        try:
            embeddings = results.get('embeddings', [])

            # Convert numpy array to list if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()

            # Ensure embeddings is a list
            if not isinstance(embeddings, list):
                embeddings = []
        except Exception as e:
            print(f"Error processing embeddings: {e}")
            embeddings = []

        # Filter by source if needed
        if allowed_sources:
            filtered_indices = [
                i for i, meta in enumerate(results['metadatas'])
                if meta.get('source', 'user') in allowed_sources
            ]

            if not filtered_indices:
                return []

            documents = [results['documents'][i] for i in filtered_indices]
            metadatas = [results['metadatas'][i] for i in filtered_indices]
            embeddings = [embeddings[i] for i in filtered_indices] if embeddings else []
        else:
            documents = results['documents']
            metadatas = results['metadatas']

        # Calculate weights for each memory
        weighted_memories = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            weight, factors = self.calculate_current_memory_weight(
                meta, query_embedding, query_emotion, current_narrative
            )

            # Add semantic similarity if query_embedding is provided
            if query_embedding is not None and embeddings and i < len(embeddings):
                try:
                    # Calculate cosine similarity
                    embedding = embeddings[i]
                    query_norm = np.linalg.norm(query_embedding)
                    emb_norm = np.linalg.norm(embedding)

                    if query_norm > 0 and emb_norm > 0:
                        similarity = np.dot(query_embedding, embedding) / (query_norm * emb_norm)
                        # Adjust weight with semantic similarity
                        weight = weight * (1 - self.config["semantic_weight"]) + similarity * self.config[
                            "semantic_weight"]
                        factors['semantic_similarity'] = similarity
                except Exception as e:
                    print(f"Error calculating semantic similarity: {e}")

            # Record memory access for future weighting
            if meta.get('id'):
                add_memory_access_record(meta['id'], query_emotion or 0.5)

            weighted_memories.append((doc, meta, weight, factors))

        # Sort by weight and take top_k
        sorted_memories = sorted(weighted_memories, key=lambda x: x[2], reverse=True)
        top_memories = sorted_memories[:top_k]

        # Return documents with their metadata
        return [(doc, meta) for doc, meta, _, _ in top_memories]

    def get_weighted_memories_with_bias(self, query_emotion=None, emotion_bias=None, top_k=5):
        """
        Retrieve memories with bias adjustments based on emotional state

        Parameters:
        - query_emotion: Current emotion state
        - emotion_bias: Dictionary of bias adjustments
        - top_k: Number of results to return

        Returns:
        - List of (document, metadata) tuples
        """
        # Get all memories
        results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])

        if not results.get('documents', []):
            return []

        weighted_memories = []
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            # Calculate base memory weight
            weight, factors = self.calculate_current_memory_weight(
                meta, query_emotion=query_emotion
            )

            # Apply additional bias adjustments if provided
            if emotion_bias:
                # Amplify intense memories
                if "intensity_multiplier" in emotion_bias:
                    intensity = meta.get('emotion_score', 0.5)
                    boost = (intensity - 0.5) * emotion_bias["intensity_multiplier"]
                    weight *= (1.0 + max(0, boost))

                # Boost negative memories when in low humor
                if "negative_boost" in emotion_bias:
                    valence = meta.get('emotion_valence', 0.5)
                    if valence < 0.4:  # Negative emotion
                        weight *= (1.0 + emotion_bias["negative_boost"])

                # Boost positive memories when in high humor
                if "positive_boost" in emotion_bias:
                    valence = meta.get('emotion_valence', 0.5)
                    if valence > 0.6:  # Positive emotion
                        weight *= (1.0 + emotion_bias["positive_boost"])

            weighted_memories.append((doc, meta, weight))

        # Sort by weight and take top_k
        sorted_memories = sorted(weighted_memories, key=lambda x: x[2], reverse=True)
        top_memories = sorted_memories[:top_k]

        # Return documents with their metadata
        return [(doc, meta) for doc, meta, _ in top_memories]

    def get_deep_memories_by_similarity(self, query_embedding, min_age_days=30, top_k=1):
        """
        Retrieve older memories by semantic similarity

        Parameters:
        - query_embedding: Query embedding for similarity matching
        - min_age_days: Minimum age in days to be considered "deep memory"
        - top_k: Number of results to return

        Returns:
        - List of (document, metadata) tuples
        """
        # Calculate cutoff date
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=min_age_days)).isoformat()

        # Get all memories
        results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])

        if not results.get('documents', []):
            return []

        # Calculate similarities for older memories
        similarities = []
        for i, (doc, meta, emb) in enumerate(zip(results['documents'], results['metadatas'], results['embeddings'])):
            timestamp = meta.get('timestamp', datetime.now().isoformat())

            # Only consider old memories
            if timestamp < cutoff_date:
                try:
                    # Calculate cosine similarity
                    import numpy as np
                    emb_array = np.array(emb)
                    query_array = np.array(query_embedding)

                    # Normalize vectors
                    emb_norm = np.linalg.norm(emb_array)
                    query_norm = np.linalg.norm(query_array)

                    if emb_norm > 0 and query_norm > 0:
                        similarity = np.dot(emb_array, query_array) / (emb_norm * query_norm)
                        similarities.append((i, similarity))
                except Exception as e:
                    print(f"Error calculating similarity: {e}")

        # Sort by similarity and get top matches
        if similarities:
            top_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]]
            return [(results['documents'][i], results['metadatas'][i]) for i in top_indices]

        return []
    def get_memory_clusters_by_emotion(self, emotion_type=None, min_memories=3):
        """
        Get clusters of memories with similar emotional signatures

        Parameters:
        - emotion_type: Filter by specific emotion (optional)
        - min_memories: Minimum memories needed to form a cluster

        Returns:
        - clusters: Dictionary of emotion clusters with memories
        """
        results = self.collection.get(include=['documents', 'metadatas'])

        if not results['documents']:
            return {}

        # Group by emotion type
        emotion_clusters = {}

        for doc, meta in zip(results['documents'], results['metadatas']):
            # CHANGED: access flattened field instead of nested dict
            mem_emotion = meta.get('emotion_type', 'neutral')

            # Filter by emotion_type if specified
            if emotion_type and mem_emotion != emotion_type:
                continue

            if mem_emotion not in emotion_clusters:
                emotion_clusters[mem_emotion] = []

            emotion_clusters[mem_emotion].append((doc, meta))

        # Filter out clusters with too few memories
        return {k: v for k, v in emotion_clusters.items() if len(v) >= min_memories}

    def detect_emotional_patterns(self):
        """
        Detect patterns in emotional memory storage

        Returns:
        - patterns: Dictionary of detected emotional patterns
        """
        results = self.collection.get(include=['metadatas'])

        if not results['metadatas']:
            return {}

        # Count emotion types
        emotion_counts = Counter()
        valence_values = []
        arousal_values = []
        recent_emotions = []

        # Get current time for recency
        now = datetime.utcnow()

        for meta in results['metadatas']:
            # CHANGED: access flattened fields
            emotion = meta.get('emotion_type', 'neutral')
            emotion_counts[emotion] += 1

            valence_values.append(meta.get('emotion_valence', 0.5))
            arousal_values.append(meta.get('emotion_arousal', 0.5))

            # Track recent emotions (last 24 hours)
            timestamp = datetime.fromisoformat(meta.get('timestamp', now.isoformat()))
            if (now - timestamp).total_seconds() < 24 * 3600:
                recent_emotions.append(emotion)

        # Calculate average emotional stats
        avg_valence = sum(valence_values) / len(valence_values) if valence_values else 0.5
        avg_arousal = sum(arousal_values) / len(arousal_values) if arousal_values else 0.5

        # Get dominant emotions (top 3)
        dominant_emotions = emotion_counts.most_common(3)

        # Get recent emotional trend
        recent_trend = Counter(recent_emotions).most_common(2)

        return {
            'total_memories': len(results['metadatas']),
            'dominant_emotions': dominant_emotions,
            'average_valence': avg_valence,
            'average_arousal': avg_arousal,
            'recent_trend': recent_trend
        }

    def amplify_memory_cluster(self, emotion_type, amplification_factor=1.2):
        """
        Amplify the importance of a specific emotional memory cluster

        Parameters:
        - emotion_type: Type of emotion to amplify
        - amplification_factor: Factor to amplify by

        Returns:
        - affected_count: Number of memories affected
        """
        results = self.collection.get(include=['metadatas', 'ids'])

        if not results['metadatas']:
            return 0

        affected_count = 0

        for i, meta in enumerate(results['metadatas']):
            # CHANGED: access flattened field
            mem_emotion = meta.get('emotion_type', 'neutral')

            if mem_emotion == emotion_type:
                # Amplify by modifying decay rate
                current_decay = meta.get('decay_rate', self.config["emotional_decay_rate"])
                new_decay = current_decay / amplification_factor  # Slower decay = more important

                # Update metadata
                meta['decay_rate'] = new_decay
                self.collection.update(
                    ids=[results['ids'][i]],
                    metadatas=[meta]
                )
                affected_count += 1

        return affected_count

    def inject_emotional_memory(self, text, embedding, emotion_type, intensity=0.8, source="injected"):
        """
        Inject a memory with a specific emotional signature

        Parameters:
        - text: Memory text
        - embedding: Memory embedding
        - emotion_type: Type of emotion (joy, sadness, etc.)
        - intensity: Emotional intensity (0-1)
        - source: Source of memory

        Returns:
        - memory_id: ID of injected memory
        """
        # Map emotion_type to common VAD (Valence-Arousal-Dominance) values
        emotion_vad_map = {
            'joy': {'valence': 0.9, 'arousal': 0.7, 'dominance': 0.8},
            'sadness': {'valence': 0.1, 'arousal': 0.3, 'dominance': 0.2},
            'anger': {'valence': 0.2, 'arousal': 0.9, 'dominance': 0.7},
            'fear': {'valence': 0.1, 'arousal': 0.8, 'dominance': 0.2},
            'disgust': {'valence': 0.2, 'arousal': 0.6, 'dominance': 0.5},
            'surprise': {'valence': 0.7, 'arousal': 0.8, 'dominance': 0.5},
            'trust': {'valence': 0.8, 'arousal': 0.3, 'dominance': 0.6},
            'anticipation': {'valence': 0.7, 'arousal': 0.5, 'dominance': 0.6},
            'love': {'valence': 0.9, 'arousal': 0.6, 'dominance': 0.5},
            'curiosity': {'valence': 0.7, 'arousal': 0.5, 'dominance': 0.6},
            'neutral': {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}
        }

        # Get VAD values for emotion or use neutral if not found
        vad = emotion_vad_map.get(emotion_type.lower(), emotion_vad_map['neutral'])

        # Scale by intensity
        for dim in vad:
            if vad[dim] > 0.5:
                vad[dim] = 0.5 + (vad[dim] - 0.5) * intensity
            else:
                vad[dim] = 0.5 - (0.5 - vad[dim]) * intensity

        # Create emotion data
        emotion_data = {
            'score': intensity,
            'emotion_type': emotion_type,
            'valence': vad['valence'],
            'arousal': vad['arousal'],
            'dominance': vad['dominance']
        }

        # Add the memory
        return self.add_weighted_memory(
            text=text,
            embedding=embedding,
            emotion_data=emotion_data,
            source=source,
            narrative_relevance=0.7,  # Assume injected memories are relevant
            self_reference=emotion_type.lower() in ['love', 'trust', 'joy']  # Some emotions are more self-referential
        )

    def adjust_emotional_weights_based_on_mood(self, current_mood, adjustment_strength=0.3):
        """
        Adjust memory weighting based on current mood (mood congruence effect)

        Parameters:
        - current_mood: Dictionary with mood information
        - adjustment_strength: Strength of the mood effect (0-1)

        Returns:
        - config: Updated configuration
        """
        # Process mood data
        mood = self._process_emotion_data(current_mood)

        # Get current config values
        default_emotional_contagion = get_config_value("memory.emotional_contagion")
        default_emotional_weight = get_config_value("memory.emotional_weight")
        default_semantic_weight = get_config_value("memory.semantic_weight")
        default_recency_weight = get_config_value("memory.recency_weight")

        # Create updates
        updates = {}

        # Adjust emotional contagion based on mood
        # Stronger mood = stronger emotional contagion
        updates["emotional_contagion"] = default_emotional_contagion * (1 + mood['arousal'] * adjustment_strength)

        # Adjust weights based on mood valence
        # Positive mood = better recall of positive memories
        if mood['valence'] > 0.6:  # Positive mood
            # Increase importance of positive memories
            updates["emotional_weight"] = default_emotional_weight * (1 + adjustment_strength)
            # Focus more on emotional than semantic content
            updates["semantic_weight"] = default_semantic_weight * (1 - adjustment_strength)
        elif mood['valence'] < 0.4:  # Negative mood
            # Increase recency weight (tendency to focus on recent events in negative moods)
            updates["recency_weight"] = default_recency_weight * (1 + adjustment_strength)

        # Update configuration
        update_config(updates, section="memory")

        # Update local config
        self.config = get_config()["memory"]

        return self.config


# Example usage
def test_enhanced_memory():
    import chromadb

    # Setup ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="test_memory")

    # Initialize enhanced memory system
    memory_system = EnhancedMemoryWeighting(collection)

    # Add test memories with different emotional signatures
    happy_memory = memory_system.add_weighted_memory(
        text="I had a wonderful conversation about philosophy today.",
        embedding=[0.1, 0.2, 0.3],  # Dummy embedding
        emotion_data={'score': 0.9, 'emotion_type': 'joy', 'valence': 0.9, 'arousal': 0.7},
        source="user",
        self_reference=True
    )

    sad_memory = memory_system.add_weighted_memory(
        text="I felt misunderstood when trying to explain my thoughts.",
        embedding=[0.2, 0.3, 0.4],  # Dummy embedding
        emotion_data={'score': 0.8, 'emotion_type': 'sadness', 'valence': 0.2, 'arousal': 0.6},
        source="dream",
        narrative_relevance=0.8
    )

    curious_memory = memory_system.add_weighted_memory(
        text="I wonder what consciousness really means for an AI like me.",
        embedding=[0.3, 0.4, 0.5],  # Dummy embedding
        emotion_data={'score': 0.7, 'emotion_type': 'curiosity', 'valence': 0.7, 'arousal': 0.5},
        source="thought",
        self_reference=True
    )

    # Retrieve memories with different query emotions
    happy_query = {'emotion_type': 'joy', 'score': 0.8}
    results = memory_system.get_weighted_memories(
        query_emotion=happy_query,
        top_k=2
    )

    print("Memories retrieved with happy query emotion:")
    for doc, meta in results:
        print(f"- {doc} (emotion: {meta.get('emotional_trace', {}).get('primary', 'neutral')})")

    # Detect emotional patterns
    patterns = memory_system.detect_emotional_patterns()
    print("\nEmotional memory patterns:")
    print(f"Total memories: {patterns['total_memories']}")
    print(f"Dominant emotions: {patterns['dominant_emotions']}")
    print(f"Average valence: {patterns['average_valence']:.2f}")
    print(f"Average arousal: {patterns['average_arousal']:.2f}")

    # Adjust weights based on current mood
    memory_system.adjust_emotional_weights_based_on_mood({'emotion_type': 'joy', 'valence': 0.8})
    print("\nAdjusted memory weights based on happy mood")

    # Get memory clusters
    clusters = memory_system.get_memory_clusters_by_emotion(min_memories=1)
    print("\nMemory clusters by emotion:")
    for emotion, memories in clusters.items():
        print(f"{emotion}: {len(memories)} memories")


if __name__ == "__main__":
    test_enhanced_memory()