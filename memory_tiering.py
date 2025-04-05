# memory_tiering.py
from datetime import datetime, timedelta
import random
import numpy as np
from emotion_classifier import classify_emotion, classify_emotion_full


class TieredMemorySystem:
    """
    A tiered memory system for Cupcake that manages:
    - Working Memory: Recent conversations (short-term)
    - Emotional Memory: Emotionally significant memories (medium-term)
    - Deep Memory: Historical memories (long-term)

    The system supports mood-influenced retrieval, personality effects,
    and "faded memories" that occasionally surface from deep storage.
    """

    def __init__(self, enhanced_memory, embed_fn, collection=None):
        """
        Initialize the tiered memory system

        Parameters:
        - enhanced_memory: EnhancedMemoryWeighting instance
        - embed_fn: Function to convert text to embeddings
        - collection: ChromaDB collection for persistence
        """
        self.enhanced_memory = enhanced_memory
        self.embed_text = embed_fn
        self.collection = collection
        self.working_memory_buffer = []  # Last N conversation turns
        self.working_memory_limit = 10  # Max items in working memory

    def add_to_working_memory(self, user_input, cupcake_response, state):
        """
        Add a conversation turn to working memory

        Parameters:
        - user_input: User's message
        - cupcake_response: Cupcake's response
        - state: Current LangGraph state
        """
        # Add to working memory buffer
        self.working_memory_buffer.append({
            "user": user_input,
            "cupcake": cupcake_response,
            "emotion": state.get("emotion", "neutra"),
            "emotion_score": state.get("emotion_score", 0.5),
            "timestamp": datetime.utcnow().isoformat()
        })

        # Prune if needed
        self._prune_working_memory(state)

    def _prune_working_memory(self, state):
        """
        Prune working memory when it exceeds limits
        Significant interactions are moved to emotional memory

        Parameters:
        - state: Current LangGraph state
        """
        if len(self.working_memory_buffer) <= self.working_memory_limit:
            return

        # Get conversations to prune
        to_prune = self.working_memory_buffer[:-self.working_memory_limit]
        self.working_memory_buffer = self.working_memory_buffer[-self.working_memory_limit:]

        # Move significant conversations to emotional memory
        for item in to_prune:
            # Only save if emotionally significant
            emotion_score = item.get("emotion_score", 0.5)
            if emotion_score > 0.6:  # Only remember significant exchanges
                conversation_text = f"Usuário: {item['user']}\nCupcake: {item['cupcake']}"
                embedding = self.embed_text(conversation_text).tolist()

                # Create emotion data
                emotion_data = {
                    'score': emotion_score,
                    'emotion_type': item.get('emotion', 'neutral'),
                    'valence': 0.7 if emotion_score > 0.7 else 0.5,  # Simplified
                    'arousal': emotion_score,
                    'dominance': 0.5  # Default neutral dominance
                }

                # Add to emotional memory
                self.enhanced_memory.add_weighted_memory(
                    text=conversation_text,
                    embedding=embedding,
                    emotion_data=emotion_data,
                    source="conversation",
                    narrative_relevance=0.5,
                    self_reference=True
                )

    def get_working_memory(self, top_k=3):
        """
        Get the most recent items from working memory

        Parameters:
        - top_k: Number of items to return

        Returns:
        - List of (text, metadata) tuples
        """
        # Format working memory
        formatted = []
        for item in self.working_memory_buffer[-top_k:]:
            text = f"Conversa recente: Usuário: {item['user']} | Cupcake: {item['cupcake']}"
            formatted.append((text, {"source": "working_memory", "timestamp": item["timestamp"]}))
        return formatted

    def retrieve_emotional_memories(self, state, top_k=3):
        """
        Retrieve emotionally significant memories based on current state
        Influenced by mood, personality, and current emotion.

        Parameters:
        - state: Current LangGraph state
        - top_k: Number of memories to retrieve

        Returns:
        - List of (text, metadata) tuples
        """
        # Extract current emotional state
        current_emotion = state.get("emotion", "neutra")
        emotion_score = state.get("emotion_score", 0.5)
        humor_level = state.get("humor", 0.5)
        personality = state.get("personality", {})

        # Adjust emotion bias based on humor
        # Low humor -> higher emphasis on negative emotions
        # High humor -> balanced or positive emphasis
        emotion_bias = {}

        if humor_level < 0.3:  # Bad mood/low humor
            # Increase retrieval weight for intense/negative memories
            emotion_bias = {
                "intensity_multiplier": 1.5,  # Amplify intense memories
                "negative_boost": 0.3,  # Boost sad/anxious memories
                "threshold_reduction": 0.2  # Lower the threshold to include more memories
            }
        elif humor_level > 0.7:  # Good mood/high humor
            # Balanced retrieval with slight preference for positive
            emotion_bias = {
                "intensity_multiplier": 1.0,  # Normal intensity weighting
                "positive_boost": 0.2,  # Slight boost to happy/joyful memories
                "threshold_reduction": 0  # Normal threshold
            }
        else:  # Neutral mood
            emotion_bias = {
                "intensity_multiplier": 1.0,  # Normal intensity weighting
                "threshold_reduction": 0.1  # Slight threshold reduction
            }

        # Personality adjustments
        # High neuroticism -> recalls negative experiences more readily
        if personality.get("neuroticism", 0.5) > 0.7:
            emotion_bias["negative_boost"] = emotion_bias.get("negative_boost", 0) + 0.2

        # High openness -> more diverse memory recall
        if personality.get("openness", 0.5) > 0.7:
            emotion_bias["diversity_factor"] = 0.3

        # Get memories with adjusted weights
        enhanced_memories = self.enhanced_memory.get_weighted_memories_with_bias(
            query_emotion={
                'emotion_type': current_emotion,
                'score': emotion_score
            },
            emotion_bias=emotion_bias,
            top_k=top_k
        )

        return enhanced_memories

    def retrieve_mixed_memories(self, state, working_memory_count=3, emotional_memory_count=3, deep_memory_count=1):
        """
        Retrieve a mix of memory types with the deep memories appearing as 'faded'

        Parameters:
        - state: Current LangGraph state
        - working_memory_count: Number of recent conversation turns to include
        - emotional_memory_count: Number of emotional memories to include
        - deep_memory_count: Number of deep memories to include

        Returns:
        - List of (text, metadata) tuples with mixed memory types
        """
        memories = {
            "working": self.get_working_memory(top_k=working_memory_count),
            "emotional": self.retrieve_emotional_memories(state, top_k=emotional_memory_count),
            "deep": []
        }

        # Random chance (30%) to include a deep memory, or always include if specifically relevant
        include_deep = random.random() < 0.3 or self._is_memory_relevant_query(state["user_input"])

        if include_deep:
            # Get query embedding for semantic search
            query_embedding = self.embed_text(state["user_input"]).tolist()

            # Retrieve deep memories by semantic similarity
            deep_memories = self.enhanced_memory.get_deep_memories_by_similarity(
                query_embedding=query_embedding,
                min_age_days=30,  # Only memories older than 30 days
                top_k=deep_memory_count
            )

            # Format deep memories as "faded" by adding a prefix
            formatted_deep_memories = []
            for doc, meta in deep_memories:
                # Calculate age factor (older = more faded)
                memory_date = datetime.fromisoformat(meta.get('timestamp', datetime.now().isoformat()))
                days_old = (datetime.now() - memory_date).days
                fade_level = min(days_old / 100, 0.9)  # Max 90% faded

                # Create faded memory text with appropriate prefix
                if fade_level > 0.7:
                    prefix = "Lembrança distante e quase esquecida: "
                elif fade_level > 0.4:
                    prefix = "Memória antiga que ainda ecoa: "
                else:
                    prefix = "Recordação que persiste: "

                formatted_memory = f"{prefix}{doc}"
                formatted_deep_memories.append((formatted_memory, meta))

            memories["deep"] = formatted_deep_memories

        # Combine all memory types
        all_memories = memories["working"] + memories["emotional"] + memories["deep"]
        return all_memories

    def handle_deep_memory_search(self, search_query, max_results=10):
        """
        Handle explicit deep memory search command

        Parameters:
        - search_query: Search terms
        - max_results: Maximum number of results to return

        Returns:
        - List of (memory_text, metadata) tuples
        """
        # Generate embedding for search query
        embedding = self.embed_text(search_query).tolist()

        # Search all memories by semantic similarity
        deep_memories = self.enhanced_memory.get_memories_by_similarity(
            query_embedding=embedding,
            top_k=max_results,
            allowed_sources=None  # Search all sources
        )

        # Format results with date information
        formatted_results = []
        for doc, meta in deep_memories:
            try:
                # Extract date and format it
                timestamp = meta.get('timestamp', datetime.now().isoformat())
                date_obj = datetime.fromisoformat(timestamp)
                date_str = date_obj.strftime("%d/%m/%Y")

                # Get emotion information
                emotion = meta.get('emotion_type', 'neutra')
                emotion_str = f" [{emotion}]" if emotion != 'neutra' else ""

                # Format the memory
                formatted_memory = f"[{date_str}]{emotion_str} {doc}"
                formatted_results.append((formatted_memory, meta))
            except:
                # Fallback if date parsing fails
                formatted_results.append((doc, meta))

        return formatted_results

    def _is_memory_relevant_query(self, user_input):
        """
        Detect if the query is asking about memories or past events

        Parameters:
        - user_input: User's query text

        Returns:
        - Boolean indicating if query is memory-related
        """
        memory_terms = [
            "lembra", "lembrar", "recordar", "memória", "aconteceu",
            "passado", "antes", "antiga", "experiência", "já sentiu",
            "remember", "recall", "memory", "happened", "past", "before",
            "experienced", "ever felt"
        ]

        return any(term in user_input.lower() for term in memory_terms)

    def forget_memory(self, memory_id):
        """
        Explicitly forget a specific memory

        Parameters:
        - memory_id: ID of the memory to forget

        Returns:
        - Boolean indicating success
        """
        if not self.collection:
            return False

        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error forgetting memory: {e}")
            return False

    def migrate_old_memories(self, days_threshold=60):
        """
        Migrate memories older than threshold to deep storage
        These memories get lower retrieval priority but are still accessible

        Parameters:
        - days_threshold: Age in days to consider a memory "old"

        Returns:
        - Number of memories migrated
        """
        if not self.collection:
            return 0

        try:
            # Get all memories
            results = self.collection.get(include=['ids', 'metadatas'])

            # Find memories older than threshold
            cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
            migration_count = 0

            for i, meta in enumerate(results['metadatas']):
                timestamp = meta.get('timestamp', datetime.now().isoformat())

                # Check if memory is old but not yet marked as deep
                if timestamp < cutoff_date and meta.get('storage_tier', '') != 'deep':
                    # Update metadata to mark as deep storage
                    meta['storage_tier'] = 'deep'
                    self.collection.update(
                        ids=[results['ids'][i]],
                        metadatas=[meta]
                    )
                    migration_count += 1

            return migration_count
        except Exception as e:
            print(f"Error migrating old memories: {e}")
            return 0

    def get_memory_statistics(self):
        """
        Get statistics about memory usage

        Returns:
        - Dictionary with memory statistics
        """
        if not self.collection:
            return {"working_memory": len(self.working_memory_buffer)}

        try:
            # Get all memories
            results = self.collection.get(include=['metadatas'])

            # Count by tier
            working_count = len(self.working_memory_buffer)
            deep_count = 0
            emotional_count = 0

            # Count by source
            source_counts = {}

            for meta in results['metadatas']:
                # Count by tier
                if meta.get('storage_tier', '') == 'deep':
                    deep_count += 1
                else:
                    emotional_count += 1

                # Count by source
                source = meta.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1

            return {
                "working_memory": working_count,
                "emotional_memory": emotional_count,
                "deep_memory": deep_count,
                "total_memories": working_count + emotional_count + deep_count,
                "sources": source_counts
            }
        except Exception as e:
            print(f"Error getting memory statistics: {e}")
            return {"working_memory": len(self.working_memory_buffer)}