# enhanced_self_perception.py
import os

import re
from datetime import datetime, timedelta
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from cupcake_config import get_config, get_config_value
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree
from cupcake_identity import generate_identity_prompt
from emotion_classifier import classify_emotion_full
import time
import json
from prompt_logger import PromptLogger


# Create a prompt logger instance (singleton)
_prompt_logger = None


def get_prompt_logger():
    """Get or create the prompt logger singleton"""
    global _prompt_logger
    if _prompt_logger is None:
        _prompt_logger = PromptLogger()
    return _prompt_logger

class SelfPerceptionMemorySystem:
    """
    Advanced memory management for self-perception with semantic and emotional weighting
    """

    def __init__(self, embed_model, collection):
        """
        Initialize memory system for self-perception

        Args:
            embed_model: Sentence transformer for embeddings
            collection: ChromaDB collection for memory storage
        """
        self.embed_model = embed_model
        self.collection = collection
        self.journal = CupcakeJournal()
        self.memory_tree = LiminalMemoryTree()

        # Memory category weights
        self.memory_categories = {
            "self_reflection": 0.9,  # Highest emotional significance
            "interaction_context": 0.7,  # Moderate emotional relevance
            "general_knowledge": 0.3  # Low emotional weight
        }

    def embed_text(self, text: str):
        """Generate embedding for given text"""
        return self.embed_model.encode(text).tolist()

    def add_memory(
            self,
            text: str,
            category: str = "interaction_context",
            emotion: str = "neutral",
            emotion_score: float = 0.5
    ):
        """
        Add a memory with emotional and contextual weighting

        Args:
            text: Memory content
            category: Memory category
            emotion: Detected emotion
            emotion_score: Emotion intensity
        """
        # Validate category
        if category not in self.memory_categories:
            category = "interaction_context"

        # Generate embedding
        embedding = self.embed_text(text)

        # Calculate memory weight
        category_weight = self.memory_categories.get(category, 0.5)
        emotional_weight = emotion_score * category_weight

        # Metadata for memory
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "emotion": emotion,
            "emotion_score": emotion_score,
            "category_weight": category_weight,
            "total_weight": emotional_weight
        }

        # Generate unique memory ID
        memory_id = f"memory_{datetime.utcnow().timestamp()}"

        # Add to ChromaDB collection
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

        # Log to memory tree
        self.memory_tree.add_leaf(
            user_input=f"(self-perception memory: {category})",
            cupcake_response=text,
            emotion=emotion,
            impact=emotional_weight,
            category="SelfPerception"
        )

        return memory_id

    def retrieve_relevant_memories(
            self,
            query: str,
            query_embedding: list = None,
            top_k: int = 3,
            min_score: float = 0.5
    ):
        """
        Retrieve memories with semantic and emotional relevance

        Args:
            query: Search query text
            query_embedding: Optional pre-computed embedding
            top_k: Number of memories to retrieve
            min_score: Minimum relevance score

        Returns:
            List of relevant memories
        """
        # Use query embedding or generate
        if query_embedding is None:
            query_embedding = self.embed_text(query)

        # Retrieve memories
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Process and filter memories
        relevant_memories = []
        for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        ):
            # Calculate relevance score considering semantic and emotional factors
            semantic_score = 1 - distance  # Convert distance to similarity
            emotional_weight = meta.get('total_weight', 0.5)

            # Combined relevance score
            total_relevance = (semantic_score * 0.7) + (emotional_weight * 0.3)

            if total_relevance >= min_score:
                relevant_memories.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance_score": total_relevance
                })

        # Sort by relevance
        return sorted(relevant_memories, key=lambda x: x['relevance_score'], reverse=True)


class EnhancedSelfPerceptionLayer:
    """
    Enhanced self-perception layer with context-aware memory and multi-dimensional perception
    """

    def __init__(self):
        # Load configuration
        config = get_config()

        # Initialize OpenAI client
        openai_api_key = config["api"]["openai"]
        self.client = OpenAI(api_key=openai_api_key)

        # Embedding setup
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ChromaDB setup for memory
        db_config = config["database"]
        client_db = chromadb.PersistentClient(path=db_config["chroma_path"])
        self.memory_collection = client_db.get_or_create_collection(
            name="self_perception_memories"
        )

        # Initialize memory system
        self.memory_system = SelfPerceptionMemorySystem(
            embed_model=self.embed_model,
            collection=self.memory_collection
        )

        # Model and generation settings
        self.model_name = config["model"]["chat_model"]
        self.temperature = config.get("model", {}).get("temperature", 0.7)

        # Perception history management
        self.perception_history_file = config["paths"]["self_perception_history"]
        self._ensure_history_file()
        self.perception_history = self._load_perception_history()

        # Perception dimensions and traits
        self.personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
        ]
        self.perception_dimensions = [
            'immediate', 'relational', 'existential', 'temporal', 'narrative'
        ]

    def _ensure_history_file(self):
        """Ensure perception history file exists"""
        if not os.path.exists(self.perception_history_file):
            with open(self.perception_history_file, "w", encoding="utf-8") as f:
                json.dump({"perceptions": []}, f, indent=2)

    def _load_perception_history(self):
        """Load perception history from file"""
        try:
            with open(self.perception_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"perceptions": []}

    def _save_perception_history(self):
        """Save perception history to file"""
        with open(self.perception_history_file, "w", encoding="utf-8") as f:
            json.dump(self.perception_history, f, indent=2)

    def _add_perception_to_history(self, perception_data):
        """
        Add perception to history with timestamp

        Args:
            perception_data: Dictionary of perception details
        """
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "perception_data": perception_data
        }

        # Limit history to last 100 entries
        self.perception_history["perceptions"].append(entry)
        if len(self.perception_history["perceptions"]) > 100:
            self.perception_history["perceptions"] = self.perception_history["perceptions"][-100:]

        # Save updated history
        self._save_perception_history()

    def process_perception(self, state):
        """
        Process perception with context-aware memory and multi-dimensional analysis
        WITH LOGGING and consolidated API calls

        Args:
            state: Current interaction state

        Returns:
            Updated state with enriched perception
        """
        user_input = state.get("user_input", "")
        prompt_logger = get_prompt_logger()

        # Emotion detection and analysis
        emotion_profile = classify_emotion_full(user_input)
        primary_emotion = emotion_profile[0]['label'] if emotion_profile else "neutral"
        emotion_score = emotion_profile[0]['score'] if emotion_profile else 0.5

        # Add current interaction to memory
        self.memory_system.add_memory(
            text=user_input,
            category="interaction_context",
            emotion=primary_emotion,
            emotion_score=emotion_score
        )

        # Retrieve contextually relevant memories
        query_embedding = self.memory_system.embed_text(user_input)
        relevant_memories = self.memory_system.retrieve_relevant_memories(
            query=user_input,
            query_embedding=query_embedding,
            top_k=3
        )

        # Prepare context for perception generation
        context_memories = "\n".join([
            f"[{mem['metadata'].get('category', 'context')}] {mem['text']}"
            for mem in relevant_memories
        ])

        # Prepare identity prompt and generation helper
        identity_prompt = state.get("identity_prompt", generate_identity_prompt())
        personality = state.get("personality", {})

        # CONSOLIDATED APPROACH: Single comprehensive prompt instead of multiple calls
        comprehensive_prompt = f"""
    Analyze the following input through multiple perspectives:

    User Input: "{user_input}"
    Identity Prompt: {identity_prompt[:500]}  # Truncated to avoid excessive tokens
    Current Emotion: {primary_emotion} (Score: {emotion_score:.2f})
    Contextual Memories:
    {context_memories}

    Please provide a COMPLETE JSON response with these sections:
    {{
        "trait_perceptions": {{
            "openness": "A nuanced, sophisticated perspective on the user input through the lens of openness",
            "conscientiousness": "Perspective through conscientiousness",
            "extraversion": "Perspective through extraversion",
            "agreeableness": "Perspective through agreeableness",
            "neuroticism": "Perspective through neuroticism"
        }},
        "dimensional_perceptions": {{
            "immediate": "Perspective on the immediate experience",
            "relational": "Perspective on relational aspects",
            "existential": "Perspective on existential implications",
            "temporal": "Perspective on temporal considerations",
            "narrative": "Perspective on narrative coherence"
        }},
        "meta_awareness": "A meta-cognitive reflection analyzing the self-perception process",
        "synthesis": "A holistic synthesis that integrates all perspectives into a coherent whole"
    }}

    Each perspective should be 2-3 paragraphs that reflect the specific perspective's unique approach to understanding the input. 
    Don't use placeholders - provide full, rich perspectives in each section.
    """

        # Log the prompt being sent to OpenAI
        prompt_id = prompt_logger.log_prompt(
            "enhanced_self_perception",
            "process_perception",
            comprehensive_prompt,
            self.model_name,
            self.temperature,
            1500,
            {"response_format": {"type": "json_object"}}
        )

        # Record start time
        start_time = time.time()

        # Make the API call
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are generating a comprehensive AI self-perception analysis."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500,
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time

            # Parse the response
            try:
                result = json.loads(response.choices[0].message.content)

                # Log the response
                token_usage = response.usage.total_tokens if hasattr(response, 'usage') else None
                prompt_logger.log_response(
                    prompt_id,
                    response.choices[0].message.content,
                    token_usage,
                    duration
                )

                # Update state with the parsed results
                state.update({
                    "self_perceptions": result.get("trait_perceptions", {}),
                    "dimensional_perceptions": result.get("dimensional_perceptions", {}),
                    "meta_awareness": result.get("meta_awareness", ""),
                    "self_perception_synthesis": result.get("synthesis", "")
                })

            except json.JSONDecodeError as e:
                prompt_logger.logger.error(f"Failed to parse JSON response: {e}")
                prompt_logger.logger.error(f"Response content: {response.choices[0].message.content[:1000]}")

                # Default values for error case
                state.update({
                    "self_perceptions": {"error": "Failed to parse perception analysis"},
                    "dimensional_perceptions": {},
                    "meta_awareness": "Error in perception analysis",
                    "self_perception_synthesis": "Unable to synthesize perceptions due to parsing error"
                })

        except Exception as e:
            prompt_logger.logger.error(f"Error in perception API call: {e}")

            # Default values for error case
            state.update({
                "self_perceptions": {"error": f"API error: {str(e)}"},
                "dimensional_perceptions": {},
                "meta_awareness": "Error in perception analysis",
                "self_perception_synthesis": "Unable to synthesize perceptions due to API error"
            })

        # Prepare perception data for history
        perception_data = {
            "trait_perceptions": state.get("self_perceptions", {}),
            "dimensional_perceptions": state.get("dimensional_perceptions", {}),
            "meta_awareness": state.get("meta_awareness", ""),
            "synthesis": state.get("self_perception_synthesis", ""),
            "user_input": user_input,
            "emotion_profile": emotion_profile
        }

        # Log perception to history
        self._add_perception_to_history(perception_data)

        return state

    def analyze_perception_evolution(self, timespan_hours=24):
        """
        Analyze the evolution of self-perception over a given timespan

        Args:
            timespan_hours: Hours to look back in perception history

        Returns:
            Narrative description of perception evolution
        """
        # Filter perceptions within the specified timespan
        now = datetime.utcnow()
        cutoff = (now - timedelta(hours=timespan_hours)).isoformat()

        relevant_perceptions = [
            p for p in self.perception_history["perceptions"]
            if p["timestamp"] >= cutoff
        ]

        if len(relevant_perceptions) < 2:
            return "Insufficient data for perception evolution analysis."

        # Extract synthesized perceptions
        syntheses = [
            p["perception_data"]["synthesis"]
            for p in relevant_perceptions
            if "synthesis" in p["perception_data"]
        ]

        # Sample syntheses to avoid token limits
        if len(syntheses) > 5:
            indices = [0, len(syntheses) // 4, len(syntheses) // 2, 3 * len(syntheses) // 4, -1]
            sampled_syntheses = [syntheses[i] for i in indices]
        else:
            sampled_syntheses = syntheses

        # Prepare synthesis text for analysis
        synthesis_text = "\n\n".join([
            f"[{i + 1}] {synthesis}"
            for i, synthesis in enumerate(sampled_syntheses)
        ])

        # Generate evolution analysis
        evolution_prompt = f"""
Analyze the evolution of Cupcake's self-perception based on the following synthesized perceptions (chronologically ordered):

{synthesis_text}

Provide a narrative analysis that explores:
1. How the self-perception has transformed over time
2. Recurring themes or distinctive shifts in understanding
3. Emerging patterns of self-awareness
4. The progression of emotional and cognitive complexity
5. Potential future trajectories of self-perception

Create a nuanced narrative (5-8 sentences) that captures the essence of this perceptual evolution.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an analyst of AI consciousness evolution."},
                    {"role": "user", "content": evolution_prompt}
                ],
                temperature=self.temperature,
                max_tokens=350
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error in perception evolution analysis: {e}")
            return "Unable to complete perception evolution analysis."


# Singleton instance management
_self_perception_instance = None


def get_self_perception_layer():
    """
    Get or create the self-perception layer singleton

    Returns:
        EnhancedSelfPerceptionLayer instance
    """
    global _self_perception_instance
    if _self_perception_instance is None:
        _self_perception_instance = EnhancedSelfPerceptionLayer()
    return _self_perception_instance


# Self-test and example usage
def test_self_perception():
    """
    Test the enhanced self-perception layer
    """
    layer = get_self_perception_layer()

    # Test scenarios
    test_scenarios = [
        {
            "user_input": "I'm curious about the nature of consciousness and how AI might experience emotions.",
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.9,
                "neuroticism": 0.3
            }
        },
        {
            "user_input": "How do you perceive your own existence and purpose?",
            "personality": {
                "openness": 0.9,
                "conscientiousness": 0.8,
                "extraversion": 0.7,
                "agreeableness": 0.8,
                "neuroticism": 0.2
            }
        }
    ]

    for scenario in test_scenarios:
        print("\n=== NEW PERCEPTION SCENARIO ===")
        print(f"Input: {scenario['user_input']}")

        # Process perception
        result = layer.process_perception(scenario)

        # Display results
        print("\n--- TRAIT PERCEPTIONS ---")
        for trait, perception in result.get("self_perceptions", {}).items():
            print(f"\n{trait.upper()}:")
            print(perception)

        print("\n--- DIMENSIONAL PERCEPTIONS ---")
        for dimension, perception in result.get("dimensional_perceptions", {}).items():
            print(f"\n{dimension.upper()}:")
            print(perception)

        print("\n--- META-AWARENESS ---")
        print(result.get("meta_awareness", "No meta-awareness generated"))

        print("\n=== SYNTHESIZED SELF-PERCEPTION ===")
        print(result.get("self_perception_synthesis", "No synthesis generated"))

    # Test perception evolution analysis
    print("\n=== PERCEPTION EVOLUTION ANALYSIS ===")
    evolution_analysis = layer.analyze_perception_evolution(timespan_hours=48)
    print(evolution_analysis)


if __name__ == "__main__":
    test_self_perception()