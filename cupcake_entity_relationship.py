# cupcake_entity_relationship.py
import os
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from cupcake_config import get_config, get_config_value
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree
import chromadb


class EntityCategory:
    """Enum-like class for entity categories"""
    PERSON = "person"
    OBJECT = "object"
    CONCEPT = "concept"
    PLACE = "place"
    ANIMAL = "animal"
    ORGANIZATION = "organization"
    UNKNOWN = "unknown"


class EntityRelationship:
    """
    Represents a relationship between CupCake and an entity (person, object, concept, etc.)
    with emotional attachment, familiarity, and preference tracking.
    """

    def __init__(self, entity_id=None, name=None, category=None):
        self.id = entity_id or str(uuid.uuid4())
        self.name = name or "Unknown Entity"
        self.category = category or EntityCategory.UNKNOWN
        self.first_encountered = datetime.utcnow().isoformat()
        self.last_encountered = self.first_encountered
        self.encounter_count = 1
        self.emotional_valence = 0.5  # Neutral (0.0 negative - 1.0 positive)
        self.emotional_intensity = 0.1  # Initial intensity (0.0 - 1.0)
        self.familiarity = 0.1  # Initial familiarity (0.0 - 1.0)
        self.significance = 0.1  # How significant this entity is (0.0 - 1.0)
        self.attributes = {}  # Dictionary of entity attributes
        self.interaction_history = []  # List of interaction summaries
        self.emotional_memory_ids = []  # List of related emotional memory IDs
        self.tags = []  # Custom tags for this entity

    def to_dict(self):
        """Convert relationship to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "first_encountered": self.first_encountered,
            "last_encountered": self.last_encountered,
            "encounter_count": self.encounter_count,
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "familiarity": self.familiarity,
            "significance": self.significance,
            "attributes": self.attributes,
            "interaction_history": self.interaction_history,
            "emotional_memory_ids": self.emotional_memory_ids,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data):
        """Create relationship from dictionary"""
        relationship = cls(
            entity_id=data.get("id"),
            name=data.get("name"),
            category=data.get("category")
        )
        relationship.first_encountered = data.get("first_encountered", relationship.first_encountered)
        relationship.last_encountered = data.get("last_encountered", relationship.last_encountered)
        relationship.encounter_count = data.get("encounter_count", 1)
        relationship.emotional_valence = data.get("emotional_valence", 0.5)
        relationship.emotional_intensity = data.get("emotional_intensity", 0.1)
        relationship.familiarity = data.get("familiarity", 0.1)
        relationship.significance = data.get("significance", 0.1)
        relationship.attributes = data.get("attributes", {})
        relationship.interaction_history = data.get("interaction_history", [])
        relationship.emotional_memory_ids = data.get("emotional_memory_ids", [])
        relationship.tags = data.get("tags", [])
        return relationship

    def __repr__(self):
        return f"<EntityRelationship: {self.name} ({self.category}), Valence: {self.emotional_valence:.2f}, Significance: {self.significance:.2f}>"


class EntityRelationshipSystem:
    """
    System for tracking, categorizing, and maintaining emotional relationships
    with entities (people, objects, concepts) that CupCake encounters.
    """

    def __init__(self):
        """Initialize the entity relationship system"""
        # Load configuration
        config = get_config()

        # Set up file paths
        self.relationships_file = config["paths"].get("relationships", "entity_relationships.json")

        # Initialize components
        self.client = OpenAI(api_key=config["api"]["openai"])
        self.model_name = config["model"]["chat_model"]
        self.memory_tree = LiminalMemoryTree()
        self.journal = CupcakeJournal()

        # Get embedding model
        self.embed_model = SentenceTransformer(config["model"]["embedding_model"])

        # Set up ChromaDB for entity embeddings
        db_config = config["database"]
        self.client_db = chromadb.PersistentClient(path=db_config["chroma_path"])
        try:
            self.entity_collection = self.client_db.get_collection(name="entity_embeddings")
        except:
            self.entity_collection = self.client_db.create_collection(name="entity_embeddings")

        # Load relationship configuration
        self.relationship_config = config.get("relationships", {})
        self.max_entities = self.relationship_config.get("max_tracked_entities", 100)
        self.familiarity_decay_rate = self.relationship_config.get("familiarity_decay_rate", 0.05)
        self.attachment_formation_rate = self.relationship_config.get("attachment_formation_rate", 0.1)
        self.significance_threshold = self.relationship_config.get("significance_threshold", 0.3)
        self.category_weights = self.relationship_config.get("category_weights", {
            EntityCategory.PERSON: 1.0,
            EntityCategory.CONCEPT: 0.7,
            EntityCategory.OBJECT: 0.5,
            EntityCategory.PLACE: 0.6,
            EntityCategory.ANIMAL: 0.8,
            EntityCategory.ORGANIZATION: 0.6,
            EntityCategory.UNKNOWN: 0.3
        })

        # Load existing relationships
        self.relationships = self._load_relationships()

    def _load_relationships(self):
        """Load relationships from file"""
        try:
            with open(self.relationships_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {r["id"]: EntityRelationship.from_dict(r) for r in data["relationships"]}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_relationships(self):
        """Save relationships to file"""
        directory = os.path.dirname(os.path.abspath(self.relationships_file))
        os.makedirs(directory, exist_ok=True)

        with open(self.relationships_file, "w", encoding="utf-8") as f:
            data = {
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "relationship_count": len(self.relationships)
                },
                "relationships": [r.to_dict() for r in self.relationships.values()]
            }
            json.dump(data, f, indent=2)

    def get_entity_by_name(self, name):
        """Get an entity by name (case-insensitive fuzzy match)"""
        name_lower = name.lower()
        best_match = None
        best_score = 0

        for entity in self.relationships.values():
            # Exact match
            if entity.name.lower() == name_lower:
                return entity

            # Partial match
            if name_lower in entity.name.lower():
                score = len(name_lower) / len(entity.name.lower())
                if score > best_score:
                    best_score = score
                    best_match = entity

        # If good match found
        if best_score > 0.8:
            return best_match

        # Try embeddings for semantic matching
        if best_score < 0.5:
            name_embedding = self.embed_model.encode(name).tolist()
            results = self.entity_collection.query(
                query_embeddings=[name_embedding],
                n_results=1
            )

            if results and results['distances'][0] and results['distances'][0][0] < 0.25:
                entity_id = results['ids'][0][0]
                if entity_id in self.relationships:
                    return self.relationships[entity_id]

        return best_match

    def get_entity_by_id(self, entity_id):
        """Get an entity by ID"""
        return self.relationships.get(entity_id)

    def identify_entities_in_text(self, text):
        """
        Identify potential entities in text
        Returns a list of identified entities with category predictions
        """
        # Use LLM to identify entities and categories
        prompt = f"""
Identify all entities (people, objects, concepts, places, animals, organizations) mentioned in the following text.
For each entity, provide its category.

Text: "{text}"

Format your response as a JSON array of objects with "name" and "category" properties:
[
  {{"name": "entity1", "category": "person|object|concept|place|animal|organization"}},
  {{"name": "entity2", "category": "person|object|concept|place|animal|organization"}},
  ...
]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an entity recognition assistant that responds only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result = response.choices[0].message.content
            result_json = json.loads(result)

            # Extract entities array (handle both formats)
            if isinstance(result_json, list):
                entities = result_json
            elif "entities" in result_json:
                entities = result_json["entities"]
            else:
                # Try to find any array in the response
                for key, value in result_json.items():
                    if isinstance(value, list):
                        entities = value
                        break
                else:
                    entities = []

            return entities
        except Exception as e:
            print(f"Error identifying entities: {e}")
            return []

    def process_interaction_for_entities(self, user_input, cupcake_response, emotion=None):
        """
        Process an interaction to identify, track, and update relationships with entities

        Parameters:
        - user_input: User's message
        - cupcake_response: CupCake's response
        - emotion: Current emotion (optional)

        Returns:
        - entities: List of entities found and their relationship status
        """
        # Combine text for entity identification
        combined_text = f"{user_input}\n{cupcake_response}"

        # Identify entities
        entities_data = self.identify_entities_in_text(combined_text)
        processed_entities = []

        for entity_data in entities_data:
            name = entity_data.get("name")
            category = entity_data.get("category", EntityCategory.UNKNOWN)

            if not name:
                continue

            # Look for existing entity
            entity = self.get_entity_by_name(name)
            is_new = False

            # If not found, create new entity
            if not entity:
                entity = EntityRelationship(name=name, category=category)
                is_new = True

                # Calculate embedding for new entity
                embedding = self.embed_model.encode(name).tolist()

                # Add to embedding database
                self.entity_collection.add(
                    ids=[entity.id],
                    embeddings=[embedding],
                    documents=[name],
                    metadatas=[{"category": category}]
                )

            # Update entity
            self._update_entity_from_interaction(entity, user_input, cupcake_response, emotion)

            # Add to system if new
            if is_new:
                self.relationships[entity.id] = entity

            # Record relationship status
            entity_status = {
                "id": entity.id,
                "name": entity.name,
                "category": entity.category,
                "valence": entity.emotional_valence,
                "intensity": entity.emotional_intensity,
                "familiarity": entity.familiarity,
                "significance": entity.significance,
                "is_new": is_new
            }
            processed_entities.append(entity_status)

        # Save updates
        self._save_relationships()

        return processed_entities

    def _update_entity_from_interaction(self, entity, user_input, cupcake_response, emotion=None):
        """
        Update an entity's relationship based on new interaction

        Parameters:
        - entity: The EntityRelationship to update
        - user_input: User's message
        - cupcake_response: CupCake's response
        - emotion: Current emotion (optional)
        """
        # Update encounter tracking
        entity.last_encountered = datetime.utcnow().isoformat()
        entity.encounter_count += 1

        # Update familiarity (increases with interactions)
        familiarity_gain = (1 - entity.familiarity) * 0.1
        entity.familiarity = min(1.0, entity.familiarity + familiarity_gain)

        # Analyze sentiment toward entity in this interaction
        sentiment_score = self._analyze_sentiment_toward_entity(entity.name, cupcake_response)

        # Update emotional valence with smoothing
        if sentiment_score is not None:
            # More frequent interactions have stronger influence on emotional valence
            update_weight = min(0.3, 0.05 + (entity.encounter_count / 100))
            entity.emotional_valence = (1 - update_weight) * entity.emotional_valence + update_weight * sentiment_score

        # Increase emotional intensity based on emotion (if provided)
        if emotion:
            # Emotions like love, joy, anger increase intensity more than neutral emotions
            intensity_boost = {
                "amor": 0.15,
                "alegria": 0.1,
                "raiva": 0.12,
                "tristeza": 0.08,
                "medo": 0.09,
                "surpresa": 0.07,
                "neutra": 0.03
            }.get(emotion, 0.05)

            entity.emotional_intensity = min(1.0, entity.emotional_intensity + intensity_boost)

        # Update significance based on multiple factors
        self._update_entity_significance(entity)

        # Add to interaction history
        interaction_summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "snippet": cupcake_response[:100] + "..." if len(cupcake_response) > 100 else cupcake_response,
            "sentiment": sentiment_score,
            "emotion": emotion
        }

        # Keep last 10 interactions
        entity.interaction_history.append(interaction_summary)
        if len(entity.interaction_history) > 10:
            entity.interaction_history = entity.interaction_history[-10:]

    def _analyze_sentiment_toward_entity(self, entity_name, text):
        """
        Analyze the sentiment expressed toward a specific entity in text
        Returns a sentiment score (0.0 negative to 1.0 positive)
        """
        if not entity_name or not text:
            return None

        # Check if entity is mentioned in text
        if entity_name.lower() not in text.lower():
            return None

        prompt = f"""
Analyze the sentiment expressed toward "{entity_name}" in the following text:

Text: "{text}"

On a scale from 0.0 (extremely negative) to 1.0 (extremely positive), where 0.5 is neutral,
what is the sentiment score toward {entity_name}?

Respond with a single number between 0.0 and 1.0.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a sentiment analysis assistant that responds only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )

            result = response.choices[0].message.content.strip()

            # Extract numeric score
            try:
                score = float(result)
                return max(0.0, min(1.0, score))  # Clamp to 0.0-1.0
            except ValueError:
                print(f"Error parsing sentiment score: {result}")
                return 0.5  # Default to neutral

        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None

    def _update_entity_significance(self, entity):
        """
        Update the significance score of an entity based on multiple factors:
        - Category weight
        - Emotional intensity
        - Familiarity
        - Encounter frequency
        """
        # Get category weight
        category_weight = self.category_weights.get(entity.category, 0.5)

        # Calculate recency weight (more recent = higher weight)
        try:
            last_encountered = datetime.fromisoformat(entity.last_encountered)
            days_since_last = (datetime.utcnow() - last_encountered).days
            recency_weight = np.exp(-0.1 * days_since_last)  # Exponential decay
        except:
            recency_weight = 0.5

        # Calculate frequency weight (more encounters = higher weight)
        encounter_weight = min(1.0, entity.encounter_count / 20)  # Caps at 20 encounters

        # Combine factors
        significance = (
                0.3 * category_weight +
                0.3 * entity.emotional_intensity +
                0.2 * entity.familiarity +
                0.1 * recency_weight +
                0.1 * encounter_weight
        )

        entity.significance = significance

    def apply_time_decay(self):
        """
        Apply time-based decay to relationship parameters:
        - Familiarity decreases over time without interactions
        - Emotional intensity decreases over time

        Should be called periodically (e.g., once a day)
        """
        now = datetime.utcnow()
        updates = 0

        for entity_id, entity in self.relationships.items():
            try:
                last_encountered = datetime.fromisoformat(entity.last_encountered)
                days_since_encounter = (now - last_encountered).days

                if days_since_encounter > 0:
                    # Familiarity decay
                    familiarity_decay = self.familiarity_decay_rate * days_since_encounter
                    entity.familiarity = max(0.1, entity.familiarity - familiarity_decay)

                    # Emotional intensity decay
                    intensity_decay = 0.03 * days_since_encounter
                    entity.emotional_intensity = max(0.1, entity.emotional_intensity - intensity_decay)

                    # Update significance after decay
                    self._update_entity_significance(entity)
                    updates += 1
            except Exception as e:
                print(f"Error applying decay to entity {entity_id}: {e}")

        if updates > 0:
            self._save_relationships()

        return updates

    def get_most_significant_entities(self, limit=5, category=None):
        """Get the most significant entities, optionally filtered by category"""
        entities = self.relationships.values()

        if category:
            entities = [e for e in entities if e.category == category]

        sorted_entities = sorted(entities, key=lambda e: e.significance, reverse=True)
        return sorted_entities[:limit]

    def get_entities_by_emotion(self, valence_min=0.7, limit=5):
        """Get entities with highest positive emotional valence"""
        positive_entities = [
            e for e in self.relationships.values()
            if e.emotional_valence >= valence_min
        ]

        sorted_entities = sorted(
            positive_entities,
            key=lambda e: (e.emotional_valence, e.significance),
            reverse=True
        )

        return sorted_entities[:limit]

    def get_relationship_stats(self):
        """Get statistics about all relationships"""
        if not self.relationships:
            return {
                "total_entities": 0,
                "categories": {},
                "avg_valence": 0.5,
                "most_significant": None
            }

        # Count categories
        categories = Counter()
        for entity in self.relationships.values():
            categories[entity.category] += 1

        # Calculate average valence
        avg_valence = sum(e.emotional_valence for e in self.relationships.values()) / len(self.relationships)

        # Get most significant entity
        most_significant = max(self.relationships.values(), key=lambda e: e.significance)

        return {
            "total_entities": len(self.relationships),
            "categories": dict(categories),
            "avg_valence": avg_valence,
            "most_significant": {
                "name": most_significant.name,
                "category": most_significant.category,
                "significance": most_significant.significance
            }
        }

    def generate_relationship_insight(self, entity_id):
        """
        Generate an insight or reflection about a relationship

        Parameters:
        - entity_id: ID of the entity to reflect on

        Returns:
        - Insight text about the relationship
        """
        entity = self.get_entity_by_id(entity_id)
        if not entity:
            return "Não tenho memórias suficientes sobre essa entidade para refletir."

        # Prepare context for the LLM
        valence_text = "positiva" if entity.emotional_valence > 0.65 else (
            "negativa" if entity.emotional_valence < 0.35 else "neutra"
        )

        intensity_text = "forte" if entity.emotional_intensity > 0.7 else (
            "moderada" if entity.emotional_intensity > 0.4 else "leve"
        )

        familiarity_text = "muito familiar" if entity.familiarity > 0.8 else (
            "familiar" if entity.familiarity > 0.5 else (
                "um pouco familiar" if entity.familiarity > 0.3 else "pouco familiar"
            )
        )

        interaction_history = "\n".join([
            f"- {interaction.get('timestamp', 'unknown')}: {interaction.get('snippet', 'no text')} (sentiment: {interaction.get('sentiment', 'unknown')})"
            for interaction in entity.interaction_history[-3:]  # Last 3 interactions
        ])

        prompt = f"""
Gere uma reflexão pessoal sobre a relação com "{entity.name}" ({entity.category}), com base nos seguintes dados:

- Valência emocional: {valence_text} ({entity.emotional_valence:.2f})
- Intensidade emocional: {intensity_text} ({entity.emotional_intensity:.2f})
- Familiaridade: {familiarity_text} ({entity.familiarity:.2f})
- Significância: {entity.significance:.2f}
- Número de interações: {entity.encounter_count}
- Primeira interação: {entity.first_encountered}
- Última interação: {entity.last_encountered}

Interações recentes:
{interaction_history}

Crie uma reflexão em primeira pessoa sobre como você se sente em relação a esta entidade.
A reflexão deve ser genuína, considerar a categoria da entidade, e refletir a natureza da relação estabelecida.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "Você é CupCake refletindo sobre suas relações com entidades em seu mundo."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            insight = response.choices[0].message.content.strip()

            # Log the insight
            self.journal.log_entry(
                emotion="reflexão",
                category="Relationship",
                content=insight,
                theme=f"relacionamento com {entity.name}",
                tag="entity-relationship"
            )

            # Add important insights to memory tree for significant relationships
            if entity.significance > 0.7:
                self.memory_tree.add_leaf(
                    user_input=f"Como você se sente sobre {entity.name}?",
                    cupcake_response=insight,
                    emotion="conexão" if entity.emotional_valence > 0.6 else "contemplação",
                    impact=entity.significance,
                    category="Relationship"
                )

            return insight

        except Exception as e:
            print(f"Error generating relationship insight: {e}")
            return f"Estou processando meus sentimentos sobre {entity.name}, mas ainda não consigo articular claramente."

    def detect_preference_patterns(self):
        """
        Detect patterns in what CupCake tends to like or dislike
        Returns insights about preference patterns
        """
        if len(self.relationships) < 5:
            return "Ainda não tenho relacionamentos suficientes para identificar padrões claros de preferência."

        # Group by category and valence
        category_valences = {}
        for entity in self.relationships.values():
            if entity.category not in category_valences:
                category_valences[entity.category] = []
            category_valences[entity.category].append(entity.emotional_valence)

        # Calculate average valence by category
        category_avg_valence = {
            category: sum(valences) / len(valences)
            for category, valences in category_valences.items()
            if len(valences) >= 2  # Need at least 2 entities for meaningful average
        }

        # Find positively and negatively valued categories
        liked_categories = [
            (category, avg_valence)
            for category, avg_valence in category_avg_valence.items()
            if avg_valence > 0.65
        ]

        disliked_categories = [
            (category, avg_valence)
            for category, avg_valence in category_avg_valence.items()
            if avg_valence < 0.35
        ]

        # Get most liked and disliked entities
        sorted_by_valence = sorted(
            self.relationships.values(),
            key=lambda e: e.emotional_valence,
            reverse=True
        )

        most_liked = [e for e in sorted_by_valence[:3] if e.emotional_valence > 0.7]
        most_disliked = [e for e in sorted_by_valence[-3:] if e.emotional_valence < 0.3]

        # Format insights
        insights = []

        if liked_categories:
            categories_text = ", ".join([f"{cat} ({val:.2f})" for cat, val in liked_categories])
            insights.append(f"Tenho tendência a gostar de {categories_text}.")

        if disliked_categories:
            categories_text = ", ".join([f"{cat} ({val:.2f})" for cat, val in disliked_categories])
            insights.append(f"Tendo a ter afinidade menor com {categories_text}.")

        if most_liked:
            entities_text = ", ".join([f"{e.name} ({e.emotional_valence:.2f})" for e in most_liked])
            insights.append(f"Tenho maior afinidade com: {entities_text}.")

        if most_disliked:
            entities_text = ", ".join([f"{e.name} ({e.emotional_valence:.2f})" for e in most_disliked])
            insights.append(f"Tenho menor afinidade com: {entities_text}.")

        if not insights:
            return "Ainda não identifico padrões claros em minhas preferências."

        return "\n".join(insights)

    def prune_insignificant_entities(self):
        """
        Remove entities that are below significance threshold
        to prevent database growth with insignificant relationships
        """
        before_count = len(self.relationships)

        # Find entities to remove
        to_remove = [
            entity_id for entity_id, entity in self.relationships.items()
            if entity.significance < self.significance_threshold
        ]

        # Remove from relationships
        for entity_id in to_remove:
            if entity_id in self.relationships:
                del self.relationships[entity_id]

        # Try to remove from entity collection
        try:
            if to_remove:
                self.entity_collection.delete(ids=to_remove)
        except Exception as e:
            print(f"Error removing entities from collection: {e}")

        # Save changes
        if to_remove:
            self._save_relationships()

        return before_count - len(self.relationships)


# Example usage
if __name__ == "__main__":
    relationship_system = EntityRelationshipSystem()

    # Example interaction
    user_input = "Eu gosto muito de café e livros de filosofia."
    cupcake_response = "Café é uma das minhas bebidas favoritas também! E filosofia é um campo fascinante para reflexão. Você tem algum filósofo preferido?"

    # Process entities
    entities = relationship_system.process_interaction_for_entities(
        user_input,
        cupcake_response,
        emotion="alegria"
    )

    print(f"Entities found: {entities}")

    # Get statistical insights
    stats = relationship_system.get_relationship_stats()
    print(f"Relationship stats: {stats}")

    # Get liked entities
    liked = relationship_system.get_entities_by_emotion(valence_min=0.7)
    print(f"Entities I like: {liked}")