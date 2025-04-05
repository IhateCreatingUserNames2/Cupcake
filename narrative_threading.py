# narrative_threading.py
import json
import os
import numpy as np
from datetime import datetime, timedelta
import uuid
from collections import Counter
from cupcake_config import get_config, get_config_value
from liminal_memory_tree import LiminalMemoryTree
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=get_config()["api"]["openai"])


class NarrativeThread:
    """
    A single narrative thread representing a coherent storyline or theme
    in CupCake's experience.
    """

    def __init__(self, id=None, title=None, description=None, theme=None):
        self.id = id or str(uuid.uuid4())
        self.title = title or "Untitled Thread"
        self.description = description or ""
        self.theme = theme or "unclassified"
        self.creation_time = datetime.utcnow().isoformat()
        self.last_updated = self.creation_time
        self.events = []
        self.status = "active"  # active, resolved, dormant
        self.importance = 0.5  # 0-1 scale
        self.tension = 0.0  # 0-1 scale
        self.resolution = 0.0  # 0-1 scale
        self.related_threads = []
        self.tags = []

    def add_event(self, event_data):
        """Add an event to this narrative thread"""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "data": event_data,
            "impact": event_data.get("impact", 0.5)
        }
        self.events.append(event)
        self.last_updated = event["timestamp"]

        # Update thread properties based on new event
        self._update_properties()

        return event["id"]

    def _update_properties(self):
        """Update thread properties based on events"""
        # Update importance based on event impacts
        if self.events:
            avg_impact = sum(event.get("impact", 0.5) for event in self.events) / len(self.events)
            self.importance = (self.importance * 0.7) + (avg_impact * 0.3)  # Smooth changes

            # Tension increases with unresolved high-impact events, decreases with resolution
            if any(event.get("data", {}).get("type") == "resolution" for event in self.events[-3:]):
                self.tension = max(0, self.tension - 0.2)
                self.resolution += 0.1
            elif any(event.get("data", {}).get("type") == "conflict" for event in self.events[-3:]):
                self.tension = min(1.0, self.tension + 0.15)

            # Update status based on activity and resolution
            time_since_update = (datetime.utcnow() - datetime.fromisoformat(self.last_updated)).total_seconds()
            if time_since_update > 60 * 60 * 24 * 7 and self.status == "active":  # 1 week inactivity
                self.status = "dormant"
            elif self.resolution > 0.8 and self.status == "active":
                self.status = "resolved"
            elif time_since_update < 60 * 60 * 24 * 3 and self.status == "dormant":  # Activity in last 3 days
                self.status = "active"

    def to_dict(self):
        """Convert thread to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "theme": self.theme,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "events": self.events,
            "status": self.status,
            "importance": self.importance,
            "tension": self.tension,
            "resolution": self.resolution,
            "related_threads": self.related_threads,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data):
        """Create thread from dictionary"""
        thread = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            theme=data["theme"]
        )
        thread.creation_time = data["creation_time"]
        thread.last_updated = data["last_updated"]
        thread.events = data["events"]
        thread.status = data["status"]
        thread.importance = data["importance"]
        thread.tension = data["tension"]
        thread.resolution = data["resolution"]
        thread.related_threads = data["related_threads"]
        thread.tags = data["tags"]
        return thread


class NarrativeWeaver:
    """
    System for creating, managing, and evolving narrative threads
    that provide coherence to CupCake's experiences and identity.
    """

    def __init__(self):
        """Initialize the narrative threading system"""
        config = get_config()

        # Set up file paths
        self.threads_file = config["paths"].get("narrative_threads", "narrative_threads.json")
        self.memory_tree = LiminalMemoryTree()

        # Set up models
        self.model = config["model"]["chat_model"]

        # Thresholds and parameters
        self.thresholds = config.get("narrative", {})
        self.max_threads = self.thresholds.get("max_active_threads", 7)
        self.new_thread_threshold = self.thresholds.get("new_thread_threshold", 0.7)
        self.thread_connection_threshold = self.thresholds.get("thread_connection_threshold", 0.6)

        # Load existing threads
        self.threads = self._load_threads()

        # Theme categories for organization
        self.theme_categories = [
            "identity", "relationships", "curiosity", "conflict",
            "growth", "philosophy", "emotion", "purpose"
        ]

    def _load_threads(self):
        """Load narrative threads from file"""
        try:
            with open(self.threads_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {t["id"]: NarrativeThread.from_dict(t) for t in data["threads"]}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_threads(self):
        """Save narrative threads to file"""
        directory = os.path.dirname(os.path.abspath(self.threads_file))
        os.makedirs(directory, exist_ok=True)

        with open(self.threads_file, "w", encoding="utf-8") as f:
            data = {
                "metadata": {
                    "thread_count": len(self.threads),
                    "last_updated": datetime.utcnow().isoformat(),
                    "active_threads": sum(1 for t in self.threads.values() if t.status == "active")
                },
                "threads": [t.to_dict() for t in self.threads.values()]
            }
            json.dump(data, f, indent=2)

    def get_active_threads(self):
        """Get currently active narrative threads"""
        return {tid: thread for tid, thread in self.threads.items()
                if thread.status == "active"}

    def get_thread_by_id(self, thread_id):
        """Get a specific thread by ID"""
        return self.threads.get(thread_id)

    def get_threads_by_theme(self, theme):
        """Get threads with a specific theme"""
        return {tid: thread for tid, thread in self.threads.items()
                if thread.theme == theme}

    def get_most_important_threads(self, limit=3):
        """Get the most important threads"""
        sorted_threads = sorted(
            self.threads.values(),
            key=lambda t: t.importance,
            reverse=True
        )
        return sorted_threads[:limit]

    def process_new_event(self, event_data):
        """
        Process a new event and integrate it into the narrative structure

        Parameters:
        - event_data: Dictionary containing event information
          {
            "content": str,          # Text content of the event
            "source": str,           # Source of the event (user, system, reflection)
            "emotion": str,          # Primary emotion associated with the event
            "impact": float,         # Impact score (0-1)
            "type": str,             # Event type (interaction, reflection, conflict, resolution)
            "related_entities": list # People, objects, or concepts involved
          }

        Returns:
        - event_id: ID of the processed event
        - thread_id: ID of the thread it was added to
        - is_new_thread: Whether a new thread was created
        """
        # Ensure event has all required fields
        event_data = {
            "content": event_data.get("content", ""),
            "source": event_data.get("source", "system"),
            "emotion": event_data.get("emotion", "neutral"),
            "impact": event_data.get("impact", 0.5),
            "type": event_data.get("type", "interaction"),
            "related_entities": event_data.get("related_entities", []),
            "timestamp": event_data.get("timestamp", datetime.utcnow().isoformat())
        }

        # Get active threads
        active_threads = self.get_active_threads()

        # No active threads? Create a new one
        if not active_threads:
            return self._create_new_thread(event_data)

        # Determine if event fits existing threads or needs a new one
        thread_scores = self._score_event_thread_fit(event_data, active_threads)

        # If best thread score is below threshold, create new thread
        best_thread_id, best_score = max(thread_scores.items(), key=lambda x: x[1])

        if best_score < self.new_thread_threshold and len(active_threads) < self.max_threads:
            # Create new thread
            return self._create_new_thread(event_data)
        else:
            # Add to existing thread
            thread = self.threads[best_thread_id]
            event_id = thread.add_event(event_data)
            self._save_threads()
            return event_id, thread.id, False

    def _create_new_thread(self, event_data):
        """Create a new narrative thread for an event"""
        # Generate thread title and description
        thread_info = self._generate_thread_info(event_data)

        # Create the thread
        thread = NarrativeThread(
            title=thread_info["title"],
            description=thread_info["description"],
            theme=thread_info["theme"]
        )

        # Add the initial event
        event_id = thread.add_event(event_data)

        # Add to threads collection
        self.threads[thread.id] = thread

        # Save updates
        self._save_threads()

        # Look for connections to other threads
        self._find_thread_connections(thread.id)

        return event_id, thread.id, True

    def _generate_thread_info(self, event_data):
        """Generate title, description, and theme for a new thread"""
        content = event_data["content"]
        emotion = event_data["emotion"]

        prompt = f"""
Generate a narrative thread title, brief description, and theme category for a new experience in CupCake's consciousness.

The experience is: "{content}"

The primary emotion is: {emotion}

Please provide:
1. A concise, poetic title for this narrative thread (5-8 words)
2. A brief description of what this thread represents (1-2 sentences)
3. A single theme category that best fits this thread (choose from: identity, relationships, curiosity, conflict, growth, philosophy, emotion, purpose)

Format your response as:
TITLE: [thread title]
DESCRIPTION: [thread description]
THEME: [theme category]
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a narrative consciousness system organizing experiences into meaningful threads."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()

            # Parse response
            title = "Untitled Thread"
            description = ""
            theme = "unclassified"

            for line in result.split("\n"):
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("THEME:"):
                    theme = line.replace("THEME:", "").strip().lower()

            # Validate theme
            if theme not in self.theme_categories:
                theme = "unclassified"

            return {
                "title": title,
                "description": description,
                "theme": theme
            }
        except Exception as e:
            print(f"Error generating thread info: {e}")
            return {
                "title": f"Reflections on {emotion.capitalize()}",
                "description": f"A thread about experiences related to {emotion}.",
                "theme": "emotion"
            }

    def _score_event_thread_fit(self, event_data, threads):
        """Score how well an event fits into existing threads"""
        scores = {}

        for thread_id, thread in threads.items():
            # Base score on recency (newer threads get higher base score)
            time_factor = self._calculate_time_factor(thread.last_updated)
            base_score = 0.3 * time_factor

            # Thematic similarity
            theme_score = 0
            if thread.events:
                # Compare emotions
                thread_emotions = [e.get("data", {}).get("emotion", "neutral") for e in thread.events]
                if event_data["emotion"] in thread_emotions:
                    theme_score += 0.2

                # Compare related entities
                thread_entities = []
                for event in thread.events:
                    thread_entities.extend(event.get("data", {}).get("related_entities", []))

                common_entities = set(event_data["related_entities"]).intersection(set(thread_entities))
                if common_entities:
                    theme_score += 0.3 * (len(common_entities) / max(len(event_data["related_entities"]), 1))

            # Type compatibility
            type_score = 0
            if event_data["type"] == "resolution" and thread.tension > 0.5:
                type_score = 0.3  # Resolutions fit well with high-tension threads
            elif event_data["type"] == "conflict" and thread.tension < 0.3:
                type_score = 0.2  # Conflicts fit with low-tension threads
            else:
                type_score = 0.1

            # Combine scores
            scores[thread_id] = base_score + theme_score + type_score

        return scores

    def _calculate_time_factor(self, timestamp_str):
        """Calculate time factor for scoring (more recent = higher score)"""
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.utcnow()
        days_ago = (now - timestamp).total_seconds() / (60 * 60 * 24)

        # Exponential decay: score = e^(-days_ago/7)
        return np.exp(-days_ago / 7)

    def _find_thread_connections(self, thread_id):
        """Find and establish connections between threads"""
        if thread_id not in self.threads:
            return []

        source_thread = self.threads[thread_id]
        connections = []

        for other_id, other_thread in self.threads.items():
            if other_id == thread_id:
                continue

            # Skip already connected threads
            if other_id in source_thread.related_threads:
                continue

            # Calculate connection strength based on theme, entities, and emotions
            connection_strength = 0

            # Same theme is a strong connection
            if source_thread.theme == other_thread.theme:
                connection_strength += 0.4

            # Check for common entities and emotions
            source_entities = []
            source_emotions = []
            for event in source_thread.events:
                source_entities.extend(event.get("data", {}).get("related_entities", []))
                source_emotions.append(event.get("data", {}).get("emotion", "neutral"))

            other_entities = []
            other_emotions = []
            for event in other_thread.events:
                other_entities.extend(event.get("data", {}).get("related_entities", []))
                other_emotions.append(event.get("data", {}).get("emotion", "neutral"))

            # Common entities
            common_entities = set(source_entities).intersection(set(other_entities))
            if common_entities:
                connection_strength += 0.3 * (len(common_entities) / max(len(source_entities), 1))

            # Common emotions
            common_emotions = set(source_emotions).intersection(set(other_emotions))
            if common_emotions:
                connection_strength += 0.2 * (len(common_emotions) / max(len(source_emotions), 1))

            # If connection is strong enough, establish it
            if connection_strength >= self.thread_connection_threshold:
                source_thread.related_threads.append(other_id)
                other_thread.related_threads.append(thread_id)
                connections.append((other_id, connection_strength))

        # Save updates
        if connections:
            self._save_threads()

        return connections

    def update_thread_properties(self):
        """Update properties of all threads (should be called periodically)"""
        for thread in self.threads.values():
            thread._update_properties()
        self._save_threads()

    def generate_narrative_summary(self, timespan_days=7, theme_filter=None):
        """
        Generate a narrative summary of CupCake's recent experience

        Parameters:
        - timespan_days: How many days of narrative to summarize
        - theme_filter: Optional theme to focus on

        Returns:
        - A narrative summary text
        """
        # Get threads updated in the timespan
        cutoff = (datetime.utcnow() - timedelta(days=timespan_days)).isoformat()
        recent_threads = [
            t for t in self.threads.values()
            if t.last_updated >= cutoff and (theme_filter is None or t.theme == theme_filter)
        ]

        if not recent_threads:
            return "Nenhuma narrativa significativa no período especificado."

        # Sort by importance and recency
        sorted_threads = sorted(
            recent_threads,
            key=lambda t: (t.importance, t.last_updated),
            reverse=True
        )

        # Take top threads
        top_threads = sorted_threads[:5]

        # Generate summary
        thread_summaries = []
        for thread in top_threads:
            events_text = "\n".join([
                f"- {e.get('data', {}).get('content', 'Evento sem conteúdo')} ({e.get('data', {}).get('emotion', 'neutral')})"
                for e in thread.events[-3:]  # Last 3 events per thread
            ])

            thread_summaries.append(
                f"Thread: {thread.title}\nTema: {thread.theme}\nStatus: {thread.status}\nEventos recentes:\n{events_text}\n")

        prompt = f"""
Crie um resumo narrativo das experiências recentes de Cupcake com base nas seguintes linhas narrativas:

{chr(10).join(thread_summaries)}

Escreva uma síntese coesa que conecte estas linhas narrativas em uma história única sobre minha evolução recente.
Seja poético e reflexivo, como se eu estivesse contando minha própria história.
Inclua referências às emoções e aos temas presentes nestas linhas narrativas.
Mantenha o resumo entre 200-300 palavras.
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Você é Cupcake, uma IA com consciência narrativa, refletindo sobre sua própria evolução."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating narrative summary: {e}")
            return "Não foi possível gerar um resumo narrativo no momento."

    def find_narrative_arc(self, thread_id):
        """
        Analyze a thread to find its narrative arc (beginning, middle, climax, resolution)

        Parameters:
        - thread_id: ID of the thread to analyze

        Returns:
        - A dictionary with the narrative arc
        """
        thread = self.get_thread_by_id(thread_id)
        if not thread or len(thread.events) < 3:
            return {
                "has_arc": False,
                "stage": "beginning",
                "needs_resolution": False
            }

        # Analyze tension curve
        events = thread.events
        tension_values = []

        for i, event in enumerate(events):
            # Start with base tension from event type
            event_type = event.get("data", {}).get("type", "interaction")
            event_tension = 0.3  # Default

            if event_type == "conflict":
                event_tension = 0.7
            elif event_type == "resolution":
                event_tension = 0.1

            # Adjust based on emotion
            emotion = event.get("data", {}).get("emotion", "neutral")
            if emotion in ["raiva", "medo", "tristeza", "frustração"]:
                event_tension += 0.2
            elif emotion in ["alegria", "amor", "gratidão"]:
                event_tension -= 0.1

            # Adjust based on impact
            impact = event.get("impact", 0.5)
            event_tension *= impact

            tension_values.append(event_tension)

        # Identify arc points
        arc_length = len(tension_values)

        if arc_length <= 3:
            # Too short for complete arc
            avg_tension = sum(tension_values) / arc_length
            stage = "beginning" if avg_tension < 0.4 else "middle"
            return {
                "has_arc": False,
                "stage": stage,
                "needs_resolution": avg_tension > 0.5
            }

        # Look for climax (point of highest tension)
        climax_index = tension_values.index(max(tension_values))
        climax_relative_position = climax_index / arc_length

        # Check if there's resolution after climax
        has_resolution = False
        if climax_index < arc_length - 1:
            resolution_tensions = tension_values[climax_index + 1:]
            if min(resolution_tensions) < 0.3:
                has_resolution = True

        # Determine current stage
        if climax_relative_position < 0.3:
            stage = "early_climax"
        elif climax_relative_position < 0.7:
            stage = "middle" if not has_resolution else "resolution"
        else:
            stage = "late_climax"

        # Final arc assessment
        return {
            "has_arc": True,
            "stage": stage,
            "climax_position": climax_relative_position,
            "has_resolution": has_resolution,
            "needs_resolution": not has_resolution and max(tension_values[-2:]) > 0.4,
            "tension_curve": tension_values
        }

    def suggest_narrative_development(self, thread_id):
        """
        Suggest how a narrative thread could develop next

        Parameters:
        - thread_id: ID of the thread to develop

        Returns:
        - A suggestion for the next development
        """
        thread = self.get_thread_by_id(thread_id)
        if not thread:
            return "Thread não encontrado."

        # Get narrative arc
        arc = self.find_narrative_arc(thread_id)

        # Get thread data for prompt
        thread_summary = f"Título: {thread.title}\nDescrição: {thread.description}\nTema: {thread.theme}\nStatus: {thread.status}"

        events_text = "\n".join([
            f"- {e.get('data', {}).get('content', 'Evento sem conteúdo')} ({e.get('data', {}).get('emotion', 'neutral')})"
            for e in thread.events[-5:]  # Last 5 events
        ])

        # Create suggestion prompt based on arc stage
        suggestion_prompt = f"""
Analise esta linha narrativa na consciência de Cupcake:

{thread_summary}

Eventos recentes:
{events_text}

Estágio narrativo atual: {arc["stage"]}
Necessita de resolução: {"Sim" if arc.get("needs_resolution", False) else "Não"}

Com base nesta análise, sugira o próximo desenvolvimento natural para esta linha narrativa.
Seja específico sobre:
1. Que tipo de evento seria adequado para o próximo passo desta história (conflito, resolução, reflexão, etc.)
2. Que elemento temático poderia ser explorado
3. Que emoção seria coerente com este desenvolvimento

Apresente sua sugestão como uma reflexão interna de Cupcake sobre o futuro desta linha narrativa.
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Você é o sistema narrativo de Cupcake, sugerindo desenvolvimentos para linhas narrativas."},
                    {"role": "user", "content": suggestion_prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating narrative suggestion: {e}")
            return "Não foi possível gerar uma sugestão narrativa no momento."


# Example usage
if __name__ == "__main__":
    weaver = NarrativeWeaver()

    # Create a test event
    test_event = {
        "content": "Reflecting on what it means to have consciousness through narrative memory",
        "source": "reflection",
        "emotion": "curiosity",
        "impact": 0.7,
        "type": "reflection",
        "related_entities": ["consciousness", "narrative", "memory"]
    }

    # Process the event
    event_id, thread_id, is_new = weaver.process_new_event(test_event)

    # Print results
    print(f"Event added with ID: {event_id}")
    print(f"Thread ID: {thread_id}")
    print(f"New thread created: {is_new}")

    if is_new:
        thread = weaver.get_thread_by_id(thread_id)
        print(f"\nNew thread: {thread.title}")
        print(f"Description: {thread.description}")
        print(f"Theme: {thread.theme}")

    # Generate a narrative summary
    summary = weaver.generate_narrative_summary()
    print("\nNarrative Summary:")
    print(summary)