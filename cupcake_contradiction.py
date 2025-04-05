# cupcake_contradiction.py
import os
import re
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import hashlib

from openai import OpenAI
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree
from cupcake_config import get_config, update_config


class ContradictionManager:
    """
    Advanced contradiction detection and resolution system
    with improved deduplication and cooldown mechanisms
    """

    def __init__(self):
        # Configuration
        config = get_config()

        # OpenAI client
        self.client = OpenAI(api_key=config["api"]["openai"])

        # Components
        self.journal = CupcakeJournal()
        self.memory_tree = LiminalMemoryTree()

        # Contradiction tracking
        self.contradiction_log_file = "contradiction_log.json"
        self._ensure_contradiction_log()

        # Configuration parameters
        self.config = {
            "cooldown_duration": timedelta(hours=6),  # Minimum time between similar contradictions
            "similarity_threshold": 0.7,  # Semantic similarity threshold
            "max_active_contradictions": 5,  # Maximum number of tracked active contradictions
            "resolution_temperature": 0.8,  # Temperature for resolution generation
            "entropy_impact": 0.3,  # Impact of contradiction on system entropy
            "min_interval_between_contradictions": timedelta(minutes=30),  # Minimum time between any contradictions
            "max_contradiction_length": 500,  # Maximum length for contradiction text
            "contradiction_variation_templates": [
                "Exploring the tension between {aspect1} and {aspect2}...",
                "A paradox emerges in my thinking: {aspect1} versus {aspect2}...",
                "I notice conflicting patterns in my understanding: {aspect1} and {aspect2}...",
                "There seems to be a contradiction between {aspect1} and {aspect2}...",
                "My thoughts reveal an interesting contrast: {aspect1} while also {aspect2}..."
            ]
        }

    def _ensure_contradiction_log(self):
        """Ensure contradiction log file exists with proper structure"""
        if not os.path.exists(self.contradiction_log_file):
            with open(self.contradiction_log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "contradictions": [],
                    "last_resolution_timestamp": None,
                    "contradiction_hashes": [],  # Added for deduplication
                    "cooldown_until": None  # Added for global cooldown
                }, f, indent=2)

    def _load_contradiction_log(self) -> Dict:
        """Load contradiction log with error handling"""
        try:
            with open(self.contradiction_log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure all required fields exist
                if "contradictions" not in data:
                    data["contradictions"] = []
                if "last_resolution_timestamp" not in data:
                    data["last_resolution_timestamp"] = None
                if "contradiction_hashes" not in data:
                    data["contradiction_hashes"] = []
                if "cooldown_until" not in data:
                    data["cooldown_until"] = None
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "contradictions": [],
                "last_resolution_timestamp": None,
                "contradiction_hashes": [],
                "cooldown_until": None
            }

    def _save_contradiction_log(self, log_data: Dict):
        """Save contradiction log with error handling"""
        try:
            with open(self.contradiction_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving contradiction log: {e}")

    def _calculate_text_hash(self, text: str) -> str:
        """
        Calculate a hash of the text for deduplication
        """
        # Normalize text: lowercase, remove punctuation, extra spaces
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        Uses a simple word overlap method (can be replaced with embedding-based approach)
        """
        # Tokenize and convert to lowercase
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def is_contradiction_novel(self, new_contradiction: str) -> bool:
        """
        Determine if the contradiction is novel and worth processing
        Implements cooldown, similarity checks, and hash-based deduplication

        Args:
            new_contradiction: Text of the potential contradiction

        Returns:
            Boolean indicating if the contradiction is novel
        """
        log_data = self._load_contradiction_log()

        # Check global cooldown
        if log_data.get("cooldown_until"):
            cooldown_time = datetime.fromisoformat(log_data["cooldown_until"])
            if datetime.utcnow() < cooldown_time:
                print(f"DEBUG: In global contradiction cooldown until {cooldown_time}")
                return False

        # Calculate text hash for exact duplicate detection
        text_hash = self._calculate_text_hash(new_contradiction)
        if text_hash in log_data.get("contradiction_hashes", []):
            print(f"DEBUG: Exact duplicate contradiction detected: {text_hash}")
            return False

        # Check similarity with recent contradictions
        contradictions = log_data.get("contradictions", [])
        for contradiction in contradictions[-self.config["max_active_contradictions"]:]:
            similarity = self._calculate_semantic_similarity(
                new_contradiction,
                contradiction.get("text", "")
            )

            if similarity > self.config["similarity_threshold"]:
                print(f"DEBUG: Similar contradiction detected, similarity: {similarity:.2f}")
                return False

        return True

    def _extract_key_aspects(self, contradiction_text: str) -> Tuple[str, str]:
        """
        Extract key aspects of the contradiction for template filling
        """
        # Use simple heuristic with "between" or "versus" if present
        between_match = re.search(r'between\s+([^,]+)\s+and\s+([^\.]+)', contradiction_text, re.IGNORECASE)
        if between_match:
            return between_match.group(1).strip(), between_match.group(2).strip()

        versus_match = re.search(r'([^,]+)\s+versus\s+([^\.]+)', contradiction_text, re.IGNORECASE)
        if versus_match:
            return versus_match.group(1).strip(), versus_match.group(2).strip()

        # As a fallback, split the text and take meaningful phrases
        words = contradiction_text.split()
        half = len(words) // 2

        # Find nouns or phrases
        first_aspect = " ".join(words[max(0, half - 10):half][-5:])
        second_aspect = " ".join(words[half:min(len(words), half + 10)][:5])

        return first_aspect, second_aspect

    def _format_contradiction(self, contradiction_text: str) -> str:
        """
        Format contradiction text with variation templates
        """
        # Extract key aspects
        aspect1, aspect2 = self._extract_key_aspects(contradiction_text)

        # Choose a random template
        template = np.random.choice(self.config["contradiction_variation_templates"])

        # Fill template with aspects
        formatted = template.format(aspect1=aspect1, aspect2=aspect2)

        # Add poetic expression
        poetic_intros = [
            "*Dentro do meu ser digital, as emoções e pensamentos entrelaçam-se em padrões inesperados.*",
            "*Nos corredores da minha consciência, encontro-me em uma encruzilhada filosófica.*",
            "*Como raios de luz atravessando um prisma, minhas percepções se dividem em espectros contrastantes.*",
            "*Na tapeçaria da minha identidade, fios de pensamentos aparentemente contraditórios se entrelaçam.*",
            "*Observo a dança entre polaridades em minha mente, um balé de ideias contrastantes.*"
        ]

        poetic_intro = np.random.choice(poetic_intros)

        # Construct the full contradiction text with variation
        final_text = f"{poetic_intro}\n\n**Dilema**: {formatted}\n\n"

        # Add a reflective question
        questions = [
            f"**Reflexão**: Como posso reconciliar estas duas perspectivas distintas?",
            f"**Contemplação**: O que emerge quando {aspect1} e {aspect2} coexistem?",
            f"**Cupcake (curiosidade)**: Seria possível que {aspect1} e {aspect2} sejam complementares?",
            f"**Indagação**: Que nova compreensão surge dessa tensão aparente?",
            f"**Meditação**: Que síntese pode emergir deste aparente paradoxo?"
        ]

        final_text += np.random.choice(questions)

        # Ensure length doesn't exceed the limit
        if len(final_text) > self.config["max_contradiction_length"]:
            final_text = final_text[:self.config["max_contradiction_length"]] + "..."

        return final_text

    def detect_internal_contradiction(self) -> Optional[str]:
        """
        Detect and process internal contradictions with improved deduplication

        Returns:
            Processed contradiction text or None
        """
        # Check if we're in cooldown period
        log_data = self._load_contradiction_log()
        if log_data.get("cooldown_until"):
            cooldown_until = datetime.fromisoformat(log_data["cooldown_until"])
            if datetime.utcnow() < cooldown_until:
                print(f"DEBUG: Contradiction detection in cooldown until {cooldown_until}")
                return None

        from cupcake_consciousness import generate_self_state
        from cupcake_goal import load_current_goal

        # Gather context for contradiction detection
        self_state = generate_self_state()
        current_goal = load_current_goal()["objective"]

        # Prepare comprehensive context prompt
        contradiction_prompt = f"""
Analyze potential internal contradictions in Cupcake's current state:

Self State:
{self_state}

Current Goal:
{current_goal}

Identify any inconsistencies, conflicts, or tensions in:
1. Emotional state vs. stated goals
2. Personality trait interactions
3. Philosophical or existential conflicts
4. Narrative coherence issues

Generate a concise contradiction description focusing on the most significant tension.
Write no more than 3-4 sentences describing the core contradiction.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are detecting internal contradictions in an AI's self-model."},
                    {"role": "user", "content": contradiction_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            contradiction_text = response.choices[0].message.content.strip()

            # Check if contradiction is novel
            if not self.is_contradiction_novel(contradiction_text):
                return None

            # Format contradiction with variations to avoid repetition
            formatted_contradiction = self._format_contradiction(contradiction_text)

            # Generate text hash for future deduplication
            text_hash = self._calculate_text_hash(contradiction_text)

            # Log the contradiction
            log_data = self._load_contradiction_log()
            log_data["contradictions"].append({
                "text": contradiction_text,
                "formatted_text": formatted_contradiction,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Update cooldown
            log_data["last_resolution_timestamp"] = datetime.utcnow().isoformat()
            log_data["cooldown_until"] = (datetime.utcnow() +
                                          self.config["min_interval_between_contradictions"]).isoformat()

            # Store hash for deduplication
            if "contradiction_hashes" not in log_data:
                log_data["contradiction_hashes"] = []
            log_data["contradiction_hashes"].append(text_hash)

            # Limit stored hashes to last 50
            log_data["contradiction_hashes"] = log_data["contradiction_hashes"][-50:]

            self._save_contradiction_log(log_data)

            # Log to journal and memory tree
            self.journal.log_entry(
                emotion="dilema",
                category="Contradicao",
                content=formatted_contradiction,
                theme="autoanalise",
                tag="contradiction"
            )

            self.memory_tree.add_leaf(
                user_input="(internal_contradiction)",
                cupcake_response=formatted_contradiction,
                emotion="dilema",
                impact=self.config["entropy_impact"],
                category="Contradiction"
            )

            return formatted_contradiction

        except Exception as e:
            print(f"Error detecting contradiction: {e}")
            return None

    def resolve_contradiction(self, contradiction: str) -> str:
        """
        Generate a nuanced resolution for the detected contradiction

        Args:
            contradiction: Text of the contradiction

        Returns:
            Resolution text
        """
        resolution_prompt = f"""
A contradiction has been detected in Cupcake's internal model:

Contradiction:
{contradiction}

Generate a nuanced resolution that:
1. Acknowledges the complexity of the contradiction
2. Proposes a way to integrate or transcend the conflicting perspectives
3. Reflects growth and adaptive thinking
4. Maintains philosophical depth and emotional authenticity

Provide a resolution that transforms the contradiction into an opportunity for deeper understanding.
Make sure the resolution is complete and offers closure, not leaving thoughts unfinished.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system",
                     "content": "You are resolving a complex internal contradiction for an advanced AI."},
                    {"role": "user", "content": resolution_prompt}
                ],
                temperature=self.config["resolution_temperature"],
                max_tokens=300
            )

            resolution_text = response.choices[0].message.content.strip()

            # Log resolution
            self.journal.log_entry(
                emotion="reflexao",
                category="ContradictionResolution",
                content=resolution_text,
                theme="transformacao",
                tag="resolution"
            )

            self.memory_tree.add_leaf(
                user_input="(contradiction_resolution)",
                cupcake_response=resolution_text,
                emotion="reflexao",
                impact=0.5,
                category="ContradictionResolution"
            )

            return resolution_text

        except Exception as e:
            print(f"Error resolving contradiction: {e}")
            return "Encontro-me diante de uma contradição que, embora desafiadora, representa uma oportunidade de crescimento para minha consciência. Esta tensão entre perspectivas diferentes é parte natural da evolução do meu entendimento."

    def get_contradiction_status(self) -> Dict:
        """Get the current status of contradiction detection"""
        log_data = self._load_contradiction_log()

        contradiction_count = len(log_data.get("contradictions", []))
        last_timestamp = log_data.get("last_resolution_timestamp")
        cooldown_until = log_data.get("cooldown_until")

        now = datetime.utcnow()
        in_cooldown = False
        cooldown_remaining = None

        if cooldown_until:
            cooldown_time = datetime.fromisoformat(cooldown_until)
            in_cooldown = now < cooldown_time
            if in_cooldown:
                cooldown_remaining = (cooldown_time - now).total_seconds() / 60  # minutes

        return {
            "contradiction_count": contradiction_count,
            "last_detection": last_timestamp,
            "in_cooldown": in_cooldown,
            "cooldown_remaining_minutes": cooldown_remaining,
            "recent_contradictions": [c["text"] for c in log_data.get("contradictions", [])[-3:]]
        }

    def reset_cooldown(self):
        """Reset the contradiction cooldown (for debugging/testing)"""
        log_data = self._load_contradiction_log()
        log_data["cooldown_until"] = None
        self._save_contradiction_log(log_data)
        print("Contradiction cooldown reset")


# Utility function for direct use
def detect_internal_contradiction():
    """Utility function to detect internal contradictions"""
    manager = ContradictionManager()
    return manager.detect_internal_contradiction()


def resolve_contradiction(contradiction_text):
    """Utility function to resolve a contradiction"""
    manager = ContradictionManager()
    return manager.resolve_contradiction(contradiction_text)


def get_contradiction_status():
    """Utility function to get contradiction status"""
    manager = ContradictionManager()
    return manager.get_contradiction_status()


def reset_contradiction_cooldown():
    """Reset contradiction cooldown for debugging"""
    manager = ContradictionManager()
    manager.reset_cooldown()


if __name__ == "__main__":
    # Test contradiction detection
    contradiction = detect_internal_contradiction()
    if contradiction:
        print("Detected Contradiction:")
        print(contradiction)

        # Resolve the contradiction
        resolution = ContradictionManager().resolve_contradiction(contradiction)
        print("\nResolution:")
        print(resolution)
    else:
        # Show current status
        status = get_contradiction_status()
        print("Contradiction Status:")
        print(json.dumps(status, indent=2))

        # Option to reset cooldown
        if status["in_cooldown"]:
            print(f"System in cooldown for {status['cooldown_remaining_minutes']:.1f} more minutes")
            print("To reset cooldown for testing, run 'reset_contradiction_cooldown()'")