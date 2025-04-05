# cupcake_sensors.py
from datetime import datetime
import os
import json
import time
from cupcake_config import get_config, get_config_value


class CupCakeSensors:
    """
    Environmental sensor system for CupCake that provides awareness of:
    - Time of day
    - Interaction patterns
    - Memory emotional distribution
    - Identity coherence
    """

    def __init__(self):
        """Initialize the sensor system with configuration values"""
        config = get_config()

        # Get file paths from config
        self.paths = config["paths"]
        self.silence_track_file = self.paths.get("last_interaction", "last_interaction.txt")
        self.memory_grove_file = self.paths.get("memory_grove", "memory_grove.json")

        # Get sensor thresholds from config
        self.sensor_config = config.get("sensors", {})
        self.neutral_emotion_threshold = self.sensor_config.get("neutral_emotion_threshold", 0.3)
        self.conflict_similarity_threshold = self.sensor_config.get("conflict_similarity_threshold", 15)

    def detect_time_of_day(self):
        """Detect the current time of day (morning, afternoon, night)"""
        hour = datetime.now().hour
        if hour >= 20 or hour < 6:
            return "noite"
        elif hour >= 12:
            return "tarde"
        else:
            return "manhã"

    def update_last_interaction(self):
        """Record the current time as the last interaction timestamp"""
        with open(self.silence_track_file, "w") as f:
            f.write(str(time.time()))

    def time_since_last_interaction(self):
        """Calculate time elapsed since the last interaction"""
        if not os.path.exists(self.silence_track_file):
            return float('inf')  # Never interacted

        try:
            with open(self.silence_track_file, "r") as f:
                last = float(f.read().strip())
            return time.time() - last
        except (ValueError, IOError) as e:
            # Handle file read errors gracefully
            print(f"Error reading interaction time: {e}")
            return float('inf')

    def count_neutral_memories(self, collection):
        """Count memories with low emotional impact"""
        try:
            memories = collection.get(include=['metadatas'])
            return sum(
                1 for meta in memories['metadatas'] if meta.get('emotion_score', 0.0) < self.neutral_emotion_threshold)
        except Exception as e:
            print(f"Error counting neutral memories: {e}")
            return 0

    def check_identity_conflicts(self):
        """Detect potential conflicts between thought patterns"""
        if not os.path.exists(self.memory_grove_file):
            return []

        try:
            with open(self.memory_grove_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get thoughts from relevant categories
            thought_categories = ("Thought", "Self Model", "Reflection")
            thoughts = [leaf for leaf in data.get("leaves", []) if leaf.get("category") in thought_categories]

            # Extract thought content
            themes = [t.get("cupcake_response", "") for t in thoughts if t.get("cupcake_response")]

            # Find potential contradictions (pairs of dissimilar thoughts)
            contradictions = []
            for i in range(len(themes)):
                for j in range(i + 1, len(themes)):
                    # Skip very short thoughts
                    if len(themes[i]) < 20 or len(themes[j]) < 20:
                        continue

                    # Compare thought beginnings (simplified heuristic)
                    similarity_length = min(len(themes[i]), len(themes[j]), self.conflict_similarity_threshold)
                    if themes[i][:similarity_length].lower() != themes[j][:similarity_length].lower():
                        contradictions.append((themes[i], themes[j]))

            # Return the most recent contradictions
            return contradictions[-3:] if contradictions else []

        except Exception as e:
            print(f"Error checking identity conflicts: {e}")
            return []

    def measure_emotional_distribution(self, collection):
        """Analyze the distribution of emotions in memories"""
        try:
            memories = collection.get(include=['metadatas'])
            if not memories or not memories['metadatas']:
                return {"distribution": {}, "dominant": None, "average_intensity": 0}

            # Count emotion types
            emotion_counts = {}
            intensity_sum = 0
            count = 0

            for meta in memories['metadatas']:
                emotion = meta.get('emotion_type', 'neutral')
                score = meta.get('emotion_score', 0.5)

                # Count emotion occurrences
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # Sum intensities for average calculation
                intensity_sum += score
                count += 1

            # Find dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else (None, 0)

            # Calculate average intensity
            avg_intensity = intensity_sum / count if count > 0 else 0

            return {
                "distribution": emotion_counts,
                "dominant": dominant_emotion[0],
                "average_intensity": avg_intensity
            }
        except Exception as e:
            print(f"Error measuring emotional distribution: {e}")
            return {"distribution": {}, "dominant": None, "average_intensity": 0}

    def detect_activity_patterns(self):
        """Analyze patterns in interaction frequency"""
        if not os.path.exists(self.silence_track_file):
            return {"pattern": "unknown", "frequency": "unknown"}

        current_time = datetime.now()
        current_hour = current_time.hour

        time_category = "morning" if 5 <= current_hour < 12 else "afternoon" if 12 <= current_hour < 18 else "evening" if 18 <= current_hour < 22 else "night"

        silence_duration = self.time_since_last_interaction()

        if silence_duration < 300:  # 5 minutes
            frequency = "active_conversation"
        elif silence_duration < 3600:  # 1 hour
            frequency = "periodic_checking"
        elif silence_duration < 21600:  # 6 hours
            frequency = "occasional"
        else:
            frequency = "rare"

        return {
            "pattern": time_category,
            "frequency": frequency,
            "silence_seconds": silence_duration
        }

    def run_all_sensors(self, collection):
        """Run all sensors and return a comprehensive environmental report"""
        time_of_day = self.detect_time_of_day()
        activity = self.detect_activity_patterns()
        silence = self.time_since_last_interaction()
        neutral_count = self.count_neutral_memories(collection)
        identity_conflicts = self.check_identity_conflicts()
        emotional_distribution = self.measure_emotional_distribution(collection)

        return {
            "temporal": {
                "time_of_day": time_of_day,
                "weekday": datetime.now().strftime("%A"),
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            "interaction": {
                "seconds_since_interaction": silence,
                "activity_pattern": activity["pattern"],
                "interaction_frequency": activity["frequency"]
            },
            "memory": {
                "neutral_memories_count": neutral_count,
                "emotional_distribution": emotional_distribution["distribution"],
                "dominant_emotion": emotional_distribution["dominant"],
                "average_emotional_intensity": emotional_distribution["average_intensity"]
            },
            "identity": {
                "conflict_count": len(identity_conflicts),
                "has_conflicts": len(identity_conflicts) > 0,
                "conflicts": identity_conflicts
            }
        }


# For backward compatibility and easier transition
# Create a singleton instance
_sensor_system = CupCakeSensors()


# Legacy function interfaces that use the singleton
def detect_time_of_day():
    return _sensor_system.detect_time_of_day()


def update_last_interaction():
    _sensor_system.update_last_interaction()


def time_since_last_interaction():
    return _sensor_system.time_since_last_interaction()


def count_neutral_memories(collection):
    return _sensor_system.count_neutral_memories(collection)


def check_identity_conflicts(path=None):
    # If path is provided, we'll just use the default from the sensor system
    return _sensor_system.check_identity_conflicts()


def run_sensors(collection):
    """Legacy function that provides basic sensor data"""
    report = _sensor_system.run_all_sensors(collection)

    # Flatten for backward compatibility
    return {
        "tempo_do_dia": report["temporal"]["time_of_day"],
        "segundos_desde_interacao": report["interaction"]["seconds_since_interaction"],
        "memorias_neutras": report["memory"]["neutral_memories_count"],
        "conflitos_identitarios": report["identity"]["conflicts"]
    }


# Example usage
if __name__ == "__main__":
    print("Running CupCake sensors test...")

    # Create sensor system
    sensors = CupCakeSensors()

    # Test basic sensors
    print(f"Time of day: {sensors.detect_time_of_day()}")
    print(f"Time since last interaction: {sensors.time_since_last_interaction():.2f} seconds")

    # Test activity pattern detection
    activity = sensors.detect_activity_patterns()
    print(f"Activity pattern: {activity['pattern']} - {activity['frequency']}")

    # Update last interaction
    sensors.update_last_interaction()
    print("Updated last interaction timestamp")
    print(f"New time since last interaction: {sensors.time_since_last_interaction():.2f} seconds")