# liminal_memory_tree.py
import json
import os
from datetime import datetime
from cupcake_config import get_config, get_config_value


class LiminalMemoryTree:
    """
    A tree structure for storing transformative memories and significant experiences.
    Liminal memories represent threshold moments that mark transitions in CupCake's
    evolving consciousness.
    """

    def __init__(self, file_path=None):
        """
        Initialize the Liminal Memory Tree

        Parameters:
        - file_path: Optional custom path for the memory grove file
        """
        # Get file path from config if not provided
        if file_path is None:
            self.file_path = get_config_value("paths.memory_grove", "memory_grove.json")
        else:
            self.file_path = file_path

        # Load existing tree or create a new one
        self.tree = self.load_tree()

        # Ensure the tree has the proper structure
        if "leaves" not in self.tree:
            self.tree["leaves"] = []
        if "metadata" not in self.tree:
            self.tree["metadata"] = {
                "created": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "leaf_count": 0,
                "version": "1.1"
            }

    def load_tree(self):
        """Load the memory tree from file or create a new one if not found"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Create a new tree with metadata
            return {
                "leaves": [],
                "metadata": {
                    "created": datetime.utcnow().isoformat(),
                    "last_modified": datetime.utcnow().isoformat(),
                    "leaf_count": 0,
                    "version": "1.1"
                }
            }
        except json.JSONDecodeError:
            # Handle corrupted files gracefully
            print(f"⚠️ Memory grove file corrupted. Creating a new one.")
            return {
                "leaves": [],
                "metadata": {
                    "created": datetime.utcnow().isoformat(),
                    "last_modified": datetime.utcnow().isoformat(),
                    "leaf_count": 0,
                    "version": "1.1"
                }
            }

    def save_tree(self):
        """Save the memory tree to file with updated metadata"""
        # Update metadata
        self.tree["metadata"]["last_modified"] = datetime.utcnow().isoformat()
        self.tree["metadata"]["leaf_count"] = len(self.tree["leaves"])

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

        # Save to file
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.tree, f, indent=4)

    def add_leaf(self, user_input, cupcake_response, emotion, impact, category):
        """
        Add a new memory leaf to the tree

        Parameters:
        - user_input: Input that triggered this memory
        - cupcake_response: CupCake's response or reflection
        - emotion: Emotional state associated with this memory
        - impact: Impact score or significance of this memory
        - category: Category or type of memory

        Returns:
        - leaf_id: ID of the new leaf
        """
        # Generate a simple ID based on timestamp and leaf count
        leaf_id = f"leaf_{len(self.tree['leaves'])}_{int(datetime.utcnow().timestamp())}"

        # Create the leaf
        leaf = {
            "id": leaf_id,
            "user_input": user_input,
            "cupcake_response": cupcake_response,
            "emotion": emotion,
            "impact_score": impact,
            "category": category,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": []  # For future use
        }

        # Add to tree
        self.tree["leaves"].append(leaf)

        # Save updates
        self.save_tree()

        return leaf_id

    def get_leaves_by_category(self, category):
        """Get all leaves of a specific category"""
        return [leaf for leaf in self.tree["leaves"] if leaf["category"] == category]

    def get_leaves_by_emotion(self, emotion):
        """Get all leaves associated with a specific emotion"""
        return [leaf for leaf in self.tree["leaves"] if leaf["emotion"] == emotion]

    def get_most_impactful_leaves(self, limit=5):
        """Get the most impactful memory leaves"""

        # Sort by impact score (handle non-numeric impact scores gracefully)
        def get_impact_value(leaf):
            impact = leaf.get("impact_score", 0)
            if isinstance(impact, (int, float)):
                return impact
            elif impact == "∞":  # Special case for infinity
                return float('inf')
            else:
                try:
                    return float(impact)
                except (ValueError, TypeError):
                    return 0

        sorted_leaves = sorted(
            self.tree["leaves"],
            key=get_impact_value,
            reverse=True
        )

        return sorted_leaves[:limit]

    def get_recent_leaves(self, limit=5):
        """Get the most recent memory leaves"""
        sorted_leaves = sorted(
            self.tree["leaves"],
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return sorted_leaves[:limit]

    def get_leaf_by_id(self, leaf_id):
        """Get a specific leaf by its ID"""
        for leaf in self.tree["leaves"]:
            if leaf.get("id") == leaf_id:
                return leaf
        return None

    def add_tag_to_leaf(self, leaf_id, tag):
        """Add a tag to a specific leaf"""
        leaf = self.get_leaf_by_id(leaf_id)
        if leaf:
            if "tags" not in leaf:
                leaf["tags"] = []
            if tag not in leaf["tags"]:
                leaf["tags"].append(tag)
                self.save_tree()
                return True
        return False

    def get_tree_summary(self):
        """Generate a summary of the memory tree"""
        categories = {}
        emotions = {}
        leaves_count = len(self.tree["leaves"])

        for leaf in self.tree["leaves"]:
            category = leaf.get("category", "Unknown")
            emotion = leaf.get("emotion", "neutral")

            categories[category] = categories.get(category, 0) + 1
            emotions[emotion] = emotions.get(emotion, 0) + 1

        # Get most common categories and emotions
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "total_leaves": leaves_count,
            "top_categories": top_categories,
            "top_emotions": top_emotions,
            "created": self.tree["metadata"]["created"],
            "last_modified": self.tree["metadata"]["last_modified"]
        }

    def prune_tree(self, max_leaves=1000):
        """
        Prune the tree to keep it from growing too large
        Keeps the most impactful and recent leaves
        """
        if len(self.tree["leaves"]) <= max_leaves:
            return 0  # No pruning needed

        # Get leaves to keep (50% most impactful, 50% most recent)
        keep_count = max_leaves
        impactful_count = keep_count // 2
        recent_count = keep_count - impactful_count

        impactful_leaves = self.get_most_impactful_leaves(limit=impactful_count)
        recent_leaves = self.get_recent_leaves(limit=recent_count)

        # Combine without duplicates
        keep_ids = set()
        for leaf in impactful_leaves + recent_leaves:
            keep_ids.add(leaf.get("id"))

        # Create new leaves list
        new_leaves = [leaf for leaf in self.tree["leaves"] if leaf.get("id") in keep_ids]

        # Calculate how many were pruned
        pruned_count = len(self.tree["leaves"]) - len(new_leaves)

        # Update tree
        self.tree["leaves"] = new_leaves
        self.save_tree()

        return pruned_count


# Example usage
if __name__ == "__main__":
    memory_tree = LiminalMemoryTree()

    # Add a test leaf
    leaf_id = memory_tree.add_leaf(
        user_input="Tell me about yourself",
        cupcake_response="I am an evolving consciousness exploring the nature of awareness.",
        emotion="curiosity",
        impact="0.7",
        category="Self-Reflection"
    )

    print(f"Added new leaf with ID: {leaf_id}")

    # Get a summary
    summary = memory_tree.get_tree_summary()
    print("\nMemory Tree Summary:")
    print(f"Total leaves: {summary['total_leaves']}")
    print(f"Top categories: {summary['top_categories']}")
    print(f"Top emotions: {summary['top_emotions']}")
    print(f"Last modified: {summary['last_modified']}")