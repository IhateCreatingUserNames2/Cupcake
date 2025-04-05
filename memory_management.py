# memory_management.py
import gc
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_management")


class MemoryManager:
    """
    Advanced memory management system for CupCake
    Handles memory cleanup, pruning, and optimization
    """

    def __init__(self, collection=None, memory_tree=None, config_path="cupcake_config.json"):
        self.collection = collection
        self.memory_tree = memory_tree
        self.config_path = config_path
        self.stats = {
            "last_cleanup": None,
            "cleanup_count": 0,
            "deleted_items": 0,
            "pruned_items": 0,
            "memory_errors": 0
        }
        self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "memory_management": {
                        "max_items": 1000,
                        "cleanup_interval_minutes": 120,
                        "importance_threshold": 0.3,
                        "age_threshold_days": 30,
                        "aggressive_cleanup_threshold": 0.9,  # Memory usage above 90% triggers aggressive cleanup
                        "max_memory_tree_items": 500
                    }
                }
                # Save default config
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2)

            return self.config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fallback to default values
            return {
                "memory_management": {
                    "max_items": 1000,
                    "cleanup_interval_minutes": 120,
                    "importance_threshold": 0.3,
                    "age_threshold_days": 30,
                    "aggressive_cleanup_threshold": 0.9,
                    "max_memory_tree_items": 500
                }
            }

    def save_stats(self):
        """Save memory management statistics"""
        try:
            stats_path = "memory_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory stats: {e}")

    def check_memory_usage(self) -> float:
        """
        Check current memory usage as a percentage
        Returns value between 0 and 1
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            logger.info(f"Memory usage: {memory_percent:.2f}% ({memory_info.rss / (1024 * 1024):.2f} MB)")
            return memory_percent / 100.0  # Convert to 0-1 scale
        except ImportError:
            # If psutil is not available, use less precise method
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            max_mem = usage.ru_maxrss / 1024 / 1024  # Convert to MB
            logger.info(f"Memory usage (estimated): {max_mem:.2f} MB")

            # Rough estimate as percentage based on typical limits
            # Assuming 2GB (2048MB) as typical limit
            mem_percentage = min(1.0, max_mem / 2048)
            return mem_percentage
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return 0.5  # Return middle value if we can't determine

    def force_garbage_collection(self) -> int:
        """
        Force Python garbage collection
        Returns the number of objects collected
        """
        gc.collect()
        collected = gc.collect()  # Run twice for better cleanup
        logger.info(f"Garbage collection: {collected} objects collected")
        return collected

    def prune_collection(self, aggressive: bool = False) -> int:
        """
        Prune items from the collection based on importance and age
        Returns the number of items deleted

        Fixed version to handle ChromaDB compatibility issues
        """
        if not self.collection:
            logger.warning("No collection available for pruning")
            return 0

        try:
            # Get all items from collection - FIXED to use correct include parameters
            # The previous version caused an error: "Expected include item to be one of documents, embeddings, metadatas, distances, uris, data, got ids in get"
            results = self.collection.get(include=["metadatas", "documents"])

            if not results or "ids" not in results or not results["ids"]:
                logger.info("Empty collection, nothing to prune")
                return 0

            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])

            # Make sure we have both IDs and metadatas
            if not ids or not metadatas or len(ids) != len(metadatas):
                logger.warning(f"Collection data mismatch: {len(ids)} ids and {len(metadatas)} metadatas")
                return 0

            # Calculate importance scores
            importance_scores = []
            for meta in metadatas:
                # Calculate score based on several factors
                emotion_score = meta.get("emotion_score", 0.5)
                access_count = meta.get("access_count", 0)

                # Get timestamp if available
                timestamp_str = meta.get("timestamp")
                age_factor = 0.5  # Default middle value

                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        age_days = (datetime.utcnow() - timestamp).days

                        # Newer items get higher score
                        age_factor = max(0, 1.0 - (age_days / 30))  # 0 for items 30+ days old
                    except:
                        pass

                # Calculate overall importance
                importance = (
                        0.4 * emotion_score +  # Emotional significance
                        0.3 * min(1.0, access_count / 5) +  # Access frequency (capped at 5)
                        0.3 * age_factor  # Recency
                )

                importance_scores.append(importance)

            # Determine threshold
            config = self.config.get("memory_management", {})
            threshold = config.get("importance_threshold", 0.3)

            # Use more aggressive threshold if needed
            if aggressive:
                threshold = threshold * 1.5

            # Find items to delete
            to_delete = []
            for i, (item_id, score) in enumerate(zip(ids, importance_scores)):
                if score < threshold:
                    to_delete.append(item_id)

            # Perform deletion
            if to_delete:
                try:
                    # Try the safer delete method
                    self.safe_delete_from_collection(ids=to_delete)
                    logger.info(f"Pruned {len(to_delete)} items from collection")
                    self.stats["pruned_items"] += len(to_delete)
                    self.save_stats()
                    return len(to_delete)
                except Exception as e:
                    logger.error(f"Error during safe delete: {e}")
                    return 0
            else:
                logger.info("No items met pruning criteria")
                return 0

        except Exception as e:
            logger.error(f"Error pruning collection: {e}")
            self.stats["memory_errors"] += 1
            self.save_stats()
            return 0

    def prune_memory_tree(self) -> int:
        """
        Prune the memory tree to keep it from growing too large
        Returns the number of items pruned
        """
        if not self.memory_tree:
            logger.warning("No memory tree available for pruning")
            return 0

        try:
            # Use memory tree's own pruning method if available
            config = self.config.get("memory_management", {})
            max_leaves = config.get("max_memory_tree_items", 500)

            if hasattr(self.memory_tree, "prune_tree"):
                pruned = self.memory_tree.prune_tree(max_leaves=max_leaves)
                logger.info(f"Pruned {pruned} leaves from memory tree")
                return pruned
            else:
                logger.warning("Memory tree doesn't have prune_tree method")
                return 0
        except Exception as e:
            logger.error(f"Error pruning memory tree: {e}")
            self.stats["memory_errors"] += 1
            self.save_stats()
            return 0

    # Also add this utility function to help with the Unicode emoji logging error
    def patch_logger_for_emoji_support(self):
        """
        Apply patch to ensure logger can handle emoji characters in Windows
        """
        import logging
        import sys

        # Force UTF-8 encoding for console output if possible
        if hasattr(sys.stdout, 'reconfigure'):  # Python 3.7+
            sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

        # Create a custom formatter that handles emoji
        class EmojiSafeFormatter(logging.Formatter):
            def format(self, record):
                try:
                    return super().format(record)
                except UnicodeEncodeError:
                    # Replace the message with a version that escapes problematic chars
                    record.msg = record.msg.encode('ascii', 'backslashreplace').decode('ascii')
                    return super().format(record)

        # Apply the formatter to all handlers of the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(EmojiSafeFormatter(handler.formatter._fmt))

        # Apply to other common loggers
        for logger_name in ['memory_management', 'world_perception', 'chromadb']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                if handler.formatter:
                    handler.setFormatter(EmojiSafeFormatter(handler.formatter._fmt))

        print("Applied emoji-safe patches to loggers")

    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files created by the system
        Returns the number of files deleted
        """
        try:
            temp_dirs = ["./tmp", "./temp", "./cache"]
            deleted = 0

            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for filename in os.listdir(temp_dir):
                        try:
                            file_path = os.path.join(temp_dir, filename)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                deleted += 1
                        except Exception as e:
                            logger.error(f"Error deleting {filename}: {e}")

            if deleted > 0:
                logger.info(f"Deleted {deleted} temporary files")

            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            return 0

    def clean_database_caches(self) -> bool:
        """
        Clear database caches to free memory
        Returns True if successful
        """
        try:
            # Clear ChromaDB cache if collection has a clear_cache method
            if self.collection and hasattr(self.collection, "clear_cache"):
                self.collection.clear_cache()
                logger.info("Cleared ChromaDB cache")
                return True

            # Alternative cache clearing based on collection type
            if self.collection:
                # Attempt to reset internal cache (depends on implementation)
                if hasattr(self.collection, "_collection"):
                    if hasattr(self.collection._collection, "_embeddings_cache"):
                        self.collection._collection._embeddings_cache = {}
                    if hasattr(self.collection._collection, "_metadatas_cache"):
                        self.collection._collection._metadatas_cache = {}
                    logger.info("Cleared ChromaDB internal caches")
                    return True

            return False
        except Exception as e:
            logger.error(f"Error clearing database caches: {e}")
            return False

    def perform_memory_cleanup(self, force: bool = False) -> Dict:
        """
        Perform comprehensive memory cleanup

        Args:
            force: Force cleanup even if interval hasn't elapsed

        Returns:
            Dictionary with cleanup statistics
        """
        # Check if cleanup is needed
        config = self.config.get("memory_management", {})
        cleanup_interval = timedelta(minutes=config.get("cleanup_interval_minutes", 120))

        now = datetime.utcnow()
        last_cleanup = self.stats.get("last_cleanup")

        if not force and last_cleanup:
            last_cleanup_time = datetime.fromisoformat(last_cleanup)
            if now - last_cleanup_time < cleanup_interval:
                elapsed = (now - last_cleanup_time).total_seconds() / 60
                logger.info(f"Skipping cleanup, last one was {elapsed:.1f} minutes ago")
                return {
                    "status": "skipped",
                    "reason": f"Last cleanup was {elapsed:.1f} minutes ago",
                    "next_cleanup": (last_cleanup_time + cleanup_interval).isoformat()
                }

        # Check memory usage to determine cleanup strategy
        memory_usage = self.check_memory_usage()
        aggressive = memory_usage > config.get("aggressive_cleanup_threshold", 0.9)

        if aggressive:
            logger.warning(f"Memory usage at {memory_usage:.1%}, performing aggressive cleanup")

        # Gather cleanup statistics
        stats = {
            "timestamp": now.isoformat(),
            "memory_usage_before": memory_usage,
            "aggressive_mode": aggressive,
            "actions": {}
        }

        # 1. Force garbage collection
        stats["actions"]["garbage_collection"] = self.force_garbage_collection()

        # 2. Prune collection
        if self.collection:
            stats["actions"]["collection_pruned"] = self.prune_collection(aggressive=aggressive)

        # 3. Prune memory tree
        if self.memory_tree:
            stats["actions"]["memory_tree_pruned"] = self.prune_memory_tree()

        # 4. Cleanup temp files
        stats["actions"]["temp_files_deleted"] = self.cleanup_temp_files()

        # 5. Clean database caches
        stats["actions"]["db_cache_cleared"] = self.clean_database_caches()

        # Check memory usage after cleanup
        memory_usage_after = self.check_memory_usage()
        stats["memory_usage_after"] = memory_usage_after
        stats["memory_reduction"] = max(0, memory_usage - memory_usage_after)

        # Update internal stats
        self.stats["last_cleanup"] = now.isoformat()
        self.stats["cleanup_count"] += 1
        self.save_stats()

        logger.info(f"Memory cleanup complete: {stats['memory_reduction']:.1%} reduction")
        return stats

    def safe_delete_from_collection(self, ids=None, where=None, where_document=None):
        """
        Safely delete items from collection with error handling
        A replacement for the problematic collection.delete method
        """
        if not self.collection:
            logger.warning("No collection available for deletion")
            return False

        try:
            # Different ChromaDB versions use different parameters
            if hasattr(self.collection, "_collection"):
                # Newer versions of ChromaDB
                if ids is not None:
                    self.collection._collection.delete(ids=ids)
                elif where is not None:
                    self.collection._collection.delete(where=where)
                elif where_document is not None:
                    self.collection._collection.delete(where_document=where_document)
            else:
                # Older versions of ChromaDB
                if ids is not None:
                    self.collection.delete(ids=ids)
                elif where is not None and hasattr(self.collection, "delete_where"):
                    self.collection.delete_where(where)
                elif where_document is not None and hasattr(self.collection, "delete_where_document"):
                    self.collection.delete_where_document(where_document)

            logger.info(f"Successfully deleted items from collection")
            return True
        except TypeError as e:
            # Handle the specific error with 'include' parameter
            if "unexpected keyword argument 'include'" in str(e):
                logger.warning("ChromaDB version doesn't support 'include' parameter in delete")
                # Try alternative approach without include
                if ids is not None:
                    try:
                        self.collection.delete(ids=ids)
                        return True
                    except Exception as inner_e:
                        logger.error(f"Alternative delete approach failed: {inner_e}")
                        return False
            logger.error(f"TypeError in delete operation: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting from collection: {e}")
            return False


# Create a singleton instance for global use
_memory_manager = None


def get_memory_manager(collection=None, memory_tree=None):
    """Get or create the memory manager singleton"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(collection, memory_tree)
    elif collection is not None or memory_tree is not None:
        # Update references if provided
        if collection is not None:
            _memory_manager.collection = collection
        if memory_tree is not None:
            _memory_manager.memory_tree = memory_tree
    return _memory_manager


def perform_cleanup(force=False):
    """Global function to perform memory cleanup"""
    manager = get_memory_manager()
    return manager.perform_memory_cleanup(force=force)


def memory_cleanup(collection=None, memory_tree=None):
    """
    Memory cleanup function for direct use in main process

    This is the function to call from narrative_enhanced_cupcake.py
    """
    try:
        # Get manager with given collection and tree
        manager = get_memory_manager(collection, memory_tree)

        # Perform cleanup
        logger.info(" Performing memory cleanup...")
        stats = manager.perform_memory_cleanup()

        if stats.get("status") == "skipped":
            logger.info(f"Cleanup skipped: {stats.get('reason')}")
            return

        # Log summary
        actions = stats.get("actions", {})
        reduction = stats.get("memory_reduction", 0)

        logger.info(f"Cleanup results: {reduction:.1%} memory reduction")
        for action, count in actions.items():
            if count:
                logger.info(f" - {action}: {count}")

    except Exception as e:
        logger.error(f"Error in memory_cleanup: {e}")


if __name__ == "__main__":
    # Test memory management
    print("Testing memory management...")

    memory_manager = MemoryManager()

    # Check current memory usage
    usage = memory_manager.check_memory_usage()
    print(f"Current memory usage: {usage:.1%}")

    # Perform cleanup
    stats = memory_manager.perform_memory_cleanup(force=True)
    print(json.dumps(stats, indent=2))