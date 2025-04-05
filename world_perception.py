# world_perception.py
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union
import random
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("perception.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("world_perception")

# List of common objects for fallback when camera is unavailable
COMMON_OBJECTS = [
    "laptop", "desk", "chair", "keyboard", "mouse", "monitor", "book",
    "pen", "notebook", "coffee mug", "water bottle", "window", "door",
    "lamp", "plant", "clock", "phone", "bag", "headphones", "charger",
    "cable", "paper", "folder", "sticky notes", "calendar", "glasses",
    "wallet", "keys", "remote control", "tablet", "wall", "ceiling", "shelf"
]

# Locations for contextual object generation
LOCATIONS = {
    "office": ["desk", "chair", "computer", "monitor", "keyboard", "mouse", "printer", "pen", "notebook", "stapler",
               "lamp", "bookshelf", "whiteboard", "coffee mug", "phone", "calendar", "sticky notes", "water bottle",
               "headphones", "bag"],
    "living_room": ["sofa", "coffee table", "TV", "remote control", "bookshelf", "lamp", "plant", "painting",
                    "curtains", "rug", "pillow", "blanket", "speaker", "magazine", "photo frame", "candle", "vase",
                    "clock", "decorative bowl", "coaster"],
    "kitchen": ["refrigerator", "stove", "microwave", "sink", "dishwasher", "counter", "table", "chair", "cabinet",
                "knife", "cutting board", "plate", "bowl", "glass", "mug", "utensils", "pan", "pot", "blender",
                "toaster"],
    "bedroom": ["bed", "pillow", "blanket", "nightstand", "lamp", "wardrobe", "mirror", "dresser", "clock", "book",
                "curtains", "rug", "clothes", "hangers", "laundry basket", "phone charger", "plant", "photo frame",
                "slippers", "water glass"],
    "bathroom": ["sink", "toilet", "shower", "bathtub", "mirror", "towel", "toothbrush", "toothpaste", "soap",
                 "shampoo", "toilet paper", "scale", "mat", "hairbrush", "razor", "hairdryer", "medicine cabinet",
                 "laundry basket", "robe", "trash can"]
}

# Environment settings
ENV_SETTINGS = {
    "headless_mode": None,  # Auto-detect
    "use_camera": True,
    "fallback_mode": "contextual",  # Options: "random", "consistent", "contextual"
    "current_location": "office",  # Default location for contextual fallback
    "time_based_objects": True,  # Add time-relevant objects
    "visual_memory": [],  # Remember what was "seen" before
    "max_memory_items": 20,  # Max items to remember
    "detection_model": "yolov8n.pt",  # Default YOLOv8 model
    "min_confidence": 0.3,  # Minimum confidence for object detection
    "max_objects": 10,  # Maximum objects to return
    "skip_frames": 2,  # Frames to skip (for performance)
    "visualization_enabled": True  # Enable visualization
}


def is_headless_environment() -> bool:
    """
    Detect if running in a headless environment
    """
    # Check common environment variables
    display_env = os.environ.get("DISPLAY", "")
    term = os.environ.get("TERM", "")
    ssh_connection = os.environ.get("SSH_CONNECTION", "")

    # Check if DISPLAY is not set or empty (common in headless environments)
    if not display_env:
        return True

    # Check for specific terminal types that indicate headless
    if term in ["dumb", "unknown"]:
        return True

    # Check if running via SSH (often headless)
    if ssh_connection:
        return True

    # Try to import GUI libraries
    try:
        import tkinter
        return False  # If tkinter imports successfully, likely not headless
    except ImportError:
        pass

    try:
        import cv2
        # Try to create a test window
        try:
            cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow("Test")
            return False
        except:
            return True
    except ImportError:
        pass

    # Default to assuming headless if can't determine
    return True


def load_config() -> Dict[str, Any]:
    """
    Load perception configuration from file
    """
    config_path = "perception_config.json"

    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(ENV_SETTINGS, f, indent=2)
        return ENV_SETTINGS.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Merge with defaults for any missing keys
        for key, value in ENV_SETTINGS.items():
            if key not in config:
                config[key] = value

        return config
    except Exception as e:
        logger.error(f"Error loading perception config: {e}")
        return ENV_SETTINGS.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save perception configuration to file
    """
    config_path = "perception_config.json"
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving perception config: {e}")
        return False


def update_config(key: str, value: Any) -> Dict[str, Any]:
    """
    Update a specific configuration value
    """
    config = load_config()
    config[key] = value
    save_config(config)
    return config


def init_perception():
    """
    Initialize perception system
    """
    global ENV_SETTINGS

    # Load configuration
    config = load_config()
    ENV_SETTINGS.update(config)

    # Auto-detect headless mode if not set
    if ENV_SETTINGS["headless_mode"] is None:
        ENV_SETTINGS["headless_mode"] = is_headless_environment()
        update_config("headless_mode", ENV_SETTINGS["headless_mode"])

    if ENV_SETTINGS["headless_mode"]:
        logger.info("Running in headless mode, camera perception disabled")
        ENV_SETTINGS["use_camera"] = False
        update_config("use_camera", False)
    else:
        logger.info("Running in graphical mode, camera perception enabled")

    # Initialize YOLO model if needed and not in headless mode
    if not ENV_SETTINGS["headless_mode"] and ENV_SETTINGS["use_camera"]:
        try:
            from ultralytics import YOLO
            global model
            model = YOLO(ENV_SETTINGS["detection_model"])
            logger.info(f"YOLO model loaded: {ENV_SETTINGS['detection_model']}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            ENV_SETTINGS["use_camera"] = False
            update_config("use_camera", False)

    # Initialize visual memory
    if not ENV_SETTINGS["visual_memory"]:
        ENV_SETTINGS["visual_memory"] = []
        update_config("visual_memory", [])

    logger.info(f"Perception system initialized: {ENV_SETTINGS}")


def get_time_relevant_objects() -> List[str]:
    """
    Get objects that are relevant to the current time of day
    """
    import datetime

    current_hour = datetime.datetime.now().hour

    if 6 <= current_hour < 10:  # Morning
        return ["coffee mug", "breakfast plate", "newspaper", "toast", "cereal bowl"]
    elif 10 <= current_hour < 12:  # Late morning
        return ["water bottle", "notebook", "pen", "laptop", "smartphone"]
    elif 12 <= current_hour < 14:  # Lunch time
        return ["lunch box", "sandwich", "salad container", "fork", "napkin"]
    elif 14 <= current_hour < 17:  # Afternoon
        return ["tea mug", "notebook", "water bottle", "snack", "headphones"]
    elif 17 <= current_hour < 20:  # Evening
        return ["dinner plate", "glass", "cooking pot", "utensils", "napkin"]
    elif 20 <= current_hour < 23:  # Night
        return ["book", "remote control", "tea mug", "blanket", "smartphone"]
    else:  # Late night
        return ["water glass", "book", "lamp", "headphones", "laptop"]


def get_consistent_objects(count: int = 5) -> List[str]:
    """
    Generate a consistent set of objects based on visual memory
    """
    config = load_config()
    memory = config.get("visual_memory", [])

    if not memory:
        # Generate new set of objects
        location = config.get("current_location", "office")
        location_objects = LOCATIONS.get(location, COMMON_OBJECTS)

        # Select a random subset
        selected = random.sample(location_objects, min(count, len(location_objects)))

        # Update memory
        config["visual_memory"] = selected
        save_config(config)

        return selected

    # Return objects from memory, with slight variations
    if len(memory) > count:
        return random.sample(memory, count)
    else:
        # Add a random new object occasionally
        if random.random() < 0.2:
            location = config.get("current_location", "office")
            location_objects = LOCATIONS.get(location, COMMON_OBJECTS)

            # Remove existing memory items
            potential_new = [obj for obj in location_objects if obj not in memory]

            if potential_new:
                new_object = random.choice(potential_new)
                memory.append(new_object)

                # Trim memory if needed
                max_items = config.get("max_memory_items", 20)
                if len(memory) > max_items:
                    memory = memory[-max_items:]

                # Update config
                config["visual_memory"] = memory
                save_config(config)

        return memory


def generate_fallback_objects(count: int = 5) -> List[str]:
    """
    Generate fallback objects when camera is unavailable
    """
    config = load_config()
    fallback_mode = config.get("fallback_mode", "contextual")

    if fallback_mode == "random":
        # Completely random objects
        return random.sample(COMMON_OBJECTS, min(count, len(COMMON_OBJECTS)))

    elif fallback_mode == "consistent":
        # Consistent objects from memory
        return get_consistent_objects(count)

    elif fallback_mode == "contextual":
        # Objects based on context (location and time)
        location = config.get("current_location", "office")
        location_objects = LOCATIONS.get(location, COMMON_OBJECTS)

        # Include time-relevant objects
        if config.get("time_based_objects", True):
            time_objects = get_time_relevant_objects()
            # Combine location and time objects, with more weight to location
            all_objects = location_objects + time_objects
        else:
            all_objects = location_objects

        # Select objects, with preference for previously seen objects
        memory = config.get("visual_memory", [])

        # If we have memory, use it to influence selection
        if memory:
            # 70% chance to include memory objects, 30% for new objects
            memory_count = int(count * 0.7)
            new_count = count - memory_count

            selected = random.sample(memory, min(memory_count, len(memory)))

            # Add new objects that aren't in memory
            potential_new = [obj for obj in all_objects if obj not in selected]
            if potential_new and new_count > 0:
                selected.extend(random.sample(potential_new, min(new_count, len(potential_new))))
        else:
            # No memory yet, just random selection
            selected = random.sample(all_objects, min(count, len(all_objects)))

        # Update memory with these objects
        new_memory = list(set(memory + selected))

        # Trim memory if needed
        max_items = config.get("max_memory_items", 20)
        if len(new_memory) > max_items:
            new_memory = new_memory[-max_items:]

        # Update config
        config["visual_memory"] = new_memory
        save_config(config)

        return selected

    # Default to common objects if mode not recognized
    return random.sample(COMMON_OBJECTS, min(count, len(COMMON_OBJECTS)))


def detect_objects_with_camera() -> List[str]:
    """
    Detect objects using camera and YOLO model
    """
    try:
        import cv2
        from ultralytics import YOLO

        config = load_config()

        # Check if we're in headless mode
        if config.get("headless_mode", False):
            logger.warning("Cannot use camera in headless mode")
            return []

        # Initialize camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Failed to open camera")
            return []

        # Skip a few frames to allow camera to adjust
        for _ in range(config.get("skip_frames", 2)):
            cap.read()

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture image")
            cap.release()
            return []

        # Load model if not already loaded
        global model
        if 'model' not in globals():
            model_path = config.get("detection_model", "yolov8n.pt")
            try:
                model = YOLO(model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                cap.release()
                return []

        # Run detection
        results = model(frame)

        # Extract detected objects
        detected_objects = []
        min_confidence = config.get("min_confidence", 0.3)
        max_objects = config.get("max_objects", 10)

        # Process results
        boxes = results[0].boxes
        classes = boxes.cls.tolist() if hasattr(boxes, 'cls') and boxes.cls is not None else []

        # Get class names and confidence scores
        for i, c in enumerate(classes):
            if hasattr(boxes, 'conf') and len(boxes.conf) > i:
                confidence = float(boxes.conf[i])
                if confidence >= min_confidence:
                    class_idx = int(c)
                    if hasattr(results[0], 'names') and class_idx in results[0].names:
                        detected_objects.append(results[0].names[class_idx])

        # Limit number of objects
        if len(detected_objects) > max_objects:
            detected_objects = detected_objects[:max_objects]

        # Try to visualize if enabled
        if config.get("visualization_enabled", True):
            try:
                # Draw bounding boxes
                for i, box in enumerate(boxes.xyxy):
                    if i < len(classes):
                        x1, y1, x2, y2 = [int(x) for x in box]
                        label = results[0].names[int(classes[i])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)

                # Show image
                cv2.imshow("Cupcake está vendo...", frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyAllWindows()
            except Exception as e:
                logger.warning(f"Could not visualize detection: {e}")

        # Release camera
        cap.release()

        # Update memory with detected objects
        config = load_config()
        memory = config.get("visual_memory", [])
        new_memory = list(set(memory + detected_objects))

        # Trim memory
        max_items = config.get("max_memory_items", 20)
        if len(new_memory) > max_items:
            new_memory = new_memory[-max_items:]

        # Update config
        config["visual_memory"] = new_memory
        save_config(config)

        return detected_objects

    except ImportError as e:
        logger.error(f"Required library not found: {e}")
        return []
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return []


def perceive_world(show: bool = True, save_snapshot: bool = False, snapshot_path: str = "snapshot.jpg") -> List[str]:
    """
    Main function to perceive objects in the environment

    Parameters:
    - show: Whether to display the camera feed
    - save_snapshot: Whether to save the snapshot
    - snapshot_path: Path to save the snapshot

    Returns:
    - List of detected objects
    """
    # Initialize if not already done
    if ENV_SETTINGS["headless_mode"] is None:
        init_perception()

    # Update config (in case it changed externally)
    config = load_config()
    ENV_SETTINGS.update(config)

    # Attempt camera detection if enabled
    if config.get("use_camera", True) and not config.get("headless_mode", False):
        try:
            detected_objects = detect_objects_with_camera()

            if detected_objects:
                # Successfully detected objects
                logger.info(f"Camera detection successful: {detected_objects}")

                # Save snapshot if requested
                if save_snapshot:
                    try:
                        import cv2
                        cap = cv2.VideoCapture(0)
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(snapshot_path, frame)
                            logger.info(f"Snapshot saved to {snapshot_path}")
                        cap.release()
                    except Exception as e:
                        logger.error(f"Failed to save snapshot: {e}")

                return detected_objects

        except Exception as e:
            logger.error(f"Camera detection failed: {e}")
            # Fall back to generated objects

    # If we get here, camera detection failed or is disabled
    # Use fallback object generation
    logger.info("Using fallback object detection")

    # Get the number of objects to generate
    object_count = random.randint(3, config.get("max_objects", 10))

    # Generate fallback objects
    fallback_objects = generate_fallback_objects(object_count)

    # Log the fallback
    if config.get("headless_mode", False):
        logger.info(f"Headless mode active, using fallback objects: {fallback_objects}")
    else:
        logger.warning(f"Camera detection unavailable, using fallback objects: {fallback_objects}")

    return fallback_objects


def set_perception_mode(mode: str = "fallback", location: str = None):
    """
    Set the perception mode

    Parameters:
    - mode: "camera" or "fallback"
    - location: optional location for contextual fallback
    """
    config = load_config()

    if mode == "camera":
        config["use_camera"] = True
    elif mode == "fallback":
        config["use_camera"] = False

    if location and location in LOCATIONS:
        config["current_location"] = location

    save_config(config)
    return config


def get_perception_status():
    """
    Get current status of perception system
    """
    config = load_config()

    # Check additional status info
    headless = config.get("headless_mode", is_headless_environment())
    camera_available = False

    if not headless:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            camera_available = cap.isOpened()
            cap.release()
        except:
            camera_available = False

    return {
        "headless_mode": headless,
        "camera_enabled": config.get("use_camera", True),
        "camera_available": camera_available,
        "fallback_mode": config.get("fallback_mode", "contextual"),
        "current_location": config.get("current_location", "office"),
        "memory_items": len(config.get("visual_memory", [])),
        "model": config.get("detection_model", "yolov8n.pt")
    }


if __name__ == "__main__":
    # Test perception
    init_perception()
    status = get_perception_status()
    print("Perception Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\nDetecting objects...")
    objects = perceive_world()
    print(f"Detected objects: {objects}")