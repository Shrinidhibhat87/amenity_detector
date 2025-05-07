"""Python file/module for the detecting amenities in an image/property."""
import logging

from typing import List, Dict, Tuple
from PIL import Image
from model.llava import LlavaModel

def get_model_instance(model_name):
    """
    Dynamically load the model class based on the model name.

    Args:
        model_name: The name of the model to load.
    Returns:
        An instance of the model class.
    """
    if model_name == "InstructBlip":
        from old_project.model.instruct_blip import InstructBlipModel
        return InstructBlipModel()
    elif model_name == "Blip2":
        from old_project.model.blip2 import Blip2Model
        return Blip2Model()
    elif model_name == "GitCausalLM":
        from old_project.model.git_causal import GitCausalLM
        return GitCausalLM()
    elif model_name == "Llava":
        return LlavaModel()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

class AmenityDetector:
    """
    Detects amenities in property images using a vision-language model.
    """
    def __init__(self, model_name: str, amenity_schema: Dict[str, List[str]], logger=None, save_dir: str = None):
        """
        Initialize the amenity detector with a specified model and schema.
        
        Args:
            model_name: The name of the model to use
            amenity_schema: Dictionary mapping room types to lists of amenities
            logger: Optional logger instance
            save_dir: Directory to save model weights
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing AmenityDetector with model: {model_name}")
        
        self.model = LlavaModel(model_name, logger=logger, save_folder=save_dir)
        self.amenity_schema = amenity_schema
        
    def detect_amenities(self, image_path: str) -> Tuple[Dict[str, Dict[str, bool]], str, Dict[str, List[str]]]:
        """
        Detect amenities in an image and generate a description.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (amenities_by_room, description, detected_amenities)
            - amenities_by_room: Dictionary mapping room types to amenities and their presence status
            - description: Generated natural language description
            - detected_amenities: Flat dictionary of amenities and their presence status
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return {}, "Error processing image", {}
            
        # Create a flattened list of all amenities for detection
        all_amenities = []
        for amenities in self.amenity_schema.values():
            all_amenities.extend(amenities)

        # Remove duplicates
        all_amenities = sorted(list(set(all_amenities)))
        
        # Detect amenities
        detected_amenities = self.model.detect_amenities(image, all_amenities)
        
        # Restructure into room-based schema
        amenities_by_room = {room_type: {} for room_type in self.amenity_schema}
        present_amenities_by_room = {room_type: [] for room_type in self.amenity_schema}
        
        # Instead of returning present_amenities_by_room, lets return detected_amenities to keep it simple
        for room_type, amenities in self.amenity_schema.items():
            for amenity in amenities:
                is_present = detected_amenities.get(amenity, False)
                amenities_by_room[room_type][amenity] = is_present
                if is_present:
                    present_amenities_by_room[room_type].append(amenity)
        
        # Generate description
        description = self.model.generate_description(img=image, detected_amenities=detected_amenities)
        
        return amenities_by_room, description, detected_amenities
