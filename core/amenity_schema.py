"""Python module for handling the property amenity schema."""
import json
import logging
from typing import Dict, List

AMENITY_SCHEMA = {
    "kitchen": [
        "refrigerator", "fridge", "oven", "microwave", "dishwasher", "sink", "stove", "toaster",
        "blender", "kettle", "coffee_maker", "cutlery", "utensils", "plates", "bowls",
        "bar_counter", "washing_machine"
    ],
    "living_room": [
        "sofa", "tv", "coffee_table", "bookshelf", "fireplace",
        "armchair", "entertainment_center", "speaker_system", "gaming_console",
        "air_conditioner", "ceiling_fan", "smart_home_system", "projector"
    ],
    "bedroom": [
        "bed", "wardrobe", "dresser", "nightstand", "desk",
        "chair", "tv", "mirror", "air_conditioner", "ceiling_fan",
        "lamp", "alarm_clock"
    ],
    "bathroom": [
        "toilet", "shower", "bathtub", "sink", "mirror",
        "towel_rack", "hair_dryer", "washing_machine", "dryer"
    ],
    "outdoor": [
        "patio", "balcony", "garden", "pool", "hot_tub",
        "bbq_grill", "outdoor_furniture", "parking_space"
    ],
    "common": [
        "wifi", "heating", "air_conditioning", "smoke_detector",
        "security_camera", "elevator", "wheelchair_accessible"
    ]
}

def load_amenity_schema(file_path: str = None) -> Dict[str, List[str]]:
    """
    Load an amenity schema from a file, or return the default schema if no file is provided.
    
    Args:
        file_path: Path to a JSON file containing an amenity schema
        
    Returns:
        Dictionary mapping room types to lists of amenities
    """
    if file_path:
        try:
            with open(file_path, 'r') as f:
                schema = json.load(f)
            logging.info(f"Loaded amenity schema from {file_path}")
            return schema
        except Exception as e:
            logging.error(f"Error loading amenity schema: {e}. Using default schema.")
    
    return AMENITY_SCHEMA

def get_all_amenities(schema: Dict[str, List[str]]) -> List[str]:
    """
    Get a flattened list of all amenities in the schema.
    
    Args:
        schema: Dictionary mapping room types to lists of amenities
        
    Returns:
        List of all unique amenities
    """
    all_amenities = []
    for amenities in schema.values():
        all_amenities.extend(amenities)

    # Remove duplicates and sort the list
    return sorted(list(set(all_amenities)))