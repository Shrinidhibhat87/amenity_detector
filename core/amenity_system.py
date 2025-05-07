"""Python file that has the system that handles the detection, storage and generation."""
import json
import pandas as pd
import logging
from omegaconf import DictConfig
from typing import Tuple, Dict
from pathlib import Path
from PIL import Image

from core.amenity_schema import load_amenity_schema
from core.amenity_detector import AmenityDetector
from core.amenity_data_manager import AmenityDataManager

class PropertyAmenitySystem:
    """
    Main system that orchestrates the detection, storage, and description generation.
    """
    def __init__(self, config: DictConfig, logger=None):
        """
        Initialize the property amenity system with configuration.
        
        Args:
            config: Hydra configuration object
            logger: Optional logger for logging messages
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Load amenity schema
        if config.amenity_schema.from_file:
            self.amenity_schema = load_amenity_schema(config.amenity_schema.file_path)
        else:
            self.amenity_schema = load_amenity_schema()
        
        # Initialize components
        self.detector = AmenityDetector(
            model_name=config.model.name,
            amenity_schema=self.amenity_schema,
            logger=self.logger,
            save_dir=config.output.model_weight_dir if hasattr(config.output, 'model_weight_dir') else None
        )

        self.data_manager = AmenityDataManager(
            config.output.directory,
            amenity_schema=self.amenity_schema,
            logger=logger
        )

        self.logger.info("Property Amenity System initialized successfully")

    def process_image(self, image_path: str) -> Tuple[Dict[str, Dict[str, bool]], str]:
        """
        Process a single image to detect amenities and generate description.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (amenities, description)
        """
        self.logger.info(f"Processing image: {image_path}")

        # Detect amenities
        amenities, description, detected_amenities = self.detector.detect_amenities(image_path)

        # Save results
        self.data_manager.save_results(image_path, amenities, description, detected_amenities)
        
        return amenities, description

    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            DataFrame with summary of processed images
        """
        self.logger.info(f"Processing all images in directory: {directory_path}")
        
        supported_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        
        # Get all image files in the directory
        for ext in supported_extensions:
            image_paths.extend(list(Path(directory_path).glob(f"*{ext}")))
        
        if not image_paths:
            self.logger.warning(f"No images found in {directory_path}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for img_path in image_paths:
            try:
                amenities, description = self.process_image(str(img_path))
                self.logger.info(f"Processed {img_path}")
                self.logger.info(f"Description: {description}")
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")

        # Return summary
        return self.data_manager.get_results_summary()

    def process_image_from_memory(self, image: Image.Image, image_name: str) -> Tuple[Dict[str, Dict[str, bool]], str]:
        """
        Process an image that's already loaded into memory.
        
        Args:
            image: PIL Image object
            image_name: Name to identify the image
            
        Returns:
            Tuple of (amenities, description)
        """
        self.logger.info(f"Processing in-memory image: {image_name}")
        
        # Save the image temporarily to process it
        temp_dir = Path(self.config.output.directory) / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_path = temp_dir / image_name
        
        try:
            image.save(temp_path)
            amenities, description = self.process_image(str(temp_path))
            return amenities, description
        finally:
            # Clean up the temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def get_all_results(self) -> pd.DataFrame:
        """
        Get all processed results as a dataframe.
        
        Returns:
            DataFrame with all processed images and their amenities
        """
        return self.data_manager.get_all_results_as_dataframe()
