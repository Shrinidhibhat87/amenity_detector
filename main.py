"""
Main Python script for the end-to-end pipeline using Hydra for configuration.

The script does the followiing:
1) Process property images online/local to detect amenities using a vision-language model.
2) Structure and store the detected amenities.
3) Generate a nice description of the property based on the detected amenities.

"""

import hydra
import os
import logging

from omegaconf import DictConfig, OmegaConf

from core.amenity_system import PropertyAmenitySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    """
    Main entry point for the property amenity detection system.
    
    Args:
        config: Hydra configuration
    """
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    system = PropertyAmenitySystem(config, logger=logger)
    
    if os.path.isdir(config.input.path):
        # Process a directory of images
        results = system.process_directory(config.input.path)
        logger.info(f"Processed {len(results)} images")
        
        if not results.empty:
            # Print summary of results
            print("\nResults Summary:")
            print(results)
    else:
        # Process a single image
        amenities, description = system.process_image(config.input.path)
        
        # Print results
        print("\nDetected Amenities:")
        for room_type, room_amenities in amenities.items():
            present = [amenity for amenity, is_present in room_amenities.items() if is_present]
            if present:
                print(f"  {room_type.capitalize()}: {', '.join(present)}")
        
        print("\nGenerated Description:")
        print(description)

    logger.info(f"Results saved to {config.output.directory}")


if __name__ == "__main__":
    main()
