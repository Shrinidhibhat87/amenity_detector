"""Python file that is meant for Llava model."""
import json
import torch
import os
import logging

from PIL import Image
from typing import List, Dict
from transformers import LlavaProcessor, LlavaForConditionalGeneration


class LlavaModel:
    """
    LlavaModel model class.
    Handles loading the model and performing generation.
    """
    def __init__(self, model_name: str = 'llava-hf/llava-1.5-7b-hf', logger=None, save_folder=None):
        super().__init__()
        self.processor = None
        self.model = None
        self.config = None
        self.logger = logger or logging.getLogger(__name__)
        save_folder = save_folder if save_folder is not None else f"/home/s.bhat/Coding/amenity_detection/model/LlavaModel/{model_name}"
        self._load_model(save_folder=save_folder, model_name=model_name)

    def _load_model(self, save_folder, model_name):
        """
        Load the LlavaModel model and processor.
        """
        # With the defined path, create the directory if it does not exist
        os.makedirs(save_folder, exist_ok=True)

        # Check if the model and processor are already saved locally
        processor_path = os.path.join(save_folder, "processor")
        model_path = os.path.join(save_folder, "model")

        if not os.path.exists(processor_path) or not os.path.exists(model_path):
            # Load from online and save locally
            self.processor = LlavaProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            )
            # Save locally
            self.processor.save_pretrained(processor_path)
            self.model.save_pretrained(model_path)
            self.logger.info(f"LLaVA model loaded successfully and saved to {save_folder}")

        else:
            # Load from local directory
            self.processor = LlavaProcessor.from_pretrained(processor_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
            )
            self.logger.info(f"LLaVA model loaded successfully from {model_path}")

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"LLaVA model loaded and moved to {self.device}")
        

    def generate_description(self, img: Image.Image, room_amenities: Dict[str, List[str]]) -> str:
        """
        Generate a natural language description of a property based on detected amenities.
        
        Args:
            img (str): Image from which to generate the description
            room_amenities: Dictionary of detected amenities by room type
            
        Returns:
            A natural language description of the property
        """
        self.logger.info("Generating property description")

        if not any(room_amenities.values()):
            return "No notable amenities were detected in this property."
        
        # Create a prompt that organizes amenities by room
        prompt = "Generate a natural and appealing property description highlighting these amenities:\n"
        
        for room_type, room_amenities in room_amenities.items():
            if room_amenities:
                prompt += f"- {room_type.replace('_', ' ')}: {', '.join(room_amenities)}\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                    ],
            },
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe the property in highlighting the amenities."},
                ],
            },
        ]

        # Use the model to generate the description
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device="cuda", dtype=torch.float16)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        description = self.processor.decode(outputs[0], skip_special_tokens=True)

        # From this we have to extract 'ASSISTANT:' part
        # Extract the text after 'ASSISTANT:'
        if "ASSISTANT:" in description:
            description = description.split("ASSISTANT:")[1].strip()

        # Extract only the generated description (remove the prompt)
        if "The description should be" in description:
            description = description.split("The description should be")[1]
            if ":" in description:
                description = description.split(":", 1)[1]

        # Clean up the description
        description = description.strip()

        return description


    def detect_amenities(self, image: Image.Image, all_amenities: List[str]) -> Dict[str, bool]:
        """
        Detect amenities in the image using the model.

        Args:
            image: The input image for the model.
            all_amenities: List of amenities to detect.

        Returns:
            Dictionary mapping amenity names to boolean values
        """
        self.logger.info("Detecting amenities in the image...")
        
        # Use a specific prompt to detect amenities
        # Create a prompt for the model
        prompt = (
            f"This is an image of a property. Please analyze the image and determine which "
            f"of the following amenities are present: {', '.join(all_amenities)}. "
            f"Please provide the answer in JSON format, where the key is the amenity name "
            f"and the value is a boolean indicating the presence of the amenity in the image. "
            f"Only include the amenities that you are confident about and give a value of false "
            f"for the amenities that are not present in the image. "
            f"Example format: {{\"amenity1\": true, \"amenity2\": false, ...}}"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                    ],
            },
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "Please follow the prompt and generate JSON like output."},
                ],
            },
        ]

        # Process the image and generate text
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device="cuda", dtype=torch.float16)

        # Because we dont want gradient calculation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Decode the response (could be batch_decode or decode)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Log the response in debug mode
        self.logger.debug(f"Model response: {response}")

        # From this we have to extract 'ASSISTANT:' part
        # Extract the text after 'ASSISTANT:'
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[1].strip()

        # Extract the JSON part from the response
        try:
            # Find the text between the first and last curly ({}) brackets
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                if self.logger:
                    self.logger.error(f"Could not find JSON like object in response.")
                    json_text = {}
            else:
                json_text = response[json_start:json_end]
                
            # Clean up potential formatting issues
            json_text = json_text.replace("'", "\"").replace("True", "true").replace("False", "false")
            # Fix escaped backslashes in property names
            json_text = json_text.replace("\\_", "_")
            # Parse the JSON response
            detected_amenities = json.loads(json_text)
            
            return detected_amenities

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"Error decoding JSON response: {e}")
                self.logger.error(f"Response: {response}")
                return {}
