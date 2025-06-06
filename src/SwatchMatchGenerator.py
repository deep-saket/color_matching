from typing import List, Union, Tuple
from PIL import Image
import Levenshtein
import logging
from common import BaseComponent, InferenceVLComponent
from config.loader import settings
from models import ModelManager
from src.helpers import SwatchDetails
import torch


class SwatchMatchGenerator(BaseComponent):
    """
    A class that uses QwenV25Infer to analyze portrait images and generate matching swatch names.
    """

    def __init__(self ,**kwargs):
        """
        Initialize the SwatchMatchGenerator.
        """
        swatch_gen_settings = settings.get('swatch_match_generator', {}).get('args', {})
        self.swatch_path = swatch_gen_settings["swatch_path"]
        self.device = torch.device(swatch_gen_settings.get("device", "cpu"))
        self.vlm_candidate = swatch_gen_settings["vlm_candidate"]
        model_classes = swatch_gen_settings.get("models", [self.vlm_candidate])

        ModelManager.initialize_models(self.device, model_classes)
        self.vlm_model = getattr(ModelManager, self.vlm_candidate)

        self.logger = logging.getLogger(__name__)

        # 2) Load swatch details using SwatDetails
        self.swatch_details = SwatchDetails(self.swatch_path, self.vlm_model)
        self.color_names = list(self.swatch_details.values())
        self.color_names_lower = [name.lower() for name in self.color_names]

    def _format_prompt(self, swatch_names: List[str]) -> str:
        """
        Format the prompt for the model with the list of available swatches.

        Args:
            swatch_names (List[str]): List of available swatch names.

        Returns:
            str: Formatted prompt for the model.
        """
        swatch_list = ", ".join(swatch_names)
        return f"Looking at this portrait image, which of the following hair color swatches would be the best match? Available swatches: {swatch_list}. Please respond with exactly one swatch name from the list."

    def match(self, image: Union[str, Image.Image, bytes]) -> str:
        """
        Generate a matching swatch name for the given portrait image.

        Args:
            image: Portrait image as file path, PIL Image, or bytes.

        Returns:
            str: Name of the matching swatch.

        Raises:
            ValueError: If inputs are invalid or empty.
        """

        if not image:
            raise ValueError("Image data cannot be empty")

        prompt = self._format_prompt(self.color_names)
        response = self.vlm_model.infer(image_data=image, prompt=prompt)

        # Ensure the response is one of the provided swatch names
        response = response.strip()
        response_lower = response.lower()

        if response in self.color_names:
            image_name = self.swatch_details.get_image_name(response)
            self.logger.info(f"Exact match found for swatch: {response} (image: {image_name})")
            return response

        # Find closest match using Levenshtein distance
        distances = [(name, Levenshtein.distance(response_lower, name.lower()))
                     for name in self.color_names]
        closest_match = min(distances, key=lambda x: x[1])

        
        if closest_match[1] <= 2:  # Allow small differences
            image_name = next((k for k, v in self.swatch_details.items() if v == closest_match[0]), None)
            self.logger.info(
                f"Found close match: '{closest_match[0]}' (image: {image_name}) for response: '{response}'")
            return image_name
        else:
            self.logger.error(f"No matching swatch found for response: '{response}'")
            raise ValueError(f"Model response '{response}' is not in the provided swatch names list")
        
