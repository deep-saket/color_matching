from typing import List, Union
from PIL import Image
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


        # 2) Load swatch details using SwatDetails
        swatch_details = SwatchDetails(self.swatch_path, self.vlm_model)
        self.color_names = list(swatch_details.values())


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
        if response not in self.color_names:
            raise ValueError(f"Model response '{response}' is not in the provided swatch names list")

        return response
