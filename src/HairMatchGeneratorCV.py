import os
import logging
from PIL import Image
from datetime import datetime
from common import BaseComponent
from models.HairSegmenter import HairSegmenter
from src.helpers import HairSwatchMatcherCV
from config.loader import settings


class HairMatchGeneratorCV(BaseComponent):
    """
    A class that segments hair from portraits and matches them to swatch images using classic CV.
    Saves input and intermediate artifacts.
    """

    def __init__(self, **kwargs):
        config = settings.get('hair_match_generator', {}).get('args', {})
        general_config = settings.get('general', {})
        self.swatch_path = config["swatch_path"]
        self.artefacts_dir = general_config.get("artefacts_dir", "./artefacts")

        self.logger = logging.getLogger(__name__)
        self.matcher = HairSwatchMatcherCV()
        self.segmenter = HairSegmenter()

        self.class_name = self.__class__.__name__
        self.artefacts_subdir = os.path.join(self.artefacts_dir, self.class_name)
        os.makedirs(self.artefacts_subdir, exist_ok=True)

        self.swatch_images = []
        for fname in os.listdir(self.swatch_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(os.path.join(self.swatch_path, fname)).convert("RGB")
                    swatch_name = os.path.splitext(fname)[0]
                    self.swatch_images.append((swatch_name, img))
                except Exception as e:
                    self.logger.warning(f"Could not load swatch {fname}: {e}")

    def match(self, image_data: str) -> str:
        """
        Segments the hair and finds the closest matching swatch from a given image path.
        Saves input and intermediate artifacts.

        Args:
            image_data (str): Path to portrait image.

        Returns:
            str: Matching swatch name.
        """
        try:
            img = Image.open(image_data).convert("RGB")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_data))[0]
            img_id = f"{self.class_name}_{base_name}_{timestamp}"

            input_path = os.path.join(self.artefacts_subdir, f"{img_id}_input.png")
            img.save(input_path)

            cropped_hair = self.segmenter.infer(img)

            mask_path = os.path.join(self.artefacts_subdir, f"{img_id}_hair_mask.png")
            cropped_hair.save(mask_path)

            match = self.matcher.match(cropped_hair, self.swatch_images)
            return match

        except Exception as e:
            self.logger.error(f"Hair matching failed: {e}")
            return "no-match"