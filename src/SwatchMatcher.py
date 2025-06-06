import os
from pathlib import Path
import torch
from io import BytesIO
from PIL import Image
from common import BaseComponent
from config.loader import settings, artifacts_dir
from models import ModelManager
from src.helpers.PatchMatcher import PatchMatcher
from typing import Union
from uuid import uuid4


class SwatchMatcher(BaseComponent):
    def __init__(self, threshold: float = None):
        super().__init__()

        # Load configuration
        cfg = settings.get("swatch_matcher", {}).get("args", {})
        swatch_path = cfg["swatch_path"]
        device = torch.device(cfg.get("device", "cpu"))
        hair_candidate = cfg["hair_segmentation_candidate"]
        embed_candidate = cfg["embedding_candidate"]
        model_classes = cfg.get("models", [hair_candidate, embed_candidate])

        # Ensure models are loaded
        ModelManager.initialize_models(device=device, model_classes=model_classes)
        self.segmenter = getattr(ModelManager, hair_candidate)
        self.embedder = getattr(ModelManager, embed_candidate)

        # Determine threshold
        self.threshold = threshold if threshold is not None else cfg.get("threshold", 0.93)

        # Load swatches directory
        swatch_dir = Path(swatch_path)
        if not swatch_dir.is_dir():
            raise ValueError(f"swatch_path must be a directory, got: {swatch_path}")

        self.swatches = []
        
        for img_file in sorted(swatch_dir.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = Image.open(img_file).convert("RGB")
            emb = self.embedder.encode_image(img)
            self.swatches.append({"name": img_file.name, "embedding": emb})

        if not self.swatches:
            raise ValueError(f"No valid swatch images found in {swatch_path}")

        # Instantiate the patch matcher
        self.patch_matcher = PatchMatcher(
            embedder=self.embedder,
            swatches=self.swatches,
            threshold=self.threshold
        )

    def match(self, image_data: Union[bytes, str, Image.Image]) -> str:
        # Load input image
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, str):
            img = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            raise TypeError(f"Unsupported image_data type: {type(image_data)}")

        if isinstance(image_data, str):
            output_dir = os.path.join(artifacts_dir, os.path.basename(image_data))
        else:
            output_dir = str(uuid4())
        os.makedirs(output_dir, exist_ok=True)

        # Segment hair and match patches
        try:
            hair_region = self.segmenter.infer(img)
        except Exception as e:
            self.logger.exception(f"Error segmenting hair: {e}")
            return "Error segmenting hair"
        hair_region.save(os.path.join(output_dir, "hair_region.png"))
        best_name, best_score = self.patch_matcher.match(hair_region)

        self.logger.info(f"Best match: {best_name} (score: {best_score:.2f})")

        return best_name