import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from common.BaseComponent import BaseComponent
from config.loader import settings

class SwatchDetails(BaseComponent, dict):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, swatches_path: str, vlm_model):
        if not hasattr(self, '_initialized'):
            super().__init__()
            super(dict, self).__init__()
            self.save_path = settings['swatch_details']['args']['save_path']
            self.swatches_path = swatches_path
            self.vlm_model = vlm_model
            self._process_swatches()
            self._initialized = True

    def _resize_image(self, image: Image.Image, reduction_factor: float = 2.0) -> Image.Image:
        w, h = image.size
        if reduction_factor >= 1.0:
            return image
        new_size = (int(w * reduction_factor), int(h * reduction_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _brightness_prefix(self, image: Image.Image) -> str:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        mean = arr.mean()
        if mean < 80:
            return "dark "
        if mean > 180:
            return "light "
        return ""

    def _process_swatches(self):
        if os.path.exists(self.save_path):
            self.load_color_mappings(self.save_path)
            return

    # 1) Gather all swatch file‐paths
        paths = [
            os.path.join(self.swatches_path, f)
            for f in os.listdir(self.swatches_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # 2) First pass: single‐image inference to get a coarse label
        self.clear()
        initial_map = {}
        for path in paths:
            img = Image.open(path).convert("RGB")
            img = self._resize_image(img)
            prefix = self._brightness_prefix(img)
            prompt = (
                f"{prefix}"
                "Analyze this hair-color swatch image and reply with exactly one concise hair color name "
                "in lowercase (e.g. \"medium ash brown\"). Do not include numbers, punctuation, adjectives "
                "beyond pure color descriptors, or any additional commentary—only the color name."
            )
            desc = self.vlm_model.infer(image_data=img, prompt=prompt).strip()
            name = os.path.basename(path)
            initial_map[name] = desc

        self.update(initial_map)

        # 3) Group by initial description and refine duplicates
        
        groups = defaultdict(list)
        for name, desc in initial_map.items():
            groups[desc].append(name)

        for desc, names in groups.items():
            if len(names) > 1:
                
                swatch_paths = [self._resize_image(Image.open(os.path.join(self.swatches_path, n))).convert("RGB") for n in names]
                prompt = (
                    f"You have {len(names)} hair-color swatch images all initially labeled “{desc}.” "
                    "Return a valid JSON array of exactly "
                    f"{len(names)} refined hair color names in lowercase, in the same order. "
                    "Each entry must be a single, pure color name (e.g. \"light auburn\"), "
                    "with no punctuation, numbers, or commentary."
                )
                try:
                    raw = self.vlm_model.infer_multi_image(swatch_paths, prompt)
                    refined = json.loads(raw)
                    if isinstance(refined, list) and len(refined) == len(names):
                        for n, new_desc in zip(names, refined):
                            self[n] = new_desc.strip()
                except Exception:
                    pass

        self.save_color_mappings(self.save_path)


    def save_color_mappings(self, output_path: str):
            """Save the color mappings to a JSON file."""
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self), f, indent=2, ensure_ascii=False)

    def load_color_mappings(self, input_path: str):
        """Load color mappings from a JSON file."""
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.clear()
                self.update(mappings)
