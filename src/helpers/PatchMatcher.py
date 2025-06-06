from common import BaseComponent
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import torch

class PatchMatcher(BaseComponent):
    def __init__(
        self,
        embedder: Any,
        swatches: List[Dict[str, Any]],
        threshold: float = 0.93
    ):
        """
        embedder: instance providing encode_image(Image) -> Tensor
        swatches: list of {"name": str, "embedding": Tensor}
        threshold: minimum cosine similarity to count as a match
        """
        super().__init__()
        self.embedder = embedder
        self.swatches = swatches
        self.threshold = threshold

    def match(
        self,
        image: Image.Image,
        patch_size: Tuple[int, int] = (64, 64),
        stride: Optional[Tuple[int, int]] = None
    ) -> Tuple[str, float]:
        """
        Splits `image` into patches, embeds each patch, and compares against all swatch embeddings.
        Returns the best‐matching swatch name and its score.
        If best score < threshold, returns ("NO_MATCH", best_score).
        """
        w, h = image.size
        pw, ph = patch_size
        sx, sy = stride if stride is not None else (pw, ph)

        best_name = "NO_MATCH"
        best_score = -1.0

        for top in range(0, h - ph + 1, sy):
            for left in range(0, w - pw + 1, sx):
                patch = image.crop((left, top, left + pw, top + ph))
                emb = self.embedder.encode_image(patch)
                for sw in self.swatches:
                    score = torch.nn.functional.cosine_similarity(
                        emb, sw["embedding"], dim=-1
                    ).item()
                    if score > best_score:
                        best_score = score
                        best_name = sw["name"]

        if best_score < self.threshold:
            return "NO_MATCH", best_score
        return best_name, best_score