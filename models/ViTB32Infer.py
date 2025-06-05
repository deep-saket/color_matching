import torch
import clip
from PIL import Image
from common import InferenceImageEmbeddingComponent

class ViTB32Infer(InferenceImageEmbeddingComponent):
    def __init__(self, model_name="ViT-B/32",  device="cuda"):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.logger.info(f"{model_name} model loaded on %s", self.device)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        self.logger.debug("Encoding image to CLIP embedding")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img_tensor)
        return emb