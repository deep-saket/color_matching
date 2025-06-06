import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from common import InferenceImageEmbeddingComponent

class ViTB32Infer(InferenceImageEmbeddingComponent):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base32",
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)

        # Load both model & processor from Hugging Face—caches locally by default
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        self.logger.info(f"{model_name} loaded on {self.device}")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        self.logger.debug("Encoding image to CLIP embedding")
        # The processor returns a dict with pixel_values already batched
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(pixel_values)
        return emb