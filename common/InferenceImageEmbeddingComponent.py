# file: common/inference_component.py

from common.BaseComponent import BaseComponent
from abc import abstractmethod


class InferenceImageEmbeddingComponent(BaseComponent):
    """
    Base class for all inference components, regardless of modality.

    Subclasses must implement:
        - infer(image_data=None, prompt=None) → str

    Where:
      - image_data: bytes | PIL.Image.Image | str (file path) | None
      - prompt: str | None

    Each subclass determines which of these inputs it requires.
    """

    @abstractmethod
    def encode_image(self, image_data=None) -> str:
        """
        Run inference on the provided inputs.

        Args:
            image_data: Optional; image bytes, PIL.Image, or file path.

        Returns:
            str: The model’s generated text (or stringified result).
        """
        raise NotImplementedError("Subclasses must implement infer()")