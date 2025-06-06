import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageOps
from io import BytesIO
from huggingface_hub import InferenceClient
from common import InferenceVLComponent
from typing import Union


class QwenV25Infer(InferenceVLComponent):
    """
    A class to perform inference using the Qwen2.5-VL model, either locally or via an API.
    """

    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='cuda'):
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.device = device
        self.client = None
        self.model = None
        self.processor = None

        if self.api_endpoint and self.api_token:
            self.client = InferenceClient(model=api_endpoint, token=api_token)
        elif model_name:
            print(f"Loading {model_name} model...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            print("Model loaded!")
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError("Either API details or a model name must be provided for inference.")

    def infer(self, image_data, prompt):
        if not image_data:
            raise ValueError("Image data cannot be None")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.client:
                response = self._infer_via_api(image_data, prompt)
                return response if isinstance(response, str) else str(response)
            else:
                return self._infer_locally(image_data, prompt)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}") from e

    def _infer_locally(self, image_data, prompt):
        # 1) Load/convert the image
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("Image must be bytes, a PIL Image, or a file path string.")

        # 2) Center-crop to 512×512
        image = ImageOps.fit(image, (512, 512), method=Image.LANCZOS)

        # 3) Build a chat‐style message list
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 4) Apply the chat template (keeps tokenizer happy) and extract vision inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # 5) Run through the processor to get final model inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Record prompt length so we can slice off prompt tokens later
        prompt_len = inputs["input_ids"].shape[-1]

        # 6) Generate and decode
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids = generated_ids[:, prompt_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _infer_via_api(self, image_data, prompt):
        image = Image.open(BytesIO(image_data)).convert("RGB")
        response = self.client.text_to_image(prompt, image=image)
        return response or {"error": "API request failed."}

    def infer_multi_image(self,
                                image_datas: list[Union[bytes, str, Image.Image]],
                                prompt: str
                                ) -> str:
        """
        One forward‐pass over N images. Returns a single text string
        describing each swatch in order (e.g. "1. light blonde, 2. dark brown, ...").
        """
        # 1) load & normalize all images
        imgs = []
        for d in image_datas:
            if isinstance(d, bytes):
                img = Image.open(BytesIO(d)).convert("RGB")
            elif isinstance(d, Image.Image):
                img = d
            elif isinstance(d, str):
                img = Image.open(d).convert("RGB")
            else:
                raise ValueError("Each item must be bytes, PIL Image, or file‐path string.")
            img = ImageOps.fit(img, (512, 512), method=Image.LANCZOS)
            imgs.append(img)

        # 2) build one chat message: a sequence of image blocks + your prompt
        content = []
        for i, img in enumerate(imgs, start=1):
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "user", "content": content}
        ]

        # 3) apply chat template & extract vision inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # 4) single‐call to processor → model.generate
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # remember how many prompt‐tokens we fed
        prompt_len = inputs["input_ids"].shape[-1]

        # 5) exactly one generate() over the whole batch
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=256)

        # 6) slice off the prompt and decode
        gen_ids = gen_ids[:, prompt_len:]
        result = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return result
