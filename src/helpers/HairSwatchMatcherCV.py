import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from common import CallableComponent

class HairSwatchMatcherCV(CallableComponent):
    def __init__(self, resize_dim=(224, 224)):
        self.resize_dim = resize_dim

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = np.array(image.convert("RGB"))
        img = cv2.resize(img, self.resize_dim)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        return img_lab

    def extract_features(self, img_lab: np.ndarray) -> np.ndarray:
        # Mean and std of L, A, B channels
        features = []
        for i in range(3):
            channel = img_lab[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
        return np.array(features)

    def match(self, query_img: Image.Image, swatch_imgs: List[Tuple[str, Image.Image]]) -> str:
        query_lab = self.preprocess(query_img)
        query_feat = self.extract_features(query_lab)

        similarities = []
        for name, swatch in swatch_imgs:
            swatch_lab = self.preprocess(swatch)
            swatch_feat = self.extract_features(swatch_lab)
            sim = self.cosine_similarity(query_feat, swatch_feat)
            similarities.append((name, sim))

        # Higher cosine similarity = more similar
        best_match = max(similarities, key=lambda x: x[1])
        return best_match[0]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def __call__(self, query_img: Image.Image, swatch_imgs: List[Tuple[str, Image.Image]]) -> str:
        return self.match(query_img, swatch_imgs)