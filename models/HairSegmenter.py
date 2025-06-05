import cv2
import numpy as np
from PIL import Image
from mediapipe import solutions as mp_solutions
from common import InferenceVisionComponent

class HairSegmenter(InferenceVisionComponent):
    def __init__(self, **kwargs):
        super().__init__()
        self.segmentor = mp_solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def segment(self, image: Image.Image) -> Image.Image:
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = self.segmentor.process(img_bgr)
        mask = result.segmentation_mask > 0.6
        h, w = mask.shape
        mask[int(h * 0.5):] = 0
        segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask.astype(np.uint8))
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = segmented[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))