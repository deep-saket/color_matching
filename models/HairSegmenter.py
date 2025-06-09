import cv2
import numpy as np
from common import InferenceVisionComponent
from PIL import Image


class HairSegmenter(InferenceVisionComponent):
    """
    Segments the hair region from a portrait image and returns a binary mask as PIL Image.
    """

    def __init__(self, haar_path=None):
        if haar_path is None:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def infer(self, image_data: Image.Image) -> Image.Image:
        # Convert PIL to OpenCV BGR
        image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            raise ValueError("No face detected.")

        # Use largest face
        (x, y, w, h) = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]

        # Define hair region
        hair_y1 = max(y - int(0.8 * h), 0)
        hair_y2 = y + int(0.2 * h)
        hair_x1 = max(x - int(0.1 * w), 0)
        hair_x2 = min(x + w + int(0.1 * w), image.shape[1])
        hair_roi = image[hair_y1:hair_y2, hair_x1:hair_x2]

        # HSV thresholding
        hsv = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])
        mask_roi = cv2.inRange(hsv, lower, upper)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

        # Embed into full-size mask
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[hair_y1:hair_y2, hair_x1:hair_x2] = mask_roi

        # Convert to PIL image
        return Image.fromarray(full_mask)