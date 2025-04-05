import cv2
import numpy as np
from PIL import Image

class SuperResWrapper:
    def __init__(self, sr_model_path, sr_model_name="edsr", scale=4):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(sr_model_path)
        self.sr.setModel(sr_model_name, scale)

    def enhance(self, pil_image):
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        enhanced_cv = self.sr.upsample(image_cv)
        enhanced_rgb = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)