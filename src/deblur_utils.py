import cv2
import numpy as np
from PIL import Image

class DeblurWrapper:
    def __init__(self, kernel_size=5, amount=1.5):
        self.kernel_size = kernel_size
        self.amount = amount

    def deblur(self, pil_image):
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(image_cv, (self.kernel_size, self.kernel_size), 0)
        sharpened = cv2.addWeighted(image_cv, 1.0 + self.amount, blurred, -self.amount, 0)
        sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(sharpened_rgb)