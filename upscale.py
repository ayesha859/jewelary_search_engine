import cv2
import numpy as np

class JewelryUpscaler:
    def __init__(self):
        pass

    def process_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        if img is None: return None
        h, w = img.shape[:2]
        # High quality 4x zoom simulation
        zoomed_img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(zoomed_img, -1, kernel)
        cv2.imwrite(output_path, sharpened)
        return output_path