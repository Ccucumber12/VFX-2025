import cv2
import numpy as np

def scale_image(mat, scale):
    height, width = mat.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(mat, new_size, interpolation=cv2.INTER_CUBIC)

def show_2d_values(mat):
    normalized = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    cv2.imshow("Normalized Image", normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()