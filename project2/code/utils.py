import cv2
import numpy as np

def scale_hd(mat):
    return scale_image(mat, min(720 / mat.shape[0], 1280 / mat.shape[1]))

def scale_image(mat, scale):
    height, width = mat.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(mat, new_size, interpolation=cv2.INTER_CUBIC)

def show_2d_values(mat, colormap = cv2.COLORMAP_BONE):
    normalized = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    image = cv2.applyColorMap(normalized, colormap)

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(mat):
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Image", 200, 100)

    # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_to_uint8(mat):
    return cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def bgr_to_grayscale(mat):
    return np.dot(mat[..., :3], [0.114, 0.587, 0.299])