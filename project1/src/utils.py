import cv2
import numpy as np
from numpy.typing import NDArray

def show_single_channel(img, colormap = cv2.COLORMAP_TURBO):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.applyColorMap(img, colormap)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_channels(img: NDArray[np.float32], skip_display = False, colormap = cv2.COLORMAP_TURBO) -> NDArray[np.float32]:
    b, g, r = cv2.split(img)

    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    b_color = cv2.applyColorMap(b_norm, colormap)
    g_color = cv2.applyColorMap(g_norm, colormap)
    r_color = cv2.applyColorMap(r_norm, colormap)

    stacked = np.hstack((b_color, g_color, r_color))

    if not skip_display:
        cv2.imshow("B | G | R Channel Intensities", stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return stacked

def normalize(mat: NDArray[np.float32]) -> NDArray[np.float32]:
    v_max = mat.max()
    v_min = mat.min()
    if v_max == v_min:
        print("[Warn] All values are identical when normalize")
        return np.ones_like(mat)
    return (mat - v_min) / (v_max - v_min)