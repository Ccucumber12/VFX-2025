import cv2
import numpy as np
from numpy.typing import NDArray

def show_channels(img: NDArray[np.float32], skip_display = False) -> NDArray[np.float32]:
    b, g, r = cv2.split(img)

    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    b_color = cv2.applyColorMap(b_norm, cv2.COLORMAP_TURBO)
    g_color = cv2.applyColorMap(g_norm, cv2.COLORMAP_TURBO)
    r_color = cv2.applyColorMap(r_norm, cv2.COLORMAP_TURBO)

    stacked = np.hstack((b_color, g_color, r_color))

    if not skip_display:
        cv2.imshow("B | G | R Channel Intensities", stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return stacked
