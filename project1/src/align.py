import cv2
import numpy as np
from numpy.typing import NDArray

from utils import *

MAX_SHIFT = 10

def shift_and_fill(img: NDArray, dx: int, dy: int) -> NDArray:
    shifted = np.roll(img, shift=(dx, dy), axis=(0, 1))
    if dx > 0: 
        shifted[:, :dx] = shifted[:, dx:dx+1]
    elif dx < 0:
        shifted[:, dx:] = shifted[:, dx-1:dx]
    if dy > 0:
        shifted[:dy, :] = shifted[dy:dy+1, :]
    elif dy < 0:
        shifted[dy:, :] = shifted[dy-1:dy, :]
    return shifted

def find_best_shift(ref: NDArray, tar: NDArray, mask: NDArray) -> tuple[int, int]:
    if ref.shape != tar.shape:
        print("[Error] image shape not matched.")
        raise RuntimeError
    min_diff = ref.shape[0] * ref.shape[1]
    for dx in range(-MAX_SHIFT, MAX_SHIFT+1):
        for dy in range(-MAX_SHIFT, MAX_SHIFT+1):
            shifted = shift_and_fill(tar, dx, dy)
            diff = np.sum((shifted != ref) & mask)
            if diff < min_diff:
                min_diff = diff
                best_dx, best_dy = dx, dy
    return best_dx, best_dy

def MtbAlign(images: list[NDArray]) -> list[NDArray]:
    bitmaps = []
    bitmasks = []
    n = len(images)
    ref = n // 2
    tolerance = 2
    for img in images:
        # b, g, r = cv2.split(img)
        # gray = (r * 54 + g * 183 + b * 19).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = np.median(gray)
        bitmaps.append((gray > threshold))
        bitmasks.append((gray < threshold - tolerance) | (gray > threshold + tolerance))

    shifted = []
    dx_max, dy_max = -MAX_SHIFT, -MAX_SHIFT
    dx_min, dy_min = MAX_SHIFT, MAX_SHIFT
    for i in range(n):
        best_dx, best_dy = find_best_shift(bitmaps[ref], bitmaps[i], bitmasks[ref] & bitmasks[i])
        print(f"best shift for image {i}: dx = {best_dx}, dy = {best_dy}")
        shifted.append(np.roll(images[i], shift=(best_dx, best_dy), axis=(0, 1)))

        dx_max = max(dx_max, best_dx)
        dy_max = max(dy_max, best_dy)
        dx_min = min(dx_min, best_dx)
        dy_min = min(dy_min, best_dx)

    dx_max = max(0, dx_max)
    dy_max = max(0, dy_max)
    dx_min = dx_min if min(0, dx_min) != 0 else None
    dy_min = dy_min if min(0, dy_min) != 0 else None
    cropped = [img[dx_max:dx_min, dy_max:dy_min] for img in shifted]
    return cropped

