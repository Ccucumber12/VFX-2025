import cv2
import numpy as np
from numpy.typing import NDArray
import time

def init_cv2_window(window_name = "CV2 Window") -> str:
    '''Move cv2 window to a better position on screen. Returns the window name. '''
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 200, 100)
    return window_name

def scale_hd(mat: NDArray) -> NDArray:
    '''Scale image to 1280 x 720 (HD).'''
    return scale_image(mat, min(720 / mat.shape[0], 1280 / mat.shape[1]))

def scale_image(mat: NDArray, scale: float, interpolation = cv2.INTER_CUBIC) -> NDArray:
    height, width = mat.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(mat, new_size, interpolation=interpolation)

def show_image(mat: NDArray, colormap = cv2.COLORMAP_BONE) -> NDArray:
    mat = mat.copy()
    if len(mat.shape) == 2: # grayscale
        if mat.dtype == np.bool_:
            mat = mat.astype(np.uint8)
        mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mat = cv2.applyColorMap(mat, colormap)

    winname = init_cv2_window()
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
    return mat

def normalize_to_uint8(mat: NDArray) -> NDArray:
    return cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def bgr_to_grayscale(mat: NDArray) -> NDArray:
    return np.dot(mat[..., :3], [0.114, 0.587, 0.299])

def draw_circles(mat: NDArray, coords: list[tuple[int, int]] | tuple[int, int], radius = 3) -> NDArray:
    if isinstance(coords, tuple):
        coords = [coords]
    mat = mat.copy()
    for x, y in coords:
        cv2.circle(mat, (y, x), radius=radius, color=(0, 0, 255), thickness=-1)
    return mat

def concat_images(mats: list[NDArray], spacing = -1) -> NDArray:
    '''
    Concat images and add spacing in between.

    Param:
        mats: list of images
        spacing: '-1' - auto spacing w.r.t. image width
                 other - given spacing
    '''
    if spacing == -1:
        spacing = max(1, int(mats[0].shape[1] / 10))
    if len(mats[0].shape) == 3:
        padding = np.ones((mats[0].shape[0], spacing, 3), dtype=mats[0].dtype) * 255
    else:
        padding = np.ones((mats[0].shape[0], spacing), dtype=mats[0].dtype) * 255
    
    spaced_mats = [mats[0]]
    for mat in mats[1:]:
        spaced_mats += [padding, mat]
    return cv2.hconcat(spaced_mats)

def random_color():
    hsv = np.array([[[
        np.random.randint(0, 180),   # H: Hue (0–179 in OpenCV)
        np.random.randint(150, 256), # S: Saturation
        np.random.randint(200, 256)  # V: Brightness (Value)
    ]]], dtype=np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)

    
def to_white_bg(mat: NDArray) -> NDArray:
    '''`mat` is bgra 4-channel image.'''
    b, g, r, a = cv2.split(mat)
    alpha = a.astype(np.float32) / 255.0
    alpha = np.stack([alpha]*3, axis=-1)

    bgr = cv2.merge([b, g, r]).astype(np.float32)

    white_bg = np.ones_like(bgr) * 255
    out = bgr * alpha + white_bg * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
    


class Timer:
    def __init__(self, decimal = 1):
        self._start_time = 0
        self._lap_time = 0
        self._decimal = decimal
    
    def start(self):
        self._start_time = self._lap_time = time.time()
    
    def lap(self):
        new_lap_time = time.time()
        ret = new_lap_time - self._lap_time
        self._lap_time = new_lap_time
        return round(ret, self._decimal)
    
    def stop(self):
        return round(time.time() - self._start_time, self._decimal)


class Animator:
    def __init__(self, height, width, out = "output.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        self.video = cv2.VideoWriter(out, fourcc, fps, (width, height))
    
    def write(self, frame: NDArray):
        self.video.write(frame)
    
    def release(self):
        self.video.release()
    
    def __del__(self):
        self.release()