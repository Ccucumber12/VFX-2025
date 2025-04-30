import cv2
import os
from scipy.ndimage import convolve
from numpy.typing import NDArray
from tqdm import tqdm

from utils import *

EMPTY = 0


class SeamAnimation:
    def __init__(self, height, width, out = "output.avi"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        self.video = cv2.VideoWriter(out, fourcc, fps, (width, height))
    
    def write(self, frame: NDArray):
        self.video.write(frame)
    
    def release(self):
        self.video.release()
    
    def __del__(self):
        self.release()


class Transform:
    def __init__(self, transpose: bool, flipud: bool):
        self._tr = transpose
        self._ud = flipud
    
    def transpose(self, img: NDArray) -> NDArray:
        if img.ndim == 2:
            return img.T
        else:
            return np.transpose(img, (1, 0, 2))
    
    def apply(self, img: NDArray) -> NDArray:
        img = img.copy()
        if self._tr: img = self.transpose(img)
        if self._ud: img = np.flipud(img)
        return img
    
    def reverse(self, img: NDArray) -> NDArray:
        img = img.copy()
        if self._ud: img = np.flipud(img)
        if self._tr: img = self.transpose(img)
        return img


def get_longest_true_segment(arr: NDArray) -> tuple[int, int]:
    padded = np.pad(arr, (1, 1), constant_values=False)
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends   = np.flatnonzero(diff == -1)

    lengths = ends - starts
    if len(lengths) == 0:
        return (0, 0)
    max_idx = np.argmax(lengths)
    return starts[max_idx], ends[max_idx]


def get_segment(I: NDArray) -> tuple[Transform, tuple[int, int]]:
    '''
    Returns the transformation that makes the segment on top horizontally.
    The second tuple indicates the left / right bound of the longest segment.
    '''
    transforms = [Transform(bool(i&1), bool(i&2)) for i in range(4)]

    best_t = transforms[0]
    best_l, best_r = 0, 0

    for t in transforms:
        row = t.apply(I)[0]
        l, r = get_longest_true_segment(row == EMPTY)
        if r - l > best_r - best_l:
            best_t = t
            best_l = l
            best_r = r

    return best_t, (best_l, best_r)


def seam_carving(img: NDArray) -> NDArray:
    INF = 1e8
    img = img.copy()
    n, m = img.shape[:2]

    anm = SeamAnimation(*scale_hd(img).shape[:2], "../output/seam.mp4")
    count = np.count_nonzero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) == EMPTY)

    with tqdm(total=count) as pbar:
        while True:
            I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            T, (l, r) = get_segment(I)
            if r - l == 0:
                break
            subI = T.apply(I)[:,l:r].astype(np.int32)
            AIx = np.abs(convolve(subI, [[-1], [0], [1]]))
            AIy = np.abs(convolve(subI, [[-1, 0, 1]]))
            E = AIx + AIy
            E[subI == 0] = INF
            
            sn, sm = E.shape
            trans = np.zeros_like(E, np.int8)
            connects = np.array([-1, 0, 1])

            for y in range(1, sm):
                col_prev = E[:, y - 1]
                prev_stack = np.full((3, sn), INF, dtype=E.dtype)

                for i, dx in enumerate(connects):
                    x_src = np.arange(sn) + dx
                    valid = (0 <= x_src) & (x_src < sn)
                    prev_stack[i, valid] = col_prev[x_src[valid]]

                min_vals = np.min(prev_stack, axis=0)
                min_indices = np.argmin(prev_stack, axis=0)

                E[:, y] = np.minimum(E[:, y] + min_vals, INF)
                trans[:, y] = connects[min_indices]
            
            seam = np.full_like(E, fill_value=False, dtype=np.bool_)
            x = np.argmin(E[:,-1])
            for y in range(sm-1, -1, -1):
                seam[x, y] = True
                x += trans[x, y]

            img = T.apply(img)
            coords = np.argwhere(seam) + [0, l]
            for x, y in coords:
                img[:x-1, y] = img[1:x, y]
                img[x, y] = ((img[np.clip(x-1, 0, n-1), y].astype(np.int16) + 
                            img[np.clip(x+1, 0, n-1), y].astype(np.int16)) // 2).astype(np.uint8)

            canvas = img.copy()
            for x, y in coords:
                cv2.circle(canvas, (y, x), 2, [0, 0, 255], -1)
            canvas = T.reverse(canvas)
            # show_image(T.reverse(canvas))
            anm.write(scale_hd(canvas))

            img = T.reverse(img)
            pbar.update(r - l)

    return img

        

if __name__ == '__main__':
    IMG_DIR = "../data/rectangling"
    src_name = "day.jpg"
    src_path = os.path.join(IMG_DIR, src_name)

    src_img = cv2.imread(src_path)

    img = seam_carving(src_img)
    cv2.imwrite("../output/day-rect.jpg", img)