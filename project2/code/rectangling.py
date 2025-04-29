import cv2
import os
from scipy.ndimage import convolve
from numpy.typing import NDArray
from tqdm import trange

from utils import *


def seam_carving(I: NDArray, img: NDArray) -> NDArray:
    n, m = I.shape

    pos = [(0, 0), (0, 0), (n-1, 0), (n-1, 0), (0, m-1), (0, m-1), (n-1, m-1), (n-1, m-1)]
    vec = [(0, 1), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (0, -1), (-1, 0)]

    EMPTY = 0
    best_dis = 0
    best_idx = 0
    for idx, ((x, y), (dx, dy)) in enumerate(zip(pos, vec)):
        dis = 0
        while I[x, y] == EMPTY:
            x += dx
            y += dy
            dis += 1
        if dis > best_dis:
            best_dis = dis
            best_idx = idx
    
    x, y = pos[best_idx]
    dx, dy = vec[best_idx]
    d = best_dis

    sx = n-d if dx == -1 else 0
    ex = d if dx == 1 else n
    sy = m-d if dy == -1 else 0
    ey = d if dy == 1 else m
    sub_img = I[sx:ex, sy:ey].astype(np.int32)

    AIx = np.abs(convolve(sub_img, [[-1], [0], [1]]))
    AIy = np.abs(convolve(sub_img, [[-1, 0, 1]]))
    E = AIx + AIy

    INF = 1e8
    E[sub_img == 0] = INF

    if best_idx & 1:
        E = np.transpose(E)
    
    sn, sm = E.shape
    trans = np.zeros_like(E, np.int8)
    connects = np.array([-1, 0, 1])

    for y in range(1, sm):
        col_prev = E[:, y - 1]
        prev_stack = np.full((3, sn), INF, dtype=E.dtype)

        for i, dx in enumerate(connects):
            x_src = np.arange(sn) + dx
            valid = (x_src >= 0) & (x_src < sn)
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
    
    if best_idx & 1:
        seam = np.transpose(seam)

    coords = np.argwhere(seam) + [sx, sy]

    canvas = img.copy()
    for x, y in coords:
        cv2.circle(canvas, (y, x), 3, [0, 0, 255], -1)
    show_image(canvas)

        

if __name__ == '__main__':
    IMG_DIR = "../data/rectangling"
    src_name = "day.jpg"
    src_path = os.path.join(IMG_DIR, src_name)

    src_img = cv2.imread(src_path)
    I = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    seam_carving(I, src_img)