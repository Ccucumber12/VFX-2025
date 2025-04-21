import numpy as np
import os
import cv2
from numpy.typing import NDArray

from match import *

# Image Matching
def ransac(matches: list[tuple[tuple[int, int], tuple[int, int]]], k: int, threshold: int) -> tuple[int, int] :
    best_inlier, best_moving = [], []
    for _ in range(k):
        sample_pair = matches[np.random.randint(len(matches))]
        moving = [sample_pair[1][0] - sample_pair[0][0], sample_pair[1][1] - sample_pair[0][1]]

        inlier = []
        for p in matches:
            _moving = [p[1][0] - p[0][0], p[1][1] - p[0][1]]
            e = np.sum(np.square(np.array(_moving) - np.array(moving)))
            if e < threshold:
                inlier.append(p)
        if len(inlier) > len(best_inlier):
            best_inlier = inlier
            best_moving = moving

    return best_moving

# Blending
def blending(best_moving: tuple[int, int], img1: NDArray, img2: NDArray) -> NDArray:
    h1, w1, c = img1.shape
    _, w2, _ = img2.shape
    x_overlap = w1 + best_moving[1]
    H, W = h1 + abs(best_moving[0]), w1 + w2 - x_overlap
    print(x_overlap, H, W)
    pano = np.zeros((H, W, c)).astype(np.float32)
    for i in range(H):
        for j in range(W):
            img1_percent = max(min(1.0, (w1 - j) * 1.0 / x_overlap), 0)
            img1_i = i - max(best_moving[0], 0)
            if img1_i >= 0 and img1_i < h1 and j < w1:
                pano[i, j] = img1_percent * img1[img1_i, j]
            img2_i, img2_j = i + min(best_moving[0], 0), j + best_moving[1]
            if img2_i >= 0 and img2_i < H and img2_j >= 0 and img2_j < w2:
                pano[i, j] += (1 - img1_percent) * img2[img2_i, img2_j]
    return pano

if __name__ == "__main__":
    IMG_DIR = "../data/"
    image_names = sorted(os.listdir(IMG_DIR))
    image_paths = [os.path.join(IMG_DIR, f) for f in image_names if f.endswith('.JPG')]
    print(image_paths)
    
    images = [scale_hd(cv2.imread(path)) for path in image_paths]
    feature_matches = match(images[0], images[1], 0.5)
    draw_matches(images[0], images[1], feature_matches)
    draw_match_vectors(*images[0].shape[:2], feature_matches)
    
    best_moving = ransac(feature_matches, 3, 1000)
    print(best_moving)
    _, w, _ = images[0].shape
    pano = blending(best_moving, images[0], images[1])
    cv2.imwrite("../output/panorama.jpg", pano)
