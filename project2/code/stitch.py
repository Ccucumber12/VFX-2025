import numpy as np
import os
import cv2
from numpy.typing import NDArray
import math

from match import *

# Image Matching
def ransac(matches: list[tuple[tuple[int, int], tuple[int, int]]], threshold: float) -> tuple[int, int] :
    max_inlier_count, best_moving = 0, []
    for sample_pair in matches:
        moving = [sample_pair[1][0] - sample_pair[0][0], sample_pair[1][1] - sample_pair[0][1]]

        inlier_count = 0
        for p in matches:
            _moving = [p[1][0] - p[0][0], p[1][1] - p[0][1]]
            e = np.sum(np.square(np.array(_moving) - np.array(moving)))
            if e < threshold:
                inlier_count += 1
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_moving = moving

    return best_moving

# Blending
def blending(best_moving: tuple[int, int], img1: NDArray, img2: NDArray) -> NDArray:
    h1, w1, c = img1.shape
    h2, w2, _ = img2.shape
    x_overlap = w1 + best_moving[1]
    H, W = h2 + abs(best_moving[0]), w1 + w2 - x_overlap
    print(x_overlap, H, W)
    pano = np.zeros((H, W, c)).astype(np.float32)
    for i in range(H):
        for j in range(W):
            img1_percent = max(min(1.0, (w1 - j) * 1.0 / x_overlap), 0)
            img1_i = i - max(best_moving[0], 0)
            if img1_i >= 0 and img1_i < h1 and j < w1:
                pano[i, j] = img1_percent * img1[img1_i, j]
            img2_i, img2_j = i + min(best_moving[0], 0), j + best_moving[1]
            if img2_i >= 0 and img2_i < h2 and img2_j >= 0 and img2_j < w2:
                pano[i, j] += (1 - img1_percent) * img2[img2_i, img2_j]
    return pano

def load_image_infos(dir: str, file_name: str = "focal.txt") -> tuple[list[str], list[float]]:
    image_names, focal_lengths = [], []
    with open(os.path.join(dir, file_name), "r") as file:
        for line in file:
            image_name, focal = line.strip().split()
            image_names.append(image_name)
            focal_lengths.append(eval(focal) * 4)
    return image_names, focal_lengths

def cylindrical_warp(image: NDArray, focal: float) -> NDArray:
    h, w, c = image.shape
    cylinder = np.zeros((h, w, c)).astype(np.float32)
    x_origin, y_origin = w // 2, h // 2
    for i in range(h):
        for j in range(w):
            x, y = j - x_origin, i - y_origin
            x_prime = int(np.round(focal * math.atan(x / focal)))
            y_prime = int(np.round(focal * y / math.sqrt(x * x + focal * focal)))
            cylinder[y_prime + y_origin][x_prime + x_origin] = image[i][j]
    x_min, x_max = int(np.round(focal * np.atan(-x_origin / focal))), int(np.round(focal * np.atan((w - x_origin) / focal)))
    print(cylinder.shape)
    return cylinder[:, (x_min + x_origin) : (x_max + x_origin)]

if __name__ == "__main__":
    IMG_DIR = "../data2"
    OUTPUT_DIR = "../output/data2"

    image_names, focal_lengths = load_image_infos(IMG_DIR)
    warp_images = [cylindrical_warp(cv2.imread(os.path.join(IMG_DIR, name)), focal) for (name, focal) in zip(image_names, focal_lengths)]
    for (name, image) in zip(image_names, warp_images):
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), image)
    feature_matches = [match(warp_images[i], warp_images[i + 1], 0.5, save=os.path.join(OUTPUT_DIR, f"match-test{i}.pkl")) for i in range(len(warp_images) - 1)]
    best_movings = [ransac(matches, 5) for matches in feature_matches]
    print(best_movings)
    panorama = warp_images[0]
    sum_moving = [0, 0]
    for i, cur_moving in enumerate(best_movings):
        sum_moving = [a + b for a, b in zip(cur_moving, sum_moving)]
        panorama = blending(sum_moving, panorama, warp_images[i + 1])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cylindrical_5.jpg"), panorama)
