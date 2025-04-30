import numpy as np
import os
import cv2
from numpy.typing import NDArray
import math
import argparse

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
            img2_i, img2_j = i + min(best_moving[0], 0), j + best_moving[1]
            in_img1 = 0 <= img1_i < h1 and j < w1
            in_img2 = 0 <= img2_i < h2 and 0 <= img2_j < w2
            if in_img1 and in_img2:
                if img1[img1_i, j, 3] == 0:
                    pano[i, j] = img2[img2_i, img2_j]
                elif img2[img2_i, img2_j, 3] == 0:
                    pano[i, j] = img1[img1_i, j]
                else:
                    pano[i, j] = img1_percent * img1[img1_i, j] + (1 - img1_percent) * img2[img2_i, img2_j]
                    pano[i, j, 3] = max(img1[img1_i, j, 3], img2[img2_i, img2_j, 3])
            elif in_img1:
                pano[i, j] = img1[img1_i, j]
            elif in_img2:
                pano[i, j] = img2[img2_i, img2_j]
    return pano

def end_to_end_alignment(image: NDArray, sum_moving_y: int) -> NDArray:
    w = image.shape[1]
    
    aligned_image = image
    shift = 0 if sum_moving_y > 0 else -sum_moving_y
    step = (sum_moving_y * 1.0) / w
    for i in range(w):
        aligned_image[:, i] = np.roll(image[:, i], int(shift), axis=0)
        shift += step

    return aligned_image

def load_image_infos(dir: str, file_name: str = "focal.txt") -> tuple[list[str], list[float]]:
    image_names, focal_lengths = [], []
    with open(os.path.join(dir, file_name), "r") as file:
        for line in file:
            image_name, focal = line.strip().split()
            image_names.append(image_name)
            focal_lengths.append(eval(focal))
    return image_names, focal_lengths

def cylindrical_warp(image: NDArray, focal: float) -> NDArray:
    h, w, _ = image.shape
    png_image = np.zeros((h, w, 4)).astype(np.uint8)
    png_image[:, :, :3] = image
    png_image[:, :, 3] = 255
    cylinder = np.zeros((h, w, 4)).astype(np.float32)
    x_origin, y_origin = w // 2, h // 2
    for i in range(h):
        for j in range(w):
            x, y = j - x_origin, i - y_origin
            x_prime = int(np.round(focal * math.atan(x / focal)))
            y_prime = int(np.round(focal * y / math.sqrt(x * x + focal * focal)))
            cylinder[y_prime + y_origin][x_prime + x_origin] = png_image[i][j]
    x_min, x_max = int(np.round(focal * np.atan(-x_origin / focal))), int(np.round(focal * np.atan((w - x_origin) / focal)))
    print(cylinder.shape)
    return cylinder[:, (x_min + x_origin) : (x_max + x_origin)]

if __name__ == "__main__":
    IMG_DIR = "../data1"
    OUTPUT_DIR = "../output/data1"
    
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--harris", action="store_true", help="use harris feature detection")
    group.add_argument("--moravec", action="store_true", help="use moravec feature detection")
    parser.add_argument("--ransac", type=float, help="the threshold used for ransac", default=5)
    args = parser.parse_args()
    feature_detection = "harris"
    if args.moravec:
        feature_detection = "moravec"
    ransac_threshold = args.ransac

    image_names, focal_lengths = load_image_infos(IMG_DIR)
    warp_images = [cylindrical_warp(cv2.imread(os.path.join(IMG_DIR, name)), focal) for (name, focal) in zip(image_names, focal_lengths)]
    for (name, image) in zip(image_names, warp_images):
        cv2.imwrite(os.path.join(OUTPUT_DIR, name[:-3] + "png"), image)
    feature_matches = [match(warp_images[i][:, :, :3], 
                             warp_images[i + 1][:, :, :3], 
                             0.5, save=os.path.join(OUTPUT_DIR, f"match_{feature_detection}_{i}.pkl"), 
                             feature_detection=feature_detection) for i in range(len(warp_images) - 1)]
    best_movings = [ransac(matches, ransac_threshold) for matches in feature_matches]
    print(best_movings)
    panorama = warp_images[0]
    sum_moving = [0, 0]
    for i, cur_moving in enumerate(best_movings):
        sum_moving = [a + b for a, b in zip(cur_moving, sum_moving)]
        panorama = blending(sum_moving, panorama, warp_images[i + 1])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "output.png"), panorama)
    aligned_panorama = end_to_end_alignment(panorama, sum_moving[0])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "output_aligned.png"), aligned_panorama)
