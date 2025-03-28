import cv2
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from utils import *

def show_img(src):
    cv2.imshow("image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_hist(array: NDArray):
    flatten = array.flatten()
    plt.hist(flatten, bins=50, color='blue', edgecolor='black')
    plt.title("Histogram of 2D Array Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def apply_color_transfer(src, target):
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    src_l, src_a, src_b = cv2.split(src_lab)
    tgt_l, tgt_a, tgt_b = cv2.split(target_lab)
    merged_lab = cv2.merge([tgt_l, src_a, src_b])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def align_images(images):
    ref_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    
    aligned_images = [images[0]]
    sz = images[0].shape
    
    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_mode = cv2.MOTION_EUCLIDEAN
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        
        _, warp_matrix = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(img, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        aligned_images.append(aligned)
    
    return aligned_images

def load_hdr(filename):
    return cv2.imread(f"../data/hdr/{filename}.hdr", flags = cv2.IMREAD_UNCHANGED)

def generate_hdr():
    img_dir = "../data/ldr/wiki-sample"
    img_fn = ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    img_list = [cv2.imread(os.path.join(img_dir, fn)) for fn in img_fn]
    exposure_times = np.array([0.0333, 0.25, 2.5, 15.0], dtype=np.float32)

    img_list = align_images(img_list)

    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
    merge_robertson = cv2.createMergeRobertson()
    hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

    return hdr_debevec

def MinMaxTonemap(irradiance: NDArray[np.float32], l_pr = 1, u_pr = 99) -> NDArray[np.float32]:
    result = np.empty_like(irradiance)
    for i in range(3):
        luminance = irradiance[:,:,i]
        luminance = np.log(1e-5 + luminance)
        flatten = luminance.flatten()
        lb = np.percentile(flatten, l_pr)
        ub = np.percentile(flatten, u_pr)
        luminance = np.clip(luminance, lb, ub)
        result[:,:,i] = (luminance - lb) / (ub - lb)       
    return result

def ReinhardEachColorTonemap(irradiance: NDArray[np.float32], alpha = 0.18, l_white = 100) -> NDArray[np.float32]:
    result = np.empty_like(irradiance)
    for i in range(3):
        l_in = irradiance[:,:,i]
        l_avg = np.exp(np.average(np.log(1e-5 + l_in)))
        l_m = l_in * alpha / l_avg
        l_out = l_m / (1 + l_m) * (1 + l_m / (l_white ** 2))
        result[:,:,i] = l_out
    return result

def ReinhardLuminanceTonemap(irradiance: NDArray[np.float32], alpha = 0.18, l_white = 100) -> NDArray[np.float32]:
    l_in = 0.2126 * irradiance[:, :, 2] + 0.7152 * irradiance[:, :, 1] + 0.0722 * irradiance[:, :, 0]
    l_avg = np.exp(np.mean(np.log(1e-5 + l_in)))
    l_m = l_in * alpha / l_avg
    l_out = l_m / (1 + l_m) * (1 + l_m / (l_white ** 2))
    result = irradiance * (l_out[..., np.newaxis] / (l_in[..., np.newaxis] + 1e-5))
    return result

def main():
    filename = "debevec"

    img = cv2.imread(f"../data/hdr/{filename}.hdr", cv2.IMREAD_UNCHANGED)
    hdr_img = load_hdr(filename)
    # toned_img = ReinhardLuminanceTonemap(hdr_img, alpha=0.35, l_white=20)
    toned_img = ReinhardEachColorTonemap(hdr_img, alpha=0.45, l_white=30)
    # toned_img = MinMaxTonemap(hdr_img)
    show_img(toned_img)
    # cv2.imwrite(f"../output/{filename}.JPG", toned_img)
    exit()


if __name__ == "__main__":
    main()