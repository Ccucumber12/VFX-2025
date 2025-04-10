import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *
from align import MtbAlign

Z_min = 0
Z_max = 255
ldr_path = "../data/ldr/cosmology-hall"
hdr_path = "../data/hdr/"

def load_images(dir_path):
    list_path = os.path.join(dir_path, "image_list.txt")
    images = []
    exposure_times = []
    with open(list_path, "r") as list:
        for line in list:
            image_name, exposure_time = line.strip().split()
            images.append(cv2.imread(os.path.join(dir_path, image_name)))
            exposure_times.append(eval(exposure_time))
    
    images = MtbAlign(images)
    return np.array(images), np.array(exposure_times, dtype=np.float32)
    
def sample_pixels(images):
    n, _, _, c = images.shape       # Number of images, height, width, channels
    num_sample = Z_max - Z_min + 1
    sampled_pixels = np.zeros((num_sample, n, c), dtype=np.uint8)
    sample_img = images[n // 2]

    for channel in range(c):
        for i in range(num_sample):
            rows, cols  = np.where(sample_img[:, :, channel] == i)
            if len(rows) != 0:
                index = np.random.randint(0, len(rows))
                sampled_pixels[i, :, channel] = images[:, rows[index], cols[index], channel]
    return sampled_pixels

def weighting(z):
    return np.where(z <= (Z_max + Z_min) / 2, z - Z_min, Z_max - z) + 1.

def solve_Debevec_g(Z, B, l=10):
    n = 256             # Intensity levels
    N = Z.shape[0]      # Number of pixels sampled
    P = Z.shape[1]      # Number of images

    A = np.zeros((N * P + 1 + 254, n + N), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)

    # Include the data-fitting equations
    k = 0
    for i in range(N):
        for j in range(P):
            z_ij = Z[i, j]
            w_ij = weighting(z_ij)
            A[k, z_ij] = w_ij
            A[k, n+i] = -w_ij
            b[k, 0] = w_ij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 127] = 1
    k += 1

    # Include the smoothness equations
    for i in range(254):
        w = weighting(i + 1)
        A[k, i] = l * w
        A[k, i + 1] = -2 * l * w
        A[k, i + 2] = l * w
        k += 1

    # Solve Ax = b
    x = np.dot(np.linalg.pinv(A), b)
    g = x[:n].flatten()
    # lnE = x[n:].flatten()

    return g

def solve_Robertson_g(images, exposure_times):
    iterate_times = 2
    n, h, w, c = images.shape
    g_curves = []
    E = np.zeros((h, w, c)).astype(np.float32)
    exposure_times_3d = exposure_times[:, np.newaxis, np.newaxis]

    # Init g_curves to be linear functions
    g_curves = [np.arange(0, 1, 1/256) for _ in range(c)]
    for channel in range(c):
        img_channel = images[..., channel]
        for i in range(iterate_times):
            # Assuming g is known, optimize for E
            g = np.array([g_curves[channel][img_channel[j]] for j in range(n)])
            weight = np.array(weighting(img_channel))
            numerator = np.sum(weight * g * exposure_times_3d, axis=0)
            denominator = np.sum(weight * exposure_times_3d * exposure_times_3d, axis=0)
            E[:, :, channel] = np.where(denominator != 0, numerator / denominator, 0)

            # Assuming E is known, optimize for g
            numerator = 0
            denominator = 0
            for m in range(256):
                index = np.where(img_channel == m)
                size = index[0].size
                numerator = np.sum([E[index[1][s]][index[2][s]][channel] * exposure_times[index[0][s]] for s in range(size)])
                denominator = size
                g_curves[channel][m] = numerator / denominator
            g_curves[channel] /= g_curves[channel][127]
    return g_curves

def save_curves_image(curves, method):
    channel_names = ["Blue", "Green", "Red"]
    _, axes = plt.subplots(1, 3, figsize=(20, 10))
    for ax, name, curve in zip(axes, channel_names, curves):
        ax.set_title(name, fontsize=25)
        ax.plot(curve, np.arange(256), c=name)
        ax.set_xlabel("Log exposure X")
        ax.set_ylabel("Pixel value Z")
        ax.grid(linestyle=":", linewidth=1)
    plt.savefig(os.path.join(ldr_path, f"{method}_curves.jpg"))

    plt.figure(figsize=(6, 10))
    plt.title("g curves", fontsize=25)
    plt.xlabel("Log exposure X")
    plt.ylabel("Pixel value Z")
    plt.grid(linestyle=":", linewidth=1)
    for name, curve in zip(channel_names, curves):
        plt.plot(curve, np.arange(256), c=name)
    plt.savefig(os.path.join(ldr_path, f"{method}_curves_1.jpg"))

def construct_hdr_radiance_map(images, log_exposure_times, g_curves):
    n, h, w, c = images.shape
    ln_E = np.zeros((h, w, c), dtype=np.float32)

    for channel in range(c):
        img_channel = images[..., channel]
        
        g = np.array([g_curves[channel][img_channel[k]] for k in range(n)])     # Shape (n, h, w)
        weight = np.array([weighting(img_channel[k]) for k in range(n)])        # Shape (n, h, w)

        numerator = np.sum(weight * (g - log_exposure_times[:, np.newaxis, np.newaxis]), axis=0)
        denominator = np.sum(weight, axis=0)
        ln_E_channel = np.zeros((h, w), dtype=float)
        ln_E_channel = np.where(denominator != 0, numerator / denominator, 0)

        ln_E[..., channel] = ln_E_channel
    return np.exp(ln_E)

def save_hdr_image(hdr_image, method):
    cv2.imwrite(os.path.join(hdr_path, f"{method}.hdr"), hdr_image.astype(np.float32))

def tone_mapping(hdr_image, method):
    tonemapDrago = cv2.createTonemapDrago(2.0, 0.75)
    ldrDrago = tonemapDrago.process(hdr_image) * 255
    cv2.imwrite(os.path.join(ldr_path, f"{method}_tonemap_drago.jpg"), ldrDrago.astype(np.uint8))

    tonemapReinhard = cv2.createTonemapReinhard(2.2, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdr_image) * 255
    cv2.imwrite(os.path.join(ldr_path, f"{method}_tonemap_reinhard.jpg"), ldrReinhard.astype(np.uint8))

    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdr_image) * 255
    cv2.imwrite(os.path.join(ldr_path, f"{method}_tonemap_mantiuk.jpg"), ldrMantiuk.astype(np.uint8))

def debevec_hdr(images, exposure_times):
    print("Debevec's method:")
    log_exposure_times = np.log(exposure_times)

    G_CURVES_SAVE = "../data/g_curves.npy"
    if os.path.exists(G_CURVES_SAVE):
        g_curves = np.load(G_CURVES_SAVE)
        print("g loaded from file")
    else:
        samples = sample_pixels(images)
        print("After sampling images, samples.shape: ", samples.shape)
        g_curves = []
        for channel in range(3):
            g = solve_Debevec_g(samples[:, :, channel], log_exposure_times)
            g_curves.append(g)
            print("After solving g for channel ", channel)
        save_curves_image(g_curves, "debevec")
        # np.save(G_CURVES_SAVE, g_curves)

    hdr = construct_hdr_radiance_map(images, log_exposure_times, g_curves)
    print("After constructing radiance map, hdr shape: ", hdr.shape)
    save_hdr_image(hdr, "debevec")
    print("After saving hdr")
    # tone_mapping(hdr, "debevec")

def robertson_hdr(images, exposure_times):
    print("Robertson's method:")
    g_curves = solve_Robertson_g(images, exposure_times)
    print("After solving g")
    save_curves_image(g_curves, "robertson")
    hdr = construct_hdr_radiance_map(images, np.log(exposure_times), g_curves)
    print("After constructing radiance map, hdr shape: ", hdr.shape)
    save_hdr_image(hdr, "robertson")
    print("After saving hdr")
    # tone_mapping(hdr, "robertson")


def main():
    images, exposure_times = load_images(ldr_path)
    print("After loading images, images.shape: ", images.shape)
    # debevec_hdr(images, exposure_times)
    robertson_hdr(images, exposure_times)

    return

if __name__ == '__main__':
    main()
