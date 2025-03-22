import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

Z_min = 0
Z_max = 255

def load_images(dir_path):
    list_path = os.path.join(dir_path, "image_list.txt")
    images = []
    exposure_times = []
    with open(list_path, "r") as list:
        for line in list:
            image_name, exposure_time = line.strip().split()
            images.append(cv2.imread(os.path.join(dir_path, image_name)))
            exposure_times.append(eval(exposure_time))
    return np.array(images), np.array(exposure_times, dtype=np.float32)
    
def sample_pixels(images, num_sample = Z_max - Z_min):
    n, h, w, c = images.shape   # Number of images, height, width, channels
    sampled_pixels = np.zeros((num_sample, n, c), dtype = np.uint8)
    sample_img = images[n // 2]

    for channel in range(c):
        for i in range(num_sample):
            rows, cols  = np.where(sample_img[:, :, channel] == i)
            if len(rows) != 0:
                index = np.random.randint(0, len(rows))
                sampled_pixels[i, :, channel] = images[:, rows[index], cols[index], channel]
    return sampled_pixels
        
    for channel in range(c):
        for i in range(num_sample):
            sample_h, sample_w = np.random.randint(0, h), np.random.randint(0, w)
            intensities = images[:, sample_h, sample_w, channel]
            sampled_pixels[i, :, channel] = intensities
    return sampled_pixels   # Number of sample, number of images, channels

def weighting(z):
    return np.where(z <= (Z_max + Z_min) / 2, z - Z_min, Z_max - z)

def solve_g(Z, B, l=1):
    n = 256             # Intensity levels
    N = Z.shape[0]      # Number of pixels sampled
    P = Z.shape[1]      # Number of images

    A = np.zeros((N * P + 1 + 254, n + N), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)

    # Include the data-fitting equations
    k = 0
    for i in range(N):
        for j in range(P):
            k = i * P + j
            z_ij = Z[i, j]
            w_ij = weighting(z_ij)
            A[k, z_ij] = w_ij
            A[k, n+i] = -w_ij
            b[k, 0] = w_ij * B[j]
            # k += 1

    # Fix the curve by setting its middle value to 0
    A[N * P, 127] = 1
    # k += 1

    # Include the smoothness equations
    for i in range(254):
        k = N * P + 1 + i
        w = weighting(i + 1)
        A[k, i] = l * w
        A[k, i + 1] = -2 * l * w
        A[k, i + 2] = l * w
        # k += 1

    # Solve Ax = b
    x = np.dot(np.linalg.pinv(A), b)
    g = x[:n].flatten()
    lnE = x[n:].flatten()

    return g, lnE

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


def save_hdr_image(hdr_image, filename = "output.hdr"):
    cv2.imwrite(filename, hdr_image.astype(np.float32))

def tone_mapping(hdr_image, filename = "output.jpg"):
    tonemapDrago = cv2.createTonemapDrago(2.0, 0.7)
    ldrDrago = tonemapDrago.process(hdr_image) * 255
    print("ldrDrago: ", np.min(ldrDrago), np.max(ldrDrago))
    print(hdr_image)
    print(ldrDrago)
    cv2.imwrite("./tonemap_drago.jpg", ldrDrago.astype(np.uint8))

    tonemapReinhard = cv2.createTonemapReinhard(3, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdr_image) * 255
    cv2.imwrite("./tonemap_reinhard.jpg", ldrReinhard.astype(np.uint8))

    tonemapMantiuk = cv2.createTonemapMantiuk(3.5, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdr_image) * 255
    cv2.imwrite("./tonemap_mantiuk.jpg", ldrMantiuk.astype(np.uint8))

def main():
    images, exposure_times = load_images("./data")
    log_exposure_times = np.log(exposure_times)
    print(log_exposure_times)
    print(np.min(images), np.max(images))
    # alignMTB = cv2.createAlignMTB()
    # alignMTB.process(images, images)
    print("After loading images, images.shape: ", images.shape)
    samples = sample_pixels(images)
    print("After sampling images, samples.shape: ", samples.shape)
    g_curves = []
    for channel in range(3):
        g, _ = solve_g(samples[:, :, channel], log_exposure_times)
        g_curves.append(g)
        print("After solving g for channel ", channel)
    # Show images 
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    channel_name = ['Blue', 'Green', 'Red']
    for channel in range(3):
        c_n = channel_name[channel]
        ax[channel].set_title( f"{c_n}", fontsize=25)
        ax[channel].plot(g_curves[channel], np.arange(256), c=c_n)
        ax[channel].set_xlabel('log exposure X ')
        ax[channel].set_ylabel('pixel value Z')
        ax[channel].grid(linestyle=':', linewidth=1)
    plt.show()
    hdr = construct_hdr_radiance_map(images, log_exposure_times, g_curves)
    print(np.min(hdr), np.max(hdr))
    print("After constructing radiance map, hdr shape: ", hdr.shape)
    save_hdr_image(hdr)
    print("After saving hdr")
    tone_mapping(hdr)

    return

if __name__ == '__main__':
    main()
