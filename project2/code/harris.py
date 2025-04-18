import os
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter, convolve
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt

from utils import *

def get_R(intensity, sigma = 1.0, k = 0.04):
    Kx = np.array([[-1, 0, 1]])
    Ix = convolve(intensity, Kx)

    Ky = np.array([[-1], [0], [1]])
    Iy = convolve(intensity, Ky)

    Sx2 = gaussian_filter(Ix * Ix, sigma)
    Sy2 = gaussian_filter(Iy * Iy, sigma)
    Sxy = gaussian_filter(Ix * Iy, sigma)

    R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2 + Sy2) ** 2
    return R


def get_threshold(R):
    # thresh = threshold_otsu(R)
    # print(f"Otsu threshold : {thresh}")
    # return threshold_otsu(R)

    counter = [0] * 256
    for r in np.nditer(R):
        counter[r] += 1
    thresh = np.argmax(counter)
    print(f"Auto harris threshold: {thresh}")
    return np.argmax(counter)


def harris(intensity, sigma = 1.0, k = 0.04, thresh = None, min_distance = 3, max_n = None):
    '''
    Returns the coordinates of detected features.

    Param:
        intensity: A single channel image that stores intensity information.
        sigma: size of Gaussian window.
        k: constant for R.
        thresh: threshold for R.
        min_distance: minimum distance between two feature points.
    '''
    R = get_R(intensity, sigma, k)
    mask = (R == maximum_filter(R, min_distance * 2))
    R = normalize_to_uint8(R)

    if not thresh:
        thresh = get_threshold(R)
    
    if not max_n:
        max_n = R.size

    is_feature = (R > thresh) & mask
    coords = np.argwhere(is_feature)
    if len(coords) > max_n:
        rs = R[coords[:, 0], coords[:, 1]]
        top_indices = np.argsort(rs)[::-1][:max_n]
        coords = coords[top_indices]
    return [(x, y) for x, y in coords]


def interactive_harris(intensity, image, sigma = 1.0, k = 0.04):
    '''
    Shows the detected feature points on the original image. Provides a slider
    to interactively set the threshold and see the results.
    '''
    R = get_R(intensity, sigma, k)
    R = normalize_to_uint8(R)

    thresh = get_threshold(R)

    cv2.namedWindow("Features")

    def redraw_features(val):
        global marked_image
        marked_image = image.copy()

        for x in range(R.shape[0]):
            for y in range(R.shape[1]):
                if R[x, y] > val:
                    cv2.circle(marked_image, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Features", marked_image)

    cv2.createTrackbar("Threshold", "Features", int(thresh), int(R.max()), redraw_features)
    redraw_features(thresh)

    cv2.waitKey(0)
    thresh = cv2.getTrackbarPos("Threshold", "Features")
    print(f"Final threshold = {thresh}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_DIR = "../data/feature-detection"
    IMG_NAME = "rome1.jpg"
    img_path = os.path.join(IMG_DIR, IMG_NAME)

    image = cv2.imread(img_path)
    image = scale_hd(image)

    I = bgr_to_grayscale(image)

    interactive_harris(I, image)

    # feature_points = harris(I)
    # for x, y in feature_points:
    #     cv2.circle(image, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Features", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

