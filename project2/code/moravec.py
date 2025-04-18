import os
import cv2
from scipy.ndimage import convolve, maximum_filter
from skimage.filters import threshold_otsu

from utils import *


def get_sxy_from_patch_size(patch_size):
    sx = 1 + (patch_size + 1) // 2
    sy = (patch_size + 1) // 2
    return sx, sy

def get_minE(intensity, patch_size = 3):
    n, m = intensity.shape

    shifts = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    shift_diffs = [(intensity[1+u:n-1+u,v:m-1+v] - intensity[1:-1,:-1]) ** 2 for u, v in shifts]

    kernel = np.ones((patch_size, patch_size), dtype=int)
    Es = [convolve(diff, kernel, mode='constant', cval=0) for diff in shift_diffs]
    minE = np.minimum.reduce(Es)
    return minE


def moravec(intensity, patch_size = 3, thresh = None, min_distance = 10):
    '''
    Returns the coordinates of detected features.

    Param:
        intensity: A single channel image that stores intensity information.
        patch_size: window size
        thresh: threshold for E (default using Otsu's method to find one).
        min_distance: minimum distance between two feature points.
    '''
    minE = get_minE(intensity, patch_size)
    mask = (minE == maximum_filter(minE, min_distance * 2))
    minE = normalize_to_uint8(minE)

    if thresh:
        print(f"Given threshold = {thresh}")
    else:
        thresh = threshold_otsu(minE)
        print(f"Otsu threshold = {thresh}")

    is_feature = (minE > thresh) & mask
    coords = np.argwhere(is_feature)
    sx, sy = get_sxy_from_patch_size(patch_size)
    return [(int(sx + x), int(sy + y)) for x, y in coords]
    

def interactive_moravec(intensity, image, patch_size = 3):
    '''
    Shows the detected feature points on the original image. Provides a slider
    to interactively set the threshold and see the results.
    '''
    sx, sy = get_sxy_from_patch_size(patch_size)
    minE = get_minE(intensity, patch_size)
    minE = normalize_to_uint8(minE)

    thresh = threshold_otsu(minE)
    print(f"Otsu threshold = {thresh}")

    cv2.namedWindow("Features")

    def redraw_features(val):
        global marked_image
        marked_image = image.copy()
        for x in range(minE.shape[0]):
            for y in range(minE.shape[1]):
                if minE[x, y] > val:
                    cv2.circle(marked_image, (sy + y, sx + x), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Features", marked_image)

    cv2.createTrackbar("Threshold", "Features", thresh, minE.max(), redraw_features)
    redraw_features(thresh)

    cv2.waitKey(0)
    thresh = cv2.getTrackbarPos("Threshold", "Features")
    print(f"Final threshold = {thresh}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_DIR = "../data/feature-detection"
    IMG_NAME = "toyosato.jpg"
    img_path = os.path.join(IMG_DIR, IMG_NAME)

    image = cv2.imread(img_path)
    image = scale_hd(image)

    I = bgr_to_grayscale(image)

    interactive_moravec(I, image)

    # feature_points = moravec(I)
    # for x, y in feature_points:
    #     cv2.circle(image, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Features", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()