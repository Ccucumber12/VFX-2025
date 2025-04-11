import os
import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from utils import *

def spatial_suppress(binary_array, space=5):
    coords = np.argwhere(binary_array)
    kept = []

    tree = cKDTree(coords)
    suppressed = np.zeros_like(binary_array, dtype=bool)
    used = np.zeros(len(coords), dtype=bool)

    for i, pt in enumerate(coords):
        if used[i]:
            continue
        kept.append(pt)
        nearby = tree.query_ball_point(pt, r=space)
        used[nearby] = True

    for x, y in kept:
        suppressed[x, y] = True
    return suppressed


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


def moravec(intensity, thresh = None, min_distance = 5, patch_size = 3):
    '''
    Returns the coordinates of detected features.

    Param:
        intensity: A single channel image that stores intensity information.
        thresh: threshold for E (default using otsu's method to find one).
        min_distance: minimum distance between two feature points.
        patch_size:
    '''
    minE = get_minE(intensity, patch_size)

    if thresh == None:
        thresh = threshold_otsu(minE)
        print(f"Otsu threshold = {thresh}")
    else:
        print(f"Given threshold = {thresh}")

    isFeature = minE > thresh
    coords = np.argwhere(isFeature)
    if len(coords) == 0:
        return []

    scores = minE[isFeature]
    sorted_indices = np.argsort(-scores)
    coords = coords[sorted_indices]

    tree = cKDTree(coords)
    selected = []
    suppressed = np.zeros(len(coords), dtype=bool)

    for i, pt in enumerate(coords):
        if suppressed[i]:
            continue
        selected.append(tuple(pt))
        nearby = tree.query_ball_point(pt, r=min_distance)
        suppressed[nearby] = True

    sx, sy = get_sxy_from_patch_size(patch_size)
    return [(int(sx + x), int(sy + y)) for x, y in selected]
    

def interactive_moravec(intensity, image, patch_size = 3):
    '''
    Shows the detected feature points on the original image. Provides a slider
    to interactively set the threshold and see the results.
    '''
    sx, sy = get_sxy_from_patch_size(patch_size)
    minE = get_minE(intensity, patch_size)

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
    image = scale_image(image, 0.5)

    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)

    interactive_moravec(I, image)

    # feature_points = moravec(I, min_distance=10)
    # for x, y in feature_points:
    #     cv2.circle(image, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Features", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
       