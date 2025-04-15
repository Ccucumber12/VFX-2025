import os
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.filters import threshold_otsu

from utils import *

def get_R(intensity, sigma = 1.0, k = 0.04):
    Ix = (intensity[1:,:] - intensity[:-1,:])[:,1:]
    Iy = (intensity[:,1:] - intensity[:,:-1])[1:,:]

    Sx2 = gaussian_filter(Ix * Ix, sigma)
    Sy2 = gaussian_filter(Iy * Iy, sigma)
    Sxy = gaussian_filter(Ix * Iy, sigma)

    R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2 + Sy2) ** 2
    return R


def harris(intensity, sigma = 1.0, k = 0.04, thresh = None, min_distance = 10):
    '''
    Returns the coordinates of detected features.

    Param:
        intensity: A single channel image that stores intensity information.
        sigma: size of Gaussian window.
        k: constant for R.
        thresh: threshold for R (default using Otsu's method to find one).
        min_distance: minimum distance between two feature points.
    '''
    R = get_R(intensity, sigma, k)
    mask = (R == maximum_filter(R, min_distance * 2))
    R = normalize_to_uint8(R)

    if thresh:
        print(f"Given threshold = {thresh}")
    else:
        thresh = threshold_otsu(R)
        print(f"Otsu threshold = {thresh}")

    is_feature = (R > thresh) & mask
    coords = np.argwhere(is_feature)
    return [(int(1 + x), int(1 + y)) for x, y in coords]


def interactive_harris(intensity, image, sigma = 1.0, k = 0.04):
    '''
    Shows the detected feature points on the original image. Provides a slider
    to interactively set the threshold and see the results.
    '''
    R = get_R(intensity, sigma, k)
    R = normalize_to_uint8(R)

    thresh = threshold_otsu(R)
    print(f"Otsu threshold = {thresh}")

    cv2.namedWindow("Features")

    def redraw_features(val):
        global marked_image
        marked_image = image.copy()

        for x in range(R.shape[0]):
            for y in range(R.shape[1]):
                if R[x, y] > val:
                    cv2.circle(marked_image, (1 + y, 1 + x), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Features", marked_image)

    cv2.createTrackbar("Threshold", "Features", int(thresh), int(R.max()), redraw_features)
    redraw_features(thresh)

    cv2.waitKey(0)
    thresh = cv2.getTrackbarPos("Threshold", "Features")
    print(f"Final threshold = {thresh}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_DIR = "../data/feature-detection"
    IMG_NAME = "shirakawa.jpg"
    img_path = os.path.join(IMG_DIR, IMG_NAME)

    image = cv2.imread(img_path)
    image = scale_image(image, 1440 / image.shape[1])

    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)

    interactive_harris(I, image)

    # feature_points = harris(I, min_distance=10)
    # for x, y in feature_points:
    #     cv2.circle(image, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("Features", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()