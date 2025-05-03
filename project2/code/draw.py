import os
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import maximum_filter

from moravec import get_minE as get_moravec_E
from harris import get_R as get_harris_E, get_threshold
from utils import *

def save_plt_and_show():
    TMP_FILE = "../tmp/temp-plt.png"
    plt.savefig(TMP_FILE, dpi=200, bbox_inches='tight')
    plt.close()

    img = cv2.imread(TMP_FILE)
    show_image(img)


def feature_energy_histogram():
    src_dir = "../data/feature-detection"
    src_name = "geometry.jpg"
    src_path = os.path.join(src_dir, src_name)

    img = cv2.imread(src_path)
    I = bgr_to_grayscale(img)
    E = get_harris_E(I)

    thresh = threshold_otsu(E)
    values = E.ravel()

    plt.hist(values, bins=50, log=True, color='skyblue', edgecolor='black')
    plt.axvline(x=thresh, color='red', linestyle='--', linewidth=2, label=f'x = {thresh}')
    plt.xlabel('Value')
    plt.ylabel('Log Count')
    plt.grid(True, which='both', linestyle='--', linewidth='0.5')
    plt.tight_layout()
    plt.title("Harris")
    save_plt_and_show()

def feature_candidates():
    src_dir = "../data/feature-detection"
    src_name = "nagoya.jpg"
    src_path = os.path.join(src_dir, src_name)

    img = cv2.imread(src_path)
    I = bgr_to_grayscale(img)
    E = get_harris_E(I)

    E = normalize_to_uint8(E)
    thresh = threshold_otsu(E)

    canvas = img.copy()
    for x in range(E.shape[0]):
        for y in range(E.shape[1]):
            if E[x, y] > thresh:
                cv2.circle(canvas, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    show_image(canvas)
    cv2.imwrite("../report/images/features-otsu.jpg", scale_hd(canvas))


def feature_filter():
    src_dir = "../data/feature-detection"
    src_name = "nagoya.jpg"
    src_path = os.path.join(src_dir, src_name)

    img = cv2.imread(src_path)
    I = bgr_to_grayscale(img)
    E = get_harris_E(I)

    min_distance = 5
    mask = (E == maximum_filter(E, min_distance * 2))

    E = normalize_to_uint8(E)
    thresh = get_threshold(E)

    # is_feature = (E > thresh) & mask
    is_feature = (E > thresh)
    canvas = img.copy()
    for x in range(E.shape[0]):
        for y in range(E.shape[1]):
            if is_feature[x, y]:
                cv2.circle(canvas, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    canvas = canvas[2463:3180,1984:3061,:]
    show_image(canvas)
    cv2.imwrite("../report/images/features-unfiltered.jpg", scale_hd(canvas))


if __name__ == '__main__':
    pass
