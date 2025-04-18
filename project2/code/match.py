import os
import cv2

from utils import *
from harris import *

def get_descriptor(image, x, y):
    l = 5
    hl = l // 2
    patch = np.zeros((l, l, 3), dtype=image.dtype)

    for dx in range(-hl, hl + 1):
        for dy in range(-hl, hl + 1):
            xi, yi = x + dx, y + dy
            if 0 <= xi < image.shape[0] and 0 <= yi < image.shape[1]:
                patch[dx + hl, dy + hl] = image[xi, yi]

    return patch


if __name__ == "__main__":
    IMG_DIR = "../data/feature-detection"
    IMG1_NAME = "rome1.jpg"
    IMG2_NAME = "rome2.jpg"

    img1_path = os.path.join(IMG_DIR, IMG1_NAME)
    img2_path = os.path.join(IMG_DIR, IMG2_NAME)

    image1 = cv2.imread(img1_path)
    image1 = scale_hd(image1)

    image2 = cv2.imread(img2_path)
    image2 = scale_hd(image2)
    
    I1 = bgr_to_grayscale(image1)
    I2 = bgr_to_grayscale(image2)

    f1 = harris(I1, max_n = 1000, thresh=37)
    f2 = harris(I2, max_n = 1000, thresh=37)

    t_image1 = image1.copy()
    t_image2 = image2.copy()
    for x, y in f1:
        cv2.circle(t_image1, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    for x, y in f2:
        cv2.circle(t_image2, (y, x), radius=3, color=(0, 0, 255), thickness=-1)
    show_image(cv2.hconcat([t_image1, t_image2]))

    np.set_printoptions(precision=1, suppress=True, floatmode='fixed')

    def get_diff(v1, v2):
        return np.sum((v1 - v2)**2)
    
    for tx, ty in f1:
        t_des = get_descriptor(image1, tx, ty)

        matched_idx = np.argmin([get_diff(t_des, get_descriptor(image2, x, y)) for x, y in f2])
        mx, my = f2[matched_idx]


        m_des = get_descriptor(image2, mx, my)
        min_diff = get_diff(t_des, m_des)
        if min_diff >= 3000:
            continue

        t_image1 = image1.copy()
        t_image2 = image2.copy()
        cv2.circle(t_image1, (ty, tx), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(t_image2, (my, mx), radius=3, color=(0, 0, 255), thickness=-1)
        show_image(cv2.hconcat([t_image1, t_image2]))

        def scale_up(mat, l):
            return cv2.resize(mat, (l, l), interpolation=cv2.INTER_NEAREST)

        # t_des = scale_up(t_des, 250)
        # m_des = scale_up(m_des, 250)
        # show_image(cv2.hconcat([t_des, m_des]))


