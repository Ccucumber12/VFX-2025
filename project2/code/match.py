import os
import cv2
from numpy.typing import NDArray
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from utils import *
from harris import harris


def get_descriptor(mat: NDArray, x: int, y: int) -> NDArray:
    l = 5
    hl = l // 2
    if len(mat.shape) == 3:
        patch = np.zeros((l, l, 3), dtype=mat.dtype)
    else:
        patch = np.zeros((l, l), dtype=mat.dtype)

    for dx in range(-hl, hl + 1):
        for dy in range(-hl, hl + 1):
            xi, yi = x + dx, y + dy
            if 0 <= xi < mat.shape[0] and 0 <= yi < mat.shape[1]:
                patch[dx + hl, dy + hl] = mat[xi, yi]

    patch = patch.flatten()
    return patch.astype(np.uint32)


def match(src_img: NDArray, dst_img: NDArray, overlap: float = 1, unique_thresh: float = 0.8, save: str = None) -> list[tuple[tuple[int,int], tuple[int,int]]]:
    '''
    Find feature points on `src_img` and match them with points on `dst_img`. 
    Returns a list of matches, each match consists of an xy coordinate from 
    `src_img` and another on `dst_img`.

    Param:
        `src_img`: find feature on this image, assume on the left.
        `dst_img`: match feature on this image, assume on the right.
        `overlap`: ratio of overlap part, should be a value in (0, 1].
        `unique_thresh`: a match is good if `min_diff / second_min_diff` is 
        smaller then this value, should be a value in (0, 1].
        `save`: path for the save file, will return immediately if save file 
        exists, default not loading nor saving.
    '''
    if save and os.path.exists(save):
        with open(save, 'rb') as f:
            matches = pickle.load(f)
        print(f"Matches loaded from save file {save}")
        return matches

    if not 0 < overlap <= 1:
        print(f"[Error] overlap range not in (0, 1], value = {overlap}")
        raise ValueError

    timer = Timer()
    timer.start()

    start_idx = int(src_img.shape[1] * (1 - overlap))
    src_I = bgr_to_grayscale(src_img)

    feats = harris(src_I[:,start_idx:], max_n = 1000) # only overlap part
    feats = [(x, y+start_idx) for x, y in feats]
    print(f"Fetched features! n = {len(feats)}, time = {timer.lap()}s")

    # show_image(draw_circles(src_img, feats))

    dst_n, dst_m = dst_img.shape[:2]
    dst_xys = [(x, y) for x in range(dst_n) for y in range(int(dst_m * overlap))]

    print(f"Building destination descriptor Ball-tree...")
    dst_des = [get_descriptor(dst_img, x, y) for x, y in dst_xys]
    dst_tree = BallTree(dst_des)
    print(f"Ball-tree Successfully built! time = {timer.lap()}s")
    
    matches = []
    for sx, sy in tqdm(feats):
        src_des = get_descriptor(src_img, sx, sy)

        diffs, indices = dst_tree.query([src_des], k=2)

        diffs = diffs[0]
        indices = indices[0]

        diff_ratio = round(diffs[0] / diffs[1], 3)
        if diff_ratio >= unique_thresh:
            continue

        ret_idx = indices[0]
        ret_x, ret_y = dst_xys[ret_idx]
        matches.append(((sx, sy), (ret_x, ret_y)))

        # show_image(concat_images([draw_circles(src_img, (sx, sy), radius=5), 
        #                           draw_circles(dst_img, (ret_x, ret_y), radius=5)]))

        # def patch_from_des(des):
        #     des = des.copy()
        #     des = des.reshape(5, 5, 3)
        #     return des.astype(np.uint8)

        # ret_des = dst_des[ret_idx]
        # show_image(concat_images([scale_image(patch_from_des(src_des), 40, cv2.INTER_NEAREST),
        #                           scale_image(patch_from_des(ret_des), 40, cv2.INTER_NEAREST)]))

    print(f"Match finished! Found n = {len(matches)} matches. time = {timer.lap()}s")
    
    if save:
        with open(save, "wb") as f:
            pickle.dump(matches, f)
        print(f"Matches saved at {save}")

    return matches


def draw_matches(src_img: NDArray, dst_img: NDArray, matches: list[tuple[tuple[int,int], tuple[int,int]]]):
    pad = 10
    canvas = concat_images([src_img, dst_img], spacing=pad)
    trans_dst = lambda x, y : (x, y + src_img.shape[1] + pad)

    for (sx, sy), (dx, dy) in matches:
        dx, dy = trans_dst(dx, dy)
        color = random_color()
        cv2.circle(canvas, (sy, sx), 4, color, -1)
        cv2.circle(canvas, (dy, dx), 4, color, -1)
        cv2.line(canvas, (sy, sx), (dy, dx), color, 1)

    show_image(canvas)
    cv2.imwrite(f"../output/{src_name[:-4]}-matched.jpg", canvas)


def draw_match_vectors(n, m, matches):
    size = 128
    vectors = np.array([np.array([0.5+(x2-x1)/n, (y1-y2)/m]) for (x1, y1), (x2, y2) in matches])
    vectors = (vectors * size).astype(np.uint8)

    cnt = np.zeros((size, size))
    for x, y in vectors:
        cnt[x, y] += 1

    plt.imshow(cnt, cmap='hot', interpolation='gaussian')
    plt.colorbar()
    plt.xticks(ticks=[0, size/2, size], labels=[0, int(m/2), m])
    plt.yticks(ticks=[0, size/2, size], labels=[int(-n/2), 0, int(n/2)])
    plt.savefig("../output/match_vector_heatmap.png", dpi=160)
    img = cv2.imread("../output/match_vector_heatmap.png")
    show_image(img)
    

if __name__ == "__main__":
    IMG_DIR = "../data/cks-hall"
    src_name = "148A8737.JPG"
    dst_name = "148A8739.JPG"

    src_path = os.path.join(IMG_DIR, src_name)
    dst_path = os.path.join(IMG_DIR, dst_name)

    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

    src_img = scale_hd(src_img)
    dst_img = scale_hd(dst_img)

    matches = match(src_img, dst_img, 0.5, save="../tmp/match-test.pkl")
    draw_matches(src_img, dst_img, matches)
    draw_match_vectors(*src_img.shape[:2], matches)
