import numpy as np
from scipy.stats import special_ortho_group
from tqdm import tqdm

from utils import *
from reinhard import interpolate

def random_rotation_matrix(n=3):
    """Generate a random nxn orthonormal rotation matrix."""
    return special_ortho_group.rvs(dim=n).astype(np.float32)

def transform_1d(source, target, bins=300):
    """Match the 1D histogram of source to that of target."""
    
    lower = min(source.min(), target.min())
    upper = max(source.max(), target.max())
    
    src_hist, bin_edges = np.histogram(source, bins=bins, range=[lower, upper], density=True)
    tar_hist, _ = np.histogram(target, bins=bins, range=[lower, upper], density=True)
    
    src_pdf = src_hist.cumsum().astype(np.float32)
    tar_pdf = tar_hist.cumsum().astype(np.float32)
    t = np.interp(src_pdf, tar_pdf, bin_edges[1:])
    
    return np.interp(source, bin_edges[1:], t)

def color_transfer_idt(src_img, ref_img, n_itr=20):
    src = src_img.reshape(-1, 3).astype(np.float32)
    ref = ref_img.reshape(-1, 3).astype(np.float32)
    
    for _ in range(n_itr):
        R = random_rotation_matrix()
        src_proj = src @ R
        ref_proj = ref @ R
        mapping = np.zeros(src_proj.shape)
        for i in range(3):
            mapping[:, i] = transform_1d(src_proj[:, i], ref_proj[:, i])
        src = src + (mapping - src_proj) @ R.T
    
    return np.clip(src.reshape(src_img.shape), 0, 255).astype(np.uint8)

def get_val(arr, x, y):
    if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1]:
        return arr[x, y]
    else:
        return 0.0

def update_J(J, psi, phi, I):
    h, w = J.shape
    J = J.astype(np.float32)
    J_new = np.copy(J).astype(np.float32)
    for x in range(h):
        for y in range(w):
            a1 = -(get_val(psi, x, y - 1) + get_val(psi, x, y)) / 2
            a2 = -(get_val(psi, x, y + 1) + get_val(psi, x, y)) / 2
            a3 = -(get_val(psi, x - 1, y) + get_val(psi, x, y)) / 2
            a4 = -(get_val(psi, x + 1, y) + get_val(psi, x, y)) / 2
            a5 = -(a1 + a2 + a3 + a4) + get_val(phi, x, y)
            J_new[x, y] = (get_val(phi, x, y) * get_val(I, x, y)
                           - a1 * (get_val(J, x, y - 1) - get_val(I, x, y - 1) + get_val(I, x, y))
                           - a2 * (get_val(J, x, y + 1) - get_val(I, x, y + 1) + get_val(I, x, y))
                           - a3 * (get_val(J, x - 1, y) - get_val(I, x - 1, y) + get_val(I, x, y))
                           - a4 * (get_val(J, x + 1, y) - get_val(I, x + 1, y) + get_val(I, x, y))) / a5
    return np.clip(J_new, 0, 255).astype(np.uint8)

def regrain(src_img, transfered_img, n_itr=10):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    gradient = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    phi = 30.0 / (1.0 + 10.0 * gradient)
    psi = np.where(gradient > 5, 1.0, gradient / 5.0)

    for c in range(3):
        J = transfered_img[:, :, c]
        for _ in range(n_itr):
            J = update_J(J, psi, phi, src_img[:, :, c])
        transfered_img[:, :, c] = J
    return transfered_img

def color_transfer_sequence(
        src_img: Union[str, NDArray],
        ref_img: Union[str, NDArray], 
        length: int = 50,
        result_dir: str = "idt",
        fps: int = 10,
    ) -> str:
    src_img = cv2.cvtColor(load_if_path(src_img), cv2.COLOR_RGB2BGR).astype(np.float32)
    ref_img = cv2.cvtColor(load_if_path(ref_img), cv2.COLOR_RGB2BGR).astype(np.float32)

    print("Interpolating...")
    result_img = color_transfer_idt(src_img, ref_img)
    regrain_img = regrain(src_img, result_img)
    cv2.imwrite(f"{OUT_DIR}/result.jpg", result_img)
    cv2.imwrite(f"{OUT_DIR}/result_regrain.jpg", regrain_img)
    seq = interpolate(src_img, regrain_img, length)

    print("Saving images...")
    h, w, _ = src_img.shape
    result_path = f"{OUT_DIR}/{result_dir}"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(f"{result_path}/sequence", exist_ok=True)
    anm = Animator(h, w, out=f"{result_path}/result.mp4", fps=fps)
    for i, img in enumerate(tqdm(seq)):
        cv2.imwrite(f"{result_path}/sequence/{i}.jpg", img)
        anm.write(img)
    anm.release()
    return result_path

def main():
    IMG_DIR = f"{ROOT_DIR}/images"
    src_img = cv2.imread(f"{IMG_DIR}/scotland_house.png")
    ref_img = cv2.imread(f"{IMG_DIR}/scotland_plain.png")

    result_img = color_transfer_idt(src_img, ref_img)
    cv2.imwrite(f"{OUT_DIR}/result.jpg", result_img)
    
    regrain_img = regrain(src_img, result_img)
    cv2.imwrite(f"{OUT_DIR}/result_regrain.jpg", regrain_img)

if __name__ == '__main__':
    main()
