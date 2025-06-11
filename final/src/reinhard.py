import cv2
from tqdm import tqdm
from typing import Union

from utils import *


def srgb_to_linear(img: NDArray):
    img /= 255.0
    img = np.where(img <= 0.04045,
                   img / 12.92,
                   ((img + 0.055) / 1.055) ** 2.4)
    img = np.clip(img, 0, 1)
    return img

def linear_to_srgb(img: NDArray):
    img = np.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * (img ** (1/2.4)) - 0.055
    )
    img *= 255.0
    img = np.clip(img, 0, 255)
    return img

M_rgb2xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])
M_xyz2lms = np.array([
    [0.4002, 0.7075, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182]
])
M_rgb2lms = np.dot(M_xyz2lms, M_rgb2xyz)
M_lms2rgb = np.linalg.inv(M_rgb2lms)

M_lms2lab = np.array([
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
    [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [ 1/np.sqrt(2), -1/np.sqrt(2),             0]
])
M_lab2lms = np.linalg.inv(M_lms2lab)

def bgr2lab(img: NDArray):
    shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = srgb_to_linear(img)
    img = img.reshape(-1, 3)
    img = np.dot(img, M_rgb2lms.T)
    img = np.clip(img, 1e-6, None)
    img = np.log10(img)
    img = np.dot(img, M_lms2lab.T)
    img = img.reshape(shape)
    return img

def lab2bgr(img: NDArray):
    shape = img.shape
    img = img.reshape(-1, 3)
    img = np.dot(img, M_lab2lms.T)
    img = 10.0 ** img
    img = np.dot(img, M_lms2rgb.T)
    img = img.reshape(shape)
    img = np.clip(img, 0, 1)
    img = linear_to_srgb(img)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img

def image_stats(image):
    mean, std = cv2.meanStdDev(image)
    return mean.flatten(), std.flatten()   

def color_transfer(src_img, ref_img):
    # src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_lab = bgr2lab(src_img)
    ref_lab = bgr2lab(ref_img)

    mean_src, std_src = image_stats(src_lab)
    mean_ref, std_ref = image_stats(ref_lab)

    result = (src_lab - mean_src) * (std_ref / std_src) + mean_ref

    # return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return lab2bgr(result)


def interpolate(start_img, end_img, length):
    start_img = start_img.astype(np.float32)
    end_img = end_img.astype(np.float32)

    alphas = np.linspace(0, 1, length)
    seq = [(1 - alpha) * start_img + alpha * end_img for alpha in tqdm(alphas)]
    seq = [np.clip(img, 0, 255).astype(np.uint8) for img in seq]
    return seq


def color_transfer_sequence(
        src_img: Union[str, NDArray], 
        ref_img: Union[str, NDArray], 
        length: int = 50,
        result_dir: str = "reinhard",
        fps: int = 10,
    ) -> str:
    """
    Convert the color of `src_img` to `ref_img`. This function creates an 
    interpolation sequence of `length` images and a video. The images are stored
    under the `sequence` directory numbered `0` to `length-1`. The video is 
    saved as `result.mp4`. 

    Param: 
    - `src_img`: source image path or object.
    - `ref_img`: reference image path or object. 
    - `length`: number of interpolations between source and result. 
    - `result_dir`: the results will be store under this directory under `./output`.
    - `fps`: fps of the output video. 

    Return: The output directory path.
    """
    src_img = load_if_path(src_img)
    ref_img = load_if_path(ref_img)

    print("Interpolating...")
    result_img = color_transfer(src_img, ref_img)
    seq = interpolate(src_img, result_img, length)

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
    src_img = cv2.imread(f"{IMG_DIR}/toyosato-day.jpg")
    ref_img = cv2.imread(f"{IMG_DIR}/toyosato-dusk.jpg")

    result_img = color_transfer(src_img, ref_img)

    show_image(result_img)
    # cv2.imwrite(f"{OUT_DIR}/result.jpg", result_img)   

if __name__ == '__main__':
    main()
    # color_transfer_sequence("images/shirakawa.jpg", "images/sunset-2.jpg")    
