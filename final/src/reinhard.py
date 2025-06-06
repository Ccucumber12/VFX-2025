import cv2
from tqdm import tqdm
from typing import Union

from utils import *

def image_stats(image):
    mean, std = cv2.meanStdDev(image)
    return mean.flatten(), std.flatten()


def color_transfer(src_img, ref_img):
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_src, std_src = image_stats(src_lab)
    mean_ref, std_ref = image_stats(ref_lab)

    result = (src_lab - mean_src) * (std_ref / std_src) + mean_ref
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


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
    ref_img = cv2.imread(f"{IMG_DIR}/sunset-2.jpg")

    result_img = color_transfer(src_img, ref_img)

    # show_image(result_img)
    cv2.imwrite(f"{OUT_DIR}/result.jpg", result_img)   
    # interpolate(src_img, result_img, 45)

if __name__ == '__main__':
    # main()
    color_transfer_sequence("images/shirakawa.jpg", "images/sunset-1.jpg")    
