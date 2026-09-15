import numpy as np
from PIL import Image


def save_img_u8(img: np.ndarray, path: str):
    """Save a uint8 image (H, W, C) with values in [0, 255] as PNG."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_img_f32(img: np.ndarray, path: str):
    """Save a float32 image (e.g. depth map) as TIFF."""
    Image.fromarray(img).save(path)
