"""Module 2 - Preprocessing & Augmentation (OpenCV / NumPy only)."""
import cv2
import numpy as np


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def load_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    # handles both 0/1 masks and 0/255 masks
    return ((m > 0) if m.max() <= 1 else (m > 127)).astype(np.uint8)


def resize_pair(img, mask, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def normalize(img: np.ndarray, method: str = "zscore", clahe: bool = False) -> np.ndarray:
    """uint8 image -> float32. Per-image normalisation reduces brightness/contrast
    differences between datasets (a cheap first step of domain adaptation)."""
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    x = img.astype(np.float32) / 255.0
    if method == "zscore":
        x = (x - x.mean()) / (x.std() + 1e-6)
    elif method == "minmax":
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    elif method != "none":
        raise ValueError(f"unknown normalisation: {method}")
    return x.astype(np.float32)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, level: str = "basic"):
    """uint8 in, uint8 out.
    level='none'   : nothing
    level='basic'  : flip + small affine (geometry only)  -> used for the BASELINE
    level='strong' : basic + ultrasound-style intensity / speckle / noise / blur -> a domain-adaptation method (Module 6)
    """
    if level == "none":
        return img, mask
    h, w = img.shape

    if rng.random() < 0.5:                                   # horizontal flip
        img, mask = img[:, ::-1], mask[:, ::-1]

    angle = rng.uniform(-15, 15)                             # random affine
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * (w, h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if level == "basic":
        return img, mask.astype(np.uint8)

    x = img.astype(np.float32)
    x = x * rng.uniform(0.5, 1.4) + rng.uniform(-35, 35)                      # contrast / brightness
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)         # gamma
    if rng.random() < 0.7:                                                    # depth attenuation (top bright -> bottom dark)
        x = x * np.linspace(1.0, rng.uniform(0.5, 1.0), h, dtype=np.float32)[:, None]
    if rng.random() < 0.6:                                                    # speckle (multiplicative)
        x = x * (1 + rng.normal(0, rng.uniform(0.05, 0.2), x.shape))
    if rng.random() < 0.6:                                                    # gaussian noise
        x = x + rng.normal(0, rng.uniform(3, 14), x.shape)
    if rng.random() < 0.5:                                                    # blur
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.0))
    return np.clip(x, 0, 255).astype(np.uint8), mask.astype(np.uint8)
