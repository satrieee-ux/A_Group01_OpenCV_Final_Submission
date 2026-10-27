"""Image loading, validation and preprocessing shared by training and inference."""
import cv2
import numpy as np

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

MIN_SIDE = 32  # images smaller than this (in pixels) are rejected

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


class InvalidImageError(ValueError):
    """Raised when an image cannot be decoded or is unusable."""


def load_image_bgr(path):
    """Read an image file with OpenCV. Returns None if it cannot be decoded.

    np.fromfile + imdecode avoids cv2.imread problems with non-ASCII Windows paths.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_image_bytes(data: bytes):
    """Decode an uploaded file (bytes) into a BGR image, or raise InvalidImageError."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    return validate_image(img)


def validate_image(img):
    if img is None:
        raise InvalidImageError("The file could not be decoded as an image.")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InvalidImageError(f"Image is too small ({w}x{h}); minimum side is {MIN_SIDE}px.")
    return img


def resize_rgb(img_bgr):
    """BGR -> RGB and resize to IMAGE_SIZE x IMAGE_SIZE (uint8, HxWx3).

    The image is resized directly (aspect ratio is not preserved). The same
    function is used in training and inference, so the model always sees the
    same kind of input.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def to_model_input(img_bgr):
    """Full preprocessing: resize, scale to 0-1, ImageNet-normalise, HWC -> CHW (float32)."""
    rgb = resize_rgb(img_bgr).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return np.transpose(rgb, (2, 0, 1))