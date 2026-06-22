import numpy as np
import cv2
from cv2 import Canny
from kymatio.numpy import Scattering2D

CANNY_LOW_THRESHOLD = 100 # Low threshold for Canny edge detection
CANNY_HIGH_THRESHOLD = 200 # High threshold for Canny edge detection

WST_J = 2 # Scale of the wavelet scattering transform
WST_L = 8 # Number of orientations for the wavelet scattering transform

_SCATTERING_CACHE: dict = {}  # Cache for Scattering2D objects to avoid reinitialization

def _canny(gray_img: np.ndarray) -> np.ndarray:
    """Canny edge detection."""
    
    return Canny(gray_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

def _wst(gray_img: np.ndarray) -> np.ndarray:
    """Wavelet scattering transform (placeholder)."""

    h, w = gray_img.shape[:2]
    scattering = _SCATTERING_CACHE.get((h, w))
    if scattering is None:
        scattering = Scattering2D(J=WST_J, shape=(h, w), L=WST_L, max_order=0)
        _SCATTERING_CACHE[(h, w)] = scattering
    
    coef0 = np.asarray(scattering.scattering(gray_img.astype(np.float32) / 255.0))[0]
    coef0 = cv2.resize(coef0, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(coef0.min()), float(coef0.max())
    out =  (coef0 - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(coef0)

    return out.astype(np.uint8)

def apply_channel_mode(img: np.ndarray, mode: int) -> np.ndarray:
    """Apply channel transformation based on the specified mode."""

    if mode == 1 or img.ndim != 3 or img.shape[2] != 3:
        return img  # No transformation
    
    gray_img  = img[..., 2].copy() # R -> Gray

    if mode == 2:
        img[..., 1] = _canny(gray_img) # G <- Canny
    elif mode == 3:
        img[..., 1] = _wst(gray_img) # G <- WST
    elif mode == 4:
        img[..., 0] = _canny(gray_img) # B <- Canny
        img[..., 1] = _wst(gray_img)   # G <- WST

    return img


class ChannelTransform:
    """Channel transformation for image preprocessing."""
    
    def __init__(self, mode: int):
        """
        Initialize the ChannelTransform with the specified mode.
        Args:
            mode (int): The mode of channel transformation to apply.
                        1: No transformation
                        2: Canny edge detection on the green channel
                        3: Wavelet scattering transform on the green channel
                        4: Canny edge detection on the blue channel and wavelet scattering transform on the green channel
        """
        self.mode = mode

    def __call__(self, labels: dict) -> dict:
        labels['img'] = apply_channel_mode(labels['img'], self.mode)
        return labels