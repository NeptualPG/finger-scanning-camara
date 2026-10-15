import cv2
import numpy as np


def open_camera(camera_index=0):
    """Open the configured camera device."""
    return cv2.VideoCapture(camera_index)


def prepare_frame(frame):
    """Mirror the camera image for a natural user-facing preview."""
    return cv2.flip(frame, 1)


def _chromatic_aberration(frame, shift=3):
    """Offset color channels to create a broken-lens effect."""
    blue, green, red = cv2.split(frame)
    red = np.roll(red, shift, axis=1)
    blue = np.roll(blue, -shift, axis=1)
    return cv2.merge((blue, green, red))


def _add_scanlines(frame, intensity=30, gap=3):
    """Darken regular horizontal rows like a CRT display."""
    textured = frame.copy()
    textured[::gap] = np.clip(
        textured[::gap].astype(np.int16) - intensity, 0, 255
    ).astype(np.uint8)
    return textured


def _add_grain(frame, strength=11):
    """Add restrained luminance noise for an analog texture."""
    noise = np.random.randint(
        -strength, strength + 1, frame.shape, dtype=np.int16
    )
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _add_glitch_bands(frame, probability=0.10, max_shift=24):
    """Occasionally displace horizontal bands without breaking the frame."""
    if np.random.random() > probability:
        return frame

    height, width = frame.shape[:2]
    glitched = frame.copy()
    for _ in range(np.random.randint(1, 4)):
        y_start = np.random.randint(0, height)
        band_height = np.random.randint(5, max(6, min(40, height)))
        y_end = min(y_start + band_height, height)
        shift = np.random.randint(-max_shift, max_shift + 1)
        glitched[y_start:y_end] = np.roll(
            glitched[y_start:y_end], shift, axis=1
        )
    return glitched


def _add_vignette(frame, strength=0.60):
    """Darken the edges to focus attention on the center of the feed."""
    height, width = frame.shape[:2]
    x = np.arange(width) - width / 2
    y = np.arange(height) - height / 2
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)
    max_distance = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    vignette = 1 - strength * (distance / max_distance) ** 2
    return np.clip(
        frame.astype(np.float32) * vignette[:, :, None], 0, 255
    ).astype(np.uint8)


def apply_punk_filter(frame):
    """Apply a high-contrast neon punk, CRT, and VHS-style camera treatment."""
    crushed = np.clip(frame.astype(np.int16) - 30, 0, 255).astype(np.uint8)

    contrast_lut = np.array(
        [
            int(np.clip(255 / (1 + np.exp(-0.06 * (value - 100))), 0, 255))
            for value in range(256)
        ],
        dtype=np.uint8,
    )
    filtered = cv2.LUT(crushed, contrast_lut)
    filtered = _chromatic_aberration(filtered)
    filtered = _add_scanlines(filtered)
    filtered = _add_grain(filtered)
    filtered = _add_vignette(filtered)
    return _add_glitch_bands(filtered)


apply_dark_harmony = apply_punk_filter
