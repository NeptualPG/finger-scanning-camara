import cv2


def open_camera(camera_index=0):
    """Open the configured camera device."""
    return cv2.VideoCapture(camera_index)


def prepare_frame(frame):
    """Mirror the camera image for a natural user-facing preview."""
    return cv2.flip(frame, 1)


def apply_dark_harmony(frame):
    """Apply a dark blue-violet cinematic appearance to a camera frame."""
    dark_tint = frame.copy()
    dark_tint[:, :, 0] = cv2.add(dark_tint[:, :, 0], 18)
    dark_tint[:, :, 1] = cv2.multiply(dark_tint[:, :, 1], 0.78)
    dark_tint[:, :, 2] = cv2.multiply(dark_tint[:, :, 2], 0.68)

    filtered = cv2.addWeighted(frame, 0.35, dark_tint, 0.65, -8)
    overlay = filtered.copy()
    height, width = filtered.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width, height), (18, 12, 38), -1)
    return cv2.addWeighted(filtered, 0.82, overlay, 0.18, 0)
