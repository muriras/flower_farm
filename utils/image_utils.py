"""Image processing utilities for drone camera feeds."""
import numpy as np
import cv2


def stabilize(frame: np.ndarray, prev_frame: np.ndarray | None = None) -> np.ndarray:
    """Simple frame stabilization using optical flow — compensates for drone jitter."""
    if prev_frame is None:
        return frame

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features in previous frame
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if features is None:
        return frame

    # Track features to current frame
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)

    # Filter good tracks
    good_old = features[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]

    if len(good_old) < 5:
        return frame

    # Estimate affine transform
    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)

    if transform is None:
        return frame

    # Apply stabilization
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR)


def enhance_for_detection(image: np.ndarray) -> np.ndarray:
    """Enhance image for better flower detection in varying light conditions."""
    # CLAHE on L channel for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced


def gps_to_pixel(
    gps_lat: float, gps_lon: float,
    zone_lat_min: float, zone_lon_min: float,
    zone_lat_max: float, zone_lon_max: float,
    img_width: int, img_height: int,
) -> tuple[int, int]:
    """Convert GPS coordinates to pixel position within a frame."""
    x_ratio = (gps_lon - zone_lon_min) / max(zone_lon_max - zone_lon_min, 1e-10)
    y_ratio = (gps_lat - zone_lat_min) / max(zone_lat_max - zone_lat_min, 1e-10)

    px = int(np.clip(x_ratio * img_width, 0, img_width - 1))
    py = int(np.clip((1 - y_ratio) * img_height, 0, img_height - 1))  # flip Y

    return px, py


def create_detection_overlay(
    image: np.ndarray,
    detections: list,
    color_map: dict | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    overlay = image.copy()

    default_colors = {
        "healthy": (0, 255, 0),
        "wilting": (0, 255, 255),
        "diseased": (0, 0, 255),
        "damaged": (128, 0, 255),
    }
    colors = color_map or default_colors

    for det in detections:
        x, y, w, h = det.bbox
        color = colors.get(det.health.value, (255, 255, 255))

        # Bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Label
        label = f"{det.flower_type.value} ({det.confidence:.0%}) [{det.health.value}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(overlay, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return overlay


def tile_image(image: np.ndarray, tile_size: int = 640, overlap: int = 64) -> list[dict]:
    """Split a large image into overlapping tiles for processing."""
    h, w = image.shape[:2]
    tiles = []
    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            tile = image[y_start:y_end, x_start:x_end]
            tiles.append({
                "image": tile,
                "offset": (x_start, y_start),
                "size": (x_end - x_start, y_end - y_start),
            })

    return tiles