"""Flower detection model — uses a lightweight CNN with OpenCV preprocessing.

Architecture: MobileNet-v2 backbone (lightweight, drone-friendly) with a custom
classification + bounding-box head. Designed to run on edge hardware (Jetson Nano,
Raspberry Pi 4 with Coral TPU) as well as GPU servers.

When a pretrained YOLOv8 model is available, it's preferred for detection.
The CNN fallback is for training from scratch on custom farm data.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

from configs.farm_config import (
    FlowerType, HealthStatus, DetectionResult, DroneConfig
)


# ─────────────────────────────────────────────
# Lightweight CNN for flower classification
# ─────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise separable convolution — 8x fewer params."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu6(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class FlowerCNN(nn.Module):
    """Tiny CNN: 4 depthwise-separable blocks → classify + bbox regression.

    Input:  (B, 3, 224, 224) RGB image
    Output: class_logits (B, num_classes), bbox (B, 4)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        self.bbox_head = nn.Linear(512, 4)  # x, y, w, h normalized

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.classifier(feat), self.bbox_head(feat)


# ─────────────────────────────────────────────
# Health assessment via color analysis
# ─────────────────────────────────────────────

def assess_health(image_patch: np.ndarray) -> tuple[HealthStatus, float]:
    """Analyze flower health from color distribution in HSV space.

    Healthy flowers: vibrant, saturated colors
    Wilting: desaturated, browning yellows
    Diseased: dark spots, unusual green-brown patches
    """
    hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_sat = np.mean(s) / 255.0
    mean_val = np.mean(v) / 255.0

    # Brown/dark pixels indicate disease or wilting
    brown_mask = ((h < 30) | (h > 15)) & (s < 100) & (v < 120)
    brown_ratio = np.sum(brown_mask) / brown_mask.size

    # Dark spots
    dark_mask = v < 50
    dark_ratio = np.sum(dark_mask) / dark_mask.size

    if brown_ratio > 0.3 or dark_ratio > 0.25:
        return HealthStatus.DISEASED, 0.7 + brown_ratio * 0.3
    elif mean_sat < 0.3 or (brown_ratio > 0.1 and mean_val < 0.5):
        return HealthStatus.WILTING, 0.6 + (1 - mean_sat) * 0.3
    elif mean_sat > 0.5 and mean_val > 0.4 and brown_ratio < 0.05:
        return HealthStatus.HEALTHY, 0.8 + mean_sat * 0.2
    else:
        return HealthStatus.HEALTHY, 0.5


# ─────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────

class FlowerDetector:
    """Flower detection and classification engine.

    Tries YOLOv8 first (better accuracy), falls back to CNN + contour detection.
    """

    CLASS_NAMES = [f.value for f in FlowerType if f != FlowerType.UNKNOWN]
    IMG_SIZE = 224

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.yolo_model = None
        self.cnn_model = None

        # Try loading YOLOv8
        try:
            from ultralytics import YOLO
            if model_path and Path(model_path).exists():
                self.yolo_model = YOLO(model_path)
            else:
                self.yolo_model = YOLO("yolov8n.pt")  # nano — fast on edge
            self._mode = "yolo"
        except Exception:
            self._mode = "cnn"

        # Always prepare CNN fallback
        self.cnn_model = FlowerCNN(num_classes=len(self.CLASS_NAMES))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> list[DetectionResult]:
        """Run detection on a single frame."""
        if self._mode == "yolo":
            return self._detect_yolo(image, confidence_threshold)
        return self._detect_cnn(image, confidence_threshold)

    def _detect_yolo(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        """YOLOv8 detection pipeline."""
        results = self.yolo_model(image, conf=threshold, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Map YOLO class to our FlowerType
                flower_type = self._map_class(cls_id)

                # Crop for health assessment
                patch = image[max(0, y1):y2, max(0, x1):x2]
                health, health_conf = assess_health(patch) if patch.size > 0 else (HealthStatus.HEALTHY, 0.5)

                detections.append(DetectionResult(
                    flower_type=flower_type,
                    health=health,
                    confidence=conf,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    area_cm2=self._estimate_area(x2 - x1, y2 - y1),
                ))

        return detections

    def _detect_cnn(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        """Fallback: contour-based detection + CNN classification."""
        detections = []
        preprocessed = self._preprocess_for_contours(image)
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # too small — noise
                continue

            x, y, w, h = cv2.boundingRect(contour)
            patch = image[y:y+h, x:x+w]

            if patch.size == 0:
                continue

            # Classify with CNN
            tensor = self._image_to_tensor(patch)
            with torch.no_grad():
                logits, _ = self.cnn_model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, cls_idx = probs.max(1)

            if conf.item() < threshold:
                continue

            flower_type = FlowerType(self.CLASS_NAMES[cls_idx.item()])
            health, health_conf = assess_health(patch)

            detections.append(DetectionResult(
                flower_type=flower_type,
                health=health,
                confidence=conf.item(),
                bbox=(x, y, w, h),
                area_cm2=self._estimate_area(w, h),
            ))

        return detections

    def _preprocess_for_contours(self, image: np.ndarray) -> np.ndarray:
        """Isolate flower-colored regions using HSV masking."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Broad flower color range: pinks, reds, yellows, purples, whites
        masks = []
        ranges = [
            ((0, 50, 50), (10, 255, 255)),     # reds (low)
            ((160, 50, 50), (180, 255, 255)),   # reds (high)
            ((10, 50, 50), (35, 255, 255)),     # yellows/oranges
            ((35, 30, 30), (85, 255, 255)),     # greens (leaves)
            ((125, 50, 50), (160, 255, 255)),   # purples/pinks
        ]
        for low, high in ranges:
            masks.append(cv2.inRange(hsv, np.array(low), np.array(high)))

        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        return combined

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.IMG_SIZE, self.IMG_SIZE))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _map_class(self, yolo_cls: int) -> FlowerType:
        """Map YOLO class ID to FlowerType (placeholder — needs training data)."""
        # In production, train YOLO on flower dataset and map properly
        idx = yolo_cls % len(self.CLASS_NAMES)
        return FlowerType(self.CLASS_NAMES[idx])

    def _estimate_area(self, w: int, h: int, altitude: float = 5.0, fov: float = 78.0) -> float:
        """Rough estimate of real-world area in cm² from pixel dimensions."""
        # Ground sample distance at given altitude
        import math
        fov_rad = math.radians(fov)
        ground_width_m = 2 * altitude * math.tan(fov_rad / 2)
        cm_per_pixel = (ground_width_m * 100) / 640  # assuming 640px wide
        return (w * cm_per_pixel) * (h * cm_per_pixel)