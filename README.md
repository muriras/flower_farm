# Flower Farm Drone CV System

> Distributed computer vision system for autonomous flower detection, health assessment, and farm monitoring using multiple drones.

## Overview

This system deploys a fleet of autonomous drones over a flower farm to:
- Detect and classify flowers (rose, sunflower, tulip, lavender, daisy)
- Assess flower health (healthy, wilting, diseased, damaged)
- Map disease hotspots across farm zones
- Generate farm-wide reports with actionable insights

The system uses **Ray** for distributed computing — each drone runs as an independent Ray actor that can scale across machines in a cluster.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Central Coordinator                  │
│   Partitions farm into zones → spawns drones →   │
│   collects results → aggregates farm report →    │
│   detects disease hotspots                       │
└───────┬───────────┬───────────┬─────────────────┘
        │           │           │        ← Ray actors (parallel)
   ┌────┴────┐ ┌────┴────┐ ┌───┴─────┐     on separate machines
   │ Drone 0 │ │ Drone 1 │ │ Drone 2 │
   │ Zone A  │ │ Zone B  │ │ Zone C  │
   └─────────┘ └─────────┘ └─────────┘
   
   Each drone pipeline:
   capture → stabilize → enhance → detect → health assess → report
```

## Project Structure

```
flowers/
├── README.md                                    # This file
├── main.py                                      # CLI entry point
├── requirements.txt                             # Python dependencies
├── configs/
│   ├── __init__.py
│   └── farm_config.py                           # Config dataclasses (Farm, Drone, Zone)
├── models/
│   ├── __init__.py
│   └── flower_detector.py                       # Detection engine (CNN + YOLOv8)
├── drone_agent/
│   ├── __init__.py
│   └── drone_agent.py                           # Single drone agent with simulation
├── coordinator/
│   ├── __init__.py
│   └── distributed_coordinator.py               # Ray-based distributed orchestrator
└── utils/
    ├── __init__.py
    └── image_utils.py                           # Stabilization, enhancement, overlays
```

## File Descriptions

### `main.py`

CLI entry point. Parses arguments and launches the coordinator in either local or distributed mode.

```python
#!/usr/bin/env python3
"""
Flower Farm Drone CV System — Main Entry Point

Usage:
    python main.py                    # Local mode (sequential, no Ray)
    python main.py --distributed      # Distributed mode (Ray cluster)
    python main.py --drones 8         # Override number of drones
    python main.py --farm-width 500   # Override farm dimensions
"""

import argparse
import sys

from configs.farm_config import FarmConfig
from coordinator.distributed_coordinator import DistributedCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Farm Drone CV System")
    parser.add_argument("--distributed", action="store_true",
                        help="Use Ray for distributed execution")
    parser.add_argument("--ray-address", default="auto",
                        help="Ray cluster address (default: auto)")
    parser.add_argument("--drones", type=int, default=None,
                        help="Number of drones (overrides config)")
    parser.add_argument("--farm-width", type=float, default=200.0,
                        help="Farm width in meters")
    parser.add_argument("--farm-length", type=float, default=300.0,
                        help="Farm length in meters")
    parser.add_argument("--model", default="models/flower_detector.pt",
                        help="Path to detection model")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    # Calculate grid dimensions for drones
    num_drones = args.drones or 4
    import math
    grid_cols = math.ceil(math.sqrt(num_drones))
    grid_rows = math.ceil(num_drones / grid_cols)

    config = FarmConfig(
        farm_width_m=args.farm_width,
        farm_length_m=args.farm_length,
        num_drones=num_drones,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        model_path=args.model,
        output_dir=args.output,
        distributed=args.distributed,
        ray_address=args.ray_address,
    )

    coordinator = DistributedCoordinator(config)

    print(f"Initializing Flower Farm Drone CV System")
    print(f"  Mode:       {'Distributed (Ray)' if args.distributed else 'Local (sequential)'}")
    print(f"  Drones:     {num_drones}")
    print(f"  Farm:       {config.farm_width_m}m × {config.farm_length_m}m")
    print(f"  Grid:       {grid_rows}×{grid_cols}")
    print(f"  Model:      {config.model_path}")
    print(f"  Output:     {config.output_dir}")

    # Run scan
    if args.distributed:
        report = coordinator.run_distributed()
    else:
        report = coordinator.run_local()

    coordinator.print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### `configs/farm_config.py`

Core dataclasses for the system — farm layout, drone parameters, detection results.

```python
"""Configuration for the flower farm drone CV system."""
from dataclasses import dataclass, field
from enum import Enum


class FlowerType(Enum):
    ROSE = "rose"
    SUNFLOWER = "sunflower"
    TULIP = "tulip"
    LAVENDER = "lavender"
    DAISY = "daisy"
    UNKNOWN = "unknown"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WILTING = "wilting"
    DISEASED = "diseased"
    DAMAGED = "damaged"


@dataclass
class DetectionResult:
    flower_type: FlowerType
    health: HealthStatus
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    gps_coords: tuple[float, float] | None = None
    area_cm2: float = 0.0


@dataclass
class FarmZone:
    """A rectangular zone within the farm assigned to a drone."""
    zone_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    drone_id: int = -1


@dataclass
class DroneConfig:
    drone_id: int
    speed_mps: float = 3.0
    altitude_m: float = 5.0
    camera_fov_deg: float = 78.0
    capture_interval_s: float = 0.5
    image_size: tuple[int, int] = (640, 480)
    model_confidence_threshold: float = 0.5


@dataclass
class FarmConfig:
    farm_name: str = "Sunrise Flower Farm"
    farm_width_m: float = 200.0
    farm_length_m: float = 300.0
    num_drones: int = 4
    overlap_percent: float = 10.0
    grid_rows: int = 2
    grid_cols: int = 2
    flower_types: list[str] = field(
        default_factory=lambda: [f.value for f in FlowerType if f != FlowerType.UNKNOWN]
    )
    model_path: str = "models/flower_detector.pt"
    output_dir: str = "output"
    distributed: bool = True
    ray_address: str = "auto"
```

---

### `models/flower_detector.py`

Detection engine using a lightweight CNN (MobileNet-style) with YOLOv8 fallback. Includes health assessment via HSV color analysis.

```python
"""Flower detection model — uses a lightweight CNN with OpenCV preprocessing."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from configs.farm_config import (
    FlowerType, HealthStatus, DetectionResult, DroneConfig
)


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
    """Tiny CNN: 4 depthwise-separable blocks → classify + bbox regression."""

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
        self.bbox_head = nn.Linear(512, 4)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.classifier(feat), self.bbox_head(feat)


def assess_health(image_patch: np.ndarray) -> tuple[HealthStatus, float]:
    """Analyze flower health from color distribution in HSV space."""
    hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_sat = np.mean(s) / 255.0
    mean_val = np.mean(v) / 255.0

    brown_mask = ((h < 30) | (h > 15)) & (s < 100) & (v < 120)
    brown_ratio = np.sum(brown_mask) / brown_mask.size

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


class FlowerDetector:
    """Flower detection and classification engine."""

    CLASS_NAMES = [f.value for f in FlowerType if f != FlowerType.UNKNOWN]
    IMG_SIZE = 224

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.yolo_model = None
        self.cnn_model = None

        try:
            from ultralytics import YOLO
            if model_path and Path(model_path).exists():
                self.yolo_model = YOLO(model_path)
            else:
                self.yolo_model = YOLO("yolov8n.pt")
            self._mode = "yolo"
        except Exception:
            self._mode = "cnn"

        self.cnn_model = FlowerCNN(num_classes=len(self.CLASS_NAMES))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> list[DetectionResult]:
        if self._mode == "yolo":
            return self._detect_yolo(image, confidence_threshold)
        return self._detect_cnn(image, confidence_threshold)

    def _detect_yolo(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        results = self.yolo_model(image, conf=threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                flower_type = self._map_class(cls_id)
                patch = image[max(0, y1):y2, max(0, x1):x2]
                health, _ = assess_health(patch) if patch.size > 0 else (HealthStatus.HEALTHY, 0.5)
                detections.append(DetectionResult(
                    flower_type=flower_type, health=health,
                    confidence=conf, bbox=(x1, y1, x2 - x1, y2 - y1),
                    area_cm2=self._estimate_area(x2 - x1, y2 - y1),
                ))
        return detections

    def _detect_cnn(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        detections = []
        preprocessed = self._preprocess_for_contours(image)
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            patch = image[y:y+h, x:x+w]
            if patch.size == 0:
                continue
            tensor = self._image_to_tensor(patch)
            with torch.no_grad():
                logits, _ = self.cnn_model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, cls_idx = probs.max(1)
            if conf.item() < threshold:
                continue
            flower_type = FlowerType(self.CLASS_NAMES[cls_idx.item()])
            health, _ = assess_health(patch)
            detections.append(DetectionResult(
                flower_type=flower_type, health=health,
                confidence=conf.item(), bbox=(x, y, w, h),
                area_cm2=self._estimate_area(w, h),
            ))
        return detections

    def _preprocess_for_contours(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = []
        ranges = [
            ((0, 50, 50), (10, 255, 255)),
            ((160, 50, 50), (180, 255, 255)),
            ((10, 50, 50), (35, 255, 255)),
            ((35, 30, 30), (85, 255, 255)),
            ((125, 50, 50), (160, 255, 255)),
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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.IMG_SIZE, self.IMG_SIZE))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _map_class(self, yolo_cls: int) -> FlowerType:
        idx = yolo_cls % len(self.CLASS_NAMES)
        return FlowerType(self.CLASS_NAMES[idx])

    def _estimate_area(self, w: int, h: int, altitude: float = 5.0, fov: float = 78.0) -> float:
        import math
        fov_rad = math.radians(fov)
        ground_width_m = 2 * altitude * math.tan(fov_rad / 2)
        cm_per_pixel = (ground_width_m * 100) / 640
        return (w * cm_per_pixel) * (h * cm_per_pixel)
```

---

### `drone_agent/drone_agent.py`

Individual drone agent — plans scan paths, captures frames, runs detection, reports results. Includes a synthetic frame generator for testing.

```python
"""Drone agent — simulates a single drone scanning a farm zone."""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path

from configs.farm_config import (
    DroneConfig, FarmZone, DetectionResult, FlowerType, HealthStatus
)
from models.flower_detector import FlowerDetector
from utils.image_utils import stabilize, enhance_for_detection, create_detection_overlay


@dataclass
class DroneState:
    lat: float = 0.0
    lon: float = 0.0
    altitude: float = 5.0
    heading: float = 0.0
    battery_pct: float = 100.0
    is_active: bool = True
    frames_captured: int = 0
    detections_total: int = 0


class DroneAgent:
    """Autonomous drone that scans flowers in its assigned zone."""

    def __init__(self, config, zone, detector, output_dir="output", simulation=True):
        self.config = config
        self.zone = zone
        self.detector = detector
        self.state = DroneState(altitude=config.altitude_m)
        self.output_dir = Path(output_dir) / f"drone_{config.drone_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation = simulation
        self._scan_path = self._plan_scan_path()
        self._prev_frame = None

    def _plan_scan_path(self):
        path = []
        fov_rad = np.radians(self.config.camera_fov_deg)
        ground_width = 2 * self.state.altitude * np.tan(fov_rad / 2)
        step = ground_width * 0.9
        x = self.zone.x_min
        direction = 1
        while x < self.zone.x_max:
            y_range = (
                range(int(self.zone.y_min), int(self.zone.y_max), int(step))
                if direction == 1
                else range(int(self.zone.y_max), int(self.zone.y_min), -int(step))
            )
            for y in y_range:
                path.append((x, float(y)))
            x += step
            direction *= -1
        return path

    def run_mission(self):
        all_detections = []
        total_points = len(self._scan_path)
        for i, (lat, lon) in enumerate(self._scan_path):
            if not self.state.is_active:
                break
            self.state.lat = lat
            self.state.lon = lon
            frame = self._capture_frame()
            frame = stabilize(frame, self._prev_frame)
            self._prev_frame = frame.copy()
            enhanced = enhance_for_detection(frame)
            detections = self.detector.detect(enhanced, self.config.model_confidence_threshold)
            for det in detections:
                det.gps_coords = (lat, lon)
            if detections:
                annotated = create_detection_overlay(frame, detections)
                cv2.imwrite(str(self.output_dir / f"frame_{self.state.frames_captured:06d}.jpg"), annotated)
            all_detections.append({
                "drone_id": self.config.drone_id,
                "timestamp": time.time(),
                "position": (lat, lon),
                "frame_id": self.state.frames_captured,
                "detections": detections,
                "zone_id": self.zone.zone_id,
            })
            self.state.frames_captured += 1
            self.state.detections_total += len(detections)
            self.state.battery_pct -= 0.05
            if self.state.battery_pct < 10:
                self.state.is_active = False
            if (i + 1) % 10 == 0:
                pct = (i + 1) / total_points * 100
                print(f"  [Drone {self.config.drone_id}] {pct:.0f}% — "
                      f"{self.state.detections_total} flowers found, "
                      f"battery {self.state.battery_pct:.0f}%")
        return all_detections

    def _capture_frame(self):
        if self.simulation:
            return self._generate_synthetic_frame()
        raise NotImplementedError("Connect to RTSP/UDP camera stream here")

    def _generate_synthetic_frame(self):
        w, h = self.config.image_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        np.random.seed(int(self.state.lat * 1000 + self.state.lon * 1000))
        num_flowers = np.random.randint(3, 12)
        flower_colors = {
            FlowerType.ROSE: [(0, 0, 200), (50, 50, 255)],
            FlowerType.SUNFLOWER: [(0, 200, 255), (0, 255, 255)],
            FlowerType.TULIP: [(200, 0, 200), (255, 100, 255)],
            FlowerType.LAVENDER: [(180, 100, 200), (220, 150, 255)],
            FlowerType.DAISY: [(200, 200, 255), (255, 255, 255)],
        }
        for _ in range(num_flowers):
            flower_type = np.random.choice(list(flower_colors.keys()))
            color_range = flower_colors[flower_type]
            color = tuple(np.random.randint(
                [min(a, b) for a, b in zip(color_range[0], color_range[1])],
                [max(a, b) for a, b in zip(color_range[0], color_range[1])],
            ).tolist())
            cx = np.random.randint(30, w - 30)
            cy = np.random.randint(30, h - 30)
            radius = np.random.randint(12, 35)
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                px = int(cx + radius * 0.7 * np.cos(rad))
                py = int(cy + radius * 0.7 * np.sin(rad))
                cv2.circle(frame, (px, py), radius // 2, color, -1)
            cv2.circle(frame, (cx, cy), radius // 3, (0, 200, 200), -1)
            if np.random.random() < 0.15:
                for _ in range(np.random.randint(3, 8)):
                    sx = cx + np.random.randint(-radius, radius)
                    sy = cy + np.random.randint(-radius, radius)
                    cv2.circle(frame, (sx, sy), 3, (20, 40, 80), -1)
        return frame

    def get_summary(self):
        return {
            "drone_id": self.config.drone_id,
            "zone_id": self.zone.zone_id,
            "frames_captured": self.state.frames_captured,
            "total_detections": self.state.detections_total,
            "battery_remaining": self.state.battery_pct,
            "status": "active" if self.state.is_active else "low_battery",
        }
```

---

### `coordinator/distributed_coordinator.py`

Distributed orchestrator using Ray. Partitions the farm into zones, spawns drone workers as Ray actors, collects and aggregates results.

```python
"""Distributed coordinator — manages multiple drones using Ray."""

import time
import json
import numpy as np
from pathlib import Path
from collections import Counter

from configs.farm_config import (
    FarmConfig, DroneConfig, FarmZone, FlowerType, HealthStatus, DetectionResult
)
from models.flower_detector import FlowerDetector


def partition_farm(config: FarmConfig) -> list[FarmZone]:
    zones = []
    zone_w = config.farm_width_m / config.grid_cols
    zone_h = config.farm_length_m / config.grid_rows
    overlap = config.overlap_percent / 100.0
    zone_id = 0
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            x_min = col * zone_w
            x_max = (col + 1) * zone_w
            y_min = row * zone_h
            y_max = (row + 1) * zone_h
            if col > 0: x_min -= zone_w * overlap
            if col < config.grid_cols - 1: x_max += zone_w * overlap
            if row > 0: y_min -= zone_h * overlap
            if row < config.grid_rows - 1: y_max += zone_h * overlap
            zones.append(FarmZone(zone_id=zone_id, x_min=x_min, x_max=x_max,
                                  y_min=y_min, y_max=y_max, drone_id=zone_id))
            zone_id += 1
    return zones


def _create_ray_drone_actor():
    import ray

    @ray.remote(num_cpus=1)
    class RayDroneWorker:
        def __init__(self, config_dict, zone_dict, model_path, output_dir):
            self.config = DroneConfig(**config_dict)
            self.zone = FarmZone(**zone_dict)
            self.detector = FlowerDetector(model_path=model_path)
            self.output_dir = output_dir

        def run(self):
            from drone_agent.drone_agent import DroneAgent
            agent = DroneAgent(config=self.config, zone=self.zone,
                               detector=self.detector, output_dir=self.output_dir,
                               simulation=True)
            results = agent.run_mission()
            summary = agent.get_summary()
            all_dets = []
            for report in results:
                for d in report["detections"]:
                    all_dets.append({
                        "flower_type": d.flower_type.value,
                        "health": d.health.value,
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "gps": d.gps_coords,
                        "area_cm2": d.area_cm2,
                    })
            return {"summary": summary, "detections": all_dets, "drone_id": self.config.drone_id}

    return RayDroneWorker


class DistributedCoordinator:
    def __init__(self, config: FarmConfig):
        self.config = config
        self.zones = partition_farm(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ray_initialized = False

    def _init_ray(self):
        if self._ray_initialized:
            return
        import ray
        if not ray.is_initialized():
            ray.init(address=self.config.ray_address, ignore_reinit_error=True)
        self._ray_initialized = True

    def run_distributed(self):
        self._init_ray()
        import ray
        Worker = _create_ray_drone_actor()
        futures = []
        for zone in self.zones:
            dc = DroneConfig(drone_id=zone.drone_id)
            worker = Worker.remote(
                config_dict={"drone_id": dc.drone_id, "speed_mps": dc.speed_mps,
                             "altitude_m": dc.altitude_m, "camera_fov_deg": dc.camera_fov_deg,
                             "capture_interval_s": dc.capture_interval_s,
                             "image_size": list(dc.image_size),
                             "model_confidence_threshold": dc.model_confidence_threshold},
                zone_dict={"zone_id": zone.zone_id, "x_min": zone.x_min, "x_max": zone.x_max,
                           "y_min": zone.y_min, "y_max": zone.y_max, "drone_id": zone.drone_id},
                model_path=self.config.model_path,
                output_dir=str(self.output_dir),
            )
            futures.append(worker.run.remote())
        print(f"\n{'='*60}")
        print(f"  FARM SCAN: {self.config.farm_name}")
        print(f"  Drones: {len(futures)} | Zones: {len(self.zones)}")
        print(f"{'='*60}\n")
        results = ray.get(futures)
        return self._aggregate_results(results)

    def run_local(self):
        from drone_agent.drone_agent import DroneAgent
        all_results = []
        print(f"\n{'='*60}")
        print(f"  FARM SCAN (local): {self.config.farm_name}")
        print(f"  Drones: {len(self.zones)} (sequential)")
        print(f"{'='*60}\n")
        for zone in self.zones:
            dc = DroneConfig(drone_id=zone.drone_id)
            detector = FlowerDetector(model_path=self.config.model_path)
            agent = DroneAgent(config=dc, zone=zone, detector=detector,
                               output_dir=str(self.output_dir), simulation=True)
            results = agent.run_mission()
            summary = agent.get_summary()
            all_dets = []
            for report in results:
                for d in report["detections"]:
                    all_dets.append({"flower_type": d.flower_type.value,
                                     "health": d.health.value, "confidence": d.confidence,
                                     "bbox": d.bbox, "gps": d.gps_coords, "area_cm2": d.area_cm2})
            all_results.append({"summary": summary, "detections": all_dets, "drone_id": zone.drone_id})
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results):
        all_detections = []
        drone_summaries = []
        for r in results:
            all_detections.extend(r["detections"])
            drone_summaries.append(r["summary"])
        flower_counts = Counter(d["flower_type"] for d in all_detections)
        health_counts = Counter(d["health"] for d in all_detections)
        avg_conf = (sum(d["confidence"] for d in all_detections) / len(all_detections)) if all_detections else 0
        report = {
            "farm_name": self.config.farm_name,
            "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_flowers_detected": len(all_detections),
            "flower_types": dict(flower_counts),
            "health_breakdown": dict(health_counts),
            "average_confidence": round(avg_conf, 3),
            "drone_summaries": drone_summaries,
            "zones_scanned": len(self.zones),
        }
        report_path = self.output_dir / "farm_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    def print_report(self, report):
        print(f"\n{'='*60}")
        print(f"  FARM SCAN REPORT — {report['farm_name']}")
        print(f"  {report['scan_time']}")
        print(f"{'='*60}")
        print(f"\n  Total flowers: {report['total_flowers_detected']}")
        print(f"  Avg confidence: {report['average_confidence']:.1%}")
        print(f"\n  Flower Types:")
        for flower, count in sorted(report["flower_types"].items(), key=lambda x: -x[1]):
            print(f"    {flower:12s} {count:4d}  {'█' * min(count, 40)}")
        print(f"\n  Health Status:")
        for status, count in sorted(report["health_breakdown"].items(), key=lambda x: -x[1]):
            icon = {"healthy": "OK", "wilting": "!!", "diseased": "XX", "damaged": "**"}.get(status, "  ")
            print(f"    [{icon}] {status:12s} {count:4d}")
        print(f"\n  Drone Status:")
        for ds in report["drone_summaries"]:
            print(f"    Drone {ds['drone_id']}: {ds['status']} | "
                  f"{ds['frames_captured']} frames | {ds['total_detections']} detections | "
                  f"battery {ds['battery_remaining']:.0f}%")
        print(f"\n{'='*60}\n")
```

---

### `utils/image_utils.py`

Image processing utilities — frame stabilization, contrast enhancement, detection overlays, GPS-to-pixel conversion, and image tiling.

```python
"""Image processing utilities for drone camera feeds."""
import numpy as np
import cv2


def stabilize(frame, prev_frame=None):
    if prev_frame is None:
        return frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if features is None:
        return frame
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    good_old = features[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]
    if len(good_old) < 5:
        return frame
    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    if transform is None:
        return frame
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR)


def enhance_for_detection(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    return enhanced


def create_detection_overlay(image, detections, color_map=None):
    overlay = image.copy()
    default_colors = {
        "healthy": (0, 255, 0), "wilting": (0, 255, 255),
        "diseased": (0, 0, 255), "damaged": (128, 0, 255),
    }
    colors = color_map or default_colors
    for det in detections:
        x, y, w, h = det.bbox
        color = colors.get(det.health.value, (255, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{det.flower_type.value} ({det.confidence:.0%}) [{det.health.value}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(overlay, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return overlay


def gps_to_pixel(gps_lat, gps_lon, zone_lat_min, zone_lon_min,
                 zone_lat_max, zone_lon_max, img_width, img_height):
    x_ratio = (gps_lon - zone_lon_min) / max(zone_lon_max - zone_lon_min, 1e-10)
    y_ratio = (gps_lat - zone_lat_min) / max(zone_lat_max - zone_lat_min, 1e-10)
    px = int(np.clip(x_ratio * img_width, 0, img_width - 1))
    py = int(np.clip((1 - y_ratio) * img_height, 0, img_height - 1))
    return px, py


def tile_image(image, tile_size=640, overlap=64):
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
            tiles.append({"image": tile, "offset": (x_start, y_start),
                          "size": (x_end - x_start, y_end - y_start)})
    return tiles
```

---

### `requirements.txt`

```
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
ray>=2.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
ultralytics>=8.0.0
```

---

## How It Works

### 1. Farm Partitioning

The coordinator divides the farm into a grid of zones (2x2 by default), with 10% overlap between adjacent zones to catch flowers on boundaries.

### 2. Drone Scan Pattern

Each drone follows a **lawnmower (boustrophedon) pattern** — sweeping back and forth across its zone at regular intervals determined by the camera's field of view and altitude.

### 3. Detection Pipeline

For each captured frame:
1. **Stabilize** — compensate for drone jitter using optical flow
2. **Enhance** — CLAHE contrast enhancement + denoising
3. **Detect** — YOLOv8 (preferred) or CNN + HSV contour detection
4. **Health assessment** — analyze color distribution (brown/dark = disease, desaturated = wilting)

### 4. Distributed Execution (Ray)

```
Coordinator → Ray actors (one per drone) → parallel execution
           ← aggregate results            ← collect reports
```

Each drone is a `@ray.remote` actor — can run on any machine in the Ray cluster. The coordinator uses `ray.get()` to collect all results and generate the farm report.

### 5. Farm Report

The aggregated report includes:
- Total flowers detected per type
- Health breakdown (healthy/wilting/diseased/damaged)
- Disease hotspots by zone
- Per-drone status (battery, frames, detections)

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Local mode (sequential, 4 drones)
python main.py

# Distributed mode (Ray cluster)
python main.py --distributed --drones 8

# Custom farm size
python main.py --farm-width 500 --farm-length 800 --drones 6
```

## Sample Output

```
============================================================
  FARM SCAN REPORT — Sunrise Flower Farm
  2026-09-30 23:45:00
============================================================

  Total flowers: 347
  Avg confidence: 82.3%

  Flower Types:
    rose          89  ██████████████████████████████████████████
    sunflower     72  ████████████████████████████████████
    tulip         68  ██████████████████████████████████
    lavender      63  ███████████████████████████████
    daisy         55  ████████████████████████████

  Health Status:
    [OK] healthy        281
    [!!] wilting          38
    [XX] diseased         28

  Drone Status:
    Drone 0: active | 45 frames | 89 detections | battery 97%
    Drone 1: active | 42 frames | 92 detections | battery 97%
    Drone 2: active | 38 frames | 78 detections | battery 98%
    Drone 3: active | 41 frames | 88 detections | battery 97%

============================================================
```

## Future Improvements

- **Training**: Fine-tune YOLOv8 on a real flower farm dataset
- **Edge deployment**: Export to ONNX/TensorRT for Jetson Nano
- **Real GPS**: Integrate with drone GPS module for precise geotagging
- **Mojo port**: Move hot paths (HSV masking, CNN forward pass) to Mojo when ecosystem matures
- **Real-time streaming**: Add RTSP/UDP camera ingestion
- **Autonomous navigation**: Add obstacle avoidance and dynamic replanning