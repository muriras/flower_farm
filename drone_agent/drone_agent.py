"""Drone agent — simulates a single drone scanning a farm zone.

Each drone runs its own detection pipeline, navigates a grid pattern
within its assigned zone, and reports findings back to the coordinator.
"""

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
    """Mutable state for a drone during a mission."""
    lat: float = 0.0
    lon: float = 0.0
    altitude: float = 5.0
    heading: float = 0.0       # degrees, 0 = north
    battery_pct: float = 100.0
    is_active: bool = True
    frames_captured: int = 0
    detections_total: int = 0


class DroneAgent:
    """Autonomous drone that scans flowers in its assigned zone.

    In simulation mode, generates synthetic camera frames.
    In production, ingests from a real camera stream (RTSP/UDP).
    """

    def __init__(
        self,
        config: DroneConfig,
        zone: FarmZone,
        detector: FlowerDetector,
        output_dir: str = "output",
        simulation: bool = True,
    ):
        self.config = config
        self.zone = zone
        self.detector = detector
        self.state = DroneState(altitude=config.altitude_m)
        self.output_dir = Path(output_dir) / f"drone_{config.drone_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation = simulation

        self._scan_path = self._plan_scan_path()
        self._path_idx = 0
        self._prev_frame = None

    def _plan_scan_path(self) -> list[tuple[float, float]]:
        """Generate a lawnmower (boustrophedon) scan pattern for the zone."""
        path = []
        fov_rad = np.radians(self.config.camera_fov_deg)
        ground_width = 2 * self.state.altitude * np.tan(fov_rad / 2)
        step = ground_width * (1 - 0.1)  # 10% overlap

        x = self.zone.x_min
        direction = 1  # 1 = east, -1 = west

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

    def run_mission(self) -> list[dict]:
        """Execute the full scan mission. Returns all detection reports."""
        all_detections = []
        total_points = len(self._scan_path)

        for i, (lat, lon) in enumerate(self._scan_path):
            if not self.state.is_active:
                break

            # Move drone
            self.state.lat = lat
            self.state.lon = lon

            # Capture frame
            frame = self._capture_frame()

            # Stabilize
            frame = stabilize(frame, self._prev_frame)
            self._prev_frame = frame.copy()

            # Enhance
            enhanced = enhance_for_detection(frame)

            # Detect flowers
            detections = self.detector.detect(enhanced, self.config.model_confidence_threshold)

            # Assign GPS to detections
            for det in detections:
                det.gps_coords = (lat, lon)

            # Save annotated frame
            if detections:
                annotated = create_detection_overlay(frame, detections)
                cv2.imwrite(
                    str(self.output_dir / f"frame_{self.state.frames_captured:06d}.jpg"),
                    annotated,
                )

            # Build report
            report = {
                "drone_id": self.config.drone_id,
                "timestamp": time.time(),
                "position": (lat, lon),
                "frame_id": self.state.frames_captured,
                "detections": detections,
                "zone_id": self.zone.zone_id,
            }
            all_detections.append(report)

            self.state.frames_captured += 1
            self.state.detections_total += len(detections)

            # Battery drain simulation
            self.state.battery_pct -= 0.05
            if self.state.battery_pct < 10:
                self.state.is_active = False

            # Progress
            if (i + 1) % 10 == 0:
                pct = (i + 1) / total_points * 100
                print(f"  [Drone {self.config.drone_id}] {pct:.0f}% — "
                      f"{self.state.detections_total} flowers found, "
                      f"battery {self.state.battery_pct:.0f}%")

        return all_detections

    def _capture_frame(self) -> np.ndarray:
        """Capture a camera frame. In simulation, generate synthetic flowers."""
        if self.simulation:
            return self._generate_synthetic_frame()
        # In production: read from camera stream
        raise NotImplementedError("Connect to RTSP/UDP camera stream here")

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic farm scene with flowers for testing."""
        w, h = self.config.image_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Green grass background
        frame[:, :] = (34, 139, 34)  # forest green

        # Add random grass texture
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Place synthetic flowers
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

            # Draw petals
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                px = int(cx + radius * 0.7 * np.cos(rad))
                py = int(cy + radius * 0.7 * np.sin(rad))
                cv2.circle(frame, (px, py), radius // 2, color, -1)

            # Center
            cv2.circle(frame, (cx, cy), radius // 3, (0, 200, 200), -1)

            # Simulate some unhealthy flowers
            if np.random.random() < 0.15:
                # Add brown spots
                for _ in range(np.random.randint(3, 8)):
                    sx = cx + np.random.randint(-radius, radius)
                    sy = cy + np.random.randint(-radius, radius)
                    cv2.circle(frame, (sx, sy), 3, (20, 40, 80), -1)

        return frame

    def get_summary(self) -> dict:
        """Return mission summary for this drone."""
        return {
            "drone_id": self.config.drone_id,
            "zone_id": self.zone.zone_id,
            "frames_captured": self.state.frames_captured,
            "total_detections": self.state.detections_total,
            "battery_remaining": self.state.battery_pct,
            "status": "active" if self.state.is_active else "low_battery",
        }