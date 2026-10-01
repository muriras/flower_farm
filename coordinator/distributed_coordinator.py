"""Distributed coordinator — manages multiple drones using Ray.

Architecture:
    ┌─────────────────────────────────────────┐
    │           Central Coordinator            │
    │  (Farm Manager — aggregates results)     │
    └──────────┬──────────┬──────────┬─────────┘
               │          │          │
         ┌─────┴──┐ ┌─────┴──┐ ┌────┴───┐
         │ Drone 0│ │ Drone 1│ │ Drone 2│   ← Ray actors
         │ Zone A │ │ Zone B │ │ Zone C │     (parallel)
         └────────┘ └────────┘ └────────┘

    Each drone runs as a Ray actor — independent process,
    potentially on a different machine in the cluster.
    The coordinator collects results and generates farm-wide reports.
"""

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

from configs.farm_config import (
    FarmConfig, DroneConfig, FarmZone, FlowerType, HealthStatus, DetectionResult
)
from models.flower_detector import FlowerDetector


# ─────────────────────────────────────────────
# Zone partitioning
# ─────────────────────────────────────────────

def partition_farm(config: FarmConfig) -> list[FarmZone]:
    """Divide the farm into zones, one per drone."""
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

            # Add overlap
            if col > 0:
                x_min -= zone_w * overlap
            if col < config.grid_cols - 1:
                x_max += zone_w * overlap
            if row > 0:
                y_min -= zone_h * overlap
            if row < config.grid_rows - 1:
                y_max += zone_h * overlap

            zones.append(FarmZone(
                zone_id=zone_id,
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                drone_id=zone_id,
            ))
            zone_id += 1

    return zones


# ─────────────────────────────────────────────
# Ray remote drone worker
# ─────────────────────────────────────────────

def _create_ray_drone_actor():
    """Create a Ray remote actor for a drone. Lazy import to avoid hard dep."""
    import ray

    @ray.remote(num_cpus=1)
    class RayDroneWorker:
        """Runs a drone mission as a Ray actor."""

        def __init__(self, config_dict: dict, zone_dict: dict, model_path: str, output_dir: str):
            self.config = DroneConfig(**config_dict)
            self.zone = FarmZone(**zone_dict)
            self.detector = FlowerDetector(model_path=model_path)
            self.output_dir = output_dir

        def run(self) -> dict:
            from drone_agent.drone_agent import DroneAgent

            agent = DroneAgent(
                config=self.config,
                zone=self.zone,
                detector=self.detector,
                output_dir=self.output_dir,
                simulation=True,
            )
            results = agent.run_mission()
            summary = agent.get_summary()

            # Flatten detections for transport
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

            return {
                "summary": summary,
                "detections": all_dets,
                "drone_id": self.config.drone_id,
            }

    return RayDroneWorker


# ─────────────────────────────────────────────
# Main coordinator
# ─────────────────────────────────────────────

class DistributedCoordinator:
    """Manages distributed drone fleet for flower farm scanning.

    Supports two modes:
    - Ray distributed: real parallel execution across cluster nodes
    - Local sequential: for development/testing without Ray
    """

    def __init__(self, config: FarmConfig):
        self.config = config
        self.zones = partition_farm(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ray_initialized = False

    def _init_ray(self):
        """Initialize Ray cluster connection."""
        if self._ray_initialized:
            return
        import ray
        if not ray.is_initialized():
            ray.init(address=self.config.ray_address, ignore_reinit_error=True)
        self._ray_initialized = True

    def run_distributed(self) -> dict:
        """Run the full farm scan with distributed drones via Ray."""
        self._init_ray()
        import ray

        Worker = _create_ray_drone_actor()

        # Spawn drone workers
        futures = []
        for zone in self.zones:
            drone_config = DroneConfig(drone_id=zone.drone_id)
            worker = Worker.remote(
                config_dict={
                    "drone_id": drone_config.drone_id,
                    "speed_mps": drone_config.speed_mps,
                    "altitude_m": drone_config.altitude_m,
                    "camera_fov_deg": drone_config.camera_fov_deg,
                    "capture_interval_s": drone_config.capture_interval_s,
                    "image_size": list(drone_config.image_size),
                    "model_confidence_threshold": drone_config.model_confidence_threshold,
                },
                zone_dict={
                    "zone_id": zone.zone_id,
                    "x_min": zone.x_min, "x_max": zone.x_max,
                    "y_min": zone.y_min, "y_max": zone.y_max,
                    "drone_id": zone.drone_id,
                },
                model_path=self.config.model_path,
                output_dir=str(self.output_dir),
            )
            futures.append(worker.run.remote())

        print(f"\n{'='*60}")
        print(f"  FARM SCAN: {self.config.farm_name}")
        print(f"  Drones: {len(futures)} | Zones: {len(self.zones)}")
        print(f"  Farm size: {self.config.farm_width_m}m × {self.config.farm_length_m}m")
        print(f"{'='*60}\n")

        # Collect results
        results = ray.get(futures)
        return self._aggregate_results(results)

    def run_local(self) -> dict:
        """Run sequentially (no Ray) — for dev/testing."""
        from drone_agent.drone_agent import DroneAgent

        all_results = []

        print(f"\n{'='*60}")
        print(f"  FARM SCAN (local): {self.config.farm_name}")
        print(f"  Drones: {len(self.zones)} (sequential)")
        print(f"{'='*60}\n")

        for zone in self.zones:
            drone_config = DroneConfig(drone_id=zone.drone_id)
            detector = FlowerDetector(model_path=self.config.model_path)

            agent = DroneAgent(
                config=drone_config,
                zone=zone,
                detector=detector,
                output_dir=str(self.output_dir),
                simulation=True,
            )
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

            all_results.append({
                "summary": summary,
                "detections": all_dets,
                "drone_id": zone.drone_id,
            })

        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: list[dict]) -> dict:
        """Combine results from all drones into a farm-wide report."""
        all_detections = []
        drone_summaries = []

        for r in results:
            all_detections.extend(r["detections"])
            drone_summaries.append(r["summary"])

        # Count by flower type
        flower_counts = Counter(d["flower_type"] for d in all_detections)

        # Count by health status
        health_counts = Counter(d["health"] for d in all_detections)

        # Average confidence
        avg_conf = (
            sum(d["confidence"] for d in all_detections) / len(all_detections)
            if all_detections else 0
        )

        # Disease hotspots (zones with high disease ratio)
        zone_health = {}
        for d in all_detections:
            # Find which zone this detection belongs to
            if d["gps"]:
                for z in self.zones:
                    if z.x_min <= d["gps"][0] <= z.x_max and z.y_min <= d["gps"][1] <= z.y_max:
                        if z.zone_id not in zone_health:
                            zone_health[z.zone_id] = {"healthy": 0, "diseased": 0, "wilting": 0, "total": 0}
                        zone_health[z.zone_id]["total"] += 1
                        if d["health"] in zone_health[z.zone_id]:
                            zone_health[z.zone_id][d["health"]] += 1

        # Find disease hotspots
        hotspots = []
        for zone_id, stats in zone_health.items():
            if stats["total"] > 0:
                disease_ratio = stats.get("diseased", 0) / stats["total"]
                if disease_ratio > 0.2:
                    hotspots.append({
                        "zone_id": zone_id,
                        "disease_ratio": disease_ratio,
                        "total_flowers": stats["total"],
                    })

        report = {
            "farm_name": self.config.farm_name,
            "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_flowers_detected": len(all_detections),
            "flower_types": dict(flower_counts),
            "health_breakdown": dict(health_counts),
            "average_confidence": round(avg_conf, 3),
            "disease_hotspots": hotspots,
            "drone_summaries": drone_summaries,
            "zones_scanned": len(self.zones),
        }

        # Save report
        report_path = self.output_dir / "farm_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def print_report(self, report: dict):
        """Pretty-print the farm scan report."""
        print(f"\n{'='*60}")
        print(f"  FARM SCAN REPORT")
        print(f"  {report['farm_name']}")
        print(f"  {report['scan_time']}")
        print(f"{'='*60}")

        print(f"\n  Total flowers detected: {report['total_flowers_detected']}")
        print(f"  Average confidence:     {report['average_confidence']:.1%}")
        print(f"  Zones scanned:          {report['zones_scanned']}")

        print(f"\n  🌸 Flower Types:")
        for flower, count in sorted(report["flower_types"].items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 40)
            print(f"    {flower:12s} {count:4d}  {bar}")

        print(f"\n  🏥 Health Status:")
        for status, count in sorted(report["health_breakdown"].items(), key=lambda x: -x[1]):
            icon = {"healthy": "✅", "wilting": "⚠️ ", "diseased": "🔴", "damaged": "💥"}.get(status, "  ")
            print(f"    {icon} {status:12s} {count:4d}")

        if report["disease_hotspots"]:
            print(f"\n  ⚠️  DISEASE HOTSPOTS:")
            for spot in report["disease_hotspots"]:
                print(f"    Zone {spot['zone_id']}: {spot['disease_ratio']:.0%} diseased "
                      f"({spot['total_flowers']} flowers)")

        print(f"\n  🚁 Drone Status:")
        for ds in report["drone_summaries"]:
            print(f"    Drone {ds['drone_id']}: {ds['status']} | "
                  f"{ds['frames_captured']} frames | "
                  f"{ds['total_detections']} detections | "
                  f"🔋 {ds['battery_remaining']:.0f}%")

        print(f"\n{'='*60}")
        print(f"  Report saved to: {self.config.output_dir}/farm_report.json")
        print(f"{'='*60}\n")