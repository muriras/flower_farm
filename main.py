#!/usr/bin/env python3
"""
Flower Farm Drone CV System — Main Entry Point

Distributed computer vision system for autonomous flower detection,
health assessment, and farm monitoring using multiple drones.

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