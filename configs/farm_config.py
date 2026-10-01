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
    speed_mps: float = 3.0          # meters per second
    altitude_m: float = 5.0         # flight altitude
    camera_fov_deg: float = 78.0    # camera field of view
    capture_interval_s: float = 0.5 # seconds between captures
    image_size: tuple[int, int] = (640, 480)
    model_confidence_threshold: float = 0.5


@dataclass
class FarmConfig:
    farm_name: str = "Sunrise Flower Farm"
    farm_width_m: float = 200.0
    farm_length_m: float = 300.0
    num_drones: int = 4
    overlap_percent: float = 10.0   # zone overlap for edge detection
    grid_rows: int = 2
    grid_cols: int = 2
    flower_types: list[str] = field(
        default_factory=lambda: [f.value for f in FlowerType if f != FlowerType.UNKNOWN]
    )
    model_path: str = "models/flower_detector.pt"
    output_dir: str = "output"
    distributed: bool = True
    ray_address: str = "auto"       # "auto" for local cluster, or "ray://host:10001"