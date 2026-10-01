# Flower Farm Drone CV System — Diagrams

## System Context (C4 Level 1)

Who uses the system and what does it interact with?

```mermaid
C4Context
    title System Context Diagram — Flower Farm Drone CV System

    Person(farmer, "Farm Manager", "Monitors flower health, reviews reports, takes action on disease hotspots")
    Person(pilot, "Drone Operator", "Deploys and manages drone fleet, handles takeoff/landing")

    System(cv_system, "Flower Farm CV System", "Distributed computer vision system that detects flowers, assesses health, and generates farm-wide reports")

    System_Ext(drones, "Drone Fleet", "Physical drones with cameras flying lawnmower patterns over the farm")
    System_Ext(gps, "GPS Module", "Provides real-time lat/lon coordinates for each drone")
    System_Ext(cloud, "Ray Cluster", "Distributed compute cluster for parallel drone processing")
    System_Ext(storage, "Object Storage", "Stores annotated frames, reports, and model weights (S3/GCS/local)")

    Rel(farmer, cv_system, "Views reports, configures farm parameters")
    Rel(pilot, cv_system, "Launches scan missions, monitors drone status")
    Rel(cv_system, drones, "Sends flight commands, receives camera feed")
    Rel(cv_system, gps, "Reads GPS coordinates per frame")
    Rel(cv_system, cloud, "Spawns Ray actors for parallel processing")
    Rel(cv_system, storage, "Writes detection frames and JSON reports")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Container Diagram (C4 Level 2)

What are the major running parts?

```mermaid
C4Container
    title Container Diagram — Flower Farm CV System

    Person(farmer, "Farm Manager", "Reviews reports and disease alerts")
    Person(pilot, "Drone Operator", "Manages drone fleet")

    System_Boundary(system, "Flower Farm CV System") {
        Container(cli, "CLI (main.py)", "Python, argparse", "Entry point — parses args, launches coordinator")
        Container(coordinator, "Distributed Coordinator", "Python, Ray", "Partitions farm, spawns drone workers, aggregates results, generates reports")
        Container(drone_agent, "Drone Agent × N", "Python, OpenCV", "Runs scan path, captures frames, runs detection pipeline, reports findings")
        Container(detector, "Flower Detector", "Python, PyTorch, YOLOv8", "CNN + YOLOv8 detection engine, health assessment via HSV analysis")
        Container(image_utils, "Image Utils", "Python, OpenCV", "Frame stabilization, CLAHE enhancement, overlay rendering")
        Container(config, "Farm Config", "Python dataclasses", "Typed configuration for farm layout, drone params, detection thresholds")
        ContainerDb(output, "Output Directory", "Filesystem", "Annotated frames (JPEG), farm report (JSON)")
    }

    System_Ext(drones, "Drone Fleet", "Physical drones with cameras")
    System_Ext(ray_cluster, "Ray Cluster", "Distributed compute nodes")

    Rel(pilot, cli, "Invokes with CLI flags")
    Rel(farmer, output, "Reads JSON reports, views annotated frames")
    Rel(cli, coordinator, "Passes FarmConfig, triggers scan")
    Rel(coordinator, ray_cluster, "Spawns Ray actors")
    Rel(coordinator, drone_agent, "Creates one per zone")
    Rel(drone_agent, detector, "Sends frames for detection")
    Rel(drone_agent, image_utils, "Stabilizes and enhances frames")
    Rel(drone_agent, config, "Reads drone and zone config")
    Rel(drone_agent, output, "Writes annotated frames")
    Rel(coordinator, output, "Writes farm_report.json")
    Rel(drones, drone_agent, "Camera feed (RTSP/UDP)")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Component Diagram (C4 Level 3)

How does the Drone Agent's detection pipeline work internally?

```mermaid
C4Component
    title Component Diagram — Drone Agent Detection Pipeline

    Container_Boundary(agent, "Drone Agent") {
        Component(path_planner, "Scan Path Planner", "numpy", "Generates boustrophedon (lawnmower) waypoints from zone bounds and camera FOV")
        Component(frame_capture, "Frame Capture", "cv2 / simulation", "Captures frame from RTSP stream or generates synthetic flowers")
        Component(stabilizer, "Frame Stabilizer", "cv2, Lucas-Kanade", "Optical flow tracking → affine warp to compensate drone jitter")
        Component(enhancer, "CLAHE Enhancer", "cv2, LAB color space", "Adaptive histogram equalization on L channel, denoising")
        Component(yolo_detector, "YOLOv8 Detector", "ultralytics, PyTorch", "Object detection backbone — CSPDarknet + PANet + decoupled head")
        Component(cnn_detector, "CNN Fallback Detector", "PyTorch, depthwise-separable", "HSV contour isolation + MobileNet-style classification")
        Component(health_assessor, "Health Assessor", "numpy, HSV analysis", "Color distribution analysis → healthy/wilting/diseased/damaged")
        Component(annotator, "Overlay Annotator", "cv2", "Draws bounding boxes, labels, and health colors on frames")
        Component(reporter, "Mission Reporter", "dataclass", "Collects all detections, builds per-drone summary")
    }

    Rel(path_planner, frame_capture, "Provides (lat, lon) waypoints")
    Rel(frame_capture, stabilizer, "Raw frame")
    Rel(stabilizer, enhancer, "Stabilized frame")
    Rel(enhancer, yolo_detector, "Enhanced frame")
    Enhancer-->cnn_detector: "Fallback if YOLO unavailable"
    Rel(yolo_detector, health_assessor, "Detected flower patches")
    Rel(cnn_detector, health_assessor, "Detected flower patches")
    Rel(yolo_detector, annotator, "Bounding boxes + labels")
    Rel(health_assessor, annotator, "Health status per detection")
    Rel(annotator, reporter, "Annotated detections")
```

## Dynamic Diagram (C4 Level 4)

What's the runtime sequence of a farm scan?

```mermaid
sequenceDiagram
    autonumber
    participant Pilot as Drone Operator
    participant CLI as main.py
    participant Coord as Distributed Coordinator
    participant Ray as Ray Cluster
    participant D0 as Drone Agent 0
    participant D1 as Drone Agent 1
    participant Det as Flower Detector
    participant Out as Output Directory

    Pilot->>CLI: python main.py --distributed --drones 4
    CLI->>Coord: new DistributedCoordinator(config)
    Coord->>Coord: partition_farm() → 4 zones with overlap

    par Parallel Drone Spawning
        Coord->>Ray: ray.remote(RayDroneWorker).remote(zone_0)
        Coord->>Ray: ray.remote(RayDroneWorker).remote(zone_1)
        Coord->>Ray: ray.remote(RayDroneWorker).remote(zone_2)
        Coord->>Ray: ray.remote(RayDroneWorker).remote(zone_3)
    end

    par Parallel Mission Execution
        Ray->>D0: run()
        Ray->>D1: run()

        loop For each waypoint in scan path
            D0->>D0: capture_frame()
            D0->>D0: stabilize(frame, prev)
            D0->>D0: enhance_for_detection(frame)
            D0->>Det: detect(enhanced, threshold)
            Det-->>D0: detections[]
            D0->>D0: assess_health(patches)
            D0->>Out: write annotated frame (JPEG)
        end

        loop Same for Drone 1
            D1->>D1: capture → stabilize → enhance
            D1->>Det: detect(enhanced, threshold)
            Det-->>D1: detections[]
            D1->>Out: write annotated frame
        end
    end

    D0-->>Coord: {summary, detections[]}
    D1-->>Coord: {summary, detections[]}

    Coord->>Coord: aggregate_results()
    Coord->>Coord: find_disease_hotspots()
    Coord->>Out: write farm_report.json
    Coord-->>Pilot: print_report() to console
```

## Class Diagram

What are the core data structures and their relationships?

```mermaid
classDiagram
    class FarmConfig {
        +str farm_name
        +float farm_width_m
        +float farm_length_m
        +int num_drones
        +float overlap_percent
        +int grid_rows
        +int grid_cols
        +list~str~ flower_types
        +str model_path
        +str output_dir
        +bool distributed
        +str ray_address
    }

    class DroneConfig {
        +int drone_id
        +float speed_mps
        +float altitude_m
        +float camera_fov_deg
        +float capture_interval_s
        +tuple image_size
        +float model_confidence_threshold
    }

    class FarmZone {
        +int zone_id
        +float x_min
        +float x_max
        +float y_min
        +float y_max
        +int drone_id
    }

    class DetectionResult {
        +FlowerType flower_type
        +HealthStatus health
        +float confidence
        +tuple bbox
        +tuple gps_coords
        +float area_cm2
    }

    class FlowerType {
        <<enumeration>>
        ROSE
        SUNFLOWER
        TULIP
        LAVENDER
        DAISY
        UNKNOWN
    }

    class HealthStatus {
        <<enumeration>>
        HEALTHY
        WILTING
        DISEASED
        DAMAGED
    }

    class DroneState {
        +float lat
        +float lon
        +float altitude
        +float heading
        +float battery_pct
        +bool is_active
        +int frames_captured
        +int detections_total
    }

    class DroneAgent {
        +DroneConfig config
        +FarmZone zone
        +FlowerDetector detector
        +DroneState state
        +run_mission() list~dict~
        +get_summary() dict
        -_plan_scan_path() list~tuple~
        -_capture_frame() ndarray
        -_generate_synthetic_frame() ndarray
    }

    class FlowerDetector {
        +str _mode
        +detect(image, threshold) list~DetectionResult~
        -_detect_yolo(image, threshold) list~DetectionResult~
        -_detect_cnn(image, threshold) list~DetectionResult~
        -_preprocess_for_contours(image) ndarray
        -_image_to_tensor(image) Tensor
        -_estimate_area(w, h) float
    }

    class DistributedCoordinator {
        +FarmConfig config
        +list~FarmZone~ zones
        +run_distributed() dict
        +run_local() dict
        +print_report(report)
        -_aggregate_results(results) dict
    }

    class FlowerCNN {
        +DepthwiseSequential features
        +Sequential classifier
        +Linear bbox_head
        +forward(x) tuple
    }

    FarmConfig "1" --> "*" FarmZone : partition_farm()
    FarmConfig "1" --> "*" DroneConfig : creates
    FarmZone "1" --> "1" DroneConfig : assigned to
    DroneAgent "1" --> "1" DroneConfig : has
    DroneAgent "1" --> "1" FarmZone : scans
    DroneAgent "1" --> "1" DroneState : tracks
    DroneAgent "1" --> "1" FlowerDetector : uses
    FlowerDetector --> FlowerCNN : fallback mode
    DetectionResult --> FlowerType : classifies
    DetectionResult --> HealthStatus : assesses
    DistributedCoordinator "1" --> "*" DroneAgent : spawns via Ray
    DistributedCoordinator "1" --> FarmConfig : configured by
```

## Deployment Diagram

How does this deploy across machines?

```mermaid
C4Deployment
    title Deployment Diagram — Multi-Node Ray Cluster

    Deployment_Node(laptop, "Operator Laptop", "macOS / Linux") {
        Container(cli, "CLI", "main.py", "Entry point, config, report display")
    }

    Deployment_Node(head, "Head Node", "Ubuntu 22.04, 8 cores, 16GB RAM") {
        Container(coordinator, "Coordinator", "Ray driver", "Farm partitioning, result aggregation, report generation")
        Container(ray_head, "Ray Head", "ray start --head", "Ray GCS + dashboard")
        Container(drone0, "Drone Actor 0", "Ray actor, 1 CPU", "Zone A scanner")
        Container(drone1, "Drone Actor 1", "Ray actor, 1 CPU", "Zone B scanner")
    }

    Deployment_Node(worker1, "Worker Node 1", "Ubuntu 22.04, 4 cores, 8GB RAM") {
        Container(ray_w1, "Ray Worker", "ray start --address=...", "Registers with head")
        Container(drone2, "Drone Actor 2", "Ray actor, 1 CPU", "Zone C scanner")
    }

    Deployment_Node(worker2, "Worker Node 2", "NVIDIA Jetson Nano") {
        Container(ray_w2, "Ray Worker", "ray start --address=...", "GPU-accelerated inference")
        Container(drone3, "Drone Actor 3", "Ray actor, 1 CPU + GPU", "Zone D scanner with TensorRT")
    }

    Rel(cli, coordinator, "Launches scan via Ray")
    Rel(coordinator, ray_head, "Spawns actors")
    Rel(ray_head, ray_w1, "Schedules drone 2")
    Rel(ray_head, ray_w2, "Schedules drone 3")
    Rel(drone0, drone1, "Parallel execution")
    Rel(drone2, drone3, "Parallel execution")
```

## Data Flow Diagram

How does data move through the system?

```mermaid
flowchart TD
    subgraph Input
        CAMERA["Camera Feed<br/>(RTSP/UDP) or<br/>Synthetic Generator"]
        GPS["GPS Module<br/>(lat, lon, alt)"]
        CONFIG["Farm Config<br/>(width, length, drones, threshold)"]
    end

    subgraph Drone Pipeline
        CAPTURE["Frame Capture<br/>640×480 BGR"]
        STABILIZE["Optical Flow Stabilization<br/>Lucas-Kanade → Affine Warp"]
        ENHANCE["CLAHE Enhancement<br/>LAB color space, denoise"]
        DETECT{"Detection Engine"}
        YOLO["YOLOv8-nano<br/>CSPDarknet + PANet"]
        CNN["CNN Fallback<br/>HSV Contour + MobileNet"]
        HEALTH["Health Assessment<br/>HSV color analysis"]
        ANNOTATE["Overlay Annotator<br/>BBoxes + labels + colors"]
    end

    subgraph Distributed Layer
        COORD["Coordinator<br/>(farm partitioning, zone assignment)"]
        RAY["Ray Actors<br/>(one per drone)"]
    end

    subgraph Output
        FRAMES["Annotated Frames<br/>(JPEG per drone)"]
        REPORT["farm_report.json<br/>(counts, health, hotspots)"]
        CONSOLE["Console Report<br/>(human-readable)"]
    end

    CONFIG --> COORD
    COORD --> RAY
    RAY --> CAPTURE
    CAMERA --> CAPTURE
    GPS --> CAPTURE
    CAPTURE --> STABILIZE
    STABILIZE --> ENHANCE
    ENHANCE --> DETECT
    DETECT -->|YOLOv8 available| YOLO
    DETECT -->|fallback| CNN
    YOLO --> HEALTH
    CNN --> HEALTH
    HEALTH --> ANNOTATE
    ANNOTATE --> FRAMES
    ANNOTATE --> REPORT
    RAY -->|aggregate| COORD
    COORD --> REPORT
    COORD --> CONSOLE

    style DETECT fill:#f9f,stroke:#333,stroke-width:2px
    style COORD fill:#bbf,stroke:#333,stroke-width:2px
    style RAY fill:#bfb,stroke:#333,stroke-width:2px
```

## State Diagram

Drone lifecycle during a mission.

```mermaid
stateDiagram-v2
    [*] --> Idle: Drone created

    Idle --> Planning: Mission assigned
    Planning --> Scanning: Scan path generated

    state Scanning {
        [*] --> Capturing
        Capturing --> Stabilizing: frame captured
        Stabilizing --> Enhancing: frame stabilized
        Enhancing --> Detecting: frame enhanced
        Detecting --> AssessingHealth: flowers detected
        AssessingHealth --> Annotating: health assessed
        Annotating --> Reporting: frame annotated
        Reporting --> Capturing: next waypoint
        Reporting --> [*]: path complete
    }

    Scanning --> Reporting: mission complete
    Scanning --> LowBattery: battery < 10%

    LowBattery --> EmergencyReturn: auto-land
    EmergencyReturn --> [*]

    Reporting --> Aggregating: results sent to coordinator
    Aggregating --> [*]: farm report generated

    note right of Scanning
        Each waypoint:
        capture → stabilize → enhance
        → detect → health → annotate
    end note
```

## Gitgraph — Development Flow

```mermaid
gitgraph
    commit id: "init" tag: "v0.1"
    commit id: "farm_config"
    commit id: "flower_detector"
    branch feature/distributed
    commit id: "ray_coordinator"
    commit id: "drone_agent"
    commit id: "image_utils"
    checkout main
    merge feature/distributed id: "merge_distributed" tag: "v0.5"
    commit id: "cli_main"
    commit id: "simulation_mode"
    commit id: "readme_enriched" tag: "v1.0"
    commit id: "diagrams" tag: "v1.1"
```