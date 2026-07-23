# IEEE SMCS Autonomous SAR Competition - Comprehensive Architecture Guide

This document serves as an exhaustive architectural guide for the Autonomous Search and Rescue (SAR) controller used in the IEEE SMCS competition. It details the project structure, the purpose of each file, and exactly how the underlying algorithms, sensor pipelines, and Multi-Robot Systems (MRS) interact.

---

## 📂 1. Directory Structure & Global File Responsibilities

The simulation environment uses a centralized scoring supervisor and decentralized robot controllers. Our core logic is isolated entirely within the `proposed_solution` directory.

### `controllers/proposed_solution/`
This directory contains the autonomous code uploaded to the competition server.

* **`proposed_solution.py`**
  * **Role:** The Master Robot Executable. 
  * **Responsibility:** This is the single monolithic script that runs independently on every robot in the swarm. It handles the entire robotics stack: reading sensors, maintaining local maps, planning paths, steering motors, and broadcasting radio packets to peers.
* **`sim_logs/`** (Artifact Directory)
  * **`map_estimate.png`**: A noisy overhead visual representation of the maze derived from pre-mission "flyover" data. The robots parse this image at boot to initialize their internal occupancy grids, giving them a rough idea of wall placements before exploring.
  * **`victim_location_estimates.csv`**: A shared CSV file. It starts with highly inaccurate victim coordinates (from the noisy flyover). Our robots dynamically overwrite this file with precise Cartesian coordinates once they physically locate a victim using their Lidar, which dramatically improves the "Video Information Extraction" score.
  * **`robot1_path_log.csv` & `robot2_path_log.csv`**: Telemetry files that record the continuous `(x, y)` path of each robot. Used strictly for post-mission visualization and trajectory debugging.

### `controllers/sar_marking_supervisor/`
* **`sar_marking_supervisor.py`**
  * **Role:** The Competition Referee (External).
  * **Responsibility:** A Webots Supervisor node that tracks the true global ground-truth of the simulation. It listens to our robots' scoring messages, reads our generated CSV files, validates our victim location claims against the hidden ground-truth, and calculates the final JSON grades.

---

## 🧠 2. Deep Dive: Architectural Components (`proposed_solution.py`)

The `proposed_solution.py` script is structured using object-oriented, decoupled classes that represent the distinct layers of a modern autonomous robotics stack.

### Layer 1: Hardware Abstraction (`ROSbotHardwareInterface`)
* **Purpose:** Acts as the Hardware Abstraction Layer (HAL).
* **Technical Mechanics:** Webots provides raw, low-level sensor node APIs. This class wraps the Webots API and exposes clean, typed Python methods to the higher-level logic. For example, instead of querying individual motor joints, the controller calls `set_motor_speeds(v, omega)`, and the HAL handles the differential drive kinematics required to spin the left and right tracks.
* **Competition Domain Focus:** Hardware Control, Radio Communication (handling both the Supervisor emitter/receiver and the Robot-to-Robot peer network).

### Layer 2: Localization (`CompassOdometry`)
* **Purpose:** Continuously calculates the robot's exact global coordinate `(x, y, theta)`.
* **Technical Mechanics:** 
  * **Translation:** Calculates distance traveled by reading the rotational deltas of the left and right **Wheel Encoders**.
  * **Rotation:** Reads the **IMU Compass** to determine absolute global heading. 
  * **Why Sensor Fusion?:** Pure wheel odometry suffers from cumulative "rotational drift" (where tiny slips in the wheels compound into massive angle errors). By fusing wheel translation with the absolute compass heading, the robot maintains near-perfect global localization without relying on an external GPS.
* **Competition Domain Focus:** Navigation, Odometry.

### Layer 3: Dynamic Mapping (`GridMap`)
* **Purpose:** Maintains a 2D matrix (Occupancy Grid) of the physical environment in real-time.
* **Technical Mechanics:** 
  * **Grid Matrix:** The map is discretized into a 600x600 grid where each cell represents 5cm (`resolution = 0.05m`). Cells hold values representing their state: `0` (Unknown), `1` (Free Space), `2` (Safety Buffer), and `3` (Solid Wall/Victim).
  * **Lidar Raytracing:** During the mission, the class subsamples the 360° lidar array. It projects lines from the robot's center to the laser hits using **Bresenham's Line Algorithm**. It paints the lines as `1` (Free Space) and the terminal hit points as `3` (Solid).
  * **Obstacle Inflation:** To prevent the physical chassis of the robot from clipping corners, every solid hit (3) spawns a circular radius of safety cells (2) around it. Planners are forbidden from entering these cells, guaranteeing physical clearance.
* **Competition Domain Focus:** Map Generation, Sensor Fusion, Video Information Extraction.

### Layer 4: Global Path Planning (`AStarPlanner`)
* **Purpose:** Calculates the mathematically optimal route from the robot to a target coordinate.
* **Technical Mechanics:** 
  * Implements the **A* (A-Star) Search Algorithm** utilizing an 8-way directional array (allowing diagonal movement at a cost of `sqrt(2)`).
  * **Heuristics & Escapes:** Uses a Euclidean distance heuristic to guarantee optimal shortest paths. If the robot accidentally finds itself slightly inside an inflated safety zone (2), the planner applies a heavy 100x cost penalty rather than throwing a hard block. This "soft penalty" ensures the robot can always calculate a valid escape route out of a tight corner.
* **Competition Domain Focus:** Algorithm Design, Global Routing.

### Layer 5: Local Steering Execution (`PurePursuitController`)
* **Purpose:** Translates the jagged, grid-based A* path into smooth physical motor commands.
* **Technical Mechanics:** 
  * Implements a **Pure Pursuit** steering algorithm.
  * Instead of driving point-to-point and stopping at every grid cell, Pure Pursuit projects a "lookahead" point `0.45m` down the path.
  * It computes the heading error to that lookahead point, applies a non-linear geometric curvature formula, and generates proportional steering velocities. This allows the robot to gracefully sweep around corners without stopping. If the heading error ever exceeds 45 degrees, it halts forward momentum and spins in place for maximum safety.
* **Competition Domain Focus:** Navigation, Local Trajectory Tracking, Efficiency.

### Layer 6: The Master State Machine (`AutonomousSARController`)
* **Purpose:** The top-level orchestrator that connects all underlying layers and drives the mission loop.
* **Technical Mechanics:** Operates on a continuous Finite State Machine (FSM) tied to the Webots simulation tick:
  * **`PLANNING`:** Triggers `AStarPlanner` to find a route to an assigned victim.
  * **`DRIVE`:** Feeds the active path to `PurePursuitController`. Simultaneously calls `GridMap.update_from_lidar()` so the map updates dynamically as the robot moves.
  * **`RECOVERY`:** If the physical front-bump sensors trigger (indicating an undetected obstacle), the robot enters Recovery. It reverses, spins away, and forces a new A* path calculation. 
    * *Failsafe Feature:* If the robot hits an obstacle 4 times back-to-back, it programmatically drops a massive 13x13 blockade of virtual obstacles on its internal map, permanently forcing A* to route completely around the hazardous area.
  * **`VICTIM_IDENTIFICATION`:** When the robot gets within 0.4 meters of its assigned victim, it mathematically projects the exact physical coordinates using its Lidar distance, overwrites `victim_location_estimates.csv` with surgical precision, and broadcasts a standardized "FOUND" radio message to the supervisor.
* **Competition Domain Focus:** Identification, Multi-Robot Coordination, Mission Logic.

---

## 📡 3. Multi-Robot Systems (MRS) Coordination
To maximize the "Efficiency" and "Coordination" scores, the robots do not search blindly.

1. **Task Partitioning:** The pre-mission flyover CSV detects two rough areas of interest. Robot 1 automatically assigns itself to Victim 1, and Robot 2 assigns itself to Victim 2. They operate completely independently, avoiding redundant exploration.
2. **Peer-to-Peer Radio:** Once a robot discovers its victim, it broadcasts the exact global coordinates over the inter-robot radio channel. The other robot receives this, marks it internally to avoid accidentally wasting time re-identifying it, and continues its own mission.
