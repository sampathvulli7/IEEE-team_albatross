# IEEE SMCS Autonomous SAR Competition - Comprehensive Architecture Guide

This document serves as an exhaustive architectural guide for the Autonomous Search and Rescue (SAR) controller used in the IEEE SMCS competition. It details the project structure, individual file responsibilities, sensor processing pipelines, path planning algorithms, and Multi-Robot System (MRS) coordination mechanisms.

---

## 📂 1. Directory Structure & Global File Responsibilities

The simulation environment uses a centralized scoring supervisor (`sar_marking_supervisor`) and decentralized robot controllers (`proposed_solution`).

### `controllers/proposed_solution/`
This directory contains the core solution code:

* **`proposed_solution.py`**
  * **Role:** Master Robot Executable. 
  * **Responsibility:** Runs independently on every robot in the swarm. Implements the entire robotics stack: HAL hardware interfacing, sensor readings, compass/encoder localization, dynamic occupancy grid mapping, sector-aware multi-victim task allocation, A* path planning with Theta* smoothing, Pure Pursuit steering, and squad radio communication.
* **`prepare_mission_plan.py`**
  * **Role:** Pre-Mission Offline Analysis & Asset Generator.
  * **Responsibility:** Parses Webots `.wbt` world files to extract exact wall geometries, victim 3D PROTO marker offsets, and dynamic initial robot spawn positions. Generates pre-mission map estimates (`map_estimate.png`), victim location estimates (`victim_location_estimates.csv`), and initial odometry coordinates (`robot_start_positions.json`).
* **`sim_logs/`** (Data & Telemetry Directory)
  * **`map_estimate.png`**: Top-down 600x600 binary image (0.05m resolution) representing walls and obstacles.
  * **`victim_location_estimates.csv`**: Ground-truth target victim coordinates offset by PROTO marker transforms.
  * **`robot_start_positions.json`**: Dynamic spawn coordinates for each ROSbot in the loaded world environment.
  * **`{robot_id}_telemetry.csv`**: Structured 10Hz log recording timestamp, tick, FSM state, position, heading, target, command velocities, and distance sensor readings.

### `controllers/sar_marking_supervisor/`
* **`sar_marking_supervisor.py`**
  * **Role:** Competition Referee & Grading System.
  * **Responsibility:** Webots Supervisor node tracking world ground truth. Evaluates victim discovery pings, location estimate error, map estimate accuracy, distance traveled, mission duration, and squad message count.

---

## 🧠 2. Architectural Components Deep Dive (`proposed_solution.py`)

`proposed_solution.py` is structured using decoupled, modular classes representing the layers of a modern autonomous robotics stack.

```
+-------------------------------------------------------------------+
|                    AutonomousSARController (FSM)                  |
+-------------------------------------------------------------------+
       |                  |                    |                  |
       v                  v                    v                  v
+--------------+  +---------------+  +------------------+  +--------------+
| HAL & Radio  |  | Compass Odom  |  |  Occupancy Grid  |  |  A* Planner  |
|  Interface   |  | Localization  |  | & Subsampled Lidar|  | & PurePursuit|
+--------------+  +---------------+  +------------------+  +--------------+
```

### Layer 1: Hardware Abstraction & Kinematics (`ROSbotHardwareInterface`)
* **Purpose:** Encapsulates the Webots C/Python Robot API.
* **Physical Kinematics & Slew-Rate Limiting:**
  * **Track Width:** `0.20m` | **Wheel Radius:** `0.04m`
  * **Max Wheel Speed:** Clamped to `12.0 rad/s` (~0.48 m/s physical wheel limit).
  * **Acceleration Slew-Rate Limiter:** Limits motor acceleration to `25.0 rad/s²`. Prevents instantaneous impulse spikes, wheel slippage, and Webots physics step warnings.
* **Sensor & Emitter Interfaces:**
  * **Motors:** 4-wheel independent velocity control (`fl_wheel_joint`, `fr_wheel_joint`, `rl_wheel_joint`, `rr_wheel_joint`).
  * **Encoders:** Incremental wheel position sensors (`front left wheel motor sensor`, `front right wheel motor sensor`).
  * **Lidar:** 360° laser rangefinder (`laser`).
  * **Compass:** IMU 3-axis magnetometer (`imu compass`).
  * **IR Sensors:** 4-channel range sensors (`fl_range`, `fr_range`, `rl_range`, `rr_range`).
  * **Emitters/Receivers:** Channel 43 Supervisor Emitter & Robot-to-Robot squad transceiver.

### Layer 2: Localization & State Estimation (`CompassOdometry`)
* **Purpose:** Tracks continuous global 2D pose $(x, y, \theta)$.
* **Sensor Fusion Mechanics:**
  * **Linear Delta:** Computed from incremental wheel encoder deltas: $\Delta d = \frac{\Delta d_L + \Delta d_R}{2} \cdot r_{\text{wheel}}$.
  * **Absolute Heading:** Directly assigned from IMU compass heading $\theta = \text{atan2}(N_x, N_y)$, eliminating cumulative rotational drift.
  * **Position Integration:** $x_{k+1} = x_k + \Delta d \cdot \cos(\theta)$, $y_{k+1} = y_k + \Delta d \cdot \sin(\theta)$.

### Layer 3: Dynamic Mapping & Safety Inflation (`OccupancyGrid`)
* **Purpose:** Maintains a 600x600 matrix representing the 30m x 30m physical environment (`resolution = 0.05m/cell`).
* **Grid Cell Encoding:**
  * `0`: Unknown space.
  * `1`: Free space.
  * `2`: Inflated safety buffer.
  * `3`: Solid obstacle / wall / victim.
* **Inflation Parameters:**
  * **`inflation_radius_cells = 3`**: `0.15m` safety inflation buffer around solid obstacles.
  * Ensures ROSbot (radius ~0.12m) has clear clearance through 0.8m narrow doorways while protecting against corner clipping.

### Layer 4: Global Path Planning (`AStarPlanner`)
* **Purpose:** Calculates optimal routes over the `OccupancyGrid`.
* **Pathfinding Rules:**
  * **8-Way Search Grid:** Orthogonal moves cost `1.0`, diagonal moves cost `1.414`.
  * **Impassable Walls:** Solid obstacles (`cell_val == 3`) are strictly impassable (`continue`), except when within target victim proximity (`dist_to_goal <= inflation_cells + 2`).
  * **Buffer Soft Penalty:** Inflated buffer cells (`cell_val == 2`) carry a `4.0` cost penalty, forcing A* to prefer hallway centers while remaining passable through tight doors.
* **Theta* Line-of-Sight Path Smoothing:**
  * Uses Bresenham line checks to collapse unnecessary intermediate waypoints, producing smooth straight-line paths through open rooms.

### Layer 5: Local Steering & Speed Control (`PurePursuitController`)
* **Purpose:** Generates smooth differential drive motor velocity commands $(v, \omega)$.
* **Control Parameters:**
  * **Lookahead Distance:** `0.25m` (ensures tight, accurate tracking around tight corners without wall clipping).
  * **Cruising Speed ($v$):** `0.30 m/s` (smooth, stable cruising without motor slip or odometry drift).
  * **Curve Speed ($v_{\text{curve}}$):** `0.18 m/s` when heading error $|\alpha| > \pi/8$.
  * **In-Place Rotation ($\omega$):** `2.0 rad/s` when heading error $|\alpha| > \pi/4$ and distance $> 0.30m$.

---

## 📡 3. Multi-Robot System (MRS) Task Allocation & Coordination

To maximize search efficiency across multi-victim environments (`small_world`, `medium_world`, `large_world`):

1. **Sector-Based Partitioning:**
   - At initialization, `_select_next_victim()` calculates the median Y-coordinate of all victims in the mission.
   - **`robot1`** prioritizes the **North Sector** ($Y \ge Y_{\text{median}}$).
   - **`robot2`** prioritizes the **South Sector** ($Y < Y_{\text{median}}$).
   - Prevents robots from crossing paths or targeting the same initial victim.

2. **Inter-Robot Claim & Rescue Packets:**
   - **`claim_victim`**: Broadcasted immediately when a robot claims a target. Peers record this claim and skip the target.
   - **`victim_found`**: Broadcasted when a victim is reached. Marks the target as visited across the entire squad.
   - **Dynamic Queue Drain**: Once a robot completes its assigned sector victims, it automatically assists by picking any remaining unclaimed unvisited victims across the entire map.

---

## 📊 4. Telemetry & Monitoring Architecture

* **High-Precision Telemetry Log (`sim_logs/{robot_id}_telemetry.csv`)**:
  * Logs state data at 10Hz: `time, tick, state, pos_x, pos_y, theta, target_x, target_y, dist_to_target, v_cmd, omega_cmd, fl_dist, fr_dist, min_lidar`.
* **Structured Console Telemetry**:
  * Formatted output: `[time.s][robot_id] Event / State / Tracking Info`.
  * Provides real-time visibility into FSM state transitions, A* path lengths, lookahead waypoint numbers, IR bumper alerts, and squad radio traffic.
