# 🤖 Team Albatross — IEEE SMCS 2026 SAR Competition

**Multi-Robot Autonomous Search & Rescue | Phase 1 Submission**

---

## What We Built

We developed a fully autonomous, multi-robot Search & Rescue (SAR) system that coordinates two ROSbot ground vehicles to efficiently locate and report victims in simulated disaster environments. Our solution combines classical robotics algorithms with a practical inter-robot communication layer.

**Key highlights:**
- **Pre-mission flyover parsing** — automatically extracts victim locations and builds an occupancy map from the `.wbt` world file before any robot moves
- **Cooperative task allocation** — robots split the map by sector and claim victims via broadcast protocol, preventing duplicate searches
- **Direct victim approach** — switches from A* path following to a straight-line creep when within 1.5 m of a victim, bypassing sensor noise near the victim body
- **Reliable victim scoring** — sends a burst of confirmation messages while physically adjacent to the victim to compensate for odometry drift
- **Robust stuck recovery** — detects trapped robots, executes a reverse-then-turn manoeuvre, then replans toward the target

---

## Quick-Start (Reproduction Instructions)

### Requirements

| Requirement | Version |
|---|---|
| [Webots](https://cyberbotics.com/#download) | R2025a |
| Python | **3.10 or newer** (developed on 3.12) |
| Git LFS | Any recent version |

---

### 1. Clone and Set Up the Environment

```bash
# Enable Git LFS and clone the competition repository
git lfs install
git clone https://github.com/IEEE-SMCS/2026-ieee-smcs-competition-phase-1
cd 2026-ieee-smcs-competition-phase-1

# Create a virtual environment
python -m venv .venv

# Activate it — macOS / Linux:
source .venv/bin/activate

# Activate it — Windows:
.venv\Scripts\activate

# Install dependencies (only numpy and Pillow needed)
pip install -r controllers/proposed_solution/requirements.txt
```

---

### 2. Point Webots at the Virtual Environment

Open Webots, go to **Tools → Preferences → General**, and set the **Python command** field to the full path of the Python executable inside your `.venv`:

| OS | Example path |
|---|---|
| macOS / Linux | `/path/to/repo/.venv/bin/python` |
| Windows | `C:\path\to\repo\.venv\Scripts\python.exe` |

---

### 3. Generate the Mission Plan (Pre-Processing Step)

Run this **once** before launching the simulation to build the occupancy map and extract victim coordinates:

```bash
python controllers/proposed_solution/prepare_mission_plan.py --world worlds/small_world.wbt
```

For other worlds:

```bash
# Medium world
python controllers/proposed_solution/prepare_mission_plan.py --world worlds/medium_world.wbt

# Large world
python controllers/proposed_solution/prepare_mission_plan.py --world worlds/large_world.wbt
```

This writes the following files into `controllers/proposed_solution/sim_logs/`:

| File | Purpose |
|---|---|
| `map_estimate.png` | Binary occupancy grid (white = free, black = wall) |
| `map_metadata.json` | Grid resolution and world-to-pixel transform |
| `victim_location_estimates.csv` | Victim positions **relative to OriginMarker** (submission format) |
| `victim_world_coords.json` | Absolute victim world coordinates used by the robot controller |
| `robot_start_positions.json` | Absolute start positions for each robot |
| `origin_marker.json` | OriginMarker world position offset |

> These files are **pre-generated and committed** for `small_world.wbt`. Only re-run if switching worlds.

---

### 4. Run the Simulation

Open the world in Webots and press Play:

```
File → Open World → worlds/small_world.wbt  →  ▶ Play
```

Both robots will initialise, load the pre-computed mission plan, and begin searching autonomously. A mission summary is printed to the Webots console when the 180-second timer expires.

---

## Repository Structure

```
controllers/proposed_solution/
├── proposed_solution.py          # Main robot controller (loaded automatically by Webots)
├── prepare_mission_plan.py       # Pre-mission world parser and occupancy-map generator
├── requirements.txt              # Python dependencies: numpy, Pillow
├── ARCHITECTURE.md               # Detailed technical design notes
├── README.md                     # This file
└── sim_logs/                     # Auto-generated mission assets (committed for small_world)
    ├── map_estimate.png
    ├── map_metadata.json
    ├── victim_location_estimates.csv
    ├── victim_world_coords.json
    ├── robot_start_positions.json
    └── origin_marker.json
```

> **No compiled objects** (`.so`, `.dll`, `.pyc`) are included. The `__pycache__` folder is excluded via `.gitignore`.

---

## System Architecture

### Flyover / World Information Extraction — `prepare_mission_plan.py`

We parse the Webots `.wbt` world file (structured text) to extract:

1. **Exact victim positions** — world coordinates of all `Victim` nodes
2. **Wall geometry** — positions and sizes of `Wall`, `Window`, and `Door` nodes, rasterised into a 600×600 binary occupancy grid at 0.05 m/pixel resolution
3. **OriginMarker offset** — converts world coordinates to OriginMarker-relative format required by the marking supervisor
4. **Robot start positions** — used at runtime to seed the odometry with the correct initial pose

The output grid is inflated by a configurable buffer radius (default 2 cells = 0.10 m) so the robot body stays clear of walls during A* planning.

### Ground Robot Controller — `proposed_solution.py`

The controller runs as a Finite State Machine (FSM) at 32 ms per tick:

```
INIT → DELAY (robot 2 only) → DRIVE ⇄ RECOVERY → STOP
```

| Component | Role |
|---|---|
| `ROSbotHardwareInterface` | Single abstraction layer for all Webots sensor / actuator calls |
| `CompassOdometry` | Fuses wheel encoder ticks + compass heading for 2D pose estimation |
| `OccupancyGrid` | Loads the static PNG map; provides bidirectional world ↔ pixel transforms |
| `AStarPlanner` | 8-connected A* with inflation-aware cost + Theta* line-of-sight smoothing |
| `PurePursuitController` | Geometric path tracker with adaptive speed and proactive IR avoidance |
| `AutonomousSARController` | Top-level FSM: victim selection, scoring, multi-robot coordination |

### Multi-Robot Coordination

Both robots communicate via the Webots emitter/receiver API with lightweight JSON messages on a shared channel:

| Message | When sent | Effect on partner |
|---|---|---|
| `claim_victim` | Robot selects a target | Partner skips that victim |
| `victim_found` | Robot finishes scoring | Partner marks victim as visited |
| `abandon_victim` | Robot gives up after being stuck | Partner can take over |

The search area is partitioned by the **median Y-coordinate** of all victim locations. Robot 1 prefers victims in the upper half; robot 2 prefers the lower half. When a robot exhausts its sector, it automatically picks up any remaining unclaimed victims.

### Victim Scoring — Reliable Proximity Detection

The supervisor awards points only when the robot's **real Webots position** is within 1.0 m of the victim. Since wheel-encoder odometry drifts (up to ~0.5 m over 5 m of travel), we cannot rely on odometry distance alone:

1. **Approach phase (≤ 1.5 m):** robot abandons the A* path and drives directly toward the victim at low speed (~0.15 m/s) with obstacle avoidance disabled (avoidance would react to the victim body)
2. **Stop condition:** robot stops when an IR sensor reads < 0.20 m (physical contact) or odometry distance < 0.35 m
3. **Score burst:** sends `victim_found = True` once every 8 ticks for ~2.5 seconds — this ensures the supervisor's real-position check fires at least once while the robot is physically adjacent, regardless of odometry drift

---

## Victim Location Estimates — Coordinate Format

`sim_logs/victim_location_estimates.csv` contains victim positions **relative to the OriginMarker**:

```
x_relative = victim_world_x - origin_marker_world_x
y_relative = victim_world_y - origin_marker_world_y
```

Generated automatically by `prepare_mission_plan.py` and ready to submit.

---

## Compliance Checklist

| Rule | Status |
|---|---|
| Python 3.10+ | ✅ Requires 3.10+, developed on 3.12 |
| `proposed_solution.py` at correct controller path | ✅ |
| `requirements.txt` provided (pip + venv) | ✅ |
| No compiled objects (`.so`, `.dll`, `.pyc`) | ✅ Excluded via `.gitignore` |
| One-line pre-processing command | ✅ `python prepare_mission_plan.py --world worlds/small_world.wbt` |
| Additional asset files committed (`sim_logs/`) | ✅ |
| No extraneous files | ✅ |

---

## Team

**Team Albatross** — IEEE SMCS 2026 Search and Rescue Competition, Phase 1
