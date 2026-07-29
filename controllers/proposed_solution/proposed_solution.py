"""
IEEE SMCS Autonomous Search and Rescue - Proposed Solution
===========================================================
Multi-robot SAR controller with A* path planning, lidar mapping,
Pure Pursuit path following, and inter-robot coordination.

Architecture:
  ROSbotHardwareInterface  - Webots sensor/actuator abstraction
  CompassOdometry          - Wheel encoder + compass pose tracking
  OccupancyGrid            - 2D grid map with static + dynamic layers
  AStarPlanner             - 8-way A* with inflation-aware cost and Theta* smoothing
  PurePursuitController    - Geometric path follower with adaptive speed
  AutonomousSARController  - Top-level FSM mission controller
"""

import sys
import math
import json
import logging
import heapq
import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    from controller import Robot
except ImportError:
    Robot = None

# ==========================================
# SYSTEM SETUP: Unbuffered I/O for Webots
# ==========================================
class Unbuffered(object):
    """Forces immediate stdout/stderr flushing so Webots console shows output in real time."""
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger("AutonomousSAR")

# Type aliases for clarity
Pose = Tuple[float, float, float]       # (x, y, theta) in meters/radians
Coordinate = Tuple[float, float]         # (x, y) in meters


# ==========================================
# HARDWARE ABSTRACTION LAYER (Webots API)
# ==========================================
class ROSbotHardwareInterface:
    """
    Provides a clean Python interface to the Webots ROSbot hardware.
    Encapsulates all motors, encoders, lidar, compass, IR sensors, 
    and radio emitter/receiver devices.
    """
    def __init__(self):
        logger.info("Initializing ROSbot Hardware Interface...")
        if Robot is None:
            self.robot = None
            self.robot_id = "test_robot"
            return

        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_id = self.robot.getName()

        # --- Drive Motors (4WD differential drive) ---
        self.fl_motor = self.robot.getDevice("fl_wheel_joint")
        self.fr_motor = self.robot.getDevice("fr_wheel_joint")
        self.rl_motor = self.robot.getDevice("rl_wheel_joint")
        self.rr_motor = self.robot.getDevice("rr_wheel_joint")
        for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
            if motor:
                motor.setPosition(float('inf'))  # Velocity control mode
                motor.setVelocity(0.0)

        # --- Wheel Encoders (for odometry) ---
        self.left_sensor = self.robot.getDevice("front left wheel motor sensor")
        self.right_sensor = self.robot.getDevice("front right wheel motor sensor")
        if self.left_sensor: self.left_sensor.enable(self.timestep)
        if self.right_sensor: self.right_sensor.enable(self.timestep)

        # --- 360° Lidar Scanner ---
        self.lidar = self.robot.getDevice("laser")
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

        # --- IMU Compass (absolute heading) ---
        self.compass = self.robot.getDevice("imu compass")
        if self.compass:
            self.compass.enable(self.timestep)

        # --- IR Distance Sensors (front-left, front-right, rear-left, rear-right) ---
        self.fl_range = self.robot.getDevice("fl_range")
        self.fr_range = self.robot.getDevice("fr_range")
        self.rl_range = self.robot.getDevice("rl_range")
        self.rr_range = self.robot.getDevice("rr_range")
        for sensor in [self.fl_range, self.fr_range, self.rl_range, self.rr_range]:
            if sensor: sensor.enable(self.timestep)

        # --- Radio: Supervisor scoring channel (channel 43) ---
        self.emitter = self.robot.getDevice("supervisor emitter")
        if self.emitter:
            self.emitter.setChannel(43)

        # --- Radio: Robot-to-robot squad communication ---
        self.squad_receiver = self.robot.getDevice("robot to robot receiver")
        self.squad_emitter = self.robot.getDevice("robot to robot emitter")
        if self.squad_receiver:
            self.squad_receiver.enable(self.timestep)

        # Track current motor speeds for smooth acceleration
        self.curr_left_rads = 0.0
        self.curr_right_rads = 0.0

    def step(self) -> bool:
        """Advance simulation by one timestep. Returns False if simulation ended."""
        if self.robot is None: return False
        return self.robot.step(self.timestep) != -1

    def get_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.robot.getTime() if self.robot else 0.0

    def read_encoders(self) -> Tuple[float, float]:
        """Read left/right wheel encoder positions in radians."""
        l = self.left_sensor.getValue() if self.left_sensor else 0.0
        r = self.right_sensor.getValue() if self.right_sensor else 0.0
        return l, r

    def read_compass_heading(self) -> float:
        """Read absolute heading from IMU compass (radians, 0=North)."""
        if not self.compass: return 0.0
        north = self.compass.getValues()
        return math.atan2(north[0], north[1])

    def read_lidar(self) -> List[float]:
        """Read 360° lidar range image (list of distances in meters)."""
        if not self.lidar: return []
        return self.lidar.getRangeImage()

    def read_front_distances(self) -> Tuple[float, float]:
        """Read front-left and front-right IR distance sensors (meters)."""
        fl = self.fl_range.getValue() if self.fl_range else 2.0
        fr = self.fr_range.getValue() if self.fr_range else 2.0
        return fl, fr

    def set_motor_speeds(self, linear_velocity: float, angular_velocity: float) -> None:
        """
        Convert (v, omega) to differential drive wheel speeds with smooth acceleration.
        Uses slew-rate limiting to prevent physics engine instability.
        """
        # ROSbot parameters
        track_width = 0.20   # Distance between left and right wheels (meters)
        wheel_radius = 0.04  # Wheel radius (meters)
        max_rads = 12.0      # Motor max speed (rad/s ≈ 0.48 m/s)

        # Differential drive kinematics: v_wheel = v ± omega * track_width/2
        v_left = linear_velocity - (angular_velocity * track_width / 2.0)
        v_right = linear_velocity + (angular_velocity * track_width / 2.0)

        # Convert linear speed to motor angular speed and clamp
        target_left = max(-max_rads, min(max_rads, v_left / wheel_radius))
        target_right = max(-max_rads, min(max_rads, v_right / wheel_radius))

        # Smooth acceleration: limit rate of change to prevent impulse spikes
        dt = (self.timestep / 1000.0) if self.timestep else 0.032
        max_delta = 25.0 * dt  # 25 rad/s² acceleration limit

        d_left = target_left - self.curr_left_rads
        d_right = target_right - self.curr_right_rads
        self.curr_left_rads += max(-max_delta, min(max_delta, d_left))
        self.curr_right_rads += max(-max_delta, min(max_delta, d_right))

        # Apply to all four motors
        if self.fl_motor: self.fl_motor.setVelocity(self.curr_left_rads)
        if self.rl_motor: self.rl_motor.setVelocity(self.curr_left_rads)
        if self.fr_motor: self.fr_motor.setVelocity(self.curr_right_rads)
        if self.rr_motor: self.rr_motor.setVelocity(self.curr_right_rads)

    def send_score_message(self, robot_id: str, position: List[float]) -> None:
        """Send victim FOUND message to supervisor for scoring.
        ONLY call this when the robot is confident it has found a victim.
        The victim_found=True flag is what triggers scoring."""
        if not self.emitter: return
        confidence = position[2] if len(position) > 2 else 1.0
        msg = {
            "timestamp": self.robot.getTime(),
            "robot_id": robot_id,
            "position": position[:2],
            "victim_found": True,
            "victim_confidence": confidence
        }
        self.emitter.send(json.dumps(msg).encode('utf-8'))

    def send_status_message(self, robot_id: str, position: List[float]) -> None:
        """Send a heartbeat/status message with victim_found=False.
        Use this for regular periodic status updates that don't claim
        a victim has been found. This does NOT affect confidence scoring."""
        if not self.emitter: return
        msg = {
            "timestamp": self.robot.getTime(),
            "robot_id": robot_id,
            "position": position[:2],
            "victim_found": False,
            "victim_confidence": 0.0
        }
        self.emitter.send(json.dumps(msg).encode('utf-8'))

    def send_squad_message(self, message: dict) -> None:
        """Broadcast a message to other robots via squad radio."""
        if not self.squad_emitter: return
        self.squad_emitter.send(json.dumps(message).encode('utf-8'))

    def receive_squad_messages(self) -> List[dict]:
        """Read all pending messages from the squad radio queue."""
        messages = []
        if not self.squad_receiver: return messages
        while self.squad_receiver.getQueueLength() > 0:
            try:
                data = self.squad_receiver.getString()
                messages.append(json.loads(data))
            except:
                pass
            self.squad_receiver.nextPacket()
        return messages


# ==========================================
# COMPASS-FUSED ODOMETRY
# ==========================================
class CompassOdometry:
    """
    Tracks robot pose (x, y, theta) using wheel encoders for distance
    and an IMU compass for absolute heading. The compass prevents
    rotational drift that accumulates with pure dead reckoning.
    """
    def __init__(self, start_x: float, start_y: float):
        self.x = start_x
        self.y = start_y
        self.theta = 0.0
        self.last_left_enc = None
        self.last_right_enc = None
        self.wheel_radius = 0.04  # meters
        self.track_width = 0.2    # meters

    def update(self, left_enc: float, right_enc: float, compass_heading: float) -> None:
        """Update pose from encoder deltas and compass heading."""
        self.theta = compass_heading
        if self.last_left_enc is None:
            self.last_left_enc = left_enc
            self.last_right_enc = right_enc
            return
        # Convert encoder deltas to linear displacement
        dl = (left_enc - self.last_left_enc) * self.wheel_radius
        dr = (right_enc - self.last_right_enc) * self.wheel_radius
        dc = (dl + dr) / 2.0
        # Integrate position using current heading
        self.x += dc * math.cos(self.theta)
        self.y += dc * math.sin(self.theta)
        self.last_left_enc = left_enc
        self.last_right_enc = right_enc

    def get_pose(self) -> Pose:
        return (self.x, self.y, self.theta)


# ==========================================
# OCCUPANCY GRID MAP
# ==========================================
class OccupancyGrid:
    """
    2D occupancy grid for the environment. Uses a two-layer approach:
    
    STATIC LAYER: Loaded from map_estimate.png (permanent walls from world file parsing).
                  Never modified by lidar. This preserves safety buffers.
    
    DYNAMIC LAYER: Updated by lidar at runtime. Detects furniture, doors, 
                   and other objects not in the static map.
    
    Cell values:
      0 = Unknown/unexplored
      1 = Free space (confirmed by map or lidar)
      2 = Inflated safety buffer (around obstacles)
      3 = Solid obstacle (wall/object)
    """
    def __init__(self, rows: int = 600, cols: int = 600, resolution: float = 0.05):
        self.resolution = resolution
        self.origin_x = -15.0  # World coordinate of grid cell (0,0)
        self.origin_y = -15.0
        self.cols = cols
        self.rows = rows
        # Inflation radius in grid cells (3 cells * 0.05m = 0.15m clearance)
        self.inflation_radius_cells = 3
        
        # Combined grid used by A* planner
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        # Static layer: loaded from PNG, never modified
        self.static_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        
        self.load_map_from_png()

    def load_map_from_png(self):
        """
        Load the pre-computed map from map_estimate.png.
        Black pixels = walls (3), white pixels = free space (1).
        Then inflate all wall cells with a safety buffer.
        """
        map_path = os.path.join(os.path.dirname(__file__), "sim_logs", "map_estimate.png")
        if not os.path.exists(map_path):
            logger.warning(f"No map_estimate.png found, starting with empty grid.")
            return

        try:
            img = Image.open(map_path).convert('L')
            if img.size != (600, 600):
                img = img.resize((600, 600))
            pixels = np.array(img)

            # Threshold: < 128 is wall (black), >= 128 is free (white)
            self.grid[pixels < 128] = 3   # Solid wall
            self.grid[pixels >= 128] = 1  # Free space
            
            # Save static copy before inflation
            self.static_grid = np.copy(self.grid)

            # Inflate walls: add safety buffer cells around every wall cell
            # Uses scipy-style binary dilation via manual circle kernel
            wall_mask = (self.grid == 3)
            r = self.inflation_radius_cells
            for row in range(self.rows):
                for col in range(self.cols):
                    if wall_mask[row, col]:
                        for dr in range(-r, r + 1):
                            for dc in range(-r, r + 1):
                                if dr*dr + dc*dc <= r*r:
                                    nr, nc = row + dr, col + dc
                                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                        if self.grid[nr, nc] < 2:
                                            self.grid[nr, nc] = 2  # Safety buffer
            
            logger.info("Successfully loaded and inflated map_estimate.png")
        except Exception as e:
            logger.error(f"Failed to load map PNG: {e}")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid cell indices (col, row)."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Coordinate:
        """Convert grid cell indices back to world coordinates (meters)."""
        x = (col + 0.5) * self.resolution + self.origin_x
        y = (row + 0.5) * self.resolution + self.origin_y
        return x, y

    def in_bounds(self, col: int, row: int) -> bool:
        """Check if grid indices are within map boundaries."""
        return 0 <= col < self.cols and 0 <= row < self.rows

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Compute all grid cells along a line using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def update_from_lidar(self, pose: Pose, lidar_data: List[float], max_range=3.5) -> bool:
        """
        Update the dynamic map layer using lidar scan data.
        
        KEY DESIGN DECISION: Lidar can only ADD new obstacles to the map.
        It NEVER removes obstacles from the static (PNG) layer. This prevents
        lidar noise from erasing the pre-computed wall safety buffers.
        
        Returns True if any new obstacles were added.
        """
        if not lidar_data:
            return False
        rx, ry, rtheta = pose
        r_col, r_row = self.world_to_grid(rx, ry)
        if not self.in_bounds(r_col, r_row):
            return False

        n = len(lidar_data)
        updated = False
        
        # Process every 4th lidar ray (sufficient for 0.05m grid at typical ranges)
        for i in range(0, n, 4):
            dist = lidar_data[i]
            
            # Skip invalid readings
            if math.isinf(dist) or math.isnan(dist):
                continue
            if dist >= max_range:
                continue

            # Compute hit point in world coordinates
            # Lidar index 0 = rear of robot, sweeps CCW
            angle = rtheta - math.pi + (2 * math.pi * i / n)
            hit_x = rx + dist * math.cos(angle)
            hit_y = ry + dist * math.sin(angle)
            
            h_col, h_row = self.world_to_grid(hit_x, hit_y)
            
            # Mark the hit cell as obstacle IF it's not already known
            if self.in_bounds(h_col, h_row):
                if self.grid[h_row, h_col] < 2:
                    self.grid[h_row, h_col] = 3
                    updated = True
                    # Inflate around the new dynamic obstacle
                    r = self.inflation_radius_cells
                    for dr in range(-r, r + 1):
                        for dc in range(-r, r + 1):
                            if dr*dr + dc*dc <= r*r:
                                nr, nc = h_row + dr, h_col + dc
                                if self.in_bounds(nc, nr) and self.grid[nr, nc] < 2:
                                    self.grid[nr, nc] = 2

        return updated

    def is_path_blocked(self, path: List[Coordinate], start_idx: int) -> bool:
        """
        Check if any waypoint in the path crosses a SOLID wall (value 3).
        Inflated buffers (2) do NOT invalidate the path since A* accounts for them.
        """
        for i in range(start_idx, len(path)):
            c, r = self.world_to_grid(*path[i])
            if self.in_bounds(c, r) and self.grid[r, c] == 3:
                return True
        return False


# ==========================================
# A* PATH PLANNER WITH THETA* SMOOTHING
# ==========================================
class AStarPlanner:
    """
    A* pathfinding over the OccupancyGrid with:
    - 8-directional movement (orthogonal + diagonal)
    - Cost penalties for inflated zones (prefers hallway centers)
    - Theta*-inspired line-of-sight smoothing (removes jagged zig-zags)
    - Dense waypoint interpolation (forces tight path tracking)
    """
    def __init__(self, grid_map: OccupancyGrid):
        self.grid_map = grid_map
        # 8-way movement: (dx, dy, base_cost)
        self.directions = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414)
        ]

    def plan(self, start_world: Coordinate, goal_world: Coordinate) -> List[Coordinate]:
        """
        Plan a path from start to goal using A* search.
        Returns a list of (x,y) world coordinates forming the path.
        """
        start = self.grid_map.world_to_grid(*start_world)
        goal = self.grid_map.world_to_grid(*goal_world)

        if not self.grid_map.in_bounds(*start) or not self.grid_map.in_bounds(*goal):
            return [goal_world]

        # If goal is inside a wall, find the nearest free cell as the actual goal
        if self.grid_map.grid[goal[1], goal[0]] == 3:
            goal = self._find_nearest_free(goal)
            if goal is None:
                return [goal_world]

        # A* search with priority queue
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        expansions = 0
        max_expansions = 40000  # Prevent CPU freeze on unreachable goals

        while frontier and expansions < max_expansions:
            expansions += 1
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                break

            for dx, dy, move_cost in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.grid_map.in_bounds(*neighbor):
                    continue

                cell = self.grid_map.grid[neighbor[1], neighbor[0]]
                
                # Solid walls are ALWAYS impassable
                if cell == 3:
                    continue

                # Cost: inflated zones cost 5x more, encouraging center-of-hallway paths
                traversal_cost = 5.0 if cell == 2 else 1.0
                new_cost = cost_so_far[current] + move_cost * traversal_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    # Euclidean heuristic (admissible, never overestimates)
                    h = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(frontier, (new_cost + h, neighbor))
                    came_from[neighbor] = current

        # If exact goal not reached, find closest reachable cell
        if goal not in came_from:
            min_dist = float('inf')
            best = None
            for node in came_from:
                d = math.hypot(goal[0] - node[0], goal[1] - node[1])
                if d < min_dist:
                    min_dist = d
                    best = node
            if best is None:
                return [goal_world]
            goal = best

        # Reconstruct path from goal back to start
        path_grid = []
        current = goal
        while current is not None and current != start:
            path_grid.append(current)
            current = came_from[current]
        path_grid.reverse()

        if not path_grid:
            return [goal_world]

        # Convert grid path to world coordinates
        path = [self.grid_map.grid_to_world(*p) for p in path_grid]

        # Apply Theta* line-of-sight smoothing
        path = self._smooth_path(path)

        # Densify: insert waypoints every 0.20m for tight tracking
        path = self._densify_path(path, spacing=0.20)

        return path

    def _find_nearest_free(self, goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell to a blocked goal using BFS spiral search."""
        from collections import deque
        visited = {goal}
        queue = deque([goal])
        while queue:
            c, r = queue.popleft()
            if self.grid_map.grid[r, c] < 2:  # Free or unknown
                return (c, r)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nc, nr = c+dx, r+dy
                if (nc, nr) not in visited and self.grid_map.in_bounds(nc, nr):
                    visited.add((nc, nr))
                    queue.append((nc, nr))
        return None

    def _line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if a straight line between two grid cells is obstacle-free.
        Only rejects paths through solid walls (3).
        Allows passing through inflated zones (2) since the robot CAN fit through
        doorways — it just shouldn't PLAN to drive there unless necessary.
        """
        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while x0 != x1 or y0 != y1:
            if self.grid_map.in_bounds(x0, y0):
                if self.grid_map.grid[y0, x0] == 3:
                    return False  # Blocked by solid wall
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def _smooth_path(self, path: List[Coordinate]) -> List[Coordinate]:
        """
        Theta*-style smoothing: try to skip intermediate waypoints
        by checking direct line-of-sight between non-adjacent waypoints.
        This removes zig-zag artifacts from grid-aligned A* paths.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            best_skip = i + 1
            # Try to skip as far ahead as possible
            for j in range(len(path) - 1, i + 1, -1):
                p1 = self.grid_map.world_to_grid(*smoothed[-1])
                p2 = self.grid_map.world_to_grid(*path[j])
                if self._line_of_sight(p1, p2):
                    best_skip = j
                    break
            smoothed.append(path[best_skip])
            i = best_skip
        return smoothed

    def _densify_path(self, path: List[Coordinate], spacing: float) -> List[Coordinate]:
        """
        Insert intermediate waypoints so no two consecutive points are
        further than `spacing` meters apart. This forces the Pure Pursuit
        controller to tightly track the planned path through narrow spaces.
        """
        if len(path) < 2:
            return path
        dense = [path[0]]
        for i in range(1, len(path)):
            prev = dense[-1]
            curr = path[i]
            dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            if dist > spacing:
                n_inserts = int(dist / spacing)
                for j in range(1, n_inserts + 1):
                    t = j / (n_inserts + 1)
                    dense.append((prev[0] + t*(curr[0]-prev[0]),
                                  prev[1] + t*(curr[1]-prev[1])))
            dense.append(curr)
        return dense


# ==========================================
# PURE PURSUIT PATH FOLLOWER
# ==========================================
class PurePursuitController:
    """
    Geometric path following controller with GENTLE proactive obstacle avoidance.
    
    Two-layer approach:
    1. Pure Pursuit: geometric curvature toward next waypoint
    2. Reactive Avoidance: IR + narrow lidar arc steering correction
       that kicks in ONLY when obstacles are very close (< 0.20m).
    
    KEY DESIGN: Avoidance is completely DISABLED within 1.2m of the victim
    target, since the "obstacle" detected at close range IS the victim body.
    """
    def __init__(self, lookahead: float = 0.25):
        self.lookahead = lookahead

    def get_velocity(self, robot_pose: Pose, target: Coordinate,
                     fl_dist: float = 2.0, fr_dist: float = 2.0,
                     lidar_data: list = None,
                     dist_to_goal: float = 999.0) -> Tuple[float, float]:
        """
        Compute (linear_vel, angular_vel) to steer toward target.
        Incorporates gentle proactive obstacle avoidance using IR + lidar.
        
        :param robot_pose: (x, y, theta) current robot pose
        :param target: (x, y) waypoint to track
        :param fl_dist: Front-left IR sensor distance (meters)
        :param fr_dist: Front-right IR sensor distance (meters)
        :param lidar_data: Full 360° lidar range image (optional)
        :param dist_to_goal: Distance to final victim target (meters).
                             When < 1.2m, all avoidance is disabled so
                             the robot can approach the victim body.
        :return: (linear_velocity, angular_velocity)
        """
        rx, ry, rtheta = robot_pose

        # Heading error: angle between current heading and target direction
        alpha = math.atan2(target[1] - ry, target[0] - rx) - rtheta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

        dist = math.hypot(target[0] - rx, target[1] - ry)

        # Turn in place if heading error > 45° and target is not trivially close
        if abs(alpha) > math.pi / 4 and dist > 0.20:
            return 0.0, (2.5 if alpha > 0 else -2.5)

        # Base cruising speed
        v = 0.35

        # Pure Pursuit geometric curvature
        L = max(dist, self.lookahead)
        omega = (2 * v * math.sin(alpha)) / L

        # Slow down on sharp curves for stability
        if abs(alpha) > math.pi / 8:
            v = 0.20

        # =============================================================
        # APPROACH MODE: When close to the victim target, DISABLE all
        # obstacle avoidance. The "obstacle" the sensors see IS the victim.
        # Just drive straight to it at a moderate speed.
        # =============================================================
        if dist_to_goal < 1.2:
            # Slow down gently for the final approach, but keep moving
            v = 0.20
            return v, omega

        # =============================================================
        # PROACTIVE OBSTACLE AVOIDANCE (using front IR sensors)
        # Only triggers within 20cm — gentle enough to not slow hallway travel
        # =============================================================
        min_front = min(fl_dist, fr_dist)

        if min_front < 0.20:
            # Close obstacle: slow down proportionally
            # 0.20m -> 70% speed, 0.08m -> near minimum
            speed_factor = max(0.3, (min_front - 0.05) / 0.15)
            v *= speed_factor

            # Steer AWAY from the closer obstacle
            avoidance_strength = 1.5 * (1.0 - min_front / 0.20)
            if fl_dist < fr_dist:
                omega -= avoidance_strength  # Turn right (away from left obstacle)
            else:
                omega += avoidance_strength  # Turn left (away from right obstacle)

        # =============================================================
        # LIDAR-BASED FRONT NARROW CHECK (±15° arc only)
        # Only checks for obstacles directly ahead, not side walls.
        # =============================================================
        if lidar_data and len(lidar_data) > 0:
            n = len(lidar_data)
            # Narrow front arc: ±15° (±n/24 indices) to avoid triggering on side walls
            arc_half = n // 24
            center = n // 2
            front_left_min = 2.0
            front_right_min = 2.0

            for i in range(center - arc_half, center + arc_half):
                idx = i % n
                d = lidar_data[idx]
                if math.isinf(d) or math.isnan(d):
                    continue
                if i < center:
                    front_right_min = min(front_right_min, d)
                else:
                    front_left_min = min(front_left_min, d)

            lidar_min_front = min(front_left_min, front_right_min)
            if lidar_min_front < 0.18:
                # Very close obstacle directly ahead — reduce speed and steer
                v = min(v, 0.15)
                steer = 1.2 * (1.0 - lidar_min_front / 0.18)
                if front_left_min < front_right_min:
                    omega -= steer
                else:
                    omega += steer

        # Enforce minimum speed floor to prevent crawling through corridors
        v = max(v, 0.12)

        return v, omega


# ==========================================
# MAIN MISSION CONTROLLER (FSM)
# ==========================================
class AutonomousSARController:
    """
    Top-level mission controller implementing a Finite State Machine (FSM):
    
    States:
      INIT  -> Initial target selection and first path planning
      DELAY -> Robot2 waits briefly for Robot1 to clear the start area
      DRIVE -> Follow A* path toward assigned victim
      RECOVERY -> Reverse + spin to unwedge from collision
      STOP  -> All victims found or mission complete
    
    Features:
      - Pre-loaded map from prepare_mission_plan.py
      - A* path planning with Theta* smoothing
      - Lidar-based dynamic obstacle detection (additive only)
      - Inter-robot victim claiming to prevent duplicate searches
      - Pure Pursuit path following with adaptive speed
    """
    def __init__(self):
        logger.info("Initializing Autonomous SAR Controller...")
        self.hardware = ROSbotHardwareInterface()

        # --- Load starting position from JSON ---
        start_x, start_y = -11.875, -7.125  # Default for robot1
        start_pos_file = os.path.join(os.path.dirname(__file__), "sim_logs", "robot_start_positions.json")
        if os.path.exists(start_pos_file):
            try:
                with open(start_pos_file, 'r') as f:
                    sp_data = json.load(f)
                    if self.hardware.robot_id in sp_data:
                        start_x, start_y = sp_data[self.hardware.robot_id]
                        logger.info(f"Loaded start position for {self.hardware.robot_id}: ({start_x}, {start_y})")
            except Exception as e:
                logger.error(f"Failed to load start positions: {e}")

        # --- Initialize subsystems ---
        self.odometry = CompassOdometry(start_x, start_y)
        self.grid_map = OccupancyGrid()
        self.planner = AStarPlanner(self.grid_map)
        self.pursuit = PurePursuitController()

        # --- Path state ---
        self.current_path: List[Coordinate] = []
        self.path_idx = 0
        self.last_replan_tick = 0
        self.min_replan_interval = 60  # At least ~2s between replans

        # --- FSM state ---
        self.state = "INIT"
        self.tick_counter = 0
        self.delay_counter = 0

        # --- Victim management ---
        self.victims: List[Coordinate] = []
        self.assigned_victim: Optional[Coordinate] = None
        self.visited_victims = set()
        self.claimed_victims = {}  # robot_id -> coordinate
        self.load_victims()

        # --- Recovery state ---
        self.recovery_timer = 0
        self.recovery_direction = 1.0
        self.recovery_count = 0         # Count consecutive recoveries
        self.last_recovery_tick = 0     # When last recovery happened

        # --- Stuck detection watchdog ---
        self.last_progress_pos = (0.0, 0.0)   # Position at last progress check
        self.last_progress_tick = 0            # Tick when progress was last confirmed
        self.stuck_replan_count = 0            # How many times we've replanned due to stuck

        # --- Telemetry ---
        self.path_log: List[Coordinate] = []
        self.telemetry_log_path = os.path.join(os.path.dirname(__file__), "sim_logs", f"{self.hardware.robot_id}_telemetry.csv")
        try:
            os.makedirs(os.path.dirname(self.telemetry_log_path), exist_ok=True)
            with open(self.telemetry_log_path, "w") as f:
                f.write("time,tick,state,pos_x,pos_y,theta,target_x,target_y,dist_to_target,v,omega,fl,fr,min_lidar\n")
        except Exception:
            pass

    def load_victims(self):
        """
        Load victim WORLD coordinates for A* navigation.
        
        Priority:
        1. victim_world_coords.json (world coords, generated by prepare_mission_plan.py)
        2. victim_location_estimates.csv + origin_marker.json offset (fallback)
        
        The CSV contains OriginMarker-relative coords for the scoring pipeline.
        The robot needs WORLD coords for navigation, so we convert if necessary.
        """
        base_dir = os.path.join(os.path.dirname(__file__), "sim_logs")
        
        # Try loading world coordinates directly (preferred)
        world_json = os.path.join(base_dir, "victim_world_coords.json")
        if os.path.exists(world_json):
            try:
                with open(world_json, 'r') as f:
                    coords = json.load(f)
                    for c in coords:
                        self.victims.append((c[0], c[1]))
                logger.info(f"Loaded {len(self.victims)} victims from world_coords JSON.")
                return
            except Exception as e:
                logger.error(f"Failed to load world_coords JSON: {e}")
        
        # Fallback: load CSV (OriginMarker-relative) and add offset
        csv_path = os.path.join(base_dir, "victim_location_estimates.csv")
        origin_path = os.path.join(base_dir, "origin_marker.json")
        
        # Load OriginMarker offset for coordinate conversion
        origin_x, origin_y = 0.0, 0.0
        if os.path.exists(origin_path):
            try:
                with open(origin_path, 'r') as f:
                    om = json.load(f)
                    origin_x, origin_y = om.get("x", 0.0), om.get("y", 0.0)
                    logger.info(f"OriginMarker offset: ({origin_x}, {origin_y})")
            except Exception:
                pass
        
        if not os.path.exists(csv_path):
            logger.warning("No victim data files found!")
            return
        try:
            with open(csv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('x') or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        # CSV is OriginMarker-relative; convert to world coords
                        x = float(parts[0]) + origin_x
                        y = float(parts[1]) + origin_y
                        self.victims.append((x, y))
            logger.info(f"Loaded {len(self.victims)} victims from CSV (with origin offset).")
        except Exception as e:
            logger.error(f"Failed to load victims: {e}")

    def _select_next_victim(self, pose: Pose) -> Optional[Coordinate]:
        """
        Select the nearest unvisited, unclaimed victim.
        Uses sector-based partitioning: Robot1 prefers one half,
        Robot2 prefers the other, to minimize path overlap.
        """
        # Filter out visited and claimed-by-partner victims
        other_claims = {coord for rid, coord in self.claimed_victims.items() if rid != self.hardware.robot_id}
        available = [v for v in self.victims if v not in self.visited_victims and v not in other_claims]

        if not available:
            # Fallback: try any unvisited victim (partner may have failed)
            available = [v for v in self.victims if v not in self.visited_victims]

        if not available:
            return None

        # Sector partitioning: split by median Y
        if len(self.victims) >= 2:
            median_y = float(np.median([v[1] for v in self.victims]))
            is_robot1 = "1" in self.hardware.robot_id
            sector = [v for v in available if (v[1] >= median_y if is_robot1 else v[1] < median_y)]
            if sector:
                available = sector

        # Pick nearest victim by Euclidean distance
        return min(available, key=lambda v: math.hypot(pose[0] - v[0], pose[1] - v[1]))

    def _claim_victim(self, target: Coordinate):
        """Claim a victim target and broadcast to partner robot."""
        self.assigned_victim = target
        self.claimed_victims[self.hardware.robot_id] = target
        self.hardware.send_squad_message({
            "type": "claim_victim",
            "robot_id": self.hardware.robot_id,
            "target": [target[0], target[1]]
        })

    def _broadcast_victim_found(self, victim_id: str, pose: Pose, target: Coordinate):
        """Notify partner robot that a victim has been found.
        NOTE: Do NOT send a score message here — that's already done
        in the DRIVE state when the victim is first confirmed close."""
        self.hardware.send_squad_message({
            "type": "victim_found",
            "robot_id": self.hardware.robot_id,
            "victim_id": victim_id,
            "target": [target[0], target[1]]
        })

    def _process_squad_messages(self):
        """Process incoming messages from partner robot."""
        for msg in self.hardware.receive_squad_messages():
            sender = msg.get("robot_id", "")
            if sender == self.hardware.robot_id:
                continue  # Ignore our own messages

            mtype = msg.get("type")
            if mtype == "victim_found":
                target_coord = msg.get("target")
                if target_coord:
                    self.visited_victims.add((target_coord[0], target_coord[1]))
                logger.info(f"[{self.hardware.get_time():.1f}s][{self.hardware.robot_id}] SQUAD RX: Partner {sender} found {msg.get('victim_id')}")
            elif mtype == "claim_victim":
                target_coord = msg.get("target")
                if target_coord:
                    self.claimed_victims[sender] = (target_coord[0], target_coord[1])
                    logger.info(f"[{self.hardware.get_time():.1f}s][{self.hardware.robot_id}] SQUAD RX: Partner {sender} claimed {target_coord}")

    def _plan_path(self, pose: Pose, target: Coordinate) -> List[Coordinate]:
        """Plan an A* path and log the result."""
        path = self.planner.plan((pose[0], pose[1]), target)
        self.last_replan_tick = self.tick_counter
        path_len = sum(
            math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) 
            for i in range(1, len(path))
        ) if len(path) > 1 else 0.0
        logger.info(f"[{self.hardware.get_time():.1f}s][{self.hardware.robot_id}] A* Planned: {len(path)} WPs ({path_len:.1f}m) -> {target}")
        return path

    def _log_telemetry(self, pose, target, v, omega, fl, fr, min_lidar):
        """Append one row to the telemetry CSV."""
        try:
            t = self.hardware.get_time()
            tx, ty = (target[0], target[1]) if target else (0, 0)
            dist = math.hypot(pose[0]-tx, pose[1]-ty) if target else 0
            with open(self.telemetry_log_path, "a") as f:
                f.write(f"{t:.3f},{self.tick_counter},{self.state},{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},{tx:.3f},{ty:.3f},{dist:.3f},{v:.3f},{omega:.3f},{fl:.3f},{fr:.3f},{min_lidar:.3f}\n")
        except Exception:
            pass

    # ====================
    # MAIN MISSION LOOP
    # ====================
    def execute_mission(self) -> None:
        """Run the FSM mission loop until all victims are found or simulation ends."""
        logger.info("Starting Mission Execution Loop...")

        while self.hardware.step():
            self.tick_counter += 1
            self._process_squad_messages()
            t = self.hardware.get_time()

            # === STATE: INIT ===
            if self.state == "INIT":
                if not self.victims:
                    logger.warning(f"[{t:.1f}s][{self.hardware.robot_id}] No victims loaded! -> STOP")
                    self.state = "STOP"
                    continue

                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                pose = self.odometry.get_pose()

                # Select first target victim
                next_target = self._select_next_victim(pose)
                if not next_target:
                    self.state = "STOP"
                    continue

                self._claim_victim(next_target)

                # **CRITICAL**: Plan path IMMEDIATELY before moving!
                self.current_path = self._plan_path(pose, next_target)
                self.path_idx = 0

                if "1" in self.hardware.robot_id:
                    self.state = "DRIVE"
                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] INIT -> DRIVE | Target: {next_target}")
                else:
                    self.state = "DELAY"
                    self.delay_counter = 0
                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] INIT -> DELAY | Target: {next_target}")

            # === STATE: DELAY (robot2 waits for robot1 to clear start area) ===
            elif self.state == "DELAY":
                self.hardware.set_motor_speeds(0.0, 0.0)
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                self.delay_counter += 1
                if self.delay_counter > 100:  # ~3.2 seconds
                    pose = self.odometry.get_pose()
                    # Replan since we've been waiting
                    self.current_path = self._plan_path(pose, self.assigned_victim)
                    self.path_idx = 0
                    self.state = "DRIVE"
                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] DELAY -> DRIVE")

            # === STATE: DRIVE (follow A* path toward victim) ===
            elif self.state == "DRIVE":
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                pose = self.odometry.get_pose()
                target = self.assigned_victim

                if not target:
                    next_target = self._select_next_victim(pose)
                    if next_target:
                        self._claim_victim(next_target)
                        self.current_path = self._plan_path(pose, next_target)
                        self.path_idx = 0
                        target = self.assigned_victim
                    else:
                        self.state = "STOP"
                        continue

                dist_to_target = math.hypot(pose[0] - target[0], pose[1] - target[1])

                # Log path for post-analysis
                if self.tick_counter % 5 == 0:
                    self.path_log.append((pose[0], pose[1]))

                # Read sensors
                fl, fr = self.hardware.read_front_distances()
                lidar = self.hardware.read_lidar()
                min_lidar = min(lidar) if lidar else 2.0

                # ==========================================================
                # VICTIM DETECTION & SCORING
                # Strategy:
                # - Send periodic STATUS messages (victim_found=False) as heartbeat
                # - Send ONE definitive SCORE message (victim_found=True) when
                #   within 0.7m (by odometry), giving margin for odometry drift
                # - Keep driving toward victim until 0.3m then stop and advance
                #
                # CONFIDENCE SCORING: score = correct_verdicts / total_victim_found
                # Sending fewer, more accurate victim_found=True messages → higher confidence
                # ==========================================================
                
                # Send periodic status heartbeat (every 100 ticks ≈ 3.2s)
                # These have victim_found=False so they DON'T hurt confidence
                if self.tick_counter % 100 == 0:
                    self.hardware.send_status_message(
                        self.hardware.robot_id,
                        [pose[0], pose[1]]
                    )
                
                # Victim approach zone: within 0.7m by odometry
                if dist_to_target < 0.7:
                    # Send exactly ONE victim_found=True score message
                    # Only send if we haven't already scored this victim
                    if target not in self.visited_victims:
                        self.hardware.send_score_message(
                            self.hardware.robot_id,
                            [target[0], target[1], 1.0]
                        )
                        logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] SCORE MSG SENT for ({target[0]:.2f},{target[1]:.2f}) | Dist: {dist_to_target:.2f}m")

                # Stop and advance to next victim when very close
                if dist_to_target < 0.3:
                    self.hardware.set_motor_speeds(0.0, 0.0)
                    self.visited_victims.add(target)

                    victim_idx = self.victims.index(target) + 1 if target in self.victims else len(self.visited_victims)
                    self._broadcast_victim_found(f"victim{victim_idx}", pose, target)
                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] VICTIM SCORED! victim{victim_idx} at ({target[0]:.2f},{target[1]:.2f}) | Dist: {dist_to_target:.2f}m")

                    # Select next victim
                    next_target = self._select_next_victim(pose)
                    if next_target:
                        self._claim_victim(next_target)
                        self.current_path = self._plan_path(pose, next_target)
                        self.path_idx = 0
                        self.stuck_replan_count = 0
                        self.recovery_count = 0
                        logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] Next target: {next_target}")
                    else:
                        logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] All victims found! -> STOP")
                        self.state = "STOP"
                    continue

                # ==========================================================
                # STUCK DETECTION WATCHDOG
                # If the robot hasn't moved > 0.15m in the last 80 ticks
                # (~2.5 seconds), it's stuck. Force a recovery maneuver
                # (reverse + spin) and replan from the new position.
                # ==========================================================
                if self.tick_counter - self.last_progress_tick > 80:
                    moved = math.hypot(
                        pose[0] - self.last_progress_pos[0],
                        pose[1] - self.last_progress_pos[1]
                    )
                    if moved < 0.15:
                        # STUCK! Force recovery
                        self.stuck_replan_count += 1
                        logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] STUCK DETECTED! Moved only {moved:.2f}m in 80 ticks. Forcing recovery #{self.stuck_replan_count}")

                        # Alternate spin direction each time to avoid repeating the same dead end
                        self.recovery_direction = 1.0 if (self.stuck_replan_count % 2 == 0) else -1.0
                        self.recovery_timer = 35  # Longer recovery: more reverse + spin
                        self.state = "RECOVERY"
                        continue
                    else:
                        # Made progress — reset watchdog
                        self.last_progress_pos = (pose[0], pose[1])
                        self.last_progress_tick = self.tick_counter

                # ==========================================================
                # HARD COLLISION RECOVERY (IR sensors)
                # Triggers when robot is physically pressed against a wall.
                # SKIP when close to the victim target.
                # ==========================================================
                if (fl < 0.10 or fr < 0.10) and dist_to_target > 1.5:
                    # Check for recovery loop: too many recoveries in a short time
                    if self.tick_counter - self.last_recovery_tick < 60:
                        self.recovery_count += 1
                    else:
                        self.recovery_count = 1
                    self.last_recovery_tick = self.tick_counter

                    self.state = "RECOVERY"
                    # Increase recovery time if we're in a loop
                    self.recovery_timer = 25 + min(self.recovery_count * 10, 30)

                    # Determine spin direction: toward the A* path waypoint
                    if self.current_path and self.path_idx < len(self.current_path):
                        wx, wy = self.current_path[self.path_idx]
                        alpha = math.atan2(wy - pose[1], wx - pose[0]) - pose[2]
                        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
                        self.recovery_direction = 1.0 if alpha > 0 else -1.0
                    else:
                        self.recovery_direction = 1.0 if fl < fr else -1.0

                    # If in a recovery loop, alternate direction
                    if self.recovery_count > 2:
                        self.recovery_direction *= -1

                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] -> RECOVERY #{self.recovery_count} | fl={fl:.2f} fr={fr:.2f} | timer={self.recovery_timer}")
                    continue

                # --- Lidar map update (every 20 ticks ≈ 0.64s, after initial settling) ---
                if self.tick_counter > 30 and self.tick_counter % 20 == 0:
                    new_obstacles = self.grid_map.update_from_lidar(pose, lidar)

                    # Check if current path is still valid
                    if self.current_path and new_obstacles:
                        if self.grid_map.is_path_blocked(self.current_path, self.path_idx):
                            self.current_path = self._plan_path(pose, target)
                            self.path_idx = 0

                # --- Ensure we have a path ---
                if not self.current_path:
                    self.current_path = self._plan_path(pose, target)
                    self.path_idx = 0

                # --- Follow the path using Pure Pursuit ---
                v_cmd, omega_cmd = 0.0, 0.0
                if self.current_path and self.path_idx < len(self.current_path):
                    waypoint = self.current_path[self.path_idx]
                    w_dist = math.hypot(pose[0] - waypoint[0], pose[1] - waypoint[1])

                    # Advance to next waypoint if close enough
                    if w_dist < 0.15:
                        self.path_idx += 1

                    if self.path_idx < len(self.current_path):
                        waypoint = self.current_path[self.path_idx]
                        v_cmd, omega_cmd = self.pursuit.get_velocity(
                            pose, waypoint, fl, fr, lidar, dist_to_target)
                    else:
                        # Path completed, drive directly to target
                        v_cmd, omega_cmd = self.pursuit.get_velocity(
                            pose, target, fl, fr, lidar, dist_to_target)
                else:
                    # No path available, drive toward target
                    v_cmd, omega_cmd = self.pursuit.get_velocity(
                        pose, target, fl, fr, lidar, dist_to_target)

                self.hardware.set_motor_speeds(v_cmd, omega_cmd)

                # --- Periodic console log ---
                if self.tick_counter % 50 == 0:
                    wp = self.path_idx
                    total = len(self.current_path)
                    logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] WP {wp}/{total} | Pos: ({pose[0]:.2f},{pose[1]:.2f}) | Target: ({target[0]:.2f},{target[1]:.2f}) | Dist: {dist_to_target:.2f}m | v={v_cmd:.2f}")

                # Telemetry
                if self.tick_counter % 5 == 0:
                    self._log_telemetry(pose, target, v_cmd, omega_cmd, fl, fr, min_lidar)

            # === STATE: RECOVERY (reverse + spin to unwedge from collision) ===
            elif self.state == "RECOVERY":
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                self.recovery_timer -= 1

                if self.recovery_timer > 0:
                    if self.recovery_timer > 15:
                        # Phase 1: Reverse straight (clears the obstacle)
                        self.hardware.set_motor_speeds(-0.40, 0.0)
                    else:
                        # Phase 2: Spin toward path direction
                        self.hardware.set_motor_speeds(0.0, 3.5 * self.recovery_direction)
                else:
                    # Recovery done: replan from new position and resume driving
                    self.state = "DRIVE"
                    pose = self.odometry.get_pose()
                    # Reset the stuck watchdog so we don't immediately re-trigger
                    self.last_progress_pos = (pose[0], pose[1])
                    self.last_progress_tick = self.tick_counter
                    if self.assigned_victim:
                        self.current_path = self._plan_path(pose, self.assigned_victim)
                        self.path_idx = 0

            # === STATE: STOP ===
            elif self.state == "STOP":
                self.hardware.set_motor_speeds(0.0, 0.0)
                # Save path log on first STOP tick
                if self.path_log:
                    try:
                        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "log")
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, f"{self.hardware.robot_id}_path_log.csv")
                        with open(log_file, "w") as f:
                            f.write("x,y\n")
                            for px, py in self.path_log:
                                f.write(f"{px:.3f},{py:.3f}\n")
                        self.path_log = []  # Clear so we don't write again
                    except Exception:
                        pass


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    try:
        controller = AutonomousSARController()
        controller.execute_mission()
    except KeyboardInterrupt:
        logger.warning("Mission aborted by user.")
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        import traceback
        traceback.print_exc()
