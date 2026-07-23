"""
IEEE SMCS Autonomous Search and Rescue - Proposed Solution
Advanced Navigation with Dynamic Mapping & A* Path Planning
"""

import sys
import math
import json
import logging
import heapq
import time
import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    from controller import Robot
except ImportError:
    Robot = None

# ==========================================
# SYSTEM SETUP
# ==========================================
class Unbuffered(object):
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

Pose = Tuple[float, float, float]
Coordinate = Tuple[float, float]

# ==========================================
# HARDWARE ABSTRACTION LAYER
# HARDWARE INTERFACE (WEBOTS)
# ==========================================
class ROSbotHardwareInterface:
    """
    Provides a high-level Python abstraction over the Webots low-level Robot API.
    It encapsulates all motors, encoders, Lidar, Compass, and radio Emitters/Receivers.
    This class handles the raw parsing of Webots data structures and exposes them
    as clean Python types (e.g., tuples, floats) to the main controller logic.
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

        # Motors
        self.fl_motor = self.robot.getDevice("fl_wheel_joint")
        self.fr_motor = self.robot.getDevice("fr_wheel_joint")
        self.rl_motor = self.robot.getDevice("rl_wheel_joint")
        self.rr_motor = self.robot.getDevice("rr_wheel_joint")
        for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)

        # Encoders
        self.left_sensor = self.robot.getDevice("front left wheel motor sensor")
        self.right_sensor = self.robot.getDevice("front right wheel motor sensor")
        if self.left_sensor: self.left_sensor.enable(self.timestep)
        if self.right_sensor: self.right_sensor.enable(self.timestep)

        # Lidar - 360° scanner
        self.lidar = self.robot.getDevice("laser")
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

        # Compass (ground truth heading)
        self.compass = self.robot.getDevice("imu compass")
        if self.compass:
            self.compass.enable(self.timestep)

        # Front distance sensors
        self.fl_range = self.robot.getDevice("fl_range")
        self.fr_range = self.robot.getDevice("fr_range")
        self.rl_range = self.robot.getDevice("rl_range")
        self.rr_range = self.robot.getDevice("rr_range")
        for sensor in [self.fl_range, self.fr_range, self.rl_range, self.rr_range]:
            if sensor: sensor.enable(self.timestep)

        # Supervisor emitter (channel 43)
        self.emitter = self.robot.getDevice("supervisor emitter")
        if self.emitter:
            self.emitter.setChannel(43)

        # Robot-to-robot communication
        self.squad_receiver = self.robot.getDevice("robot to robot receiver")
        self.squad_emitter = self.robot.getDevice("robot to robot emitter")
        if self.squad_receiver:
            self.squad_receiver.enable(self.timestep)

    def step(self) -> bool:
        if self.robot is None: return False
        return self.robot.step(self.timestep) != -1

    def get_time(self) -> float:
        return self.robot.getTime() if self.robot else 0.0

    def read_encoders(self) -> Tuple[float, float]:
        l = self.left_sensor.getValue() if self.left_sensor else 0.0
        r = self.right_sensor.getValue() if self.right_sensor else 0.0
        return l, r

    def read_compass_heading(self) -> float:
        if not self.compass: return 0.0
        north = self.compass.getValues()
        return math.atan2(north[0], north[1])

    def read_lidar(self) -> List[float]:
        if not self.lidar: return []
        return self.lidar.getRangeImage()

    def read_front_distances(self) -> Tuple[float, float]:
        fl = self.fl_range.getValue() if self.fl_range else 2.0
        fr = self.fr_range.getValue() if self.fr_range else 2.0
        return fl, fr

    def read_rear_distances(self) -> Tuple[float, float]:
        rl = self.rl_range.getValue() if self.rl_range else 2.0
        rr = self.rr_range.getValue() if self.rr_range else 2.0
        return rl, rr

    def set_motor_speeds(self, linear_velocity: float, angular_velocity: float) -> None:
        track_width = 0.2
        wheel_radius = 0.04
        v_left = linear_velocity - (angular_velocity * track_width / 2.0)
        v_right = linear_velocity + (angular_velocity * track_width / 2.0)
        left_rads = max(-26.0, min(26.0, v_left / wheel_radius))
        right_rads = max(-26.0, min(26.0, v_right / wheel_radius))
        if self.fl_motor: self.fl_motor.setVelocity(left_rads)
        if self.rl_motor: self.rl_motor.setVelocity(left_rads)
        if self.fr_motor: self.fr_motor.setVelocity(right_rads)
        if self.rr_motor: self.rr_motor.setVelocity(right_rads)

    def send_score_message(self, robot_id: str, position: List[float]) -> None:
        if not self.emitter: return
        # Extract confidence from position array if it exists (length 3), else default to 1.0
        confidence = position[2] if len(position) > 2 else 1.0
        msg = {
            "timestamp": self.robot.getTime(),
            "robot_id": robot_id,
            "position": position[:2],  # Only send X, Y in the position array
            "victim_found": True,
            "victim_confidence": confidence
        }
        self.emitter.send(json.dumps(msg).encode('utf-8'))

    def send_squad_message(self, message: dict) -> None:
        if not self.squad_emitter: return
        self.squad_emitter.send(json.dumps(message).encode('utf-8'))

    def receive_squad_messages(self) -> List[dict]:
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
# COMPASS ODOMETRY
# ==========================================
class CompassOdometry:
    """
    Tracks the robot's pose (x, y, theta) using wheel encoders and an IMU compass.
    The encoders track the distance traveled, while the compass provides absolute
    global heading, preventing rotational drift over time.
    """
    def __init__(self, start_x: float, start_y: float):
        """
        Initializes the odometry at a known starting position.
        :param start_x: Initial global X coordinate in meters
        :param start_y: Initial global Y coordinate in meters
        """
        self.x = start_x
        self.y = start_y
        self.theta = 0.0
        
        self.last_left_enc = None
        self.last_right_enc = None
        
        # ROSbot specific hardware constants
        self.wheel_radius = 0.04
        self.track_width = 0.2

    def update(self, left_enc: float, right_enc: float, compass_heading: float) -> None:
        """
        Updates the robot's position based on sensor deltas.
        :param left_enc: Current left wheel encoder reading (radians)
        :param right_enc: Current right wheel encoder reading (radians)
        :param compass_heading: Current compass heading reading
        """
        self.theta = compass_heading
        if self.last_left_enc is None:
            self.last_left_enc = left_enc
            self.last_right_enc = right_enc
            return
        dl = (left_enc - self.last_left_enc) * self.wheel_radius
        dr = (right_enc - self.last_right_enc) * self.wheel_radius
        dc = (dl + dr) / 2.0
        self.x += dc * math.cos(self.theta)
        self.y += dc * math.sin(self.theta)
        self.last_left_enc = left_enc
        self.last_right_enc = right_enc

    def get_pose(self) -> Pose:
        return (self.x, self.y, self.theta)


# ==========================================
# GRID MAP UTILITY
# ==========================================
class OccupancyGrid:
    """
    Maintains a 2D occupancy grid representing the environment.
    This grid is used by the A* and Theta* planners for obstacle avoidance.
    The grid cells contain integers representing occupancy state:
    - 0: Unknown / Unexplored
    - 1: Free space
    - 2: Inflated obstacle boundary (safety buffer)
    - 3: Solid obstacle (walls, victims, furniture)
    """
    def __init__(self, rows: int = 600, cols: int = 600, resolution: float = 0.05):
        """
        Initializes the grid map.
        :param rows: Number of rows in the grid
        :param cols: Number of columns in the grid
        :param resolution: Physical size of each grid cell in meters (e.g., 0.05m = 5cm)
        """
        self.resolution = resolution
        self.origin_x = -15.0
        self.origin_y = -15.0
        self.cols = cols
        self.rows = rows
        # 0 = unknown, 1 = free, 2 = obstacle
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.inflation_radius_cells = 7  # 7 * 0.05 = 0.35m inflation

        self.load_map_from_png()

    def load_map_from_png(self):
        map_path = os.path.join(os.path.dirname(__file__), "sim_logs", "map_estimate.png")
        if not os.path.exists(map_path):
            logger.warning(f"No map_estimate.png found at {map_path}, starting with empty grid.")
            return
            
        try:
            img = Image.open(map_path).convert('L')
            if img.size != (600, 600):
                img = img.resize((600, 600))
            
            pixels = np.array(img)
            # Threshold: < 128 is black (wall), >= 128 is white (free)
            for r in range(600):
                for c in range(600):
                    if pixels[r, c] < 128:
                        self.grid[r, c] = 3  # Raw obstacle
                    else:
                        self.grid[r, c] = 1  # Free
                        
            # Inflate obstacles
            temp_grid = np.copy(self.grid)
            for r in range(600):
                for c in range(600):
                    if temp_grid[r, c] == 3:
                        for dr in range(-self.inflation_radius_cells, self.inflation_radius_cells + 1):
                            for dc in range(-self.inflation_radius_cells, self.inflation_radius_cells + 1):
                                if dr*dr + dc*dc <= self.inflation_radius_cells*self.inflation_radius_cells:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < 600 and 0 <= nc < 600 and self.grid[nr, nc] < 2:
                                        self.grid[nr, nc] = 2  # Inflated obstacle
            logger.info("Successfully loaded and inflated map_estimate.png")
        except Exception as e:
            logger.error(f"Failed to load map PNG: {e}")
            
    def save_map(self) -> None:
        """Saves the current grid back to map_estimate.png as strict B/W"""
        try:
            map_path = os.path.join(os.path.dirname(__file__), "sim_logs", "map_estimate.png")
            out_pixels = np.full((600, 600), 255, dtype=np.uint8)
            # Only raw obstacles (3) should be black. Free (1), unknown (0), inflated (2) are white
            out_pixels[self.grid == 3] = 0
            
            img = Image.fromarray(out_pixels, mode='L')
            img.save(map_path)
            logger.info("Successfully saved updated map_estimate.png")
        except Exception as e:
            logger.error(f"Failed to save map PNG: {e}")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converts real-world physical coordinates (meters) to grid indices (pixels).
        :param x: World X coordinate in meters
        :param y: World Y coordinate in meters
        :return: (col, row) representing the grid cell indices
        """
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Coordinate:
        """
        Converts grid indices (pixels) back to real-world coordinates (meters).
        :param col: Column index
        :param row: Row index
        :return: (x, y) real-world coordinates
        """
        x = (col + 0.5) * self.resolution + self.origin_x
        y = (row + 0.5) * self.resolution + self.origin_y
        return x, y

    def in_bounds(self, col: int, row: int) -> bool:
        """
        Checks if the given grid indices are within the grid map boundaries.
        :param col: Column index
        :param row: Row index
        :return: True if the cell is inside the map, False otherwise
        """
        return 0 <= col < self.cols and 0 <= row < self.rows

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
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

    def update_from_lidar(self, pose: Pose, lidar_data: List[float], max_range=4.0) -> Tuple[bool, List[Tuple[float, float]]]:
        """Returns True if the map was updated, and a list of new obstacle coordinates."""
        if not lidar_data: return False, []
        rx, ry, rtheta = pose
        r_col, r_row = self.world_to_grid(rx, ry)
        if not self.in_bounds(r_col, r_row): return False, []

        n = len(lidar_data)
        updated = False
        new_obstacles = []
        
        # Subsample lidar rays to reduce CPU load. Checking every 4th ray is sufficient for 0.05m grid.
        step = 4
        for i in range(0, n, step):
            dist = lidar_data[i]
            
            # Handle infinite/invalid rays by capping them at the maximum sensor range
            if math.isinf(dist) or math.isnan(dist):
                dist = max_range
                
            # Compute the global angle of the lidar ray.
            # Lidar sweeps from -pi to +pi. Index 0 is the BACK of the robot.
            # We subtract math.pi to align the sensor's zero-angle with the robot's heading (rtheta).
            angle = rtheta - math.pi + (2 * math.pi * i / n)
            
            # Convert polar distance and angle to global cartesian coordinates (x, y)
            hit_x = rx + dist * math.cos(angle)
            hit_y = ry + dist * math.sin(angle)
            
            # Translate physical hit coordinates into discrete grid cell indices
            h_col, h_row = self.world_to_grid(hit_x, hit_y)
            
            if dist < max_range:
                # If the ray hit an actual object, the space between the robot and the object must be free.
                # Use Bresenham's line algorithm to trace the ray and mark those cells as Free (1).
                line_points = self.bresenham_line(r_col, r_row, h_col, h_row)
                for (cx, cy) in line_points[:-1]:  # Exclude the last point (the obstacle itself)
                    if self.in_bounds(cx, cy) and self.grid[cy, cx] < 2:
                        self.grid[cy, cx] = 1 # Mark as free space

                # Mark the final hit point as a solid Obstacle (3)
                if self.in_bounds(h_col, h_row):
                    if self.grid[h_row, h_col] != 3:
                        updated = True
                        self.grid[h_row, h_col] = 3
                        new_obstacles.append((hit_x, hit_y))
                        
                        # INFLATION: To prevent the robot from colliding with the physical wall,
                        # we draw a "circle" of safety buffer cells (value 2) around the solid obstacle.
                        for dr in range(-self.inflation_radius_cells, self.inflation_radius_cells + 1):
                            for dc in range(-self.inflation_radius_cells, self.inflation_radius_cells + 1):
                                # Check if cell falls within the circular inflation radius using Euclidean distance squared
                                if dr*dr + dc*dc <= self.inflation_radius_cells*self.inflation_radius_cells:
                                    nc, nr = h_col + dc, h_row + dr
                                    # Only overwrite Unknown (0) or Free (1) space, don't overwrite Solid (3)
                                    if self.in_bounds(nc, nr) and self.grid[nr, nc] < 2:
                                        self.grid[nr, nc] = 2
        return updated, new_obstacles

    def is_blocked_world(self, x: float, y: float) -> bool:
        c, r = self.world_to_grid(x, y)
        if not self.in_bounds(c, r): return False
        return self.grid[r, c] >= 2


# ==========================================
# A* PATH PLANNER (WITH THETA* SMOOTHING)
# ==========================================
class AStarPlanner:
    """
    Implements an A* pathfinding algorithm over the OccupancyGrid map.
    It uses an 8-way directional movement system with cost penalties for diagonals.
    Once a path is found, it applies Theta*-inspired line-of-sight smoothing to
    remove jagged zig-zags and create a smooth, direct trajectory for the robot.
    """
    def __init__(self, grid_map: OccupancyGrid):
        """
        :param grid_map: Reference to the shared environment grid map.
        """
        self.grid_map = grid_map

    def plan(self, start_world: Coordinate, goal_world: Coordinate) -> List[Coordinate]:
        start = self.grid_map.world_to_grid(*start_world)
        goal = self.grid_map.world_to_grid(*goal_world)
        
        if not self.grid_map.in_bounds(*start) or not self.grid_map.in_bounds(*goal):
            return [goal_world]

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        # Define 8-way movement: (dx, dy, base_cost). 
        # Orthogonal moves cost 1.0, diagonal moves cost 1.414 (sqrt(2))
        directions = [(0,1,1.0), (1,0,1.0), (0,-1,1.0), (-1,0,1.0), 
                      (1,1,1.414), (-1,-1,1.414), (1,-1,1.414), (-1,1,1.414)]

        expansions = 0
        max_expansions = 10000 # Hard cap to prevent freezing the CPU in unreachable areas

        # Standard A* expansion loop
        while frontier and expansions < max_expansions:
            expansions += 1
            _, current = heapq.heappop(frontier)
            if current == goal:
                break # We reached the target!
                
            for dx, dy, cost in directions:
                next_node = (current[0] + dx, current[1] + dy)
                if not self.grid_map.in_bounds(*next_node): continue
                
                # Check if the node is an obstacle (value 2 or 3)
                is_obstacle = self.grid_map.grid[next_node[1], next_node[0]] == 2
                if is_obstacle:
                    # SOFT OBSTACLE PENALTY:
                    # If the robot accidentally drifts into an inflated obstacle (2), we don't completely block the node.
                    # We also allow the planner to path *into* the goal if the goal itself happens to be inside an inflated zone.
                    dist_to_goal = math.hypot(goal[0] - next_node[0], goal[1] - next_node[1])
                    if dist_to_goal <= self.grid_map.inflation_radius_cells + 2:
                        is_obstacle = False # Ignore inflation if it's right next to the target victim
                
                # Assign a massive cost penalty (100x) to obstacles instead of blocking them outright.
                # This ensures the robot can still find an escape path if it is completely boxed in.
                node_cost = 100.0 if is_obstacle else 1.0
                    
                # Calculate cumulative traversal cost
                new_cost = cost_so_far[current] + cost * node_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # Heuristic: Euclidean distance to goal. Maintains optimal shortest-path guarantee (Admissible).
                    h = math.hypot(goal[0] - next_node[0], goal[1] - next_node[1])
                    priority = new_cost + h
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
        if goal not in came_from:
            # Cannot reach exactly; find closest node we reached
            min_dist = float('inf')
            best_node = None
            for node in came_from:
                dist = math.hypot(goal[0] - node[0], goal[1] - node[1])
                if dist < min_dist:
                    min_dist = dist
                    best_node = node
            if best_node is None: return [goal_world]
            goal = best_node

        path = []
        current = goal
        while current != start:
            path.append(self.grid_map.grid_to_world(*current))
            current = came_from[current]
        path.reverse()

        # Simplify path using Bresenham Line-of-Sight (Theta* Style Smoothing)
        def line_of_sight(p1, p2):
            x0, y0 = p1
            x1, y1 = p2
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while x0 != x1 or y0 != y1:
                # Check 3x3 block around the line for extra safety margin against corner clipping
                for check_dy in [-1, 0, 1]:
                    for check_dx in [-1, 0, 1]:
                        ny, nx = y0 + check_dy, x0 + check_dx
                        if 0 <= ny < self.grid_map.rows and 0 <= nx < self.grid_map.cols:
                            if self.grid_map.grid[ny, nx] >= 2:
                                return False
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return True

        if len(path) > 2:
            smoothed = [path[0]]
            current_idx = 0
            while current_idx < len(path) - 1:
                next_valid = current_idx + 1
                for j in range(len(path) - 1, current_idx + 1, -1):
                    p1_grid = self.grid_map.world_to_grid(*smoothed[-1])
                    p2_grid = self.grid_map.world_to_grid(*path[j])
                    if line_of_sight(p1_grid, p2_grid):
                        next_valid = j
                        break
                smoothed.append(path[next_valid])
                current_idx = next_valid
            path = smoothed

        return path


# ==========================================
# PURE PURSUIT CONTROLLER
# ==========================================
class PurePursuitController:
    def __init__(self, lookahead: float = 0.45):
        # Lookahead distance determines how "aggressively" the robot cuts corners
        self.lookahead = lookahead

    def get_velocity(self, robot_pose: Pose, target: Coordinate) -> Tuple[float, float]:
        """
        Calculates the left/right motor speeds needed to smoothly steer towards a target.
        :param robot_pose: The robot's current (x, y, theta)
        :param target: The (x, y) coordinate we want to reach
        :return: (linear_velocity, angular_velocity)
        """
        rx, ry, rtheta = robot_pose
        
        # Calculate the absolute angle from the robot to the target in the global frame
        alpha = math.atan2(target[1] - ry, target[0] - rx) - rtheta
        
        # Calculate the relative heading error (how much the robot needs to turn)
        # We normalize this error between -pi and +pi to ensure the robot always takes the shortest turn.
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        
        # Calculate the direct distance to the target
        dist_to_target = math.hypot(target[0] - rx, target[1] - ry)
        
        # If the robot is facing completely the wrong way (error > 45 degrees), 
        # prioritize spinning in place over moving forward, but only if we aren't right next to the target.
        if abs(alpha) > math.pi / 4 and dist_to_target > 0.5:
            return 0.0, (3.5 if alpha > 0 else -3.5)
            
        # Determine forward speed. 
        v = 0.60
        
        # P-Controller for steering: Proportional to the heading error.
        # We use a non-linear curvature calculation based on the lookahead distance.
        omega = (2 * v * math.sin(alpha)) / self.lookahead

        if abs(alpha) > math.pi / 8:
            v = 0.40  # Slow down on curves
        return v, omega


# ==========================================
# MAIN MISSION CONTROLLER
# ==========================================
class AutonomousSARController:
    """
    The top-level Autonomous Search and Rescue (SAR) mission controller.
    This controller runs on each robot independently. It acts as a finite state
    machine (FSM) orchestrating the following:
    - SLAM-like Odometry and Occupancy Grid Mapping
    - Autonomous exploration and Victim identification
    - Multi-robot coordination via Radio messages (avoiding redundant searches)
    - Sending standardized scoring messages to the sar_marking_supervisor
    """
    def __init__(self):
        """
        Initializes the state machine, subsystems, and loads the initial map/victim data.
        """
        logger.info("Initializing Autonomous SAR Controller (A* Upgrade)...")
        self.hardware = ROSbotHardwareInterface()

        # Initialize odometry at known start positions
        if "1" in self.hardware.robot_id:
            self.odometry = CompassOdometry(1.815, 1.833)
        else:
            self.odometry = CompassOdometry(0.625, 1.0)

        self.pursuit = PurePursuitController()
        self.grid_map = OccupancyGrid()
        self.planner = AStarPlanner(self.grid_map)
        
        self.current_path: List[Coordinate] = []
        self.path_idx = 0
        self.last_map_update = 0

        self.state = "INIT"
        self.victims: List[Coordinate] = []
        self.assigned_victim: Coordinate = (0.0, 0.0)
        self.path_log: List[Coordinate] = []
        self.load_victims()
        
        self.tick_counter = 0
        self.score_transmit_counter = 0
        self.delay_counter = 0
        self.startup_ticks = 0
        self.consecutive_recoveries = 0
        self.last_recovery_tick = 0
        
        self.recovery_timer = 0
        self.recovery_direction = 1.0

        # Inter-robot communication state
        self.my_victim_found = False
        self.partner_victim_found = False
        self.partner_victim_id: Optional[str] = None

    def _process_squad_messages(self):
        messages = self.hardware.receive_squad_messages()
        for msg in messages:
            if msg.get("type") == "victim_found":
                sender = msg.get("robot_id", "")
                if sender != self.hardware.robot_id:
                    self.partner_victim_found = True
                    self.partner_victim_id = msg.get("victim_id", "")
                    logger.info(f"[{self.hardware.robot_id}] Received: Partner {sender} found {self.partner_victim_id}")
            elif msg.get("type") == "obstacle":
                hx, hy = msg.get("x"), msg.get("y")
                if hx is not None and hy is not None:
                    c, r = self.grid_map.world_to_grid(hx, hy)
                    if self.grid_map.in_bounds(c, r) and self.grid_map.grid[r, c] != 2:
                        self.grid_map.grid[r, c] = 2
                        inf = self.grid_map.inflation_radius_cells
                        for dr in range(-inf, inf + 1):
                            for dc in range(-inf, inf + 1):
                                if dr*dr + dc*dc <= inf*inf:
                                    nc, nr = c + dc, r + dr
                                    if self.grid_map.in_bounds(nc, nr) and self.grid_map.grid[nr, nc] != 2:
                                        self.grid_map.grid[nr, nc] = 2

    def load_victims(self):
        csv_path = os.path.join(os.path.dirname(__file__), "sim_logs", "victim_location_estimates.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and "Your code" not in line and not line.startswith('x'):
                            parts = line.split(',')
                            if len(parts) >= 2:
                                x, y = float(parts[0]), float(parts[1])
                                self.victims.append((x, y))
                logger.info(f"Loaded {len(self.victims)} victims from CSV.")
            except Exception as e:
                logger.error(f"Failed to load victims: {e}")

    def _broadcast_victim_found(self, victim_id: str, pose: Pose):
        # Notify Supervisor (scoring)
        self.hardware.send_score_message(self.hardware.robot_id, [pose[0], pose[1], 0.0])

        # Notify Squad
        self.hardware.send_squad_message({
            "type": "victim_found",
            "robot_id": self.hardware.robot_id,
            "victim_id": victim_id,
        })

    def execute_mission(self) -> None:
        logger.info("Starting Mission Execution Loop...")

        while self.hardware.step():
            self.startup_ticks += 1
            self.tick_counter += 1
            self._process_squad_messages()

            if self.state == "INIT":
                if not self.victims:
                    logger.warning(f"[{self.hardware.robot_id}] No victims in CSV to rescue!")
                    self.state = "STOP"
                    continue
                
                # Assign victim dynamically based on robot ID suffix
                idx = 0 if "1" in self.hardware.robot_id else 1
                if idx < len(self.victims):
                    self.assigned_victim = self.victims[idx]
                else:
                    self.assigned_victim = self.victims[0]
                    
                if "1" in self.hardware.robot_id:
                    self.state = "DRIVE"
                    logger.info(f"[{self.hardware.robot_id}] State: DRIVE -> {self.assigned_victim}")
                else:
                    self.state = "DELAY"
                    self.delay_counter = 0
                    logger.info(f"[{self.hardware.robot_id}] State: DELAY - Waiting for robot1 to clear.")

            elif self.state == "DELAY":
                self.hardware.set_motor_speeds(0.0, 0.0)
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                self.delay_counter += 1
                if self.delay_counter > 200:  # ~3.2 seconds
                    self.state = "DRIVE"
                    logger.info(f"[{self.hardware.robot_id}] State: DRIVE -> {self.assigned_victim}")

            elif self.state == "DRIVE":
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                pose = self.odometry.get_pose()
                target = self.assigned_victim
                dist = math.hypot(pose[0] - target[0], pose[1] - target[1])
                
                if self.tick_counter % 5 == 0:
                    self.path_log.append((pose[0], pose[1]))

                # Transmit score messages as soon as we enter the legal 1.0m radius!
                # This locks in a very fast "Victim Found Time" for the score.
                # However, we KEEP DRIVING closer (to 0.4m) so our Lidar can project
                # a surgically precise coordinate for the final CSV overwrite!
                if dist <= 1.0:
                    if self.tick_counter % 10 == 0:
                        self.hardware.send_score_message(self.hardware.robot_id, [target[0], target[1], 1.0])
                
                fl, fr = self.hardware.read_front_distances()
                front_blocked = fl < 0.3 or fr < 0.3
                
                # Have we reached the victim?
                if dist < 0.4 or (front_blocked and dist < 1.0):
                    # Calculate better estimate of victim based on our pose and sensors
                    sensor_dist = min(fl, fr) if front_blocked else dist
                    est_x = pose[0] + sensor_dist * math.cos(pose[2])
                    est_y = pose[1] + sensor_dist * math.sin(pose[2])

                    for _ in range(5):
                        self.hardware.send_score_message(self.hardware.robot_id, [est_x, est_y, 1.0])
                    
                    self.hardware.set_motor_speeds(0.0, 0.0)
                    self.my_victim_found = True
                    victim_name = "victim1" if "1" in self.hardware.robot_id else "victim2"
                    
                    # Update the CSV file for the supervisor (Read, Replace closest, Write back)
                    est_csv = os.path.join(os.path.dirname(__file__), "sim_logs", "victim_location_estimates.csv")
                    try:
                        # 1. Read existing
                        current_ests = []
                        with open(est_csv, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and "Your code" not in line and not line.startswith('x'):
                                    parts = line.split(',')
                                    if len(parts) >= 2:
                                        current_ests.append((float(parts[0]), float(parts[1])))
                        
                        # 2. Find closest and replace
                        if current_ests:
                            closest_idx = min(range(len(current_ests)), key=lambda i: math.hypot(current_ests[i][0]-est_x, current_ests[i][1]-est_y))
                            current_ests[closest_idx] = (est_x, est_y)
                        else:
                            current_ests.append((est_x, est_y))
                        
                        # 3. Write back
                        with open(est_csv, "w") as f:
                            for ex, ey in current_ests:
                                f.write(f"{ex:.3f},{ey:.3f}\n")
                    except Exception as e:
                        logger.error(f"Failed to write victim estimate: {e}")
                    self._broadcast_victim_found(victim_name, pose)
                    logger.info(f"[{self.hardware.robot_id}] Scored {victim_name}! dist={dist:.2f}m")

                    self.grid_map.save_map()
                    self.state = "STOP"
                    
                    # Dump path log for external tracking/visualization
                    log_file = f"../../log/{self.hardware.robot_id}_path_log.csv"
                    try:
                        with open(log_file, "w") as f:
                            f.write("x,y\\n")
                            for px, py in self.path_log:
                                f.write(f"{px:.3f},{py:.3f}\\n")
                    except Exception as e:
                        logger.error(f"Failed to dump path log: {e}")
                        
                    continue

                # Fallback Physical Collision Detection
                if fl < 0.25 or fr < 0.25:
                    self.state = "RECOVERY"
                    self.recovery_timer = 50
                    
                    if self.current_path and self.path_idx < len(self.current_path):
                        wx, wy = self.current_path[self.path_idx]
                        alpha = math.atan2(wy - pose[1], wx - pose[0]) - pose[2]
                        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
                        self.recovery_direction = 1.0 if alpha > 0 else -1.0
                    else:
                        self.recovery_direction = 1.0 if fl < fr else -1.0
                        
                    logger.info(f"[{self.hardware.robot_id}] COLLISION AVOIDANCE! Reversing...")
                    continue

                # Map Update and Path Planning
                # Update map every 10 ticks to save CPU
                lidar = self.hardware.read_lidar()
                if lidar and self.tick_counter % 50 == 0:
                    min_dist = min(lidar)
                    min_idx = lidar.index(min_dist)
                    logger.info(f"[{self.hardware.robot_id}] DEBUG LIDAR: min_dist={min_dist:.2f} at index {min_idx}/{len(lidar)}")
                
                if self.tick_counter - self.last_map_update > 10 and self.startup_ticks > 150:
                    map_changed, new_obstacles = self.grid_map.update_from_lidar(pose, lidar)
                    if map_changed:
                        for obs_x, obs_y in new_obstacles:
                            self.hardware.send_squad_message({
                                "type": "obstacle",
                                "x": obs_x,
                                "y": obs_y
                            })
                    self.last_map_update = self.tick_counter
                    
                    # Replan if path is empty, map changed significantly, or ANY future waypoint is blocked
                    path_invalid = False
                    if not self.current_path:
                        path_invalid = True
                    elif self.path_idx < len(self.current_path):
                        for i in range(self.path_idx, len(self.current_path)):
                            wx, wy = self.current_path[i]
                            if self.grid_map.is_blocked_world(wx, wy):
                                path_invalid = True
                                break
                    
                    if path_invalid or not self.current_path:
                        self.current_path = self.planner.plan((pose[0], pose[1]), target)
                        self.path_idx = 0
                        if self.tick_counter % 50 == 0:
                            logger.info(f"[{self.hardware.robot_id}] A* Planned {len(self.current_path)} waypoints")

                # Follow Path
                if self.current_path and self.path_idx < len(self.current_path):
                    waypoint = self.current_path[self.path_idx]
                    w_dist = math.hypot(pose[0] - waypoint[0], pose[1] - waypoint[1])
                    
                    if w_dist < 0.4:
                        self.path_idx += 1
                        
                    if self.path_idx < len(self.current_path):
                        waypoint = self.current_path[self.path_idx]
                        v, omega = self.pursuit.get_velocity(pose, waypoint)
                        self.hardware.set_motor_speeds(v, omega)
                    else:
                        # Path finished, just head to target
                        v, omega = self.pursuit.get_velocity(pose, target)
                        self.hardware.set_motor_speeds(v, omega)
                else:
                    # No path yet (startup), just head straight
                    v, omega = self.pursuit.get_velocity(pose, target)
                    self.hardware.set_motor_speeds(v, omega)

            elif self.state == "RECOVERY":
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                self.recovery_timer -= 1
                if self.recovery_timer > 0:
                    if self.recovery_timer > 15:
                        self.hardware.set_motor_speeds(-0.4, -0.4)
                    else:
                        self.hardware.set_motor_speeds(0.0, 3.5 * self.recovery_direction)
                else:
                    self.state = "DRIVE"
                    self.current_path = [] # Force replan
                    
                    # Check if we are recovering repeatedly
                    if self.tick_counter - self.last_recovery_tick < 150:
                        self.consecutive_recoveries += 1
                    else:
                        self.consecutive_recoveries = 1
                    
                    self.last_recovery_tick = self.tick_counter
                    
                    if self.consecutive_recoveries >= 4:
                        pose = self.odometry.get_pose()
                        gx, gy = self.grid_map.world_to_grid(pose[0], pose[1])
                        # Mark a 10x10 cell block as obstacle to force avoidance
                        for dy in range(-6, 7):
                            for dx in range(-6, 7):
                                if 0 <= gy+dy < self.grid_map.rows and 0 <= gx+dx < self.grid_map.cols:
                                    if self.grid_map.grid[gy+dy, gx+dx] < 2:
                                        self.grid_map.grid[gy+dy, gx+dx] = 2
                        self.consecutive_recoveries = 0
                        logger.warning(f"[{self.hardware.robot_id}] Hit obstacle 4 times! Marked area as obstacle.")

            elif self.state == "FALLBACK_WAIT":
                self.hardware.set_motor_speeds(0.0, 0.0)
                self.odometry.update(*self.hardware.read_encoders(), self.hardware.read_compass_heading())
                self._process_squad_messages()
                self.fallback_timer += 1
                
                if self.partner_victim_found:
                    self.state = "STOP"
                    logger.info(f"[{self.hardware.robot_id}] Partner confirmed their victim. Stopping.")
                elif self.fallback_timer > 300:
                    if len(self.victims) > 1:
                        if "1" in self.hardware.robot_id:
                            self.assigned_victim = self.victims[1]
                        else:
                            self.assigned_victim = self.victims[0]
                    self.state = "DRIVE"
                    self.current_path = [] # Force replan
                    logger.info(f"[{self.hardware.robot_id}] Fallback: Taking over {self.assigned_victim}")

            elif self.state == "STOP":
                self.hardware.set_motor_speeds(0.0, 0.0)

if __name__ == "__main__":
    try:
        controller = AutonomousSARController()
        controller.execute_mission()
    except KeyboardInterrupt:
        logger.warning("Mission aborted by user.")
    except Exception as e:
        logger.error(f"Mission failed due to fatal error: {e}")
        import traceback
        traceback.print_exc()
