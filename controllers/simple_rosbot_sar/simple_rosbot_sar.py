"""
Basic Rosbot Controller for Search and Rescue Simulation
This controller implements an optimized exploration, anti-stuck, and victim detection strategy.
"""

from controller import Robot
import json
import random
import math
from typing import Tuple, List


class BasicRosbotController:
    def __init__(self):
        """Initialize the robot controller"""
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get robot name for identification
        self.robot_id = self.robot.getName()

        # Initialize sensors and actuators
        self._init_devices()

        # Robot state and navigation
        self.last_decision_time = 0
        self.decision_interval = 2.0  # Reduced to 2s for quicker, sharper adjustments
        self.stuck_counter = 0

        # Victim detection
        self.victim_confidence_threshold = 0.7

        # Navigation constants
        self.max_speed = 4.0          # Safe operating velocity limits
        self.obstacle_threshold = 0.4 # Alert zone boundary distance (meters)

        print(f"[{self.robot_id}] Initialized - Ready for search and rescue mission")

    def _init_devices(self):
        """Initialize robot sensors and actuators"""
        # All 4 Motors (Must be synchronized for Rosbot to slide or rotate cleanly)
        self.front_left_motor = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.robot.getDevice("rr_wheel_joint")
        
        self.front_left_motor.setPosition(float("inf"))
        self.front_right_motor.setPosition(float("inf"))
        self.rear_left_motor.setPosition(float("inf"))
        self.rear_right_motor.setPosition(float("inf"))
        
        self.set_wheels_velocity(0, 0)

        # Wheel position sensors
        self.front_left_position_sensor = self.robot.getDevice("front left wheel motor sensor")
        self.front_right_position_sensor = self.robot.getDevice("front right wheel motor sensor")
        self.rear_left_position_sensor = self.robot.getDevice("rear left wheel motor sensor")
        self.rear_right_position_sensor = self.robot.getDevice("rear right wheel motor sensor")
        
        if self.front_left_position_sensor: self.front_left_position_sensor.enable(self.timestep)
        if self.front_right_position_sensor: self.front_right_position_sensor.enable(self.timestep)
        if self.rear_left_position_sensor: self.rear_left_position_sensor.enable(self.timestep)
        if self.rear_right_position_sensor: self.rear_right_position_sensor.enable(self.timestep)

        # RGB Camera with Recognition Enabled (Vital correction to detect victims!)
        try:
            self.camera_rgb = self.robot.getDevice("camera rgb")
            if self.camera_rgb:
                self.camera_rgb.enable(self.timestep)
                self.camera_rgb.recognitionEnable(self.timestep) # Activates direct target scanning
        except:
            self.camera_rgb = None
            print(f"[{self.robot_id}] Warning: No RGB camera found")

        # Depth camera
        try:
            self.camera_depth = self.robot.getDevice("camera depth")
            if self.camera_depth: self.camera_depth.enable(self.timestep)
        except:
            self.camera_depth = None

        # Lidar sensor
        try:
            self.lidar = self.robot.getDevice("laser")
            if self.lidar: self.lidar.enable(self.timestep)
        except:
            self.lidar = None

        # IMU Components
        try:
            self.accelerometer = self.robot.getDevice("imu accelerometer")
            if self.accelerometer: self.accelerometer.enable(self.timestep)
            self.gyro = self.robot.getDevice("imu gyro")
            if self.gyro: self.gyro.enable(self.timestep)
            self.compass = self.robot.getDevice("imu compass")
            if self.compass: self.compass.enable(self.timestep)
        except:
            self.accelerometer = None
            self.gyro = None
            self.compass = None

        # Distance sensors array parsing
        self.distance_sensors = []
        sensor_names = ["fl_range", "fr_range", "rl_range", "rr_range"]
        for name in sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
            except:
                print(f"[{self.robot_id}] Warning: No {name} sensor found")

        # Supervisor Emitter Channel 43 (Coordinates with your sar_marking_supervisor)
        try:
            self.supervisor_emitter = self.robot.getDevice("supervisor emitter")
        except:
            self.supervisor_emitter = None

        # Robot-to-Robot Swarm Hardware
        try:
            self.squad_receiver = self.robot.getDevice("robot to robot receiver")
            if self.squad_receiver: self.squad_receiver.enable(self.timestep)
            self.squad_emitter = self.robot.getDevice("robot to robot emitter")
        except:
            self.squad_receiver = None
            self.squad_emitter = None

    def set_wheels_velocity(self, left: float, right: float):
        """Helper to cleanly apply differential speeds across all 4 wheels"""
        self.front_left_motor.setVelocity(left)
        self.rear_left_motor.setVelocity(left)
        self.front_right_motor.setVelocity(right)
        self.rear_right_motor.setVelocity(right)

    def get_orientation(self) -> float:
        """Get current robot orientation in radians"""
        if self.compass:
            north = self.compass.getValues()
            return math.atan2(north[0], north[1])
        return 0.0

    def get_distance_readings(self) -> List[float]:
        """Get safe distance sensor readings"""
        readings = []
        for sensor in self.distance_sensors:
            readings.append(sensor.getValue())
        return readings

    def detect_obstacles(self) -> Tuple[bool, str]:
        """Detect obstacles safely mapped from device readings"""
        distances = self.get_distance_readings()

        if len(distances) < 2:
            return False, "clear path"

        # Check front left (index 0) and front right (index 1) range bounds
        if distances[0] < self.obstacle_threshold and distances[1] < self.obstacle_threshold:
            return True, "obstacle ahead"
        elif distances[0] < self.obstacle_threshold:
            return True, "obstacle front left"
        elif distances[1] < self.obstacle_threshold:
            return True, "obstacle front right"

        return False, "clear path"

    def detect_victim(self) -> Tuple[bool, float]:
        """Uses real camera target tracking to replace your faulty random guess mechanism"""
        if self.camera_rgb:
            # Safely fetch all concrete objects within the lens boundary
            objects = self.camera_rgb.getRecognitionObjects()
            for obj in objects:
                try:
                    model_name = obj.getModel().decode('utf-8')
                except:
                    model_name = str(obj.getModel())

                # Exact structure checks matching your 'man_1' and 'woman_3' victims
                if "man" in model_name or "woman" in model_name or "victim" in model_name:
                    print(f"[{self.robot_id}] Visual Confirmation: Found victim type '{model_name}'!")
                    return True, 0.95 # Highly confident direct target acquisition
                    
        return False, 0.0

    def send_victim_found_message(self, victim_detected: bool = False, confidence: float = 0.0):
        """Package telemetry data and ping the mapping supervisor channel"""
        if not self.supervisor_emitter:
            return

        # Build telemetry data
        request = {
            "timestamp": self.robot.getTime(),
            "robot_id": self.robot_id,
            "position": [0.0, 0.0, 0.0], # Standard initial target layout anchor
            "victim_found": victim_detected,
            "victim_confidence": confidence,
        }

        message = json.dumps(request)
        # Emit data package out across Webots simulation stack
        self.supervisor_emitter.send(message.encode('utf-8'))

    def run(self):
        """Main operational execution sequence loop"""
        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()

            # 1. LIVE VICTIM DETECTION ENGINE
            victim_found, confidence = self.detect_victim()
            if victim_found and confidence >= self.victim_confidence_threshold:
                # Stop and register with supervisor immediately
                self.set_wheels_velocity(0.0, 0.0)
                self.send_victim_found_message(True, confidence)
                print(f"[{self.robot_id}] SUCCESS: Reporting victim location data packages.")
                self.robot.step(2000) # Lock wheels down for 2 seconds to seal points
                continue

            # 2. PROXIMITY TELEMETRY & ANTI-STUCK SYSTEM
            is_obstacle, condition = self.detect_obstacles()

            if is_obstacle:
                self.stuck_counter += 1
                # If stuck facing a wall for too long, execute high-speed breakout spin
                if self.stuck_counter > 25:
                    self.set_wheels_velocity(-self.max_speed * 0.7, self.max_speed * 0.7)
                else:
                    # Swerve to escape the wall boundary based on orientation paths
                    if condition == "obstacle front left":
                        self.set_wheels_velocity(self.max_speed * 0.6, -self.max_speed * 0.4)
                    elif condition == "obstacle front right" or condition == "obstacle ahead":
                            self.set_wheels_velocity(-self.max_speed * 0.4, self.max_speed * 0.6)
                    else :
                        # Clear route pathing: Advance forward smoothlyself.stuck_counter = 0# Introduce structured micro-adjustments every cycle to find optimal pathingif current_time - self.last_decision_time > self.decision_interval:self.last_decision_time = current_time# Sligh