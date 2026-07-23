"""
Marking supervisor for search and rescue simulation
This supervisor logs robot actions.
"""

from controller import Supervisor, Node
import json
from datetime import datetime
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings
import uuid
from io import BytesIO
from urllib import error as urllib_error
from urllib import request as urllib_request
import numpy as np
from PIL import Image, ImageDraw
import ssl

# Bypass macOS Python SSL certificate issues for the marking server
ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class SupervisorMessageLogEntry:
    """A log entry by the supervisor"""

    robot_id: str  # Identifier of robot
    sent_timestamp: float  # Timestamp when message was sent by robot
    received_timestamp: float  # Timestamp when message was received by supervisor
    position: List  # Position of the robot at the time of sending the message
    reported_position: (
        List  # Position reported by robot at the time of sending the message
    )
    victim_found: bool  # Robot report of finding a victim
    victim_confidence: float  # Confidence on victim found report
    victim_found_verdict: bool  # True if robot was correct about finding a victim


class MarkingSupervisor:
    def __init__(
        self,
        log_file: str = None,
        max_duration: int = 180,
    ):
        """
        Initialize the marking supervisor

        Args:
            log_file: Path to save mission logs
            max_duration: Mission duration in seconds
        """
        self.supervisor = Supervisor()
        self.log_file = log_file
        self.max_duration = max_duration
        self.message_log: List[SupervisorMessageLogEntry] = []

        # Communication channels
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(int(self.supervisor.getBasicTimeStep()))

        # Get all robots in the simulation
        self.robots = {}
        self._discover_robots()

        # Get all victims in the simulation
        self.victims = {}
        self._discover_victims()

        # Get origin marker
        self.origin_marker = self._find_nodes_by_name("OriginMarker")
        self.coordinate_offset = (
            self.origin_marker[0].getPosition()
            if self.origin_marker[0]
            else [0.0, 0.0, 0.0]
        )
        print(
            f"[SUPERVISOR] Coordinate offset from OriginMarker: {self.coordinate_offset}"
        )

        # Generate ground truth map
        self.ground_truth_map = self._generate_ground_truth_map()

        # Create logs file
        self.logs_dir = "../../log"
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        print(
            f"[SUPERVISOR] Found {len(self.robots)} robots and {len(self.victims)} victims"
        )

    def _find_nodes_by_name(
        self, name: str, matching_nodes: List[Node] = None, node: Node = None
    ) -> List[Node]:
        """Helper function to find a node by its name field"""

        if matching_nodes is None:
            matching_nodes = []

        if node is None:
            node = self.supervisor.getRoot()
            children_field = node.getField("children")
        else:
            children_field = node.getField("children")

        if children_field is None:
            return matching_nodes

        for i in range(children_field.getCount()):
            child_node = children_field.getMFNode(i)
            if child_node and child_node.getTypeName() == name:
                matching_nodes.append(child_node)

            if child_node:
                self._find_nodes_by_name(name, matching_nodes, child_node)

        return matching_nodes

    def _discover_robots(self):
        """Find all Rosbot robots in the simulation"""

        rosbot_nodes = self._find_nodes_by_name("Rosbot")
        for robot in rosbot_nodes:
            robot_name = robot.getField("name").getSFString()
            self.robots[robot_name] = {
                "node": robot,
                "victims_found": 0,
                "distance_travelled": 0.0,
                "last_position": None,
                "messages": [],
            }
            print(
                f"[SUPERVISOR] Discovered robot: {robot_name} at {robot.getPosition()}"
            )

    def _get_victim_marker_position(self, victim_node: Node) -> Optional[List]:
        """Get victim marker world position from DEF VICTIM_MARKER inside Victim PROTO."""
        if victim_node is None:
            return None

        marker_node = victim_node.getFromProtoDef("VICTIM_MARKER")
        if marker_node is None:
            return None

        return marker_node.getPosition()

    def _discover_victims(self):
        """Find all victims in the simulation"""
        victim_nodes = self._find_nodes_by_name("Victim")
        for victim in victim_nodes:
            victim_name = victim.getField("name").getSFString()
            position = self._get_victim_marker_position(victim)
            if position is None:
                print(
                    f"[SUPERVISOR] Warning: Could not find marker position for victim {victim_name}. Defaulting to robot position."
                )
                position = victim.getPosition()
            self.victims[victim_name] = {
                "node": victim,
                "position": position,
                "found": False,
                "found_by": None,
                "found_time": None,
            }
            print(f"[SUPERVISOR] Discovered victim: {victim_name} at {position}")

    def _get_wall_bounding_box(
        self, wall_node: Node
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Get the four wall corners in world coordinates using the node orientation matrix.
        Returns list of (x, y) tuples for the four corners, or None if unable to determine.
        """
        try:
            position = wall_node.getPosition()

            size_field = wall_node.getField("size")
            if not size_field:
                return None

            size = size_field.getSFVec3f()
            dx, dy = size[0] / 2, size[1] / 2

            # Webots returns a 3x3 orientation matrix flattened as 9 values.
            orientation_flat = wall_node.getOrientation()
            if orientation_flat is None or len(orientation_flat) != 9:
                return [
                    (position[0] - dx, position[1] - dy),
                    (position[0] + dx, position[1] - dy),
                    (position[0] + dx, position[1] + dy),
                    (position[0] - dx, position[1] + dy),
                ]

            rotation_matrix = np.array(orientation_flat, dtype=float).reshape((3, 3))

            local_corners = [
                np.array([-dx, -dy, 0.0]),
                np.array([dx, -dy, 0.0]),
                np.array([dx, dy, 0.0]),
                np.array([-dx, dy, 0.0]),
            ]

            world_corners = []
            for local_corner in local_corners:
                rotated = rotation_matrix @ local_corner
                world_corners.append(
                    (position[0] + rotated[0], position[1] + rotated[1])
                )

            return world_corners

        except Exception as e:
            print(f"[SUPERVISOR] Warning: Could not get bounding box for wall: {e}")
            return None

    def _generate_ground_truth_map(
        self, map_width: int = 600, map_height: int = 600, resolution: float = 0.05
    ) -> Optional[np.ndarray]:
        """
        Generate ground truth map as binary image (white=free, black=obstacle).
        Renders all walls from a top-down view, accounting for wall rotation.

        Args:
            map_width: Width of output image in pixels (default 1000)
            map_height: Height of output image in pixels (default 1000)
            resolution: Meters per pixel (default 0.05)

        Returns:
            image_array or None if no walls found
        """
        # Find all walls
        wall_nodes = self._find_nodes_by_name("Wall")
        wall_nodes.extend(self._find_nodes_by_name("Window"))
        wall_nodes.extend(self._find_nodes_by_name("Door"))

        if not wall_nodes:
            return None

        # Get all wall corner lists (each wall is a list of 4 corners)
        wall_polygons = []
        all_points = []
        skipped_elevated = 0
        for wall in wall_nodes:
            wall_pos = wall.getPosition()
            if wall_pos[2] > 1.0:  # Wall elements are usually positioned at z=0.0
                skipped_elevated += 1
                continue
            corners = self._get_wall_bounding_box(wall)
            if corners:
                wall_polygons.append(corners)
                all_points.extend(corners)

        if not wall_polygons or not all_points:
            return None

        # Calculate bounding box of all walls
        all_points = np.array(all_points)
        all_min_x = all_points[:, 0].min()
        all_max_x = all_points[:, 0].max()
        all_min_y = all_points[:, 1].min()
        all_max_y = all_points[:, 1].max()

        # Calculate center of all walls
        center_x = (all_min_x + all_max_x) / 2
        center_y = (all_min_y + all_max_y) / 2

        # Calculate world space dimensions covered by image
        world_width = map_width * resolution
        world_height = map_height * resolution

        # Calculate world bounds centered on walls
        world_min_x = center_x - world_width / 2
        world_max_y = center_y + world_height / 2

        # Create white image (all free space)
        image = Image.new("L", (map_width, map_height), 255)
        draw = ImageDraw.Draw(image)

        # Draw walls as black polygons (accounting for rotation)
        for corners in wall_polygons:
            # Convert world coordinates to image coordinates
            img_corners = []
            for wx, wy in corners:
                img_x = int((wx - world_min_x) / resolution)
                # World +Y points up, but image +Y points down.
                # Flip Y during projection (equivalent to top-down 180 deg rotation about X).
                img_y = int((world_max_y - wy) / resolution)
                img_corners.append((img_x, img_y))

            # Draw filled polygon (black = 0)
            if len(img_corners) >= 3:
                draw.polygon(img_corners, fill=0)

        img_array = np.array(image)

        return img_array

    def _check_victim_proximity(
        self, robot_pos: List, detection_radius: float = 1.0
    ) -> tuple:
        """
        Check if robot is near any victims
        Returns: (victim_found, victim_id)
        """
        if not robot_pos:
            return False, None

        for victim_id, victim_data in self.victims.items():
            if victim_data["found"]:
                continue

            victim_pos = victim_data["position"]
            distance = (
                (robot_pos[0] - victim_pos[0]) ** 2
                + (robot_pos[1] - victim_pos[1]) ** 2
            ) ** 0.5

            if distance < detection_radius:
                return True, victim_id

        return False, None

    def process_robot_messages(self, data: Dict):
        """
        Process a message from a robot

        Expected data format:
        {
            "timestamp": float
            "robot_id": str,
            "position": list,
            "victim_found": bool,
            "victim_confidence": float,
        }
        """

        # Get robot position
        robot_id = data.get("robot_id", "undefined")
        robot = self.robots.get(robot_id, None)
        if not robot:
            print(
                f"[SUPERVISOR] Warning: Received message from unknown robot {robot_id}"
            )
            return
        robot_pos = robot["node"].getPosition()

        # Check for nearby victims
        super_victim_found, super_victim_id = self._check_victim_proximity(robot_pos)

        # Log the message
        logEntry = SupervisorMessageLogEntry(
            robot_id=robot_id,
            sent_timestamp=data.get("timestamp", -1.0),
            received_timestamp=self.supervisor.getTime(),
            position=robot_pos if robot_pos else [-1000.0, -1000.0, -1000.0],
            reported_position=data.get("position", [-10000.0, -10000.0, -10000.0]),
            victim_found=data.get("victim_found", False),
            victim_confidence=data.get("victim_confidence", 0.0),
            victim_found_verdict=super_victim_found and data.get("victim_found", False),
        )
        self.message_log.append(logEntry)

        # Update robot stats
        if robot_id in self.robots:
            self.robots[robot_id]["messages"].append(logEntry)

            # Mark victim as found if applicable
            if (
                logEntry.victim_found
                and super_victim_found
                and super_victim_id
                and not self.victims[super_victim_id]["found"]
            ):
                self.victims[super_victim_id]["found"] = True
                self.victims[super_victim_id]["found_by"] = robot_id
                self.victims[super_victim_id]["found_time"] = logEntry.sent_timestamp
                self.robots[robot_id]["victims_found"] += 1
                print(
                    f"[SUPERVISOR] VICTIM FOUND: {super_victim_id} by {robot_id} (confidence: {logEntry.victim_confidence:.2f})"
                )

        # Print message
        print(
            f"[SUPERVISOR] MSG RECEIVED - Robot: {robot_id}, Found verdict: {logEntry.victim_found_verdict}"
        )

    def handle_communications(self):
        """Check for incoming messages from robots"""
        while self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            self.receiver.nextPacket()

            try:
                # Parse JSON message from robot
                data = json.loads(message)

                # Process the request
                self.process_robot_messages(data)

            except json.JSONDecodeError:
                print(f"[SUPERVISOR] Error decoding message: {message}")
            except Exception as e:
                print(f"[SUPERVISOR] Error processing request: |{e}| ")

    def save_logs(self):
        """Save mission logs to file"""
        mission_data = {
            "mission_duration": self.supervisor.getTime(),
            "total_messages": len(self.message_log),
            "total_victims": len(self.victims),
            "victims_found": sum(1 for v in self.victims.values() if v["found"]),
            "victim_location_estimates": [],
            "robots": {},
            "victims": {},
            "messages": [asdict(d) for d in self.message_log],
        }

        # Add information extraction data if available
        if os.path.exists(
            "../proposed_solution/sim_logs/victim_location_estimates.csv"
        ):
            with open(
                "../proposed_solution/sim_logs/victim_location_estimates.csv", "r"
            ) as f:
                try:
                    victim_location_estimates = [
                        [float(x) for x in line.strip().split(",")]
                        for line in f.readlines()
                    ]
                    mission_data["victim_location_estimates"] = [
                        [
                            estimate[0] + self.coordinate_offset[0],
                            estimate[1] + self.coordinate_offset[1],
                        ]
                        for estimate in victim_location_estimates
                    ]
                except ValueError:
                    warnings.warn(
                        "Invalid numeric value in victim_location_estimates.csv. Each line should contain comma-separated numeric values representing x and y coordinates."
                    )
                    mission_data["victim_location_estimates"] = []
                except IndexError:
                    warnings.warn(
                        "Invalid format in victim_location_estimates.csv. Each line should contain at least two comma-separated values representing x and y coordinates."
                    )
                    mission_data["victim_location_estimates"] = []

        # Add robot statistics
        for robot_id, robot_data in self.robots.items():
            mission_data["robots"][robot_id] = {
                "victims_found": robot_data["victims_found"],
                "distance_travelled": robot_data["distance_travelled"],
                "num_messages": len(robot_data["messages"]),
            }

        # Add victim statistics
        for victim_id, victim_data in self.victims.items():
            mission_data["victims"][victim_id] = {
                "position": victim_data["position"],
                "found": victim_data["found"],
                "found_by": victim_data["found_by"],
                "found_time": victim_data["found_time"],
            }

        # Save to file
        if self.log_file:
            log_file = self.log_file
        else:
            fname = datetime.now().strftime("SupervisorLog_%Y_%h_%d___%H_%M_%S")
            log_file = f"{self.logs_dir}/{fname}.json"

        with open(log_file, "w") as f:
            json.dump(mission_data, f, indent=2)

        print(f"[SUPERVISOR] Mission logs saved to {log_file}")

        log_file_stem = os.path.splitext(os.path.basename(log_file))[0]
        marked_response_file = os.path.join(
            self.logs_dir, f"Marked_{log_file_stem}.txt"
        )

        def _write_marking_response(response_text: str):
            try:
                with open(marked_response_file, "w", encoding="utf-8") as f:
                    f.write(response_text)
                print(
                    f"[SUPERVISOR] Marking server response saved to {marked_response_file}"
                )
            except Exception as exc:
                warnings.warn(f"Could not save marking response text: {exc}")

        # Send request to marking server as multipart/form-data.
        url = "https://ieee-sar-hackathon.ts.r.appspot.com/mark-log"
        boundary = f"----SARBoundary{uuid.uuid4().hex}"

        def _multipart_text_part(field_name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")

        def _multipart_file_part(
            field_name: str,
            file_name: str,
            content_type: str,
            data: bytes,
        ) -> bytes:
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            return header + data + b"\r\n"

        parts = []
        parts.append(_multipart_text_part("sar_log", json.dumps(mission_data)))

        # Optional estimate map from proposed solution.
        map_estimate_path = "../proposed_solution/sim_logs/map_estimate.png"
        if os.path.exists(map_estimate_path):
            with open(map_estimate_path, "rb") as f:
                parts.append(
                    _multipart_file_part(
                        "map_estimate_image",
                        "map_estimate.png",
                        "image/png",
                        f.read(),
                    )
                )

        # Optional ground truth map generated by supervisor (from in-memory array).
        if self.ground_truth_map is not None:
            png_buffer = BytesIO()
            Image.fromarray(np.asarray(self.ground_truth_map, dtype=np.uint8)).save(
                png_buffer, format="PNG"
            )
            parts.append(
                _multipart_file_part(
                    "map_ground_truth_image",
                    "ground_truth_map.png",
                    "image/png",
                    png_buffer.getvalue(),
                )
            )

        payload = b"".join(parts) + f"--{boundary}--\r\n".encode("utf-8")
        request = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request) as response:
                response_text = response.read().decode("utf-8", errors="replace")
                _write_marking_response(response_text)
                if response.status != 200:
                    warnings.warn(
                        "Error sending log to marking server! Please double check log structure, make sure there are no missing fields"
                    )
                    warnings.warn(f"Server returned status code: {response.status}")
                    warnings.warn(f"Server error reason: {response.reason}")
                    warnings.warn(f"Server error text: {response_text}")
                else:
                    print(response_text)
        except urllib_error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            _write_marking_response(response_text)
            warnings.warn(
                "Error sending log to marking server! Please double check log structure, make sure there are no missing fields"
            )
            warnings.warn(f"Server returned status code: {exc.code}")
            warnings.warn(f"Server error reason: {exc.reason}")
            warnings.warn(f"Server error text: {response_text}")
        except urllib_error.URLError as exc:
            _write_marking_response(str(exc))
            warnings.warn(f"Error sending log to marking server: {exc}")

    def print_mission_summary(self):
        """Print summary of the mission"""
        duration = self.supervisor.getTime()
        victims_found = sum(1 for v in self.victims.values() if v["found"])

        print("\n" + "=" * 60)
        print("MISSION SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration:.3f} seconds")
        print(f"Victims Found: {victims_found}/{len(self.victims)}")
        print(f"Total Number of Messages: {len(self.message_log)}")

        print("\nRobot Performance:")
        for robot_id, robot_data in self.robots.items():
            total = len(robot_data["messages"])
            print(f"  {robot_id}:")
            print(f"    - Victims Found: {robot_data['victims_found']}")
            print(f"    - Distance Travelled: {robot_data['distance_travelled']:.1f}m")
            print(f"    - Number of Messages: {total}")

        print("\nVictim Status:")
        for victim_id, victim_data in self.victims.items():
            if victim_data["found"]:
                print(
                    f"  {victim_id}: FOUND by {victim_data['found_by']} at t={victim_data['found_time']:.1f}s"
                )
            else:
                print(f"  {victim_id}: NOT FOUND")
        print("=" * 60)

    def update_distance_travelled(self):
        """Updates distance travelled for all Rosbots"""
        for robot_id, robot_data in self.robots.items():
            robot_pos = robot_data["node"].getPosition()
            if robot_data["last_position"] and robot_pos:
                last_pos = robot_data["last_position"]
                dist = (
                    (robot_pos[0] - last_pos[0]) ** 2
                    + (robot_pos[1] - last_pos[1]) ** 2
                ) ** 0.5
                self.robots[robot_id]["distance_travelled"] += dist
            self.robots[robot_id]["last_position"] = robot_pos

    def run(self):
        """Main supervisor loop"""
        timestep = int(self.supervisor.getBasicTimeStep())
        self.last_update_time = 0
        print("[SUPERVISOR] Starting mission supervision...")

        while self.supervisor.step(timestep) != -1:
            current_time = self.supervisor.getTime()

            # Handle incoming communications
            self.handle_communications()
            self.update_distance_travelled()

            # Periodic status update (every 30 seconds)
            if (
                int(current_time) > self.last_update_time
                and int(current_time) % 30 == 0
            ):
                self.last_update_time = int(current_time)
                victims_found = sum(1 for v in self.victims.values() if v["found"])
                print(
                    f"[SUPERVISOR] Status - Time: {current_time:.3f}s, "
                    f"Victims: {victims_found}/{len(self.victims)}, "
                    f"Num Messages: {len(self.message_log)}"
                )

            # Stop if all victims found
            if sum(1 for v in self.victims.values() if v["found"]) == len(self.victims):
                print("[SUPERVISOR] All victims found! Mission complete.")
                break

            # Stop if timer is done
            if current_time > self.max_duration:
                print("[SUPERVISOR] Mission timeout reached.")
                break

        # Mission complete
        self.supervisor.simulationSetMode(0)
        self.print_mission_summary()
        self.save_logs()
        self.supervisor.simulationReset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Marking supervisor for SAR simulation"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to save mission logs",
    )

    args = parser.parse_args()

    supervisor = MarkingSupervisor(log_file=args.log_file)
    supervisor.run()
