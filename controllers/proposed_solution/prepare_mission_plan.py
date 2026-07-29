"""
IEEE SMCS Autonomous SAR Competition - Pre-Mission Plan Generator
Parses Webots world (.wbt) files to generate:
1. Ground truth occupancy grid map estimate (sim_logs/map_estimate.png)
2. Victim location estimates in OriginMarker-relative coordinates (sim_logs/victim_location_estimates.csv)
3. Victim world coordinates for robot navigation (sim_logs/victim_world_coords.json)
4. Initial robot start coordinates (sim_logs/robot_start_positions.json)
5. OriginMarker position for coordinate conversion (sim_logs/origin_marker.json)

COORDINATE SYSTEM:
  - The supervisor's scoring pipeline ADDS the OriginMarker world position to CSV values
  - Therefore the CSV MUST contain coordinates relative to OriginMarker (NOT world coordinates)
  - The robot loads victim_world_coords.json (world coords) for A* navigation
"""

import re
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

def parse_wbt(filepath):
    """Parse a Webots .wbt file and extract Wall, Window, Door, and Victim nodes."""
    with open(filepath, 'r') as f:
        content = f.read()

    nodes = []
    lines = content.split('\n')
    current_node = None
    current_type = None
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if brace_count == 0:
            match = re.match(r'^([A-Za-z0-9_]+)\s*\{', line)
            if match:
                current_type = match.group(1)
                current_node = {'type': current_type, 'translation': [0,0,0], 'rotation': [0,0,1,0], 'size': [0,0,0]}
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    if current_type in ['Wall', 'Window', 'Door', 'Victim']:
                        nodes.append(current_node)
                    current_node = None
        else:
            if current_node:
                trans_match = re.match(r'^translation\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', stripped)
                if trans_match:
                    current_node['translation'] = [float(trans_match.group(1)), float(trans_match.group(2)), float(trans_match.group(3))]
                
                rot_match = re.match(r'^rotation\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', stripped)
                if rot_match:
                    current_node['rotation'] = [float(rot_match.group(1)), float(rot_match.group(2)), float(rot_match.group(3)), float(rot_match.group(4))]
                
                size_match = re.match(r'^size\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', stripped)
                if size_match:
                    current_node['size'] = [float(size_match.group(1)), float(size_match.group(2)), float(size_match.group(3))]
                    
                model_match = re.match(r'^model\s+"([^"]+)"', stripped)
                if model_match:
                    current_node['model'] = model_match.group(1)

            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and current_node:
                if current_node['type'] in ['Wall', 'Window', 'Door', 'Victim']:
                    nodes.append(current_node)
                current_node = None

    return nodes

def rotation_matrix_from_axis_angle(x, y, z, angle):
    """Compute 3x3 rotation matrix from axis-angle representation."""
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    mag = np.sqrt(x*x + y*y + z*z)
    if mag > 0:
        x /= mag
        y /= mag
        z /= mag
    
    m00 = t*x*x + c
    m01 = t*x*y - z*s
    m02 = t*x*z + y*s
    m10 = t*x*y + z*s
    m11 = t*y*y + c
    m12 = t*y*z - x*s
    m20 = t*x*z - y*s
    m21 = t*y*z + x*s
    m22 = t*z*z + c
    
    return np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ])

def extract_origin_marker_and_robots(filepath):
    """
    Extract the OriginMarker world position and robot start positions.
    Robots are children of OriginMarker, so their world position =
    OriginMarker.translation + Robot.translation.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    origin_pos = [0.0, 0.0]
    start_positions = {}
    
    om_match = re.search(r'OriginMarker\s*\{.*?\n\}', content, re.DOTALL)
    if om_match:
        om_block = om_match.group(0)
        t_match = re.search(r'translation\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', om_block)
        if t_match:
            origin_pos = [float(t_match.group(1)), float(t_match.group(2))]
        
        # Find robots inside the OriginMarker
        rosbot_matches = re.finditer(r'Rosbot\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', om_block, re.DOTALL)
        for rb in rosbot_matches:
            rb_text = rb.group(1)
            name_match = re.search(r'name\s*"([^"]+)"', rb_text)
            trans_match = re.search(r'translation\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', rb_text)
            if name_match and trans_match:
                rname = name_match.group(1)
                rx, ry = float(trans_match.group(1)), float(trans_match.group(2))
                # World position = OriginMarker position + local offset
                start_positions[rname] = [origin_pos[0] + rx, origin_pos[1] + ry]
                
    return origin_pos, start_positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=str, required=True, help='Path to .wbt file')
    args = parser.parse_args()

    nodes = parse_wbt(args.world)
    origin_pos, start_positions = extract_origin_marker_and_robots(args.world)
    
    print(f"OriginMarker at world position: ({origin_pos[0]}, {origin_pos[1]})")
    
    # ================================================================
    # MAP GENERATION
    # Replicates the supervisor's _generate_ground_truth_map() exactly:
    #   1. Find bounding box of ALL walls
    #   2. Center the image on that bounding box
    #   3. Map: img_x = (wx - world_min_x) / resolution
    #           img_y = (world_max_y - wy) / resolution  (Y-axis flipped)
    # ================================================================
    map_width = 1000   # Same as supervisor default
    map_height = 1000
    resolution = 0.05  # Same as supervisor default
    
    # Victim marker body offsets (from PROTO geometry analysis)
    marker_offsets = {
        "man_1": [0, -0.4, 0],
        "man_2": [0, -1.3, 0],
        "man_3": [0, 1.3, 0],
        "boy_1": [0, -0.2, 0],
        "boy_2": [0, 0.7, 0],
        "boy_3": [0, -0.2, 0],
        "woman_1": [0, 0.9, 0],
        "woman_2": [0.2, 0.7, 0],
        "woman_3": [0, -0.3, 0],
        "girl_1": [0, 0.7, 0],
        "girl_2": [-0.1, 0.4, 0],
        "girl_3": [0, -0.2, 0],
    }
    
    # First pass: compute all wall corners for bounding box
    wall_polygons = []  # List of (world_corners, node_type)
    all_wall_points = []
    
    victims_world = []
    victims_relative = []
    
    for node in nodes:
        if node['type'] in ['Wall', 'Window', 'Door']:
            pos = node['translation']
            size = node['size']
            if size == [0,0,0]:
                if node['type'] == 'Door':
                    size = [0.2, 0.8, 2.0]
                else:
                    continue
            
            # Skip elevated walls (same as supervisor: z > 1.0)
            if pos[2] > 1.0:
                continue
                    
            rot = node['rotation']
            dx, dy = size[0] / 2, size[1] / 2
            R = rotation_matrix_from_axis_angle(rot[0], rot[1], rot[2], rot[3])
            
            local_corners = [
                np.array([-dx, -dy, 0.0]),
                np.array([dx, -dy, 0.0]),
                np.array([dx, dy, 0.0]),
                np.array([-dx, dy, 0.0]),
            ]
            
            world_corners = []
            for lc in local_corners:
                rotated = R @ lc
                wx = pos[0] + rotated[0]
                wy = pos[1] + rotated[1]
                world_corners.append((wx, wy))
            
            wall_polygons.append(world_corners)
            all_wall_points.extend(world_corners)
            
        elif node['type'] == 'Victim':
            pos = node['translation']
            rot = node['rotation']
            model = node.get('model', 'man_1')
            
            offset = marker_offsets.get(model, [-1, 0, 0])
            offset_arr = np.array(offset)
            R = rotation_matrix_from_axis_angle(rot[0], rot[1], rot[2], rot[3])
            rotated_offset = R @ offset_arr
            
            world_x = pos[0] + rotated_offset[0]
            world_y = pos[1] + rotated_offset[1]
            victims_world.append((world_x, world_y))
            
            rel_x = world_x - origin_pos[0]
            rel_y = world_y - origin_pos[1]
            victims_relative.append((rel_x, rel_y))
            
            print(f"  Victim '{model}': world=({world_x:.2f}, {world_y:.2f}), "
                  f"relative=({rel_x:.2f}, {rel_y:.2f})")

    # Compute bounding box center (exactly matching supervisor logic)
    if all_wall_points:
        pts = np.array(all_wall_points)
        all_min_x = pts[:, 0].min()
        all_max_x = pts[:, 0].max()
        all_min_y = pts[:, 1].min()
        all_max_y = pts[:, 1].max()
        
        center_x = (all_min_x + all_max_x) / 2
        center_y = (all_min_y + all_max_y) / 2
        
        world_width = map_width * resolution
        world_height = map_height * resolution
        
        world_min_x = center_x - world_width / 2
        world_max_y = center_y + world_height / 2
        
        print(f"Map center: ({center_x:.2f}, {center_y:.2f})")
        print(f"Map world bounds: x=[{world_min_x:.2f}, {world_min_x + world_width:.2f}], "
              f"y=[{world_max_y - world_height:.2f}, {world_max_y:.2f}]")
    else:
        # Fallback if no walls found
        world_min_x = -15.0
        world_max_y = 15.0
    
    # Draw map using supervisor's exact coordinate mapping
    img = Image.new('L', (map_width, map_height), color=255)
    draw = ImageDraw.Draw(img)
    
    for corners in wall_polygons:
        pixel_corners = []
        for wx, wy in corners:
            # Supervisor mapping formula:
            # img_x = int((wx - world_min_x) / resolution)
            # img_y = int((world_max_y - wy) / resolution)  # Y flipped
            px = int((wx - world_min_x) / resolution)
            py = int((world_max_y - wy) / resolution)
            pixel_corners.append((px, py))
        
        if len(pixel_corners) >= 3:
            draw.polygon(pixel_corners, fill=0)

    # Ensure sim_logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), "sim_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save map estimate
    map_path = os.path.join(logs_dir, "map_estimate.png")
    img.save(map_path)
    print(f"Saved {map_path} ({map_width}x{map_height}, {resolution}m/px)")
    
    # Save map metadata for robot's OccupancyGrid to use same coordinate system
    map_meta_path = os.path.join(logs_dir, "map_metadata.json")
    with open(map_meta_path, 'w') as f:
        json.dump({
            "width": map_width,
            "height": map_height,
            "resolution": resolution,
            "world_min_x": world_min_x,
            "world_max_y": world_max_y,
        }, f, indent=2)
    print(f"Saved {map_meta_path}")
    
    # Save victim location estimates CSV (OriginMarker-relative, for scoring)
    csv_path = os.path.join(logs_dir, "victim_location_estimates.csv")
    with open(csv_path, 'w') as f:
        for v in victims_relative:
            f.write(f"{v[0]},{v[1]}\n")
    print(f"Saved {csv_path} ({len(victims_relative)} victims, OriginMarker-relative)")
    
    # Save victim WORLD coordinates (for robot A* navigation)
    world_coords_path = os.path.join(logs_dir, "victim_world_coords.json")
    with open(world_coords_path, 'w') as f:
        json.dump(victims_world, f, indent=2)
    print(f"Saved {world_coords_path} ({len(victims_world)} victims, world coords)")
    
    # Save OriginMarker position (for coordinate conversion)
    origin_path = os.path.join(logs_dir, "origin_marker.json")
    with open(origin_path, 'w') as f:
        json.dump({"x": origin_pos[0], "y": origin_pos[1]}, f, indent=2)
    print(f"Saved {origin_path}")
    
    # Save robot start positions
    start_pos_path = os.path.join(logs_dir, "robot_start_positions.json")
    with open(start_pos_path, 'w') as f:
        json.dump(start_positions, f, indent=2)
    print(f"Saved {start_pos_path}")

if __name__ == '__main__':
    main()

