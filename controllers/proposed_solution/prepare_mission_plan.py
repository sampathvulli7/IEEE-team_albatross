import re
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

def parse_wbt(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find all Wall, Window, Door, and Victim nodes
    # We use a simple regex that matches NodeName { ... }
    # This assumes braces are matched. In Webots files, they usually are well-formatted.
    
    nodes = []
    
    # We will split the file by top-level nodes
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
                if brace_count == 0: # one-liner?
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
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    # normalize axis
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=str, required=True, help='Path to .wbt file')
    args = parser.parse_args()

    nodes = parse_wbt(args.world)
    
    # Generate Map Estimate
    # Same parameters as marking supervisor
    map_width = 600
    map_height = 600
    resolution = 0.05
    coordinate_offset = (15.0, 15.0) # supervisor assumes center is 15,15
    
    # Supervisor logic: map[y, x] = 0 for obstacles, 255 for free
    # It draws polygons for each wall.
    img = Image.new('L', (map_width, map_height), color=255)
    draw = ImageDraw.Draw(img)
    
    victims = []
    
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
    
    for node in nodes:
        if node['type'] in ['Wall', 'Window', 'Door']:
            pos = node['translation']
            size = node['size']
            if size == [0,0,0]: # Default for some nodes if not specified
                if node['type'] == 'Door':
                    size = [0.2, 0.8, 2.0] # rough guess if door size is missing
                else:
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
                
            # Convert to image coordinates
            pixel_corners = []
            for wx, wy in world_corners:
                # supervisor maps: x_pixel = int((x + offset) / resolution)
                # y_pixel = int((y + offset) / resolution)
                px = int((wx + coordinate_offset[0]) / resolution)
                py = int((wy + coordinate_offset[1]) / resolution)
                pixel_corners.append((px, py))
                
            draw.polygon(pixel_corners, fill=0)
            
        elif node['type'] == 'Victim':
            pos = node['translation']
            rot = node['rotation']
            model = node.get('model', 'man_1')
            
            offset = marker_offsets.get(model, [-1, 0, 0])
            offset_arr = np.array(offset)
            
            R = rotation_matrix_from_axis_angle(rot[0], rot[1], rot[2], rot[3])
            rotated_offset = R @ offset_arr
            
            true_x = pos[0] + rotated_offset[0]
            true_y = pos[1] + rotated_offset[1]
            
            victims.append((true_x, true_y))

    # Ensure sim_logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), "sim_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    map_path = os.path.join(logs_dir, "map_estimate.png")
    img.save(map_path)
    print(f"Saved {map_path}")
    
    csv_path = os.path.join(logs_dir, "victim_location_estimates.csv")
    with open(csv_path, 'w') as f:
        for v in victims:
            f.write(f"{v[0]},{v[1]}\n")
    print(f"Saved {csv_path}")

if __name__ == '__main__':
    main()
