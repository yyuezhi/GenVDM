import os
import time
import numpy as np
import cv2
import binvox_rw_customized as binvox_rw  # Ensure this is your customized version
import cutils
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import trimesh
mask_colors = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (128, 128, 0),  # Olive
    (0, 128, 128),  # Teal
    (128, 0, 0)     # Maroon
]

# Initialize camera parameters
cam_alpha = 0.0
cam_beta = 0.0
cam_pan_x = 0.0
cam_pan_y = 0.0
cam_zoom = 1.0

# Initialize current mode
current_mode = 'zoom'  # Modes: 'zoom', 'select'

# Initialize current label index
current_label_index = 0  # Range from 0 to 9

# Capture mouse events
mouse_xyd = np.zeros([3], np.int32)
mouse_xyd_backup = np.zeros([3], np.int32)
selected_point = None  # For select mode

# Specify window dimensions
window_width = 800   # Set your desired window width
window_height = 800  # Set your desired window height

def mouse_ops(event, x, y, flags, param):
    global current_mode, selected_point, cam_alpha, cam_beta, cam_zoom, mouse_xyd, mouse_xyd_backup

    if event == cv2.EVENT_RBUTTONDOWN:
        mouse_xyd[2] = 1
        mouse_xyd[0] = x
        mouse_xyd[1] = y
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse_xyd[2] = 2
        mouse_xyd[0] = x
        mouse_xyd[1] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_xyd[2] == 1:
            dx = x - mouse_xyd[0]
            dy = y - mouse_xyd[1]
            cam_alpha -= dx / 200.0
            cam_beta += dy / 200.0
            cam_beta = max(min(cam_beta, 1.2), -1.2)
            mouse_xyd[0] = x
            mouse_xyd[1] = y
        elif mouse_xyd[2] == 2:
            dy = y - mouse_xyd[1]
            cam_zoom *= 1 + dy / 200.0
            cam_zoom = max(min(cam_zoom, 5.0), 0.1)
            mouse_xyd[0] = x
            mouse_xyd[1] = y
    elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
        mouse_xyd[2] = 0

    if current_mode == 'select':
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)

# Initialize OpenCV window
Window_name = "Explorer"
cv2.namedWindow(Window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(Window_name, window_width, window_height)  # Set the window size
cv2.setMouseCallback(Window_name, mouse_ops)

# Directory containing voxel models
target_dir = "./voxel/"
render_resolution = 1024
render_radius = 0.003

# Directory to save labels
labels_dir = "./label/"
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Retrieve and sort voxel filenames
obj_names = [n for n in os.listdir(target_dir) if n.endswith(".binvox")]

if not obj_names:
    raise ValueError(f"No .binvox files found in the directory: {target_dir}")

# Initialize current index
current_index = 0

def load_voxel(index):
    """Load voxel data from the specified index.

    Returns:
        bool: True if loading is successful, False otherwise.
    """
    global px, py, pz, selected_label, render_labels, voxel_positions, voxel_tree, current_model_name
    this_name = os.path.join(target_dir, obj_names[index])
    current_model_name = os.path.splitext(obj_names[index])[0]
    print(f"Loading voxel: {this_name}")
    try:
        with open(this_name, 'rb') as voxel_model_file:
            batch_voxels = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(bool)
            px, pz, py = np.nonzero(batch_voxels)
            px = (px.astype(np.float32) + 0.5) / 1024 - 0.5
            py = (py.astype(np.float32) + 0.5) / 1024 - 0.5
            pz = (pz.astype(np.float32) + 0.5) / 1024 - 0.5

            points = np.stack((px, py, pz), axis=1)

            selected_label = [[] for _ in range(10)]  # Initialize selected_label as list of lists
            render_labels = np.full(len(px), -1, dtype=np.int32)  # Initialize render_labels to -1

            # Build KD-tree for voxel positions
            voxel_positions = np.stack((px, py, pz), axis=1)
            voxel_tree = cKDTree(voxel_positions)

            # Try to load existing labels
            label_filename = os.path.join(labels_dir, current_model_name + "_labels.npz")
            if os.path.exists(label_filename):
                npz_file = np.load(label_filename)
                indices = npz_file['indices']
                offsets = npz_file['offsets']
                selected_label = []
                for i in range(len(offsets) - 1):
                    start = offsets[i]
                    end = offsets[i+1]
                    selected_label.append(indices[start:end].tolist())
                print(f"Loaded existing labels from {label_filename}")
                # Need to update render_labels for the loaded labels
                for label_idx in range(len(selected_label)):
                    if len(selected_label[label_idx]) > 0:
                        update_render_labels(label_idx)
            else:
                print("No existing labels found for this model.")

            del batch_voxels
            voxel_model_file.close()
            return True
    except Exception as e:
        print(f"Failed to load voxel '{this_name}': {e}")
        return False

def attempt_load_voxel(index, direction='next'):
    """Attempt to load a voxel. If it fails, try the next (or previous) voxel.

    Args:
        index (int): The current index.
        direction (str): 'next' to move forward, 'prev' to move backward.

    Returns:
        int: The new index after attempting to load.
    """
    max_attempts = len(obj_names)
    attempts = 0
    new_index = index

    while attempts < max_attempts:
        success = load_voxel(new_index)
        if success:
            return new_index
        else:
            if direction == 'next':
                new_index += 1
                if new_index >= len(obj_names):
                    print("Reached the end of the voxel list.")
                    return index  # Return original index if no more voxels
            elif direction == 'prev':
                new_index -= 1
                if new_index < 0:
                    print("Reached the beginning of the voxel list.")
                    return index  # Return original index if no more voxels
            attempts += 1
            print(f"Attempting to load {'next' if direction == 'next' else 'previous'} voxel: Index {new_index}")
    print("No valid voxels found after multiple attempts.")
    return index  # Return original index if all attempts fail

def save_labels():
    """Save the selected_label data to an .npz file."""
    global selected_label, current_model_name
    label_filename = os.path.join(labels_dir, current_model_name + "_labels.npz")
    # Concatenate all indices and create offsets
    indices_list = []
    offsets = [0]
    for label in selected_label:
        indices_list.extend(label)
        offsets.append(len(indices_list))
    indices_array = np.array(indices_list, dtype=np.int32)
    offsets_array = np.array(offsets, dtype=np.int32)
    np.savez(label_filename, indices=indices_array, offsets=offsets_array)
    print(f"Labels saved to {label_filename}")

def update_render_labels(current_label_index):
    """Update render_labels for the current label index."""
    global render_labels, selected_label, voxel_tree, voxel_positions
    # Remove previous assignments for the current label
    render_labels[render_labels == current_label_index] = -1

    selected_ids = selected_label[current_label_index]
    if len(selected_ids) == 0:
        return

    selected_positions = voxel_positions[selected_ids]

    # Query KD-tree for voxels within render_radius around selected positions
    indices_list = voxel_tree.query_ball_point(selected_positions, r=render_radius)

    # Collect indices
    indices_within_sphere = set()
    for indices in indices_list:
        indices_within_sphere.update(indices)

    indices_within_sphere = np.array(list(indices_within_sphere), dtype=int)

    # Update render_labels
    render_labels[indices_within_sphere] = current_label_index


def render_img_with_camera_pose(px, py, pz, cam_alpha=0.785, cam_beta=0.785,
                                cam_pan_x=0.0, cam_pan_y=0.0, cam_zoom=1.0,
                                get_depth=False, ray_x=0, ray_y=0, ray_z=1, steep_threshold=16):

    global render_labels
    # Get mask and depth
    sin_alpha = float(np.sin(cam_alpha))
    cos_alpha = float(np.cos(cam_alpha))
    sin_beta = float(np.sin(cam_beta))
    cos_beta = float(np.cos(cam_beta))

    new_x2 = cos_alpha*px - sin_alpha*py
    new_y2 = sin_alpha*px + cos_alpha*py
    new_z2 = pz

    new_x3 = sin_beta*new_x2 + cos_beta*new_z2
    new_y3 = new_y2
    new_z3 = cos_beta*new_x2 - sin_beta*new_z2

    # Apply panning
    new_x3 += cam_pan_x
    new_y3 += cam_pan_y

    # Apply zooming
    new_x3 *= cam_zoom
    new_y3 *= cam_zoom

    # Map to image coordinates
    new_x = ((new_x3 + 0.5) * render_resolution).astype(np.int32)
    new_y = ((new_y3 + 0.5) * render_resolution).astype(np.int32)

    depth = np.full([render_resolution, render_resolution], 10, np.float32)
    ids = np.full([render_resolution, render_resolution], -1, np.int32)

    cutils.z_buffering_points(new_x, new_y, new_z3, depth, ids)
    mask = ids >= 0
    depth = depth * render_resolution

    # Visualize
    if get_depth:  # Depth
        output = 255 + (np.min(depth) - depth) / 4
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:  # Surface
        dx = depth[:, :-1] - depth[:, 1:]
        dy = depth[:-1, :] - depth[1:, :]
        dxp = 0
        dyp = 0
        counter = 0
        # 1. get normal
        for iii in range(12):
            if iii == 0:  # /\ 12 25
                partial_dx = dx[:-2, :-1]
                partial_dy = dy[:-1, 1:-1]
            elif iii == 1:  # /\ 45 14
                partial_dx = dx[1:-1, :-1]
                partial_dy = dy[:-1, :-2]
            elif iii == 2:  # /\ 45 25
                partial_dx = dx[1:-1, :-1]
                partial_dy = dy[:-1, 1:-1]
            elif iii == 3:  # /\ 23 25
                partial_dx = dx[:-2, 1:]
                partial_dy = dy[:-1, 1:-1]
            elif iii == 4:  # /\ 56 25
                partial_dx = dx[1:-1, 1:]
                partial_dy = dy[:-1, 1:-1]
            elif iii == 5:  # /\ 56 36
                partial_dx = dx[1:-1, 1:]
                partial_dy = dy[:-1, 2:]
            elif iii == 6:  # /\ 56 69
                partial_dx = dx[1:-1, 1:]
                partial_dy = dy[1:, 2:]
            elif iii == 7:  # /\ 56 58
                partial_dx = dx[1:-1, 1:]
                partial_dy = dy[1:, 1:-1]
            elif iii == 8:  # /\ 89 58
                partial_dx = dx[2:, 1:]
                partial_dy = dy[1:, 1:-1]
            elif iii == 9:  # /\ 78 58
                partial_dx = dx[2:, :-1]
                partial_dy = dy[1:, 1:-1]
            elif iii == 10:  # /\ 45 58
                partial_dx = dx[1:-1, :-1]
                partial_dy = dy[1:, 1:-1]
            elif iii == 11:  # /\ 45 47
                partial_dx = dx[1:-1, :-1]
                partial_dy = dy[1:, :-2]
            partial_m = (np.abs(partial_dx) < steep_threshold) & (np.abs(partial_dy) < steep_threshold)
            dxp = dxp + partial_dx * partial_m
            dyp = dyp + partial_dy * partial_m
            counter = counter + partial_m

        counter = np.maximum(counter, 1)
        dxp = dxp / counter
        dyp = dyp / counter

        ds = np.sqrt(dxp**2 + dyp**2 + 1)
        dxp = dxp / ds
        dyp = dyp / ds
        dzp = 1.0 / ds

        output = dxp * ray_x + dyp * ray_y + dzp * ray_z

        output = output * 220 + (1 - mask[1:-1, 1:-1]) * 256
        output = np.clip(output, 0, 255).astype(np.uint8)

        # Create color output image
        color_output = np.ones((render_resolution - 2, render_resolution - 2, 3), dtype=np.uint8) * 255

        # Prepare for coloring based on render_labels
        ids_trimmed = ids[1:-1, 1:-1]
        mask_ids = ids_trimmed >= 0
        ids_flat = ids_trimmed[mask_ids]
        render_labels_flat = render_labels[ids_flat]
        output_flat = output[mask_ids]

        color_flat = np.zeros((len(ids_flat), 3), dtype=np.uint8)

        # For voxels with labels, set color based on label
        for label_index in range(len(mask_colors)):
            indices_label = render_labels_flat == label_index
            if np.any(indices_label):
                color = mask_colors[label_index]
                color_flat[indices_label] = color  # BGR
        # For voxels without label (-1), set grayscale color based on shading
        indices_unlabeled = render_labels_flat == -1
        if np.any(indices_unlabeled):
            gray_colors = output_flat[indices_unlabeled][:, None] * np.array([1, 1, 1], dtype=np.uint8)
            color_flat[indices_unlabeled] = gray_colors

        # Assign colors back to color_output
        color_output[mask_ids] = color_flat

    return color_output, depth, ids

def add_loop_voxels(label_index):
    """Add voxels along the loop defined by the ordered selected voxels."""
    selected_ids = selected_label[label_index]
    num_selected = len(selected_ids)
    if num_selected < 3:
        print(f"Not enough points to form a loop in label {label_index}.")
        return

    # Assuming selected_ids are in order forming a loop (last connects to first)
    path_voxel_ids = []

    for i in range(num_selected):
        id1 = selected_ids[i]
        id2 = selected_ids[(i + 1) % num_selected]  # Wrap around to form a loop
        pos1 = voxel_positions[id1]
        pos2 = voxel_positions[id2]
        # Compute line between pos1 and pos2
        distance = np.linalg.norm(pos2 - pos1)
        num_steps = int(distance / (1/1024))
        num_steps = max(num_steps, 2)
        t_values = np.linspace(0, 1, num_steps)
        line_positions = pos1[None, :] + (pos2 - pos1)[None, :] * t_values[:, None]

        # Query nearest voxel for each position
        distances_to_voxels, indices = voxel_tree.query(line_positions)

        # Collect voxel IDs in order
        path_voxel_ids.extend(indices.tolist())

    # Remove duplicates while preserving order
    seen = set()
    ordered_unique_ids = []
    for vid in path_voxel_ids:
        if vid not in seen:
            seen.add(vid)
            ordered_unique_ids.append(vid)

    # Update selected_label[label_index] with the new ordered_unique_ids
    selected_label[label_index] = ordered_unique_ids
    print(f"Loop voxels added to label {label_index}. Total voxels in loop: {len(selected_label[label_index])}")


# Load the initial voxel
if not load_voxel(current_index):
    print("Initial voxel failed to load. Attempting to load the next voxel.")
    current_index = attempt_load_voxel(current_index, direction='next')

# Initialize previous camera parameters
previous_cam_beta = cam_beta
previous_cam_alpha = cam_alpha
previous_cam_pan_x = cam_pan_x
previous_cam_pan_y = cam_pan_y
previous_cam_zoom = cam_zoom
UI_image = None
needs_update = False  # Flag to update render_labels

# UI loop
while True:
    # Update render_labels if needed
    if needs_update:
        update_render_labels(current_label_index)
        needs_update = False

    # Render image if camera parameters have changed or if a new voxel is loaded
    if (UI_image is None or
        previous_cam_beta != cam_beta or
        previous_cam_alpha != cam_alpha or
        previous_cam_pan_x != cam_pan_x or
        previous_cam_pan_y != cam_pan_y or
        previous_cam_zoom != cam_zoom):

        previous_cam_beta = cam_beta
        previous_cam_alpha = cam_alpha
        previous_cam_pan_x = cam_pan_x
        previous_cam_pan_y = cam_pan_y
        previous_cam_zoom = cam_zoom

        UI_image, depth, ids = render_img_with_camera_pose(
            px, py, pz, cam_alpha, cam_beta, cam_pan_x, cam_pan_y, cam_zoom, get_depth=False)
        #print(f"Camera Angles - Beta: {cam_beta:.3f}, Alpha: {cam_alpha:.3f}, Pan: ({cam_pan_x:.3f}, {cam_pan_y:.3f}), Zoom: {cam_zoom:.3f}")

    # Display the image
    cv2.imshow(Window_name, UI_image[::-1])

    # Wait for key press for 1 ms
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to exit
        print("Exiting the program.")
        save_labels()  # Save labels before exiting
        break
    elif key == ord(' '):  # Space key to exit (as in original code)
        print("Space key pressed. Exiting the program.")
        save_labels()  # Save labels before exiting
        break
    elif key == ord('s'):  # Next voxel
        save_labels()  # Save labels before moving to next model
        if current_index < len(obj_names) - 1:
            current_index += 1
            print(f"Attempting to load next voxel: Index {current_index}")
            new_index = attempt_load_voxel(current_index, direction='next')
            if new_index != current_index:
                current_index = new_index
            else:
                print("No valid next voxel loaded. Staying on the current voxel.")
            # Reset camera angles if desired
            cam_alpha = 0.0
            cam_beta = 0.0
            cam_pan_x = 0.0
            cam_pan_y = 0.0
            cam_zoom = 1.0
            UI_image = None  # Force re-render
            print(f"Current voxel index: {current_index + 1}/{len(obj_names)}")
        else:
            print("Already at the last voxel. Cannot move to the next voxel.")
    elif key == ord('w'):  # Previous voxel
        save_labels()  # Save labels before moving to previous model
        if current_index > 0:
            current_index -= 1
            print(f"Attempting to load previous voxel: Index {current_index}")
            new_index = attempt_load_voxel(current_index, direction='prev')
            if new_index != current_index:
                current_index = new_index
            else:
                print("No valid previous voxel loaded. Staying on the current voxel.")
            # Reset camera angles if desired
            cam_alpha = 0.0
            cam_beta = 0.0
            cam_pan_x = 0.0
            cam_pan_y = 0.0
            cam_zoom = 1.0
            UI_image = None  # Force re-render
            print(f"Current voxel index: {current_index + 1}/{len(obj_names)}")
        else:
            print("Already at the first voxel. Cannot move to the previous voxel.")
    elif key == ord('z'):  # Switch to zoom mode
        current_mode = 'zoom'
        print("Switched to zoom mode.")
    elif key == ord('x'):  # Switch to select mode
        current_mode = 'select'
        print("Switched to select mode.")
    elif key == ord('a'):  # Decrement current label
        current_label_index = (current_label_index - 1)
        if current_label_index < 0:
            current_label_index = 0
            print("there is no smaller label")
        print(f"Current label index: {current_label_index}, Color: {mask_colors[current_label_index]}, number of selected voxels: {len(selected_label[current_label_index])}")
    elif key == ord('d'):  # Increment current label
        current_label_index = (current_label_index + 1) 
        if current_label_index == len(mask_colors):
            current_label_index = len(mask_colors) - 1
            print("there is no bigger label")
        print(f"Current label index: {current_label_index}, Color: {mask_colors[current_label_index]}, number of selected voxels: {len(selected_label[current_label_index])}")
    elif key == ord('c'):  # Clear selected points in current label
        selected_label[current_label_index] = []
        needs_update = True
        UI_image = None  # Force re-render
        print(f"Cleared selected points in label {current_label_index}.")
    elif key == ord('f'):  # Form loop and add voxels along path
        if len(selected_label[current_label_index]) >= 3:
            add_loop_voxels(current_label_index)
            needs_update = True
            UI_image = None  # Force re-render
            print(f"Formed loop and added voxels along path for label {current_label_index}.")
        else:
            print(f"Not enough points to form a loop in label {current_label_index}.")

    # Handle selection in select mode
    if selected_point is not None:
        x, y = selected_point
        # Get depth value at (x, y)
        y_inverted = render_resolution - y - 1  # Invert Y-axis
        if 0 <= y_inverted < depth.shape[0] and 0 <= x < depth.shape[1]:
            id = ids[y_inverted, x]
            if id >= 0:
                # Add voxel id to selected_label at current_label_index
                selected_label[current_label_index].append(id)
                needs_update = True
                UI_image = None  # Force re-render
                print(f"Voxel id {id} at pixel ({x}, {y}) added to label {current_label_index}.")
            else:
                print(f"No voxel at pixel ({x}, {y}).")
        else:
            print(f"Selected pixel coordinate: ({x}, {y}), out of bounds")
        selected_point = None  # Reset for next selection

# Cleanup
cv2.destroyAllWindows()
