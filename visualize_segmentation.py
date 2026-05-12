import trimesh
import numpy as np

# 1. Load your raw patient STL file
print("Loading mesh...")
mesh = trimesh.load('/home/shiva/Documents/NuDent/stl/il20254263_---_default_2026-04-17-13-12-11-21-22-23-24-25-26-27-bridge_slm_cad.stl')

# 2. Simulate the AI "Selection" using a bounding box.
# You will need to tweak these coordinates based on the scale of your specific STL
# to ensure the box actually hits a tooth.
bounds = np.array([
    [-15.0, -15.0, -15.0],  # Min X, Y, Z
    [15.0, 15.0, 15.0]      # Max X, Y, Z
])
bounding_box = trimesh.primitives.Box(bounds=bounds)

# 3. Create a segmentation mask
# We check which vertices of the mesh fall inside our bounding box
print("Calculating segmentation mask...")
is_inside = bounding_box.contains(mesh.vertices)

# 4. Apply Visuals
# Set the entire mesh to a default semi-transparent gray
mesh.visual.vertex_colors = [200, 200, 200, 100] 

# Highlight the "selected" tooth vertices in bright red
mesh.visual.vertex_colors[is_inside] = [255, 50, 50, 255]

# 5. Render the result in an interactive desktop window
print("Opening 3D viewer. Use your mouse to rotate and zoom.")
mesh.show()