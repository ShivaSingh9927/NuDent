import trimesh
import numpy as np
import pyglet
from trimesh.viewer.windowed import SceneViewer

class ToothPicker(SceneViewer):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        # Initialize vertex colors (gray)
        self.base_color = [200, 200, 200, 150]
        self.mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), self.base_color, dtype=np.uint8)
        
        # Create a scene
        scene = trimesh.Scene(mesh)
        super().__init__(scene, **kwargs)
        
        # Prepare ray intersector for clicking
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    def on_mouse_press(self, x, y, buttons, modifiers):
        """Handle mouse clicks for segmentation."""
        super().on_mouse_press(x, y, buttons, modifiers)
        
        # Left Click to segment
        if buttons & pyglet.window.mouse.LEFT:
            self._segment_at_mouse(x, y)
        # Right Click to reset
        elif buttons & pyglet.window.mouse.RIGHT:
            self.mesh.visual.vertex_colors = np.full((len(self.mesh.vertices), 4), self.base_color, dtype=np.uint8)

    def _segment_at_mouse(self, x, y):
        # 1. Convert 2D mouse click to 3D Ray
        # Trimesh 4.x uses the camera object for specific coordinate rays
        # We invert Y because pyglet is bottom-up and trimesh camera is top-down
        y_fix = self.height - y
        origins, directions = self.scene.camera.camera_rays(coords=[[x, y_fix]])
        ray_origin, ray_direction = origins[0], directions[0]
        
        # 2. Find intersection with mesh
        index_tri, index_ray, locations = self.intersector.intersects_id(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            multiple_hits=False
        )
        
        if len(locations) > 0:
            click_point = locations[0]
            print(f"Clicked at: {click_point}")
            
            # 3. Segmentation Logic:
            # - Find vertices within a small horizontal radius (X, Y)
            # - And above the clicked Z coordinate
            horizontal_threshold = 6.0 # mm (width of a tooth)
            
            # Calculate horizontal distance (X, Y only)
            xy_dist = np.linalg.norm(self.mesh.vertices[:, :2] - click_point[:2], axis=1)
            
            # Condition: Close in XY and HIGHER in Z
            # (Assuming Z+ is 'up' towards the tooth crown)
            mask = (xy_dist < horizontal_threshold) & (self.mesh.vertices[:, 2] > click_point[2])
            
            # Apply Highlight Color (Red)
            new_colors = self.mesh.visual.vertex_colors.copy()
            new_colors[mask] = [255, 50, 50, 255]
            self.mesh.visual.vertex_colors = new_colors
            
            print(f"Segmented {np.sum(mask)} vertices.")

# --- Execution ---
stl_path = '/home/shiva/Documents/NuDent/stl/il20254263_---_default_2026-04-17-13-12-11-21-22-23-24-25-26-27-modelbase.stl'
print("Loading model for interactive segmentation...")
mesh = trimesh.load(stl_path)

print("\n--- Interactive Controls ---")
print("LEFT CLICK:  Select tooth (Gum -> Crown)")
print("RIGHT CLICK: Reset colors")
print("----------------------------\n")

# Start the interactive viewer
ToothPicker(mesh, width=1024, height=768, caption="NuDent Tooth Segmenter")
