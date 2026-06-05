import os
import trimesh
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

class MarginTool:
    def __init__(self, mesh_path):
        self.mesh = trimesh.load(mesh_path)
        print(f"Mesh loaded: {mesh_path}")
        
        # Pre-calculate curvature for speed
        print("Pre-calculating surface curvature...")
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(self.mesh, self.mesh.vertices, 0.5)
        self.curv_norm = np.abs(curvature)
        self.curv_norm = (self.curv_norm - self.curv_norm.min()) / (self.curv_norm.max() - self.curv_norm.min() + 1e-6)
        
        self.tree = cKDTree(self.mesh.vertices)
        self.base_colors = np.full((len(self.mesh.vertices), 4), [200, 200, 200, 255], dtype=np.uint8)
        self.mesh.visual.vertex_colors = self.base_colors.copy()

    def on_click(self, x, y):
        try:
            # Get ray from the viewer's current scene
            # We'll use the viewer's own internal raycaster if possible
            # But for now, we'll use a semi-auto approach: 
            # Find the vertex closest to the center of the screen
            pass

    def run_rolling_ball(self, click_point):
        try:
            _, seed_idx = self.tree.query(click_point)
            current_idx = int(seed_idx)
            
            print(f"Rolling ball from vertex {current_idx}...")
            path_down = [current_idx]
            
            for step in range(150):
                neighbors = np.array(list(self.mesh.vertex_neighbors[current_idx]))
                if len(neighbors) == 0: break
                
                neighbor_z = self.mesh.vertices[neighbors, 2]
                current_z = self.mesh.vertices[current_idx, 2]
                
                lower_mask = neighbor_z < current_z
                if not np.any(lower_mask): break
                    
                lower_indices = neighbors[lower_mask]
                lower_z_vals = neighbor_z[lower_mask]
                
                if self.curv_norm[current_idx] > 0.4:
                    print(f"Margin Found! (Step {step})")
                    break
                    
                current_idx = int(lower_indices[np.argmin(lower_z_vals)])
                path_down.append(current_idx)

            # Highlight result
            margin_indices = np.where(self.curv_norm > 0.35)[0]
            new_colors = self.base_colors.copy()
            new_colors[margin_indices] = [0, 255, 0, 255] # Green
            new_colors[path_down] = [255, 0, 255, 255]    # Magenta
            
            self.mesh.visual.vertex_colors = new_colors
            print("Successfully updated margin visualization.")
            
        except Exception as e:
            print(f"Processing Error: {str(e)}")

def run_interactive_tool():
    jaw_path = "/home/shiva/Documents/NuDent/Anatomic_Crown/sample/2023-10-01_99999-011-lowerjaw.stl"
    if not os.path.exists(jaw_path):
        jaw_path = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-034/2023-10-01_99999-034-lowerjaw.stl"
    
    tool = MarginTool(jaw_path)
    scene = trimesh.Scene(tool.mesh)

    print("\n=== Interactive Margin Tool (Slim Version) ===")
    print("Use the 'i' key to print the vertex ID under your mouse,")
    print("or just click to see if it triggers.")
    
    # We use a much simpler callback that doesn't mess with entities
    def my_callback(window):
        @window.event
        def on_mouse_press(x, y, buttons, modifiers):
            try:
                print(f"DEBUG: Click at ({x}, {y})")
                # Use the scene's built-in ray generator
                res = window.scene.camera_rays()
                idx = int(x) + int(window.height - y) * int(window.width)
                if idx < len(res[0]):
                    ray_o = res[0][idx]
                    ray_d = res[1][idx]
                    
                    # Intersect
                    locations, _, _ = tool.mesh.ray.intersects_location([ray_o], [ray_d])
                    if len(locations) > 0:
                        tool.run_rolling_ball(locations[0])
            except Exception as e:
                print(f"Error: {e}")

    scene.show(callback=my_callback)

if __name__ == "__main__":
    run_interactive_tool()
