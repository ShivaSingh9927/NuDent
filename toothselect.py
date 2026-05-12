import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os

# Data lists and App State
picked_points = []
point_actors = []  
line_actors = []   
tooth_actors = [] # NEW: Tracks colored teeth so we can undo them!
app_state = {"mode": "drawing"}

def draw_margin_manually(file_path):
    print("Loading mesh...")
    mesh = pv.read(file_path)

    plotter = pv.Plotter()
    jaw_actor = plotter.add_mesh(mesh, color="white", opacity=1.0, pickable=True)

    # --- 1. THE MOUSE CLICK LOGIC ---
    def pick_callback(point):
        if point is None: return 
        
        if app_state["mode"] == "extracting":
            execute_extraction(point)
            return
            
        picked_points.append(point)
        
        sphere = pv.Sphere(radius=0.4, center=point)
        p_actor = plotter.add_mesh(sphere, color='red', pickable=False)
        point_actors.append(p_actor)
        
        if len(picked_points) > 1:
            line = pv.Line(picked_points[-2], picked_points[-1])
            tube = line.tube(radius=0.15)
            l_actor = plotter.add_mesh(tube, color='red', pickable=False)
            line_actors.append(l_actor)

    def extract_mode_callback():
        if len(picked_points) < 3: return
        print("\n--- EXTRACTION MODE ACTIVATED ---")
        print("INSTRUCTION: Left-Click directly on the TOP (cusp) of the tooth you want to keep.")
        app_state["mode"] = "extracting"

    # --- 2. THE EXTRACTION ALGORITHM ---
    def execute_extraction(tip_point):
        print("\nExtracting tooth... please wait.")
        
        loop_points = np.vstack((picked_points, picked_points[0]))
        spline = pv.Spline(loop_points, 1000)

        tree = cKDTree(mesh.points)
        cut_radius = 1.2 
        distances, closest_pts = tree.query(spline.points, distance_upper_bound=cut_radius)
        points_to_cut = np.unique(closest_pts[distances != np.inf])
        print(f"Severing {len(points_to_cut)} points to create a watertight gap...")

        mask = np.ones(mesh.n_points, dtype=bool)
        mask[points_to_cut] = False
        points_to_keep = np.where(mask)[0]
        
        severed_mesh = mesh.extract_points(points_to_keep)

        print("Running region grow (this may take 5-10 seconds)...")
        click_id = severed_mesh.find_closest_point(tip_point)
        extracted_ugrid = severed_mesh.connectivity('point_seed', point_ids=[click_id])
        tooth_surface = extracted_ugrid.extract_surface()

        print(f"Extraction complete! Highlighting {tooth_surface.n_points} vertices.")

        # CRITICAL FIX: The new overlay UI logic
        # 1. Overlay the gold tooth directly on the jaw
        t_actor = plotter.add_mesh(tooth_surface, color='gold')
        tooth_actors.append(t_actor) # Save it so we can undo it later!
        
        # 2. Erase the red margin line automatically to clean up the view
        clear_callback()
        
        # 3. Reset the state so the user can immediately start drawing the next tooth
        app_state["mode"] = "drawing"
        print("\nSUCCESS! Tooth colored. You can now draw a new line, or press 'z' to undo the color.")
        
        plotter.render()

    # --- 3. STATE SAVING LOGIC ---
    def save_state_callback():
        if len(picked_points) == 0: return
        np.save("debug_margin_points.npy", np.array(picked_points))
        print(f"Saved {len(picked_points)} points to debug_margin_points.npy!")

    def load_state_callback():
        if not os.path.exists("debug_margin_points.npy"): return
        print("Loading saved margin points...")
        clear_callback() 
        loaded_points = np.load("debug_margin_points.npy")
        for pt in loaded_points:
            pick_callback(pt)
        print("Line loaded! Press 'f' to close it, then 't' to extract.")
        plotter.render()

    # --- 4. UTILITY CALLBACKS ---
    def close_loop_callback():
        if len(picked_points) < 3: return
        if len(line_actors) == len(point_actors): return
        print("Closing loop...")
        line = pv.Line(picked_points[-1], picked_points[0])
        l_actor = plotter.add_mesh(line.tube(radius=0.15), color='red', pickable=False)
        line_actors.append(l_actor)

    def toggle_visibility_callback():
        jaw_actor.visibility = not jaw_actor.visibility
        plotter.render()

    # CRITICAL FIX: Upgraded Undo Logic
    def undo_callback():
        # If there are no red points currently drawn, check if there is a gold tooth to undo!
        if len(picked_points) == 0:
            if len(tooth_actors) > 0:
                print("Undoing tooth color...")
                t_actor = tooth_actors.pop()
                plotter.remove_actor(t_actor)
                plotter.render()
            else:
                print("Nothing to undo!")
            return

        # Normal red point undo logic
        print("Undoing point/line...")
        if len(line_actors) == len(point_actors):
            plotter.remove_actor(line_actors.pop())
            return 
        if len(line_actors) > 0:
            plotter.remove_actor(line_actors.pop())
        plotter.remove_actor(point_actors.pop())
        picked_points.pop()
        plotter.render()

    def clear_callback():
        while picked_points:
            if len(line_actors) == len(point_actors): plotter.remove_actor(line_actors.pop())
            if len(line_actors) > 0: plotter.remove_actor(line_actors.pop())
            plotter.remove_actor(point_actors.pop())
            picked_points.pop()
        plotter.render()

    # --- 5. VIEWER SETUP ---
    plotter.add_key_event('z', undo_callback)
    plotter.add_key_event('u', undo_callback)
    plotter.add_key_event('c', clear_callback)
    plotter.add_key_event('f', close_loop_callback) 
    plotter.add_key_event('h', toggle_visibility_callback) 
    plotter.add_key_event('t', extract_mode_callback) 
    plotter.add_key_event('w', save_state_callback) 
    plotter.add_key_event('l', load_state_callback) 

    plotter.enable_surface_point_picking(
        callback=pick_callback, 
        left_clicking=True,
        show_point=False, 
        show_message="Click: Pick | 'f': Close | 't': Extract | 'w': Save | 'l': Load | 'z': Undo"
    )

    plotter.set_background('gray')
    plotter.show()

# Run it
stl_path = "/home/shiva/Documents/NuDent/2023-10-01_99999-011-lowerjaw.stl"
draw_margin_manually(stl_path)