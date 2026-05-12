import pyvista as pv

# Data lists to track math and visuals
picked_points = []
point_actors = []  
line_actors = []   

def draw_margin_manually(file_path):
    print("Loading mesh...")
    mesh = pv.read(file_path)

    plotter = pv.Plotter()
    
    # 1. CAPTURE THE JAW ACTOR
    # We save the output of add_mesh so we can control it later
    jaw_actor = plotter.add_mesh(mesh, color="white", opacity=1.0, pickable=True)

    # --- 2. THE DRAWING LOGIC ---
    def pick_callback(point):
        if point is None: 
            return 
            
        print(f"Point snapped at: {point}")
        picked_points.append(point)
        
        sphere = pv.Sphere(radius=0.4, center=point)
        p_actor = plotter.add_mesh(sphere, color='red', pickable=False)
        point_actors.append(p_actor)
        
        if len(picked_points) > 1:
            line = pv.Line(picked_points[-2], picked_points[-1])
            tube = line.tube(radius=0.15)
            l_actor = plotter.add_mesh(tube, color='red', pickable=False)
            line_actors.append(l_actor)

    # --- 3. THE CLOSE LOOP LOGIC ---
    def close_loop_callback():
        if len(picked_points) < 3:
            print("Need at least 3 points to close the loop!")
            return
            
        if len(line_actors) == len(point_actors):
            print("Loop is already closed.")
            return

        print("Closing the margin line...")
        line = pv.Line(picked_points[-1], picked_points[0])
        tube = line.tube(radius=0.15)
        l_actor = plotter.add_mesh(tube, color='red', pickable=False)
        line_actors.append(l_actor)

    # --- 4. THE UNDO LOGIC ---
    def undo_callback():
        if not picked_points:
            print("Nothing to undo!")
            return
            
        print("Undoing...")
        
        if len(line_actors) == len(point_actors):
            last_line_actor = line_actors.pop()
            plotter.remove_actor(last_line_actor)
            return 

        if len(line_actors) > 0:
            last_line_actor = line_actors.pop()
            plotter.remove_actor(last_line_actor)
            
        last_point_actor = point_actors.pop()
        plotter.remove_actor(last_point_actor)
        picked_points.pop()

    # --- 5. THE CLEAR ALL LOGIC ---
    def clear_callback():
        print("Clearing all points...")
        while picked_points:
            if len(line_actors) == len(point_actors):
                plotter.remove_actor(line_actors.pop())
            if len(line_actors) > 0:
                plotter.remove_actor(line_actors.pop())
            plotter.remove_actor(point_actors.pop())
            picked_points.pop()

    # --- 6. THE HIDE/SHOW LOGIC (NEW) ---
    # --- 6. THE HIDE/SHOW LOGIC (FIXED) ---
    def toggle_visibility_callback():
        # Flip the visibility state
        jaw_actor.visibility = not jaw_actor.visibility
        state = "Hidden" if not jaw_actor.visibility else "Visible"
        print(f"Jaw scan is now {state}.")
        
        # CRITICAL FIX: Force the 3D window to refresh the frame immediately!
        plotter.render()

    # --- 7. VIEWER SETUP & KEY BINDINGS ---
    print("Opening 3D viewer...")
    
    # Bind our custom functions to keyboard keys
    plotter.add_key_event('z', undo_callback)
    plotter.add_key_event('u', undo_callback)
    plotter.add_key_event('c', clear_callback)
    plotter.add_key_event('f', close_loop_callback) 
    plotter.add_key_event('h', toggle_visibility_callback) # The new Hide command
    
    plotter.enable_surface_point_picking(
        callback=pick_callback, 
        left_clicking=True,
        show_point=False, 
        show_message="Click: Pick | 'f': Close | 'h': Hide Jaw | 'z': Undo | 'c': Clear"
    )

    plotter.set_background('gray')
    plotter.show()
    
    if len(picked_points) > 0:
        print(f"\nFinal closed margin line saved with {len(picked_points)} vertices!")

# Run the manual tool
stl_path = "/home/shiva/Documents/NuDent/Full_Denture_Set_-_Upper_and_Lower_Anatomy_(STL)/NCT_DentalCAD/full-denture-upper-lower-teeth/2024-01-08_99999-001-37-36-35-34-33-32-31-41-42-43-44-45-46-47-bridge_slm_cad.stl"
draw_margin_manually(stl_path)