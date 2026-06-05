import trimesh
import numpy as np

def visualize_jaws(upper_path, lower_path):
    print(f"Loading Upper Jaw: {upper_path}")
    upper = trimesh.load(upper_path)
    # Color it Blue (Semi-transparent)
    upper.visual.face_colors = [0, 0, 255, 150]

    print(f"Loading Lower Jaw: {lower_path}")
    lower = trimesh.load(lower_path)
    # Color it Red (Semi-transparent)
    lower.visual.face_colors = [255, 0, 0, 150]

    print("Opening viewer... (Close the window to exit)")
    # Create a scene and show it
    scene = trimesh.Scene([upper, lower])
    scene.show()

if __name__ == "__main__":
    upper_stl = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-034/2023-10-01_99999-034-upperjaw.stl"
    lower_stl = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-034/2023-10-01_99999-034-lowerjaw.stl"
    
    visualize_jaws(upper_stl, lower_stl)
