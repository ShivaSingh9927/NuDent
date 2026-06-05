import os
import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
from scipy.spatial import cKDTree
import networkx as nx

class MarginStudio(QtWidgets.QMainWindow):
    def __init__(self, mesh_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NuDent Margin Studio")
        self.resize(1200, 800)

        # 1. Load Data
        print(f"Loading mesh: {mesh_path}")
        self.mesh = pv.read(mesh_path)
        
        # Pre-calculate Curvature
        print("Calculating dental features...")
        curv = self.mesh.curvature(curv_type="gaussian")
        self.curv_norm = np.abs(curv)
        self.curv_norm = (self.curv_norm - self.curv_norm.min()) / (self.curv_norm.max() - self.curv_norm.min() + 1e-6)
        
        # UI State
        self.path_down = []
        self.margin_loop = []
        
        # 2. Main Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # 3. Sidebar
        sidebar = QtWidgets.QVBoxLayout()
        layout.addLayout(sidebar, 1)

        title = QtWidgets.QLabel("NuDent Control")
        title.setStyleSheet("font-weight: bold; font-size: 18px;")
        sidebar.addWidget(title)

        self.btn_reset = QtWidgets.QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_view)
        sidebar.addWidget(self.btn_reset)

        sidebar.addStretch()
        
        self.status = QtWidgets.QLabel("Status: Ready. Click tooth top.")
        sidebar.addWidget(self.status)

        # 4. 3D Viewport
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, 4)

        # Initialize Plotter
        self.plotter.add_mesh(self.mesh, color="lightgrey", smooth_shading=True, name="jaw")
        self.plotter.enable_point_picking(callback=self.on_pick, show_message=False, use_picker=True)
        self.plotter.view_isometric()

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.clear_measurements()
        if 'margin_path' in self.plotter.renderer.actors:
            self.plotter.remove_actor('margin_path')

    def on_pick(self, point):
        self.status.setText(f"Status: Point Picked. Processing...")
        self.run_margin_detection(point)

    def run_margin_detection(self, click_point):
        try:
            # 1. Find start vertex
            # In PyVista, we can find closest point easily
            v_idx = self.mesh.find_closest_point(click_point)
            
            # 2. Rolling Ball (Gradient Descent)
            current_idx = v_idx
            path = [current_idx]
            
            for _ in range(200):
                # Get neighbors
                # PyVista meshes (PolyData) have point_neighbors
                neighbors = self.mesh.point_neighbors(current_idx)
                if not neighbors: break
                
                # Find lower neighbor
                z_vals = self.mesh.points[neighbors, 2]
                curr_z = self.mesh.points[current_idx, 2]
                
                lower = [n for n, z in zip(neighbors, z_vals) if z < curr_z]
                if not lower: break
                
                # Check Curvature
                if self.curv_norm[current_idx] > 0.4:
                    break
                
                # Move to next
                current_idx = lower[np.argmin([self.mesh.points[n, 2] for n in lower])]
                path.append(current_idx)

            self.draw_path(path)
            self.status.setText(f"Status: Margin detected at {len(path)} steps.")
            
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")

    def draw_path(self, path_indices):
        # Create a line mesh from indices
        pts = self.mesh.points[path_indices]
        # Offset for visibility
        pts[:, 2] += 0.2
        
        # Create PolyData for the line
        line = pv.MultipleLines(points=pts)
        self.plotter.add_mesh(line, color="magenta", line_width=5, name="margin_path")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Path to your mesh
    jaw_path = "/home/shiva/Documents/NuDent/Anatomic_Crown/sample/2023-10-01_99999-011-lowerjaw.stl"
    if not os.path.exists(jaw_path):
        jaw_path = "/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-034/2023-10-01_99999-034-lowerjaw.stl"

    window = MarginStudio(jaw_path)
    window.show()
    sys.exit(app.exec_())
