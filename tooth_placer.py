import os
import sys
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QGridLayout, QFrame)
from PyQt5.QtCore import Qt

class NuDentApp(QMainWindow):
    def __init__(self, jaw_path, library_dir):
        super().__init__()
        self.setWindowTitle("NuDent CAD - Pro Library Placer")
        self.resize(1200, 800) # Set a nice large default window size

        # --- DATA SETUP ---
        self.library_dir = library_dir
        self.available_teeth = sorted([f for f in os.listdir(library_dir) if f.endswith('.stl')])
        self.current_index = 0
        
        self.crown = None
        self.crown_actor = None
        self.move_step = 0.5

        # --- UI LAYOUT ---
        # The central widget holds everything
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # LEFT SIDE: The 3D Viewer (takes up 75% of the screen)
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, stretch=3)

        # RIGHT SIDE: The Control Panel (takes up 25% of the screen)
        panel = QFrame()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout(panel)
        main_layout.addWidget(panel)

        # --- BUILD THE CONTROL PANEL ---
        
        # 1. Library Status
        self.lbl_info = QLabel("Loading Jaw...")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        panel_layout.addWidget(self.lbl_info)

        # 2. Browsing Buttons
        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("<< Prev")
        btn_next = QPushButton("Next >>")
        btn_prev.clicked.connect(self.load_prev)
        btn_next.clicked.connect(self.load_next)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        panel_layout.addLayout(nav_layout)

        # 3. Translation D-Pad (Using a Grid)
        panel_layout.addWidget(QLabel("\nPosition (X / Y)"))
        grid = QGridLayout()
        btn_fwd = QPushButton("Forward")
        btn_back = QPushButton("Backward")
        btn_left = QPushButton("Left")
        btn_right = QPushButton("Right")
        
        btn_fwd.clicked.connect(lambda: self.move_crown([0, self.move_step, 0]))
        btn_back.clicked.connect(lambda: self.move_crown([0, -self.move_step, 0]))
        btn_left.clicked.connect(lambda: self.move_crown([-self.move_step, 0, 0]))
        btn_right.clicked.connect(lambda: self.move_crown([self.move_step, 0, 0]))
        
        grid.addWidget(btn_fwd, 0, 1)
        grid.addWidget(btn_left, 1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_back, 2, 1)
        panel_layout.addLayout(grid)

        # Height Controls (Z)
        panel_layout.addWidget(QLabel("\nHeight (Z)"))
        z_layout = QHBoxLayout()
        btn_up = QPushButton("Up")
        btn_down = QPushButton("Down")
        btn_up.clicked.connect(lambda: self.move_crown([0, 0, self.move_step]))
        btn_down.clicked.connect(lambda: self.move_crown([0, 0, -self.move_step]))
        z_layout.addWidget(btn_up)
        z_layout.addWidget(btn_down)
        panel_layout.addLayout(z_layout)

        # 4. Shape & Utilities
        panel_layout.addWidget(QLabel("\nShape Utilities"))
        btn_scale_up = QPushButton("Scale Up (+5%)")
        btn_scale_down = QPushButton("Scale Down (-5%)")
        btn_mirror = QPushButton("Mirror Anatomy")
        
        btn_scale_up.clicked.connect(self.scale_up)
        btn_scale_down.clicked.connect(self.scale_down)
        btn_mirror.clicked.connect(self.mirror_crown)
        
        panel_layout.addWidget(btn_scale_up)
        panel_layout.addWidget(btn_scale_down)
        panel_layout.addWidget(btn_mirror)

        # 5. Save Button (Pushed to the bottom)
        panel_layout.addStretch() # Adds blank space to push save to the bottom
        btn_save = QPushButton("SAVE PLACEMENT")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self.save_crown)
        panel_layout.addWidget(btn_save)

        # --- INITIALIZE 3D ENVIRONMENT ---
        print("Loading Jaw Mesh...")
        self.jaw = pv.read(jaw_path)
        self.jaw_center = self.jaw.center
        self.plotter.add_mesh(self.jaw, color="white", opacity=0.6)
        self.plotter.set_background('gray')
        
        self.load_tooth(0) # Load the first tooth on startup

    # --- APPLICATION LOGIC / METHODS ---

    def load_tooth(self, index):
        if self.crown_actor is not None:
            self.plotter.remove_actor(self.crown_actor)

        tooth_filename = self.available_teeth[index]
        self.lbl_info.setText(f"Active:\n{tooth_filename}")
        
        tooth_path = os.path.join(self.library_dir, tooth_filename)
        self.crown = pv.read(tooth_path)

        # Auto-Normalize the scale (Force to ~10mm)
        x_length = self.crown.bounds[1] - self.crown.bounds[0]
        if x_length < 6.0 or x_length > 15.0: 
            scale_factor = 10.0 / x_length
            self.crown.points *= scale_factor

        # Teleport to the jaw center
        c_center = self.crown.center
        self.crown.translate([
            self.jaw_center[0] - c_center[0], 
            self.jaw_center[1] - c_center[1], 
            self.jaw_center[2] - c_center[2] + 20.0 
        ], inplace=True)

        self.crown_actor = self.plotter.add_mesh(self.crown, color="gold")
        self.plotter.render()

    def load_next(self):
        self.current_index = (self.current_index + 1) % len(self.available_teeth)
        self.load_tooth(self.current_index)

    def load_prev(self):
        self.current_index = (self.current_index - 1) % len(self.available_teeth)
        self.load_tooth(self.current_index)

    def move_crown(self, direction):
        if self.crown:
            self.crown.translate(direction, inplace=True)
            self.plotter.render()

    def scale_up(self):
        if self.crown:
            c = self.crown.center
            self.crown.translate([-c[0], -c[1], -c[2]], inplace=True)
            self.crown.points *= 1.05  
            self.crown.translate(c, inplace=True)
            self.plotter.render()

    def scale_down(self):
        if self.crown:
            c = self.crown.center
            self.crown.translate([-c[0], -c[1], -c[2]], inplace=True)
            self.crown.points *= 0.95  
            self.crown.translate(c, inplace=True)
            self.plotter.render()

    def mirror_crown(self):
        if self.crown:
            self.crown.points[:, 0] *= -1
            self.crown.flip_normals()
            self.plotter.render()

    def save_crown(self):
        if self.crown:
            filename = self.available_teeth[self.current_index]
            out_name = f"placed_{filename}"
            self.crown.save(out_name)
            self.lbl_info.setText(f"SAVED SUCCESSFULLY!\n{out_name}")

# --- APP EXECUTION ---
if __name__ == '__main__':
    # Define your paths here
    jaw_stl = "/home/shiva/Documents/NuDent/2023-10-01_99999-011-lowerjaw.stl"
    lib_dir = "/home/shiva/Documents/NuDent/Tooth_Library/tooth_stl"

    app = QApplication(sys.argv)
    window = NuDentApp(jaw_stl, lib_dir)
    window.show()
    sys.exit(app.exec_())