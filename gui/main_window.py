import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QFileDialog, QLabel, QSpinBox,
                                QDoubleSpinBox, QMessageBox, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt
from typing import Optional, Dict, Any, cast
from gui.view_canvas import ViewCanvas
from core.nifti_io import load_nifti, save_nifti, world_to_voxel, voxel_to_world
from core.spline import build_spline_through_points, sample_equispaced
from core.resample import compute_curved_mpr
from core.utils import ControlPointManager
import imageio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Curved MPR Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.volume: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None
        self.header_info: Optional[dict] = None
        self.control_points = ControlPointManager()
        self.curved_mpr_result: Optional[np.ndarray] = None
        
        # Parameters
        self.width_mm = 40.0
        self.spacing_mm = 1.0
        self.cross_samples = 50
        self.interp_order = 1
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Views
        view_layout = QGridLayout()
        
        self.axial_canvas = ViewCanvas('axial')
        self.coronal_canvas = ViewCanvas('coronal')
        self.sagittal_canvas = ViewCanvas('sagittal')
        self.mpr_canvas = ViewCanvas('mpr')
        
        # Set callbacks
        for canvas in [self.axial_canvas, self.coronal_canvas, self.sagittal_canvas]:
            canvas.on_click_callback = self._on_canvas_click
            canvas.on_drag_callback = self._on_canvas_drag
        
        view_layout.addWidget(self.axial_canvas, 0, 0)
        view_layout.addWidget(self.coronal_canvas, 0, 1)
        view_layout.addWidget(self.sagittal_canvas, 1, 0)
        view_layout.addWidget(self.mpr_canvas, 1, 1)
        
        main_layout.addLayout(view_layout, stretch=3)
        
        # Right: Controls
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel widget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()

        load_btn = QPushButton("Load NIfTI")
        load_btn.clicked.connect(self._load_nifti)  # type: ignore[attr-defined]
        file_layout.addWidget(load_btn)

        self.info_label = QLabel("No volume loaded")
        self.info_label.setWordWrap(True)
        file_layout.addWidget(self.info_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Point operations
        point_group = QGroupBox("Control Points")
        point_layout = QVBoxLayout()
        
        self.point_count_label = QLabel("Points: 0")
        point_layout.addWidget(self.point_count_label)

        clear_btn = QPushButton("Clear Points")
        clear_btn.clicked.connect(self._clear_points)  # type: ignore[attr-defined]
        point_layout.addWidget(clear_btn)

        point_group.setLayout(point_layout)
        layout.addWidget(point_group)
        
        # Parameters
        param_group = QGroupBox("MPR Parameters")
        param_layout = QVBoxLayout()
        
        param_layout.addWidget(QLabel("Width (mm):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 200)
        self.width_spin.setValue(40)
        self.width_spin.valueChanged.connect(lambda v: setattr(self, 'width_mm', v))  # type: ignore[attr-defined]
        param_layout.addWidget(self.width_spin)
        
        param_layout.addWidget(QLabel("Spacing (mm):"))
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 10)
        self.spacing_spin.setValue(1.0)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.valueChanged.connect(lambda v: setattr(self, 'spacing_mm', v))  # type: ignore[attr-defined]
        param_layout.addWidget(self.spacing_spin)
        
        param_layout.addWidget(QLabel("Cross Samples:"))
        self.cross_spin = QSpinBox()
        self.cross_spin.setRange(10, 200)
        self.cross_spin.setValue(50)
        self.cross_spin.valueChanged.connect(lambda v: setattr(self, 'cross_samples', v))  # type: ignore[attr-defined]
        param_layout.addWidget(self.cross_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Generate
        self.generate_btn = QPushButton("Generate Curved MPR")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate_mpr)  # type: ignore[attr-defined]
        layout.addWidget(self.generate_btn)

        # Export
        self.export_btn = QPushButton("Save MPR (PNG/NIfTI)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_mpr)  # type: ignore[attr-defined]
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
        
        # help_label = QLabel(
        #     "Instructions:\n"
        #     "• Load NIfTI volume\n"
        #     "• Left-click to add points\n"
        #     "• Drag points to move\n"
        #     "• Right-click to delete\n"
        #     "• Scroll to change slice\n"
        #     "• Generate when ready"
        # )
        # help_label.setStyleSheet("background-color: #2a2a2a; padding: 10px; border-radius: 5px;")
        # help_label.setWordWrap(True)
        # layout.addWidget(help_label)
        
        return panel
    
    def _load_nifti(self):
        """Load NIfTI file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        
        try:
            self.volume, self.affine, self.header_info = load_nifti(path)
            
            # Update all views
            self.axial_canvas.set_volume(self.volume, self.affine)
            self.coronal_canvas.set_volume(self.volume, self.affine)
            self.sagittal_canvas.set_volume(self.volume, self.affine)
            
            # Update info
            shape = self.header_info['shape']
            pixdims = self.header_info['pixdims']
            self.info_label.setText(
                f"Shape: {shape}\n"
                f"Spacing: {pixdims[0]:.2f}, {pixdims[1]:.2f}, {pixdims[2]:.2f} mm"
            )
            
            # Set default spacing to minimum voxel size
            min_spacing = min(pixdims)
            self.spacing_spin.setValue(min_spacing)
            
            # Clear previous points
            self.control_points.clear()
            self._update_point_displays()
            
            QMessageBox.information(self, "Success", "NIfTI loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load NIfTI:\n{str(e)}")
    
    def _clear_points(self):
        """Clear all control points."""
        self.control_points.clear()
        self._update_point_displays()
        self.generate_btn.setEnabled(False)
    
    def _on_canvas_click(self, view_type: str, x: float, y: float, button: int):
        """Handle click on canvas."""
        if self.volume is None or self.affine is None:
            return
        
        # Get current slice for this view
        if view_type == 'axial':
            canvas = self.axial_canvas
            # x, y in pixel coords -> convert to world
            voxel = np.array([canvas.current_slice, y, x])
        elif view_type == 'coronal':
            canvas = self.coronal_canvas
            voxel = np.array([x, canvas.current_slice, y])
        elif view_type == 'sagittal':
            canvas = self.sagittal_canvas
            voxel = np.array([x, y, canvas.current_slice])
        else:
            return
        
        world = voxel_to_world(voxel, self.affine)
        
        if button == 1:  # Left click - add or select point
            # Check if clicking near existing point
            points_array = self.control_points.get_points_array()
            if len(points_array) > 0:
                # Project points to this view
                voxels = world_to_voxel(points_array, self.affine)
                
                # Get coords in view plane
                if view_type == 'axial':
                    view_coords = voxels[:, [2, 1]]  # [i, j] = [x, y]
                elif view_type == 'coronal':
                    view_coords = voxels[:, [0, 2]]  # [k, i] = [x, y]
                else:  # sagittal
                    view_coords = voxels[:, [0, 1]]  # [k, j] = [x, y]
                
                click_pos = np.array([x, y])
                distances = np.linalg.norm(view_coords - click_pos, axis=1)
                
                if distances.min() < 10:  # Within 10 pixels
                    # Select point
                    idx = int(np.argmin(distances))
                    self.control_points.select_point(idx)
                    self._update_point_displays()
                    return
            
            # Add new point
            self.control_points.add_point(world)
            self._update_point_displays()
            
        elif button == 3:  # Right click - delete
            # Find nearest point
            points_array = self.control_points.get_points_array()
            if len(points_array) > 0:
                voxels = world_to_voxel(points_array, self.affine)
                
                if view_type == 'axial':
                    view_coords = voxels[:, [2, 1]]
                elif view_type == 'coronal':
                    view_coords = voxels[:, [0, 2]]
                else:
                    view_coords = voxels[:, [0, 1]]
                
                click_pos = np.array([x, y])
                distances = np.linalg.norm(view_coords - click_pos, axis=1)
                
                if distances.min() < 10:
                    idx = int(np.argmin(distances))
                    self.control_points.remove_point(idx)
                    self._update_point_displays()
    
    def _on_canvas_drag(self, view_type: str, x: float, y: float):
        """Handle dragging on canvas."""
        if self.volume is None or self.affine is None:
            return
        
        selected = self.control_points.get_selected()
        if selected < 0:
            return
        
        # Get current point
        point_world = self.control_points.points_world[selected]
        
        # Update coordinates in view plane, keep orthogonal coord fixed
        if view_type == 'axial':
            canvas = self.axial_canvas
            voxel = np.array([canvas.current_slice, y, x])
        elif view_type == 'coronal':
            canvas = self.coronal_canvas
            voxel = np.array([x, canvas.current_slice, y])
        elif view_type == 'sagittal':
            canvas = self.sagittal_canvas
            voxel = np.array([x, y, canvas.current_slice])
        else:
            return
        
        new_world = voxel_to_world(voxel, self.affine)
        self.control_points.move_point(selected, new_world)
        self._update_point_displays()
    
    def _update_point_displays(self):
        """Update control point overlays on all views."""
        points_array = self.control_points.get_points_array()
        num_points = len(points_array)
        
        self.point_count_label.setText(f"Points: {num_points}")
        self.generate_btn.setEnabled(num_points >= 3)
        
        # Redraw all views with points
        for canvas in [self.axial_canvas, self.coronal_canvas, self.sagittal_canvas]:
            if canvas.volume is None:
                continue
            if self.affine is None:
                canvas.draw()
                continue
            
            canvas.update_display()
            
            if num_points == 0:
                canvas.draw()
                continue
            
            # Convert points to voxel coords
            voxels = world_to_voxel(points_array, cast(np.ndarray, self.affine))
            
            # Project to view
            if canvas.view_type == 'axial':
                coords = voxels[:, [2, 1]]  # [i, j] = [x, y]
            elif canvas.view_type == 'coronal':
                coords = voxels[:, [0, 2]]  # [k, i]
            else:  # sagittal
                coords = voxels[:, [0, 1]]  # [k, j]
            
            # Draw points
            selected = self.control_points.get_selected()
            for i, (x, y) in enumerate(coords):
                color = 'red' if i == selected else 'cyan'
                canvas.ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
                canvas.ax.text(x + 2, y + 2, str(i + 1), color='yellow', fontsize=10, fontweight='bold')
            
            # Draw connecting line
            if num_points > 1:
                canvas.ax.plot(coords[:, 0], coords[:, 1], 'c--', alpha=0.5, linewidth=1)
            
            # Try to draw spline if enough points
            if num_points >= 3:
                try:
                    from core.spline import build_spline_through_points
                    tck, _ = build_spline_through_points(points_array)
                    u_dense = np.linspace(0, 1, 200)
                    from scipy.interpolate import splev
                    spline_pts_world = np.array(splev(u_dense, tck)).T
                    spline_voxels = world_to_voxel(spline_pts_world, cast(np.ndarray, self.affine))
                    
                    if canvas.view_type == 'axial':
                        spline_coords = spline_voxels[:, [2, 1]]
                    elif canvas.view_type == 'coronal':
                        spline_coords = spline_voxels[:, [0, 2]]
                    else:
                        spline_coords = spline_voxels[:, [0, 1]]
                    
                    canvas.ax.plot(spline_coords[:, 0], spline_coords[:, 1], 'g-', alpha=0.7, linewidth=2)
                except:
                    pass
            
            canvas.draw()
    
    def _generate_mpr(self):
        """Generate curved MPR."""
        if self.volume is None or self.affine is None:
            QMessageBox.warning(self, "Error", "No volume loaded!")
            return
            
        points_array = self.control_points.get_points_array()
        
        if len(points_array) < 3:
            QMessageBox.warning(self, "Error", "Need at least 3 control points!")
            return
        
        try:
            # Build spline
            tck, _ = build_spline_through_points(points_array)
            
            # Sample equispaced
            positions, tangents = sample_equispaced(tck, self.spacing_mm)
            
            # Generate MPR
            self.curved_mpr_result = compute_curved_mpr(
                self.volume, self.affine, positions, tangents,
                self.width_mm, self.cross_samples, self.interp_order
            )
            
            # Display
            self.mpr_canvas.display_mpr(self.curved_mpr_result)
            self.export_btn.setEnabled(True)
            
            QMessageBox.information(self, "Success", 
                f"Generated Curved MPR:\n{self.curved_mpr_result.shape[0]} x {self.curved_mpr_result.shape[1]}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate MPR:\n{str(e)}")
            logger.exception("MPR generation failed")
    
    def _export_mpr(self):
        """Export curved MPR result."""
        if self.curved_mpr_result is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save MPR", "", "PNG Image (*.png);;NIfTI (*.nii)")
        if not path:
            return
        
        try:
            if path.endswith('.png'):
                # Normalize for display
                img = self.curved_mpr_result.copy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = (img * 255).astype(np.uint8)
                imageio.imwrite(path, img.T)
                
            elif path.endswith('.nii') or path.endswith('.nii.gz'):
                # Save as NIfTI with identity affine
                affine_2d = np.eye(4)
                affine_2d[0, 0] = self.spacing_mm
                affine_2d[1, 1] = self.width_mm / self.cross_samples
                save_nifti(self.curved_mpr_result, affine_2d, path)
            
            # Save metadata
            meta_path = path.rsplit('.', 1)[0] + '_metadata.json'
            metadata: Dict[str, Any] = {
                'control_points_world_mm': self.control_points.get_points_array().tolist(),
                'spacing_along_mm': self.spacing_mm,
                'width_mm': self.width_mm,
                'cross_samples': self.cross_samples,
                'interpolation_order': self.interp_order
            }
            
            if self.affine is not None:
                metadata['source_affine'] = self.affine.tolist()
            if self.header_info is not None and 'filename' in self.header_info:
                metadata['source_file'] = self.header_info['filename']
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Saved to:\n{path}\n{meta_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")