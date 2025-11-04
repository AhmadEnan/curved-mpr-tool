import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ViewCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for displaying orthogonal slices."""
    
    def __init__(self, view_type: str, parent=None):
        """
        Args:
            view_type: 'axial', 'sagittal', 'coronal', or 'mpr'
        """
        self.view_type = view_type
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        self.volume: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None
        self.current_slice: int = 0
        self.image_handle = None
        
        # Interaction callbacks
        self.on_click_callback: Optional[Callable] = None
        self.on_drag_callback: Optional[Callable] = None
        
        # Interaction state
        self.dragging_point_idx = -1
        self.press_event = None
        
        # Connect events
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('scroll_event', self._on_scroll)
        
        self.ax.set_aspect('equal')
        self.ax.axis('off')
    
    def set_volume(self, volume: np.ndarray, affine: np.ndarray):
        """Set the volume data."""
        self.volume = volume
        self.affine = affine
        
        # Set initial slice to middle
        if self.view_type == 'axial':
            self.current_slice = volume.shape[0] // 2
        elif self.view_type == 'coronal':
            self.current_slice = volume.shape[1] // 2
        elif self.view_type == 'sagittal':
            self.current_slice = volume.shape[2] // 2
        
        self.update_display()
    
    def update_display(self):
        """Redraw the current slice."""
        if self.volume is None or self.view_type == 'mpr':
            return
        
        # Extract slice based on view type
        slice_data: np.ndarray
        title: str
        
        if self.view_type == 'axial':
            slice_data = self.volume[self.current_slice, :, :]
            title = f"Axial - Slice {self.current_slice}/{self.volume.shape[0]-1}"
        elif self.view_type == 'coronal':
            slice_data = self.volume[:, self.current_slice, :]
            title = f"Coronal - Slice {self.current_slice}/{self.volume.shape[1]-1}"
        elif self.view_type == 'sagittal':
            slice_data = self.volume[:, :, self.current_slice]
            title = f"Sagittal - Slice {self.current_slice}/{self.volume.shape[2]-1}"
        else:
            # Unknown view type — nothing to display
            return
        
        self.ax.clear()
        self.image_handle = self.ax.imshow(slice_data.T, cmap='gray', origin='lower', 
                                           interpolation='nearest')
        self.ax.set_title(title, color='white', fontsize=10)
        self.ax.axis('off')
        self.draw()
    
    def display_mpr(self, mpr_image: np.ndarray):
        """Display curved MPR result."""
        if self.view_type != 'mpr':
            return
        
        self.ax.clear()
        self.ax.imshow(mpr_image.T, cmap='gray', aspect='auto', origin='lower')
        self.ax.set_title("Curved MPR", color='white', fontsize=10)
        self.ax.axis('off')
        self.draw()
    
    def _on_scroll(self, event):
        """Handle mouse wheel scrolling."""
        if self.volume is None or event.inaxes != self.ax:
            return
        
        # Get max slice for this view
        if self.view_type == 'axial':
            max_slice = self.volume.shape[0] - 1
        elif self.view_type == 'coronal':
            max_slice = self.volume.shape[1] - 1
        elif self.view_type == 'sagittal':
            max_slice = self.volume.shape[2] - 1
        else:
            return
        
        # Update slice
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, max_slice)
        elif event.button == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
        
        self.update_display()
    
    def _on_press(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.ax or self.view_type == 'mpr':
            return
        
        self.press_event = event
        
        # Check if clicking on existing point (delegated to main window)
        if self.on_click_callback:
            self.on_click_callback(self.view_type, event.xdata, event.ydata, event.button)
    
    def _on_release(self, event):
        """Handle mouse button release."""
        self.press_event = None
        self.dragging_point_idx = -1
    
    def _on_motion(self, event):
        """Handle mouse motion."""
        if self.press_event is None or event.inaxes != self.ax:
            return
        
        if self.on_drag_callback and event.xdata and event.ydata:
            self.on_drag_callback(self.view_type, event.xdata, event.ydata)
