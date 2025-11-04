import numpy as np
from scipy.ndimage import map_coordinates
from core.nifti_io import world_to_voxel
from core.spline import compute_orthonormal_frames
import logging

logger = logging.getLogger(__name__)


def compute_curved_mpr(volume: np.ndarray, affine: np.ndarray,
                       sampled_positions_world: np.ndarray,
                       tangents_world: np.ndarray,
                       width_mm: float, cross_samples: int,
                       order: int = 1) -> np.ndarray:
    """
    Generate Curved MPR image by sampling volume along curved path.
    
    Args:
        volume: 3D array of image data
        affine: 4x4 affine transformation
        sampled_positions_world: (N, 3) positions along curve in world coords
        tangents_world: (N, 3) normalized tangent vectors
        width_mm: Total width of cross-section
        cross_samples: Number of samples across width
        order: Interpolation order (1=trilinear, 3=cubic)
    
    Returns:
        curved_mpr: 2D array of shape (N, cross_samples)
    """
    num_along = len(sampled_positions_world)
    
    # Compute orthonormal frames
    normals, binormals = compute_orthonormal_frames(sampled_positions_world, tangents_world)
    
    # Create cross-section offsets
    offsets = np.linspace(-width_mm / 2, width_mm / 2, cross_samples)
    
    # Initialize output
    curved_mpr = np.zeros((num_along, cross_samples), dtype=np.float32)
    
    # For each position along curve
    for i in range(num_along):
        P = sampled_positions_world[i]  # (3,)
        N = normals[i]  # (3,)
        
        # Compute 3D positions across width
        cross_positions_world = P[np.newaxis, :] + offsets[:, np.newaxis] * N[np.newaxis, :]  # (cross_samples, 3)
        
        # Convert to voxel coordinates
        voxel_coords = world_to_voxel(cross_positions_world, affine)  # (cross_samples, 3) as [k, j, i]
        
        # Prepare for map_coordinates (needs separate arrays for each dimension)
        coords = voxel_coords.T  # (3, cross_samples)
        
        # Check bounds
        in_bounds = (
            (coords[0] >= 0) & (coords[0] < volume.shape[0]) &
            (coords[1] >= 0) & (coords[1] < volume.shape[1]) &
            (coords[2] >= 0) & (coords[2] < volume.shape[2])
        )
        
        # Sample intensities
        try:
            values = map_coordinates(volume, coords, order=order, mode='constant', cval=0.0)
            values[~in_bounds] = 0  # Explicitly zero out-of-bounds
            curved_mpr[i, :] = values
        except Exception as e:
            logger.warning(f"Sampling failed at position {i}: {e}")
            curved_mpr[i, :] = 0
    
    logger.info(f"Generated curved MPR: {curved_mpr.shape}")
    return curved_mpr