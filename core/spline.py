import numpy as np
from scipy.interpolate import splprep, splev
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def build_spline_through_points(points_world: np.ndarray, k: int = 3) -> Tuple:
    """
    Build cubic spline through control points.
    
    Args:
        points_world: Array of shape (N, 3) with world coordinates
        k: Spline degree (3 = cubic)
    
    Returns:
        tck: Tuple (t, c, k) for scipy splev
        u: Parameter values
    """
    if len(points_world) < 2:
        raise ValueError("Need at least 2 points for spline")
    
    # Adjust k if we don't have enough points
    k = min(k, len(points_world) - 1)
    
    try:
        # Build spline with no smoothing (s=0 means interpolation)
        tck, u = splprep(points_world.T, s=0, k=k)
        logger.info(f"Built spline through {len(points_world)} points (k={k})")
        return tck, u
    except Exception as e:
        logger.error(f"Spline creation failed: {e}")
        raise


def sample_equispaced(tck: Tuple, target_spacing_mm: float, 
                      num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample spline at equispaced points along arc length.
    
    Args:
        tck: Spline representation from splprep
        target_spacing_mm: Desired spacing along curve
        num_samples: If provided, use this many samples instead of spacing
    
    Returns:
        positions: Array of shape (N, 3) with sampled positions
        tangents: Array of shape (N, 3) with normalized tangent vectors
    """
    # First, sample densely to estimate arc length
    u_dense = np.linspace(0, 1, 1000)
    pts_dense = np.array(splev(u_dense, tck)).T  # (1000, 3)
    
    # Compute cumulative arc length
    diffs = np.diff(pts_dense, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = arc_lengths[-1]
    
    # Determine number of samples
    if num_samples is None:
        num_samples = max(3, int(np.ceil(total_length / target_spacing_mm)))
    
    # Create evenly spaced arc length values
    target_arc_lengths = np.linspace(0, total_length, num_samples)
    
    # Interpolate to find parameter values for target arc lengths
    u_equispaced = np.interp(target_arc_lengths, arc_lengths, u_dense)
    
    # Sample positions and derivatives
    positions = np.array(splev(u_equispaced, tck)).T  # (N, 3)
    derivatives = np.array(splev(u_equispaced, tck, der=1)).T  # (N, 3)
    
    # Normalize tangents
    tangent_norms = np.linalg.norm(derivatives, axis=1, keepdims=True)
    tangent_norms[tangent_norms < 1e-10] = 1.0  # Avoid division by zero
    tangents = derivatives / tangent_norms
    
    logger.info(f"Sampled {num_samples} points, total length: {total_length:.2f} mm")
    return positions, tangents


def compute_orthonormal_frames(positions: np.ndarray, tangents: np.ndarray,
                                up_vector: np.ndarray = np.array([0, 0, 1])) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal frames {N, B} at each curve position.
    Uses Gram-Schmidt to get stable normals.
    
    Args:
        positions: Array of shape (N, 3)
        tangents: Array of shape (N, 3), normalized
        up_vector: Reference up direction
    
    Returns:
        normals: Array of shape (N, 3) - N vectors
        binormals: Array of shape (N, 3) - B vectors
    """
    N = len(positions)
    normals = np.zeros((N, 3))
    binormals = np.zeros((N, 3))
    
    # Initial normal via Gram-Schmidt with up_vector
    for i in range(N):
        T = tangents[i]
        
        # Use previous normal if available for continuity
        if i > 0:
            U = normals[i - 1]
        else:
            U = up_vector
        
        # Project U onto plane perpendicular to T
        U_proj = U - np.dot(U, T) * T
        norm = np.linalg.norm(U_proj)
        
        if norm < 1e-6:
            # T is parallel to U, choose arbitrary perpendicular
            if abs(T[2]) < 0.9:
                U_proj = np.array([0, 0, 1]) - T[2] * T
            else:
                U_proj = np.array([1, 0, 0]) - T[0] * T
            norm = np.linalg.norm(U_proj)
        
        N_vec = U_proj / norm
        B_vec = np.cross(T, N_vec)
        
        normals[i] = N_vec
        binormals[i] = B_vec
    
    return normals, binormals
