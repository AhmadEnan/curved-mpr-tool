import numpy as np
from typing import Tuple, Dict, Any, cast
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
import logging

logger = logging.getLogger(__name__)


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load NIfTI file and return data, affine, and header info.
    
    Args:
        path: Path to NIfTI file (.nii or .nii.gz)
    
    Returns:
        data: 3D array (float32)
        affine: 4x4 transformation matrix (voxel -> world mm)
        header_info: Dict with useful header data
    """
    try:
        img = load(path)
        # some type checkers don't recognize the concrete nibabel image type
        img = cast(Nifti1Image, img)
        data = img.get_fdata(dtype=np.float32)
        affine = cast(np.ndarray, img.affine)
        
        header_info: Dict[str, Any] = {
            'pixdims': img.header.get_zooms()[:3],  # (dx, dy, dz)
            'shape': data.shape,
            'filename': path
        }
        
        logger.info(f"Loaded NIfTI: shape={data.shape}, pixdims={header_info['pixdims']}")
        return data, affine, header_info
        
    except Exception as e:
        logger.error(f"Failed to load NIfTI: {e}")
        raise


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str):
    """Save 2D or 3D array as NIfTI."""
    img = Nifti1Image(data, affine)
    save(img, path)
    logger.info(f"Saved NIfTI to {path}")


def world_to_voxel(world_xyz_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates (mm) to voxel indices.
    
    Args:
        world_xyz_mm: Array of shape (..., 3) in world coordinates [x, y, z]
        affine: 4x4 affine matrix
    
    Returns:
        voxel_kji: Array of shape (..., 3) in voxel indices [k, j, i] (z, y, x)
    """
    world_xyz_mm = np.asarray(world_xyz_mm)
    original_shape = world_xyz_mm.shape
    
    # Reshape to (N, 3)
    points = world_xyz_mm.reshape(-1, 3)
    
    # Add homogeneous coordinate
    points_h = np.column_stack([points, np.ones(len(points))])
    
    # Apply inverse affine
    inv_affine = np.linalg.inv(affine)
    voxel_ijk_h = (inv_affine @ points_h.T).T
    
    # Drop homogeneous coordinate and swap to (k, j, i)
    voxel_ijk = voxel_ijk_h[:, :3]
    voxel_kji = voxel_ijk[:, [2, 1, 0]]  # [i, j, k] -> [k, j, i]
    
    return voxel_kji.reshape(original_shape)


def voxel_to_world(voxel_kji: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel indices to world coordinates (mm).
    
    Args:
        voxel_kji: Array of shape (..., 3) in voxel indices [k, j, i] (z, y, x)
        affine: 4x4 affine matrix
    
    Returns:
        world_xyz_mm: Array of shape (..., 3) in world coordinates [x, y, z]
    """
    voxel_kji = np.asarray(voxel_kji)
    original_shape = voxel_kji.shape
    
    # Reshape to (N, 3) and swap to (i, j, k)
    points = voxel_kji.reshape(-1, 3)
    voxel_ijk = points[:, [2, 1, 0]]  # [k, j, i] -> [i, j, k]
    
    # Add homogeneous coordinate
    voxel_ijk_h = np.column_stack([voxel_ijk, np.ones(len(voxel_ijk))])
    
    # Apply affine
    world_xyz_h = (affine @ voxel_ijk_h.T).T
    
    # Drop homogeneous coordinate
    world_xyz_mm = world_xyz_h[:, :3]
    
    return world_xyz_mm.reshape(original_shape)
