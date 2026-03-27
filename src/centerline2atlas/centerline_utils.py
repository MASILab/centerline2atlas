"""
Utility functions for centerline operations including reading, resampling, and saving coordinates.
"""

import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import os

def read_centerline_ply(file_path):
    """Read points from a PLY file."""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def resample_centerline(centerline, num_points=26):
    """Resample centerline using arc length parameterization."""
    if centerline.shape[0] < 2:
        raise ValueError(f"Centerline must have at least 2 points, got {centerline.shape[0]}")
        
    if centerline.shape[0] == num_points:
        return centerline
    
    # Calculate arc lengths
    diffs = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate(([0], np.cumsum(lengths)))
    total_length = cum_lengths[-1]
    
    # # Create normalized parameter
    # u = cum_lengths / total_length
    
    # # Create spline for each dimension
    # splines = [
    #     CubicSpline(u, centerline[:, i])
    #     for i in range(3)
    # ]
    
    # # Generate new points
    # u_new = np.linspace(0, 1, num_points)
    # return np.vstack([spline(u_new) for spline in splines]).T

    # Create interpolator for each dimension using linear interpolation

    interpolators = [
        interp1d(cum_lengths, centerline[:, i], kind='linear') 
        for i in range(3)
    ]
    
    # Generate new points at equal arc length intervals
    new_lengths = np.linspace(0, cum_lengths[-1], num_points)
    return np.vstack([interp(new_lengths) for interp in interpolators]).T

def save_centerline_coordinates(points, filename):
    """Save centerline coordinates to a text file.
    
    Args:
        points: Numpy array of 3D points
        filename: Output filename (will create directories if they don't exist)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, points, fmt='%.6f', delimiter=',', header='x,y,z', comments='')
    
def load_centerline_coordinates(filename):
    """Load centerline coordinates from a text file."""
    return np.loadtxt(filename, delimiter=',', skiprows=1)

def measure_line_length(points):
    """Measure the length of a line defined by ordered points."""
    if len(points) < 2:
        return 0
    
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(lengths)

def transform_centerline(source_centerline, target_centerline):
    """Calculate transformation from source to target centerline."""
    if len(source_centerline) != len(target_centerline):
        raise ValueError("Source and target centerlines must have the same number of points")
        
    return target_centerline - source_centerline

"""
Cylindrical coordinate transformation for non-rigid mesh registration.

This module contains the core functions for transforming mesh vertices
while preserving local radii.
"""

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

def compute_frames(centerline):
    """Compute local coordinate frames along centerline."""
    # Calculate tangent vectors
    tangents = np.gradient(centerline, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    tangents = tangents / norms[:, np.newaxis]
    
    # Calculate normal vectors using reference up vector
    normals = np.zeros_like(tangents)
    up = np.array([0, 1, 0])
    
    for i in range(len(tangents)):
        normal = np.cross(tangents[i], up)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            normal = np.cross(tangents[i], np.array([1, 0, 0]))
            norm = np.linalg.norm(normal)
        normals[i] = normal / max(norm, 1e-8)
    
    # Calculate binormal vectors
    binormals = np.cross(tangents, normals)
    
    return tangents, normals, binormals

def point_to_cylindrical(point, center, tangent, normal, binormal):
    """Convert point to cylindrical coordinates."""
    # Vector from center to point
    v = point - center
    
    # Project onto plane perpendicular to tangent
    v_proj = v - np.dot(v, tangent) * tangent
    
    # Calculate r (radius)
    r = np.linalg.norm(v_proj)
    
    # Calculate theta (angle)
    theta = np.arctan2(np.dot(v_proj, binormal), np.dot(v_proj, normal))
    
    # Calculate h (height along tangent)
    h = np.dot(v, tangent)
    
    return r, theta, h

def cylindrical_to_point(r, theta, h, center, tangent, normal, binormal):
    """Convert cylindrical coordinates to Cartesian."""
    return (center + 
            r * (normal * np.cos(theta) + binormal * np.sin(theta)) + 
            h * tangent)

def transform_points(points, source_centerline, target_centerline, k=1):
    """Transform any set of points using cylindrical transformation.
    
    Args:
        points: Numpy array of points to transform (shape Nx3)
        source_centerline: Source centerline points
        target_centerline: Target centerline points
        k: Number of nearest neighbors to consider
    
    Returns:
        Transformed points
    """
    # Compute frames
    src_tangents, src_normals, src_binormals = compute_frames(source_centerline)
    tgt_tangents, tgt_normals, tgt_binormals = compute_frames(target_centerline)
    
    # Find nearest centerline points
    tree = cKDTree(source_centerline)
    _, indices = tree.query(points, k=k)
    
    # Transform points
    transformed_points = np.zeros_like(points)
    
    for i, point in enumerate(points):
        # Get nearest centerline point index
        idx = indices[i] if k == 1 else indices[i, 0]
        
        # Convert to cylindrical coordinates
        r, theta, h = point_to_cylindrical(
            point, 
            source_centerline[idx],
            src_tangents[idx], 
            src_normals[idx], 
            src_binormals[idx]
        )
        
        # Transform to target frame
        transformed_points[i] = cylindrical_to_point(
            r, theta, h,  # Preserve cylindrical coordinates
            target_centerline[idx],
            tgt_tangents[idx],
            tgt_normals[idx],
            tgt_binormals[idx]
        )
    
    return transformed_points

def transform_mesh(mesh, source_centerline, target_centerline, k=1):
    """Transform mesh while preserving local radii.
    
    Args:
        mesh: Open3D mesh to transform
        source_centerline: Source centerline points
        target_centerline: Target centerline points
        k: Number of nearest neighbors to consider
    
    Returns:
        Transformed mesh
    """
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # Transform vertices using the general point transformation function
    transformed_vertices = transform_points(vertices, source_centerline, target_centerline, k)
    
    # Create transformed mesh
    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    transformed_mesh.triangles = mesh.triangles
    
    return transformed_mesh

