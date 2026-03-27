"""
Utility class for centerline processing operations.
"""

import os
import numpy as np
import open3d as o3d
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from centerline_utils import *

class CenterlineProcessor:
    def __init__(self, base_output_dir, group_num):
        """
        Initialize a centerline processor
        
        Args:
            base_output_dir: Base directory for all outputs
            group_num: Group number/identifier for the current processing task
        """
        self.group_num = group_num
        
        # Setup directories
        self.coords_dir = os.path.join(base_output_dir, "centerlines_coordinates")
        self.output_dir = os.path.join(base_output_dir, "centerline/registered/")
        
        # Create directories if they don't exist
        os.makedirs(self.coords_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_coord_filepath(self, name, stage):
        """Generate consistent filepath for centerline coordinate files"""
        return os.path.join(self.coords_dir, f"{name}_{stage}_{self.group_num}.txt")
    
    def process_centerline(self, filepath, name, num_points=26):
        """
        Process a centerline: load, save original, resample, save resampled
        
        Args:
            filepath: Path to the centerline PLY file
            name: Identifier for this centerline (e.g., 'nerf', 'atlas')
            num_points: Number of points for resampling
            
        Returns:
            The resampled centerline points
        """
        # Load centerline
        centerline = read_centerline_ply(filepath)
        
        # Save original centerline
        save_centerline_coordinates(
            centerline, 
            self._get_coord_filepath(name, "original")
        )
        
        # Resample centerline
        resampled = resample_centerline(centerline, num_points)
        
        # Save resampled centerline
        save_centerline_coordinates(
            resampled,
            self._get_coord_filepath(name, "resampled")
        )
        
        return resampled
        
    def transform_and_save_centerline(self, points, source_centerline, target_centerline, name):
        """
        Transform centerline points and save results
        
        Args:
            points: Points to transform
            source_centerline: Source centerline for transformation
            target_centerline: Target centerline for transformation
            name: Identifier for this centerline
            
        Returns:
            The transformed centerline points
        """
        # Transform points
        transformed_points = transform_points(
            points, source_centerline, target_centerline
        )
        
        # Save transformed points
        save_centerline_coordinates(
            transformed_points,
            self._get_coord_filepath(name, "transformed")
        )
        
        return transformed_points
    
    def save_as_pointcloud(self, points, name=None, output_path=None):
        """
        Save points as a PLY point cloud file
        
        Args:
            points: The points to save
            name: Name identifier for the points (e.g., 'nerf', 'straight_line')
            output_path: Direct output path (if provided, takes precedence over name)
            
        Returns:
            The path where the file was saved
        """
        if output_path is None and name is None:
            raise ValueError("Either name or output_path must be provided")
        
        if output_path is None:
            # Construct output path based on name
            output_path = os.path.join(self.output_dir, f"registered_{name}_{self.group_num}.ply")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_path, pcd)
        
        return output_path
