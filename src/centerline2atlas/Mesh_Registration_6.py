"""
Simple non-rigid mesh registration based on centerline transformation.
"""

import numpy as np
import open3d as o3d
import os
import glob
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

# Import from local modules
from centerline_utils import *
from centerline_processor import CenterlineProcessor

def read_points(file_path):
    """Read points from a PLY file."""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def create_visualization(original_mesh, transformed_mesh, reference_mesh,
                        source_centerline, target_centerline, output_path):
    """Create interactive visualization and save as HTML."""
    # Extract mesh data
    orig_vertices = np.asarray(original_mesh.vertices)
    orig_triangles = np.asarray(original_mesh.triangles)
    
    trans_vertices = np.asarray(transformed_mesh.vertices)
    trans_triangles = np.asarray(transformed_mesh.triangles)
    
    ref_vertices = np.asarray(reference_mesh.vertices)
    ref_triangles = np.asarray(reference_mesh.triangles)
    
    fig = go.Figure()
    
    # Add meshes
    fig.add_trace(go.Mesh3d(
        x=ref_vertices[:, 0], y=ref_vertices[:, 1], z=ref_vertices[:, 2],
        i=ref_triangles[:, 0], j=ref_triangles[:, 1], k=ref_triangles[:, 2],
        opacity=0.3, color='lightpink', name='Reference Mesh'
    ))
    
    fig.add_trace(go.Mesh3d(
        x=orig_vertices[:, 0], y=orig_vertices[:, 1], z=orig_vertices[:, 2],
        i=orig_triangles[:, 0], j=orig_triangles[:, 1], k=orig_triangles[:, 2],
        opacity=0.3, color='lightblue', name='Original Mesh'
    ))
    
    fig.add_trace(go.Mesh3d(
        x=trans_vertices[:, 0], y=trans_vertices[:, 1], z=trans_vertices[:, 2],
        i=trans_triangles[:, 0], j=trans_triangles[:, 1], k=trans_triangles[:, 2],
        opacity=0.3, color='lightgreen', name='Transformed Mesh'
    ))
    
    # Add centerlines
    fig.add_trace(go.Scatter3d(
        x=source_centerline[:, 0], y=source_centerline[:, 1], z=source_centerline[:, 2],
        mode='lines+markers', line=dict(color='blue', width=3), name='Source Centerline'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=target_centerline[:, 0], y=target_centerline[:, 1], z=target_centerline[:, 2],
        mode='lines+markers', line=dict(color='red', width=3), name='Target Centerline'
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(aspectmode='data'),
        title="Non-Rigid Registration Results"
    )
    
    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    
    return fig

def process_mesh(mesh_path, source_centerline_path, atlas_mesh_path, 
                atlas_centerline_path, output_mesh_path, vis_path,
                straight_line_path=None, output_straight_line_path=None):
    """Process a single mesh for registration."""
    # Load meshes
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    atlas_mesh = o3d.io.read_triangle_mesh(atlas_mesh_path)
    
    # Extract group number for file naming
    group_num = os.path.basename(output_mesh_path).split("_")[-1].split(".")[0]
    
    # Create centerline processor
    base_output_dir = os.path.dirname(os.path.dirname(output_mesh_path))
    processor = CenterlineProcessor(base_output_dir, group_num)
    
    # Process source and atlas centerlines
    source_centerline = processor.process_centerline(source_centerline_path, "nerf")
    atlas_centerline = processor.process_centerline(atlas_centerline_path, "atlas")
    
    # Transform the source centerline and save it
    transformed_nerf_points = processor.transform_and_save_centerline(
        source_centerline, source_centerline, atlas_centerline, "nerf"
    )

    # Save transformed nerf centerline as point cloud
    processor.save_as_pointcloud(transformed_nerf_points, name="nerf_centerline")
    
    # Transform mesh
    transformed_mesh = transform_mesh(mesh, source_centerline, atlas_centerline)
    
    # Also transform straight line if provided
    if straight_line_path and output_straight_line_path:
        # Process straight line
        straight_line_points = processor.process_centerline(straight_line_path, "straight_line")
        
        # Transform straight line
        transformed_line_points = processor.transform_and_save_centerline(
            straight_line_points, source_centerline, atlas_centerline, "straight_line"
        )
        
        # Save transformed straight line as point cloud
        processor.save_as_pointcloud(transformed_line_points, name="straight_line")
    else:
        pass
    
    # Save transformed mesh
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_mesh_path, transformed_mesh)
    
    # Create visualization
    create_visualization(
        mesh, transformed_mesh, atlas_mesh,
        source_centerline, atlas_centerline, vis_path
    )
    
    return transformed_mesh

def remove_overlapping_faces(mesh, distance_threshold=0.01):
    """Remove overlapping or very similar faces from a mesh.
    
    Args:
        mesh: The input Open3D triangle mesh
        distance_threshold: Distance threshold to consider faces as overlapping
        
    Returns:
        Processed mesh with overlapping faces removed
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Calculate face centers
    face_centers = np.zeros((len(triangles), 3))
    for i, tri in enumerate(triangles):
        face_centers[i] = (vertices[tri[0]] + vertices[tri[1]] + vertices[tri[2]]) / 3.0
    
    # Calculate face normals if not already available
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)
    
    # Track which faces to keep
    keep_faces = np.ones(len(triangles), dtype=bool)
    
    # Simple spatial data structure for efficiency
    from scipy.spatial import KDTree
    tree = KDTree(face_centers)
    
    # Find potentially overlapping faces
    for i in range(len(triangles)):
        if not keep_faces[i]:
            continue
            
        # Find nearby face centers
        nearby_indices = tree.query_ball_point(face_centers[i], distance_threshold)
        
        for j in nearby_indices:
            if i == j or not keep_faces[j]:
                continue
                
            # Check if normals are similar (faces are nearly parallel)
            normal_similarity = np.abs(np.dot(face_normals[i], face_normals[j]))
            
            if normal_similarity > 0.95:  # Faces have similar orientation
                # Keep the face with better quality (you can define your quality metric)
                # Here we just keep the first face and remove the second one
                keep_faces[j] = False
    
    # Create a new mesh with only the kept faces
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles[keep_faces])
    
    return new_mesh

def main():
    """Process all meshes in the dataset."""
    # Get the base directory relative to the current script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root
    base_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Define paths
    atlas_mesh = os.path.join(base_dir, "result/PLY/mesh_from_ct_scan/ATLAS_SURFACE.ply")
    atlas_centerline = os.path.join(base_dir, "result/PLY/centerline_auto/ATLAS/centerline_resampled.ply")
    mesh_dir = os.path.join(base_dir, "result/PLY/nerf_atlas_pca_alignment")
    centerline_dir = os.path.join(base_dir, "result/PLY/centerline_auto/NERF")
    output_dir = os.path.join(base_dir, "result/PLY/centerline_auto")
    vis_dir = os.path.join(base_dir, "result/PLY/centerline_auto")
    
    # Ensure directories exist
    for directory in [output_dir, vis_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all mesh files - fix the pattern to avoid double extensions
    mesh_files = glob.glob(os.path.join(mesh_dir, "PCA_ALIGNED_NERF_TO_ATLAS.ply"))
    
    for mesh_file in mesh_files:
        try:
            # Extract group number more robustly
            filename = os.path.basename(mesh_file)
            # Handle both potential formats: aligned_pancreas_group2.ply or aligned_pancreas_group2.ply.ply
            base_name = filename.replace('.ply.ply', '.ply')  # Fix duplicate extension if present
            group_num = base_name.split("group")[-1].split(".")[0]
            
            # Define paths
            centerline_path = os.path.join(centerline_dir, f"centerline_resampled.ply")
            # straight_line_path = os.path.join(centerline_dir, f"straight_centerline_pancreas_group{group_num}.ply")
            output_path = os.path.join(output_dir, f"registered_pancreas_group{group_num}.ply")
            # output_straight_line_path = os.path.join(output_dir, f"registered_straight_line_group{group_num}.ply")
            vis_path = os.path.join(vis_dir, f"registration_group{group_num}.html")
            
            # Verify files exist before processing
            if not os.path.exists(mesh_file):
                print(f"Skipping group {group_num}: mesh file not found: {mesh_file}")
                continue
                
            if not os.path.exists(centerline_path):
                print(f"Skipping group {group_num}: centerline not found: {centerline_path}")
                continue
                
            if not os.path.exists(atlas_mesh):
                print(f"Atlas mesh not found: {atlas_mesh}")
                continue
                
            if not os.path.exists(atlas_centerline):
                print(f"Atlas centerline not found: {atlas_centerline}")
                continue
                
            print(f"Processing group {group_num}...")
            transformed_mesh = process_mesh(
                mesh_file, centerline_path,
                atlas_mesh, atlas_centerline,
                output_path, vis_path,
                # straight_line_path, output_straight_line_path
            )

            # Remove overlapping faces
            transformed_mesh = remove_overlapping_faces(transformed_mesh)

            print(f"Completed group {group_num}")
            
        except Exception as e:
            print(f"Error processing {mesh_file}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full stack trace for better debugging

if __name__ == "__main__":
    main()
