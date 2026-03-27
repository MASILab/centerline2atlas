import numpy as np
import trimesh
import open3d as o3d
from sklearn.decomposition import PCA
from pathlib import Path
import time
import glob
import os

# Define base path dynamically
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = SCRIPT_DIR.parent.parent  # Go up two levels from Mesh_manipulation/code to reach nerf_surface_to_volume

def scale_mesh(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
    """Scale a mesh by a given factor."""
    return trimesh.Trimesh(vertices=mesh.vertices * scale_factor, faces=mesh.faces)

def calculate_scaling_factor(mesh_ct: trimesh.Trimesh, mesh_nerf: trimesh.Trimesh) -> float:
    """Calculate scaling factor to match NeRF mesh to CT mesh size."""
    bbox_ct = mesh_ct.bounds
    bbox_nerf = mesh_nerf.bounds
    
    length_ct = np.linalg.norm(bbox_ct[1] - bbox_ct[0])
    length_nerf = np.linalg.norm(bbox_nerf[1] - bbox_nerf[0])
    
    return length_ct/length_nerf

def pca_alignment(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Align source point cloud to target using PCA."""
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    
    pca_source = PCA(n_components=3).fit(source_points - source_mean)
    pca_target = PCA(n_components=3).fit(target_points - target_mean)
    
    source_axes = pca_source.components_
    target_axes = pca_target.components_
    source_axes[1] = -source_axes[1] # Needs visual inspection. This is a fix for the axes orientation.
    
    rotation_matrix = target_axes.T @ source_axes
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = target_mean - rotation_matrix @ source_mean
    
    return transform

def process_single_mesh(mesh_nerf: trimesh.Trimesh, mesh_ct: trimesh.Trimesh, output_path: Path, mesh_name: str, 
                       target_mesh_path: Path = None):
    """Process a single mesh with scaling and alignment."""
    print(f"Processing {mesh_name}...")
    scale_factor = calculate_scaling_factor(mesh_ct, mesh_nerf)
    scaled_mesh_nerf = scale_mesh(mesh_nerf, scale_factor)
    
    print("Volume of scaled NeRF: " + str(scaled_mesh_nerf.volume))

    # Convert to Open3D format and sample points
    temp_path = output_path / f'temp_{mesh_name}.ply'
    scaled_mesh_nerf.export(temp_path)
    
    source_mesh = o3d.io.read_triangle_mesh(str(temp_path))
    target_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path if target_mesh_path else 
                                               BASE_PATH / 'result/PLY/mesh_from_ct_scan/pancreas_surface_for_window_100.ply'))
    


    source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=5000)

    # Align meshes
    pca_transform = pca_alignment(source_pcd, target_pcd)
    source_mesh.transform(pca_transform)

    # Apply final transformation and save
    output_file = output_path / f'aligned_{mesh_name}.ply'
    o3d.io.write_triangle_mesh(str(output_file), source_mesh)
    temp_path.unlink()  # Remove temporary file

    return output_file

def main():
    start_time = time.time()
    
    # Create output directories
    output_path_ct = BASE_PATH / 'result/PLY/scaled_pca_registration'
    output_path_atlas = BASE_PATH / 'result/PLY/nerf_atlas_pca_alignment'
    output_path_ct.mkdir(parents=True, exist_ok=True)
    output_path_atlas.mkdir(parents=True, exist_ok=True)
    
    # Load reference meshes
    print("Loading reference meshes...")
    mesh_ct = trimesh.load(BASE_PATH / 'result/PLY/mesh_from_ct_scan/spleen_surface_for_window_100.ply')

    print("Volume of CT surface: " + str(mesh_ct.volume))

    atlas_path = BASE_PATH / 'result/PLY/mesh_from_ct_scan/ATLAS_SURFACE.ply'
    
    # Process all pancreas group meshes
    input_path = BASE_PATH / 'data/reconstruction'
    mesh_files = glob.glob(str(input_path / 'spleen_group*.ply'))
    
    for mesh_file in sorted(mesh_files):
        mesh_name = Path(mesh_file).name
        print(f"\nProcessing {mesh_name}...")
        try:
            mesh_nerf = trimesh.load(mesh_file)
            
            # Align with CT mesh
            output_file_ct = process_single_mesh(mesh_nerf, mesh_ct, output_path_ct, mesh_name)
            print(f"Successfully processed CT alignment: {output_file_ct}")
            
            # Align with ATLAS mesh
            output_file_atlas = process_single_mesh(mesh_nerf, mesh_ct, output_path_atlas, mesh_name, 
                                                  target_mesh_path=atlas_path)
            print(f"Successfully processed ATLAS alignment: {output_file_atlas}")
           
        except Exception as e:
            print(f"Error processing {mesh_name}: {str(e)}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()