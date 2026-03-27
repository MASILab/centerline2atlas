import numpy as np
import nibabel as nib
from nibabel import orientations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
import trimesh
import pymeshlab
from typing import Tuple, Optional
from pathlib import Path
import time
from functools import wraps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from pathlib import Path


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@timer_decorator
def load_and_orient_images(binary_path: str, reference_path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load and reorient binary image to match reference image orientation."""
    print(f"Loading binary image from: {binary_path}")
    img_binary = nib.load(binary_path)
    print(f"Loading reference image from: {reference_path}")
    img_reference = nib.load(reference_path)
    
    # print("Reorienting binary image to match reference...")
    # ornt = orientations.axcodes2ornt(nib.aff2axcodes(img_binary.affine))
    # ornt[0, 1] = 0
    # oriented_data = nib.apply_orientation(img_binary.get_fdata(), ornt)
    
    # oriented_img = nib.Nifti1Image(oriented_data, img_reference.affine, img_reference.header)

    # print("Reference image:", nib.aff2axcodes(img_reference.affine))
    # print("Reference image:", orientations.axcodes2ornt(nib.aff2axcodes(img_reference.affine)))
    # print("Reoriented image:", nib.aff2axcodes(oriented_img.affine))
    # print("Reoriented image:", orientations.axcodes2ornt(nib.aff2axcodes(oriented_img.affine)))

    return img_reference, img_binary  #img_binary.get_fdata(), img_reference.get_fdata(), oriented_img

@timer_decorator
def visualize_slice(original: np.ndarray, oriented: np.ndarray, slice_num: int) -> None:
    """Visualize a specific slice with overlaid images using plotly."""
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Heatmap(
            z=original[slice_num, :, :],
            colorscale='Hot',
            opacity=0.5,
            showscale=True,
            name='Original'
        )
    )
    
    fig.add_trace(
        go.Heatmap(
            z=oriented[slice_num, :, :],
            colorscale='Jet',
            opacity=0.5,
            showscale=True,
            name='Oriented'
        )
    )
    
    fig.update_layout(
        title=f'Slice {slice_num}',
        width=800,
        height=800,
        showlegend=True
    )
    
    fig.show()

@timer_decorator
def calculate_binary_com(image_data: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    """Calculate center of mass of binary mask considering voxel spacing."""
    indices = np.where(image_data > 0.5)
    positions = np.array(indices).T * np.array(spacing)
    return np.mean(positions, axis=0)

@timer_decorator
def create_mesh(image_data: np.ndarray, spacing: Tuple[float, float, float], output_path: str,
                smooth: bool = False, smooth_iterations: int = 0, img: Optional[nib.Nifti1Image] = None) -> np.ndarray:
    """
    Create and save a mesh from image data, return center of mass.
    Args:
        image_data: Image data
        spacing: Voxel spacing
        output_path: Output file path
        smooth: Apply smoothing
        smooth_iterations: Number of smoothing iterations
        img: Original NIfTI image for orientation
    Returns:
        np.ndarray: Center of mass of the mesh
    """
    print("Running marching cubes algorithm...")
    verts, faces, _, _ = measure.marching_cubes(image_data, level=0.5, spacing=spacing, gradient_direction='ascent', allow_degenerate=False, step_size=1, method='lorensen')
    
    print("Creating mesh...")
    mesh_set = trimesh.Trimesh(vertices=verts, faces=faces)
    
    if smooth:
        print(f"Applying {smooth_iterations} smoothing iterations...")
        mesh_set = mesh_set.smooth_laplacian(iterations=smooth_iterations)
        
    print(f"Saving mesh to {output_path}...")
    mesh_set.export(output_path)

    volume = mesh_set.volume
    print(f"Mesh volume: {volume}")

    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Plot the NIfTI image slices
    ax1 = fig.add_subplot(121)
    ax1.imshow(image_data[:, :, image_data.shape[2] // 2], cmap='gray')
    ax1.set_title('NIfTI Image Slice')

    # Plot the 3D mesh
    ax2 = fig.add_subplot(122, projection='3d')
    mesh_set = Poly3DCollection(verts[faces], alpha=0.1, edgecolor='k')
    ax2.add_collection3d(mesh_set)
    ax2.set_xlim(0, image_data.shape[0] * spacing[0])
    ax2.set_ylim(0, image_data.shape[1] * spacing[1])
    ax2.set_zlim(0, image_data.shape[2] * spacing[2])
    ax2.set_title('3D Mesh')

    plt.tight_layout()
    # plt.show()

    # return mesh_set.center_mass

def main():
    start_time = time.time()
    print("Starting mesh generation process...")
    binary_path = Path('result/NIFTI/ORNT_CORRECTED_SPLEEN_MASK.nii.gz').resolve()
    reference_path = Path('data/noncontrast-label.nii.gz').resolve()
    output_path = Path('result/PLY/mesh_from_ct_scan/spleen_surface_for_window_100.ply').resolve()
    output_path_altas = Path('result/PLY/mesh_from_ct_scan/ATLAS_SURFACE.ply').resolve()
    
    reference_data, oriented_img = load_and_orient_images(str(binary_path), str(reference_path))
    # print("\nDisplaying slice visualization...")
    # visualize_slice(reference_data, oriented_img.get_fdata(), slice_num=400)
    
    # Calculate center of mass for binary mask
    # binary_com = calculate_binary_com(oriented_img.get_fdata(), oriented_img.header.get_zooms())
    # print(f"\nBinary mask center of mass: {binary_com}")

    voxsz = oriented_img.header.get_zooms()
    voxsz_oriented = (voxsz[2], voxsz[1], voxsz[0])  # Reorder voxel sizes to match the orientation
    print(f'NeRF: {voxsz, voxsz_oriented}')

    voxsz_atlas = reference_data.header.get_zooms()
    print(f'Atlas: {voxsz_atlas}')

    # Create mesh and get its center of mass with orientation transformation
    create_mesh(oriented_img.get_fdata(), voxsz_oriented, 
                          str(output_path), img=oriented_img)
    create_mesh(reference_data.get_fdata(), reference_data.header.get_zooms(), str(output_path_altas), img=reference_data)
    
    # print(f"Mesh center of mass: {mesh_com}")
    
    print("Process completed successfully!")
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
