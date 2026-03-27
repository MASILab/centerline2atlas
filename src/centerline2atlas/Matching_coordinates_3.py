""" 
This code is to change the coordinates of the pancreas CT scan to match the ATLAS CT scan. Since it is a pancreas on a table, the Axial, Coronal and Sagittal views are not aligned plus the images do not coincide in space. 

Done after manual segmentation of the previously generate binary mask (Step 2) using Slicer3D.
"""

import nibabel as nib
from nibabel import orientations
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from functools import wraps
import time
from dataclasses import dataclass, field
import copy
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

@dataclass
class Config: 
    # File paths
    NIFTI_NERF_PATH: str = Path('result/NIFTI/denoised_brightness10_CT1.nii.gz').resolve()
    NIFTI_NERF_BINARY: str = Path('result/NIFTI/Segmentation-spleen-label.nii.gz').resolve() # Change for different organs
    OUTPUT_DIR: str = Path('result/').resolve()
    CORRECTED_NERF_PATH: str = Path('result/NIFTI/ORNT_CORRECTED_NERF_CT_SCAN.nii.gz').resolve()
    CORRECTED_NERF_BINARY_PATH: str = Path('result/NIFTI/ORNT_CORRECTED_SPLEEN_MASK.nii.gz').resolve() # Change to save different organs
    REFERENCE_PATH = Path('data/noncontrast.nii.gz').resolve()

@timer_decorator
def data_loader():
    config = Config()

    # Load the NIFTI files
    nifti_nerf = nib.load(config.NIFTI_NERF_PATH)
    nifti_nerf_binary = nib.load(config.NIFTI_NERF_BINARY)

    # Get the data and affine matrices
    data_nerf = nifti_nerf.get_fdata()
    data_nerf_binary = nifti_nerf_binary.get_fdata()

    # Load reference image
    nifti_reference = nib.load(config.REFERENCE_PATH)
    data_reference = nifti_reference.get_fdata()

    # Print the shapes of the data arrays
    print("NeRF shape:", data_nerf.shape)
    print("NeRF binary shape:", data_nerf_binary.shape)
    print("Reference shape:", data_reference.shape)

    return nifti_nerf, nifti_nerf_binary, nifti_reference

@timer_decorator
def reoriented_views(nerf, nerf_binary, reference):
    config = Config()

    # Reorient the images
    print("Reorienting images...")
    nerf_ornt = copy.deepcopy(nerf.get_fdata())
    nerf_ornt = np.transpose(nerf_ornt, (2, 1, 0))
    nerf_ornt = nib.Nifti1Image(nerf_ornt, nerf.affine, nerf.header)

    nerf_binary_ornt = copy.deepcopy(nerf_binary.get_fdata())
    nerf_binary_ornt = np.transpose(nerf_binary_ornt, (2, 1, 0))
    ornt = orientations.axcodes2ornt(nib.aff2axcodes(reference.affine))
    # ornt[0, 1] = 0
    nerf_binary_ornt = nib.apply_orientation(nerf_binary_ornt, ornt)
    nerf_binary_ornt = nib.Nifti1Image(nerf_binary_ornt, nerf.affine, nerf.header)

    print("Reference image:", nib.aff2axcodes(reference.affine))
    print("NeRF image:", nib.aff2axcodes(nerf.affine))
    print("NeRF Corrected image:", nib.aff2axcodes(nerf_ornt.affine))
    print("NeRF binary image:", nib.aff2axcodes(nerf_binary.affine))
    print("NeRF Corrected binary image:", nib.aff2axcodes(nerf_binary_ornt.affine))

    print("Reference image:", orientations.axcodes2ornt(nib.aff2axcodes(reference.affine)))
    print("NeRF image:", orientations.axcodes2ornt(nib.aff2axcodes(nerf.affine)))
    print("NeRF Corrected image:", orientations.axcodes2ornt(nib.aff2axcodes(nerf_ornt.affine)))
    print("NeRF binary image:", orientations.axcodes2ornt(nib.aff2axcodes(nerf_binary.affine)))
    print("NeRF Corrected binary image:", orientations.axcodes2ornt(nib.aff2axcodes(nerf_binary_ornt.affine)))

    return nerf_ornt, nerf_binary_ornt

@timer_decorator
def save_corrected_nerf(nifti_image, path):
    print(f"Saving corrected NIFTI to {path}")
    nib.save(nifti_image, path)

@timer_decorator
def main():
    config = Config()
    nerf, nerf_binary, reference = data_loader()
    nerf_corrected, nerf_binary_corrected = reoriented_views(nerf, nerf_binary, reference)
    save_corrected_nerf(nerf_corrected, config.CORRECTED_NERF_PATH)
    save_corrected_nerf(nerf_binary_corrected, config.CORRECTED_NERF_BINARY_PATH)


if __name__ == "__main__":
    main()