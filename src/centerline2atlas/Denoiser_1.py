import os
import time
import nibabel as nib
import nipype.interfaces.fsl as fsl
from nipype import Node, Workflow
from pathlib import Path

# Configuration
INPUT_PATH = Path('data/CT1.nii.gz').resolve()
OUTPUT_PATH = Path('result/NIFTI/denoised_CT1.nii.gz').resolve()
WORKFLOW_DIR = Path('result/').resolve()
BRIGHTNESS_THRESHOLD = 10

def load_ct_image(file_path):
    """Load a CT image from a .nii.gz file."""
    print(f"Loading CT image from:", file_path)
    start_time = time.time()
    try:
        img = nib.load(file_path)
        print(f"Image loaded successfully in {time.time() - start_time:.2f} seconds")
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load CT image: {e}")

def calculate_fwhm(ct_img):
    """Calculate FWHM based on voxel dimensions."""
    x, y, z = ct_img.header.get_zooms()
    fwhm = x * y * z
    print(f"Calculated FWHM: {fwhm}")
    return fwhm

def run_susan_denoising(input_file, output_file, brightness, fwhm):
    """Run SUSAN denoising on the input image."""
    print(f"\nStarting SUSAN denoising...")
    print(f"Brightness threshold: {brightness}")
    print(f"FWHM: {fwhm}")
    start_time = time.time()
    
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    
    susan = Node(fsl.SUSAN(), name='susan')
    susan.inputs.in_file = input_file
    susan.inputs.brightness_threshold = brightness
    susan.inputs.fwhm = fwhm
    susan.inputs.out_file = output_file

    workflow = Workflow(name='denoise_workflow', base_dir=WORKFLOW_DIR)
    workflow.add_nodes([susan])
    workflow.run(plugin='MultiProc')
    print(f"Denoising completed in {time.time() - start_time:.2f} seconds")
    print(f"Output saved to: {output_file}")

def main():
    print("\n=== CT Image Denoising Pipeline ===")
    total_start_time = time.time()
    try:
        ct_img = load_ct_image(INPUT_PATH)
        fwhm = calculate_fwhm(ct_img)
        run_susan_denoising(INPUT_PATH, OUTPUT_PATH, BRIGHTNESS_THRESHOLD, fwhm)
        print(f"\nTotal processing time: {time.time() - total_start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during denoising: {e}")

if __name__ == "__main__":
    main()

