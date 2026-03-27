import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage import measure, morphology
from scipy.spatial import ConvexHull, QhullError
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.segmentation import flood_fill
from typing import List, Tuple, Optional
from pathlib import Path
import time
from functools import wraps

# Configuration
INPUT_PATH = Path("result/NIFTI/denoised_brightness10_CT1.nii.gz").resolve()
OUTPUT_PATH = Path("result/NIFTI/binary_mask_allorgans_window_100.nii.gz").resolve()  
SCANNER_BOTTOM_CUTOFF = 100 # Visual inspection of the CT image
VOLUME_THRESHOLD = 2000
CONTOUR_CLOSURE_THRESHOLD = 1
IMAGE_SIZE = 768 # Could be done automatically (image dimensions)
WINDOW = 100
LEVEL = 50

def intensity_seg(ct_3d_array: np.ndarray, level: float, window: float, axis: int) -> List:
    """Segment CT image based on intensity windowing along specified axis.
    
    Args:
        ct_3d_array: 3D numpy array from CT image
        level: Center of HU window
        window: Width of HU window
        axis: Axis to process (0=x, 1=y, 2=z)
    
    Returns:
        List of contours for each slice along specified axis
    """
    contours = []
    max_val = level + window/2
    min_val = level - window/2
    
    for slice_idx in range(ct_3d_array.shape[axis]):
        slice_data = ct_3d_array[:, :, slice_idx]
        clipped = slice_data.clip(min_val, max_val)
        clipped[clipped != min_val] = 1
        clipped[clipped == min_val] = 0
        contours.append(measure.find_contours(clipped, 0.95))
    
    return contours

def contour_distance(contour: np.ndarray) -> float:
    """Calculate the distance between the first and last point of a contour.
    
    Args:
        contour: Array of contour points
    
    Returns:
        Euclidean distance between first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

def set_is_closed(contour: np.ndarray) -> bool:
    """Check if a contour is closed based on distance between first and last point.
    
    Args:
        contour: Array of contour points
    
    Returns:
        True if contour is closed, False otherwise
    """
    return contour_distance(contour) < CONTOUR_CLOSURE_THRESHOLD

def find_pancreas(contours: List) -> List:
    """Filter contours to identify pancreas regions.
    
    Args:
        contours: List of detected contours
    
    Returns:
        List of contours corresponding to pancreas area
    """
    all_refined_contours = []

    for contours_group in contours:

        table_and_pancreas_contours = []
        vol_contours = []
        
        for contour in contours_group:
            try:
                # Check for valid 2-dimensional points before creating ConvexHull
                if np.unique(contour, axis=0).shape[0] < 3:
                    continue
                hull = ConvexHull(contour)

                # set some constraints for the volume
                if hull.volume > VOLUME_THRESHOLD and set_is_closed(contour):
                    table_and_pancreas_contours.append(contour)
                    vol_contours.append(hull.volume)
            except:
                continue

        # Discard body contour
        if len(table_and_pancreas_contours) <= 3:
            all_refined_contours.append(table_and_pancreas_contours)
        elif len(table_and_pancreas_contours) > 3:
            vol_contours, table_and_pancreas_contours = (list(t) for t in
                                                    zip(*sorted(zip(vol_contours, table_and_pancreas_contours))))
            table_and_pancreas_contours.pop(-1)
            all_refined_contours.append(table_and_pancreas_contours)
    return all_refined_contours

def create_mask_from_polygon(image: np.ndarray, contours: List, axis: int) -> List:
    """Create binary mask from contours.
    
    Args:
        image: Source image
        contours: List of contours
        axis: Processing axis
    
    Returns:
        List of binary masks
    """
    mask_list = []
    for contours_group in contours:
        pancreas_mask = np.array(Image.new('L', [IMAGE_SIZE, IMAGE_SIZE], 0))
        for contour in contours_group:
            x, y = contour[:, 0], contour[:, 1]
            polygon = list(zip(x, y))
            img = Image.new('L', [IMAGE_SIZE, IMAGE_SIZE], 0)
            ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
            pancreas_mask += np.array(img)
        
        pancreas_mask[pancreas_mask > 1] = 1  
        mask_list.append(pancreas_mask)
    
    return mask_list

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def find_mask_center(binary_mask: np.ndarray) -> Tuple[int, int, int]:
    """Find the center of mass of the binary mask.
    
    Args:
        binary_mask: 3D numpy array of the binary mask
    
    Returns:
        Tuple of (x, y, z) coordinates of the center point
    """
    # Get indices of non-zero elements
    coords = np.where(binary_mask > 0)
    if len(coords[0]) == 0:
        # Fallback to geometric center if mask is empty
        return tuple(dim // 2 for dim in binary_mask.shape)
    
    # Calculate center of mass
    center = tuple(int(np.mean(coord)) for coord in coords)
    return center

@timing_decorator
def main():
    start_time = time.time()
    print("Starting pancreas segmentation pipeline...")
    
    # Load and preprocess CT image
    print("Loading CT image from:", INPUT_PATH)
    ct_img = nib.load(INPUT_PATH)
    img_data = ct_img.get_fdata()
    print("Preprocessing: removing scanner bottom...")
    img_data[:,:SCANNER_BOTTOM_CUTOFF,:] = -1024
    
    # Generate segmentation(414, 528, 211)
    print("Generating initial segmentation...")
    seg_z = intensity_seg(img_data, level=LEVEL, window=WINDOW, axis=2)
    print("Identifying pancreas contours...")
    pancreas_contours = find_pancreas(seg_z)
    print("Creating binary masks...")
    binary_masks = create_mask_from_polygon(img_data, pancreas_contours, axis=2)
    
    # Post-process masks - matching working version
    print("Post-processing: rotating, aligning and filling holes...")
    binary_mask = np.flipud(np.rot90(np.stack(binary_masks, axis=-1), 1, (0, 1))).astype(np.int32)
    binary_mask[:,:SCANNER_BOTTOM_CUTOFF,:] = 0
    
    # print("Post-processing: filling holes...")
    structure = morphology.ball(radius=5)
    final_mask = ndimage.binary_fill_holes(binary_mask, structure=structure)
    
    # Calculate flood fill seed point from mask center
    flood_fill_seed = find_mask_center(final_mask)
    print(f"Flood fill seed point: {flood_fill_seed}")
    final_mask = flood_fill(final_mask, seed_point=flood_fill_seed, new_value=1)

    # Save result with original affine and header
    print(f"Saving final mask to: {OUTPUT_PATH}")
    nib.save(nib.Nifti1Image(final_mask, ct_img.affine, ct_img.header), OUTPUT_PATH)
    
    total_time = time.time() - start_time
    print(f"Segmentation completed successfully! Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()


