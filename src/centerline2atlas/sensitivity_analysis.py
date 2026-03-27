# %%
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import KDTree

# --- OUTPUT FILE PATHS (Change as needed) ---
OUTPUT_GRAPH_PATH = "snake_profile_graph.png"
OUTPUT_HEATMAP_PATH = "radial_distortion_heatmap.html"

def compute_centerline_attributes(centerline_points):
    """ Converts centerline points into Arc Length (t). """
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    return arc_length

def evaluate_snake_deformation(mesh_orig, mesh_def, center_orig, center_def):
    """ Calculates the radial distance change for every vertex. """
    # Convert Open3D objects to Numpy Arrays
    if hasattr(center_orig, 'points'):
        center_orig = np.asarray(center_orig.points)
    if hasattr(center_def, 'points'):
        center_def = np.asarray(center_def.points)
        
    pts_orig = np.asarray(mesh_orig.vertices)
    pts_def = np.asarray(mesh_def.vertices)
    
    # Build Spatial Trees
    tree_orig = KDTree(center_orig)
    tree_def = KDTree(center_def) 
    
    # Find nearest centerline point (Mapping)
    r_orig, idx_orig = tree_orig.query(pts_orig)
    r_def, idx_def = tree_def.query(pts_def)
    
    # Calculate Metrics
    radial_diff = r_def - r_orig 
    
    cl_arc_lengths = compute_centerline_attributes(center_orig)
    vertex_arc_pos = cl_arc_lengths[idx_orig]
    
    return radial_diff, vertex_arc_pos, r_orig, r_def

def save_heatmap_html(mesh, radial_diff, output_path):
    """
    Generates an interactive HTML file using Plotly.
    Handles decimation if the mesh is too large for a browser.
    """
    print(f"Generating HTML heatmap at: {output_path}")
    
    # 1. Decimation Check (Browsers crash if > 200k triangles)
    # We use a copy so we don't modify the actual data analysis mesh
    vis_mesh = mesh
    max_tris = 150000
    
    # Note: If we decimate, we must also downsample the radial_diff array to match!
    # However, tracking which vertex is which after decimation is hard.
    # STRATEGY: We will NOT decimate here to ensure heatmap accuracy, 
    # but be warned: if your NeRF is >500k verts, the HTML might be slow.
    
    verts = np.asarray(vis_mesh.vertices)
    tris = np.asarray(vis_mesh.triangles)
    
    print(f"Mesh Stats for HTML: {len(verts)} vertices, {len(tris)} triangles")

    # 2. Plotly Configuration
    # We map radial_diff directly to the color intensity
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=tris[:, 0],
            j=tris[:, 1],
            k=tris[:, 2],
            intensity=radial_diff,
            colorscale='rdbu', 
            cmin=-2.0, # Clamp min color (Blue)
            cmax=2.0,  # Clamp max color (Red)
            showscale=True,
            colorbar=dict(title="Radial Diff (mm)"),
            name='Deformed Mesh',
            flatshading=False
        )
    ])

    fig.update_layout(
        title="Radial Distortion Heatmap (Blue=Thinner, Red=Fatter)",
        scene=dict(aspectmode='data')
    )
    
    fig.write_html(output_path)
    print("HTML Generation Complete.")

def save_snake_profile_graph(vertex_arc_pos, r_orig, r_def, output_path):
    """
    Saves the 2D profile graph as a PNG instead of trying to open a window.
    """
    print(f"Saving profile graph to: {output_path}")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with high transparency
    plt.scatter(vertex_arc_pos, r_orig, alpha=0.05, s=1, c='blue', label='Original Radius')
    plt.scatter(vertex_arc_pos, r_def, alpha=0.05, s=1, c='red', label='Deformed Radius')
    
    plt.title("Snake Profile: Radius vs Arc Length")
    plt.xlabel("Distance along Centerline (mm)")
    plt.ylabel("Radius of Mesh (mm)")
    
    # Custom legend
    leg = plt.legend()
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
        
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close() # Close memory buffer

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Update these paths to match your local environment
    mesh_orig_path = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/scaled_pca_registration/NERF_REGISTRATION_TO_CT_SCAN.ply"
    mesh_def_path = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/final_mesh/NONRIGID_NERF.ply"
    center_orig_path = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/Slicer3D/pancreas_centerline_group0.ply"
    center_def_path = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/centerline/registered/registered_nerf_centerline_group0.ply"

    # 1. Load Meshes
    print("Loading meshes...")
    mesh_orig = o3d.io.read_triangle_mesh(mesh_orig_path)
    mesh_def = o3d.io.read_triangle_mesh(mesh_def_path)
    center_orig = o3d.io.read_point_cloud(center_orig_path)
    center_def = o3d.io.read_point_cloud(center_def_path)

    # 2. APPLY SMOOTHING (Taubin to preserve volume)
    print("Applying Taubin smoothing...")
    mesh_def = mesh_def.filter_smooth_laplacian(number_of_iterations=5)
    mesh_def.compute_vertex_normals()

    # 3. Run Evaluation
    print("Calculating radial differences...")
    radial_diff, arc_pos, r_orig, r_def = evaluate_snake_deformation(
        mesh_orig, mesh_def, center_orig, center_def
    )

    # 3.5 Visualize the mesh on Open3D (Optional)
    print("Visualizing heatmap in Open3D...")
    
    # Map radial_diff to colors (Blue-White-Red)
    # We clamp the color range between -2.0mm and 2.0mm to match the Plotly heatmap
    cmap = plt.get_cmap('RdBu')  # Red-White-Blue colormap
    norm = plt.Normalize(vmin=-2.0, vmax=2.0)
    
    # Generate RGB colors from the scalar values
    colors = cmap(norm(radial_diff))[:, :3]
    
    # Assign colors to the mesh
    mesh_def.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Open the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Radial Distortion Heatmap")
    vis.add_geometry(mesh_def)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_show_back_face = True
    
    vis.run()
    vis.destroy_window()



    # 4. Save the "Snake Profile" graph (PNG)
    save_snake_profile_graph(arc_pos, r_orig, r_def, OUTPUT_GRAPH_PATH)

    # 5. Save the 3D Heatmap (HTML)
    save_heatmap_html(mesh_def, radial_diff, OUTPUT_HEATMAP_PATH)
    
    print("\nDone! Download the following files to your local machine to view:")
    print(f"1. {OUTPUT_GRAPH_PATH}")
    print(f"2. {OUTPUT_HEATMAP_PATH}")

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_distortion_analysis_graphs(radial_diff):
    """
    Creates a boxplot and a histogram for the Radial Distortion data.
    """
    
    # ----------------------------------------------------
    # 1. Boxplot (Vertical)
    # ----------------------------------------------------
    plt.figure(figsize=(6, 8))
    
    # Create the boxplot with notches for median confidence
    plt.boxplot(radial_diff, notch=False, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#39AFE6', color='#0C5E85'),
                medianprops=dict(color='#0C5E85', linewidth=2),
                whiskerprops=dict(color='#0C5E85'),
                capprops=dict(color='#0C5E85'))

    # Draw a line at zero for the ideal reference
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal Radius (0 Distortion)')

    # Add quantitative markers for key statistics
    mean_val = np.mean(radial_diff)
    median_val = np.median(radial_diff)

    plt.scatter(1, mean_val, marker='o', edgecolor='#0C5E85', s=70, zorder=5, color='none',
                label=f'Mean: {mean_val:.4f} mm')

    plt.text(1.1, median_val, f'Median: {median_val:.4f} mm', 
            verticalalignment='center', color='black', weight='bold')
    

    plt.title("Boxplot of Radial Distortion ($\Delta r$) Distribution", fontsize=14)
    plt.ylabel("Radial Distortion (mm)", fontsize=12)
    plt.xticks([1], ['Error Distribution'])
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig("radial_distortion_boxplot.png")
    

# --- How to Use ---
# Call this function after the evaluation:
plot_distortion_analysis_graphs(radial_diff)

# %%
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Find all JSON files matching the pattern
base_path = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/MRI_CT_Registration"
pattern = "**/duct/duct_analysis/duct_deeds_nonlinear_resampled_to_fixed_original_resolution_analysis.json"

# Search for all matching JSON files
json_files = glob.glob(os.path.join(base_path, pattern), recursive=True)

print(f"Found {len(json_files)} JSON files")

# Keys to extract
metrics = ['mean_distance', 'min_distance', 'max_distance', 'hausdorff_distance', 'centerline_length']

# Collect data from all files
data = {metric: [] for metric in metrics}

for json_file in json_files:
    print(f"Reading: {json_file}")
    with open(json_file, 'r') as f:
        content = json.load(f)
        for metric in metrics:
            if metric in content:
                data[metric].append(content[metric])

# Create box plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    ax.boxplot(data[metric], notch=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    
    # Add mean marker
    mean_val = np.mean(data[metric])
    ax.scatter(1, mean_val, marker='D', color='darkgreen', s=70, zorder=5)
    
    # Format title
    title = metric.replace('_', ' ').title()
    ax.set_title(f"{title}\n(n={len(data[metric])})", fontsize=12)
    ax.set_ylabel("Value (mm)" if metric != 'centerline_length' else "Length (mm)", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks([])

plt.suptitle("Pancreatic Duct Analysis - Registration Metrics", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("pancreatic_duct_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
df = pd.DataFrame(data)
print(df.describe())

# %%
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
from scipy.spatial import KDTree
import copy

# --- USER PATHS ---
MESH_ORIG_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/scaled_pca_registration/NERF_REGISTRATION_TO_CT_SCAN.ply"
MESH_DEF_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/final_mesh/NONRIGID_NERF.ply"
CENTER_ORIG_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/Slicer3D/pancreas_centerline_group0.ply"
CENTER_DEF_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/centerline/registered/registered_nerf_centerline_group0.ply"

OUTPUT_PLOT_PATH = "sensitivity_analysis_boxplot.png"

# --- HELPER FUNCTIONS ---
def compute_centerline_attributes(centerline_points):
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    return arc_length

def evaluate_snake_deformation(mesh_orig, mesh_def, center_orig, center_def):
    if hasattr(center_orig, 'points'):
        center_orig = np.asarray(center_orig.points)
    if hasattr(center_def, 'points'):
        center_def = np.asarray(center_def.points)
        
    pts_orig = np.asarray(mesh_orig.vertices)
    pts_def = np.asarray(mesh_def.vertices)
    
    tree_orig = KDTree(center_orig)
    tree_def = KDTree(center_def) 
    
    r_orig, idx_orig = tree_orig.query(pts_orig)
    r_def, idx_def = tree_def.query(pts_def)
    
    radial_diff = r_def - r_orig 
    return radial_diff

# --- UPDATED PLOTTER ---
def plot_sensitivity_results(results_dict):
    """
    Plots boxplots and adds numerical labels for the medians.
    """
    iterations = list(results_dict.keys())
    data_to_plot = list(results_dict.values())
    
    plt.figure(figsize=(16, 9)) # Increased size slightly for text readability
    
    # Create the boxplot
    bplot = plt.boxplot(data_to_plot, 
                        notch=False, 
                        vert=True, 
                        patch_artist=True,
                        labels=iterations) 
    
    # --- Style the boxplots ---
    for patch in bplot['boxes']:
        patch.set_facecolor('#39AFE6')
        patch.set_edgecolor('#0C5E85')
        patch.set_alpha(0.6) # Slight transparency to make text pop
        
    for median in bplot['medians']:
        median.set_color('#0C5E85')
        median.set_linewidth(2)
        
    for whisker in bplot['whiskers']:
        whisker.set_color('#0C5E85')
        
    for cap in bplot['caps']:
        cap.set_color('#0C5E85')

    # --- ADD MEDIAN LABELS ---
    print("Adding median labels to plot...")
    for i, data in enumerate(data_to_plot):
        median_val = np.median(data)

        mean_val = np.mean(data)
        
        # X position is i + 1 (because boxplots start at x=1)
        # Y position is the median value
        # We add a slight vertical offset so it doesn't obscure the line
        
        # Calculate an offset based on the data range so it scales automatically
        y_offset = (np.max(data) - np.min(data)) * 0.02 
        
        plt.text(x=i + 1, 
                 y=median_val + y_offset, 
                 s=f'{median_val:.4f}', # 4 decimal places for precision
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=9,
                 fontweight='bold',
                 color='black',
                 rotation=45) # Rotate slightly if they are crowded
        
        plt.text(x=i + 1, 
                 y=mean_val - y_offset, 
                 s=f'Mean: {mean_val:.4f}', # 4 decimal places for precision
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontsize=8,
                 fontweight='normal',
                 color='darkgreen',
                 rotation=45) # Rotate slightly if they are crowded

    # Add reference line
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (0)')

    # Configure axes
    ax = plt.gca()
    
    # Set y-axis ticks every 0.25
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    # Remove tick labels on both axes
    plt.xticks(ticks=range(1, len(iterations) + 1), labels=[''] * len(iterations))
    ax.yaxis.set_major_formatter(NullFormatter()) # Hide y-axis labels

    plt.title("Sensitivity Analysis: Radial Distortion vs. Smoothing Iterations", fontsize=16)
    plt.xlabel("Laplacian Smoothing Iterations", fontsize=14)
    plt.ylabel("Radial Distortion (mm)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=600)
    print(f"Sensitivity graph saved to: {OUTPUT_PLOT_PATH}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data 
    print("Loading geometries...")
    mesh_orig = o3d.io.read_triangle_mesh(MESH_ORIG_PATH)
    mesh_def_raw = o3d.io.read_triangle_mesh(MESH_DEF_PATH) 
    center_orig = o3d.io.read_point_cloud(CENTER_ORIG_PATH)
    center_def = o3d.io.read_point_cloud(CENTER_DEF_PATH)

    # 2. Define Iterations
    smoothing_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = {}

    # print(f"Starting sensitivity analysis on iterations: {smoothing_iterations}")

    # 3. Loop and Process
    for iters in smoothing_iterations:
        print(f"Processing smoothing iterations: {iters}...")
        
        mesh_current = copy.deepcopy(mesh_def_raw)
        
        if iters > 0:
            mesh_current = mesh_current.filter_smooth_laplacian(number_of_iterations=iters)
        
        mesh_current.compute_vertex_normals()
        
        radial_diff = evaluate_snake_deformation(
            mesh_orig, mesh_current, center_orig, center_def
        )
        results[iters] = radial_diff

    # 4. Plot Results
    plot_sensitivity_results(results)
    print("Done!")

# %%
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
from scipy.spatial import KDTree
import copy
import csv

# --- USER PATHS ---
MESH_ORIG_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/scaled_pca_registration/NERF_REGISTRATION_TO_CT_SCAN.ply"
MESH_DEF_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/final_mesh/NONRIGID_NERF.ply"
CENTER_ORIG_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/Slicer3D/pancreas_centerline_group0.ply"
CENTER_DEF_PATH = "/home/teixeia/git_repos/NeRF/CutTrackingNerfToAtlas/result/PLY/centerline/registered/registered_nerf_centerline_group0.ply"

OUTPUT_PLOT_PATH = "sensitivity_analysis_boxplot.png"
OUTPUT_CSV_PATH = "sensitivity_analysis_summary.csv"

# --- HELPER FUNCTIONS ---
def compute_centerline_attributes(centerline_points):
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    return arc_length

def evaluate_snake_deformation(mesh_orig, mesh_def, center_orig, center_def):
    if hasattr(center_orig, 'points'):
        center_orig = np.asarray(center_orig.points)
    if hasattr(center_def, 'points'):
        center_def = np.asarray(center_def.points)

    pts_orig = np.asarray(mesh_orig.vertices)
    pts_def = np.asarray(mesh_def.vertices)

    tree_orig = KDTree(center_orig)
    tree_def = KDTree(center_def)

    r_orig, idx_orig = tree_orig.query(pts_orig)
    r_def, idx_def = tree_def.query(pts_def)

    radial_diff = r_def - r_orig
    return radial_diff

def compute_summary_stats(data, iteration):
    data = np.asarray(data)

    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)

    return {
        "iteration": iteration,
        "n": len(data),
        "min": np.min(data),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": np.max(data),
        "iqr": q3 - q1,
        "mean": np.mean(data),
        "std": np.std(data, ddof=1) if len(data) > 1 else 0.0,
    }

def export_summary_csv(results_dict, csv_path):
    rows = []
    for iteration, data in results_dict.items():
        rows.append(compute_summary_stats(data, iteration))

    fieldnames = ["iteration", "n", "min", "q1", "median", "q3", "max", "iqr", "mean", "std"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary CSV saved to: {csv_path}")

def plot_sensitivity_results(results_dict):
    """
    Clean boxplot: no value labels on the plot.
    The variability is shown visually; numeric summaries are exported to CSV.
    """
    iterations = list(results_dict.keys())
    data_to_plot = list(results_dict.values())

    plt.figure(figsize=(16, 9))

    # Boxplot already shows median + IQR.
    # Set whis=(0, 100) if you want whiskers to show min/max instead of 1.5*IQR.
    bplot = plt.boxplot(
        data_to_plot,
        notch=False,
        vert=True,
        patch_artist=True,
        labels=iterations,
        showmeans=False,
        whis=1.5
    )

    # --- Style the boxplots ---
    for patch in bplot['boxes']:
        patch.set_facecolor('#39AFE6')
        patch.set_edgecolor('#0C5E85')
        patch.set_alpha(0.6)

    for median in bplot['medians']:
        median.set_color('#0C5E85')
        median.set_linewidth(2)

    for whisker in bplot['whiskers']:
        whisker.set_color('#0C5E85')

    for cap in bplot['caps']:
        cap.set_color('#0C5E85')

    # Reference line
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (0)')

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(NullFormatter())

    # Keep x labels clean too
    plt.xticks(ticks=range(1, len(iterations) + 1), labels=[''] * len(iterations))

    plt.title("Sensitivity Analysis: Radial Distortion vs. Smoothing Iterations", fontsize=16)
    plt.xlabel("Laplacian Smoothing Iterations", fontsize=14)
    plt.ylabel("Radial Distortion (mm)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=600, bbox_inches="tight")
    print(f"Sensitivity graph saved to: {OUTPUT_PLOT_PATH}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading geometries...")
    mesh_orig = o3d.io.read_triangle_mesh(MESH_ORIG_PATH)
    mesh_def_raw = o3d.io.read_triangle_mesh(MESH_DEF_PATH)
    center_orig = o3d.io.read_point_cloud(CENTER_ORIG_PATH)
    center_def = o3d.io.read_point_cloud(CENTER_DEF_PATH)

    smoothing_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = {}

    for iters in smoothing_iterations:
        print(f"Processing smoothing iterations: {iters}...")

        mesh_current = copy.deepcopy(mesh_def_raw)

        if iters > 0:
            mesh_current = mesh_current.filter_smooth_laplacian(number_of_iterations=iters)

        mesh_current.compute_vertex_normals()

        radial_diff = evaluate_snake_deformation(
            mesh_orig, mesh_current, center_orig, center_def
        )
        results[iters] = radial_diff

    # Export stats first
    export_summary_csv(results, OUTPUT_CSV_PATH)

    # Then plot cleanly
    plot_sensitivity_results(results)

    print("Done!")


