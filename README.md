# Centerline-Guided Atlas Registration

Code for centerline-guided mesh registration of deformed ex vivo organ surfaces to 3D anatomical atlases while preserving local geometry.

## Overview

This repository contains the centerline-based registration component used in the manuscript:

**Globally Deformable, Locally Rigid Registration of Ex vivo Pancreas to a 3D Atlas Using a Centerline-Based Method**

The method was developed to map a deformed ex vivo pancreas surface to a 3D atlas using a globally deformable, locally rigid transformation. In the full pipeline, a specimen surface is reconstructed from smartphone video using a Neural Radiance Field (NeRF), scaled using ex vivo CT, and then transformed to atlas space through cylindrical coordinates defined by corresponding centerlines.

The core idea is to:
- extract a centerline from the ex vivo mesh,
- extract a centerline from the atlas mesh,
- establish a one-to-one correspondence between the two centerlines,
- represent each mesh vertex in a local coordinate system relative to the source centerline,
- transfer that representation to the atlas centerline.

This lets the specimen bend toward the atlas trajectory while preserving local radial geometry relative to the centerline.

## What this repository contains

This repository is intended to host the **centerline registration stage** of the workflow, including:
- centerline extraction helpers,
- centerline resampling,
- vertex-to-centerline correspondence,
- cylindrical-coordinate transform,
- mesh warping from specimen space to atlas space,
- distortion analysis utilities.

## Related repositories

The NeRF-based surface reconstruction pipeline used upstream in the study is available separately:

- https://github.com/MASILab/NerfOrganRecon

This repository focuses on the **registration and atlas-mapping** portion of the workflow rather than NeRF reconstruction itself.

## Method summary

Given:
- an ex vivo specimen mesh,
- a target atlas mesh,
- a source centerline,
- a target centerline,

the method:

1. Resamples the specimen and atlas centerlines to matching point counts.
2. Assigns each specimen vertex to its nearest centerline sample.
3. Expresses each vertex in a local cylindrical frame:
   - radial distance `r`
   - angular position `theta`
   - tangent-direction offset `h`
4. Transfers `(r, theta, h)` to the corresponding atlas centerline frame.
5. Reconstructs the transformed vertex in atlas space.

This produces a deformation that is globally flexible but locally constrained by the centerline frame.

## Manuscript context

In the current study, the workflow was applied to a single ex vivo human pancreas specimen. The wet-lab sampling plan consisted of:
- 25 tissue blocks,
- 100 sub-blocks,
- a 26-point centerline representation used to align cut locations with atlas coordinates.

Geometric side effects were evaluated by measuring vertex-wise radial deviation before and after transformation. Moderate Laplacian smoothing (approximately 4–6 iterations) provided the best trade-off between surface stabilization and minimizing smoothing-induced shrinkage.

## Inputs

Expected inputs will include:
- specimen surface mesh
- atlas surface mesh
- specimen centerline
- atlas centerline
- optional cut/block metadata
- optional smoothing parameters

## Outputs

Expected outputs will include:
- transformed specimen mesh in atlas space
- mapped cut/block coordinates
- centerline correspondence files
- radial distortion measurements
- summary tables and figures for quality control

## Data acquisition used in the paper

For the feasibility study described in the manuscript, specimen video was acquired manually using the main rear camera of an iPhone 14 Pro at:
- 1920 × 1080 resolution
- 30 frames/s
- H.264 encoding
- ~180.8 s duration

The upstream NeRF mesh was scaled using ex vivo CT before centerline-based registration.

## Installation

_TODO_

## How to run

_TODO_

## Example workflow

_TODO_

## Repository structure

_TODO_

## Citation

If you use this code, please cite the associated manuscript:

```bibtex
@article{hucke2026centerline,
  title={Globally Deformable, Locally Rigid Registration of Ex vivo Pancreas to a 3D Atlas Using a Centerline-Based Method},
  author={Hucke, Andre T.S. and Kim, Michael E. and Remedios, Lucas W. and others},
  journal={TBD},
  year={2026}
}
