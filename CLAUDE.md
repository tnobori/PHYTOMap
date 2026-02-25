# CLAUDE.md — PHYTOMap Codebase Guide

This document provides guidance for AI assistants working with the PHYTOMap repository.

## Project Overview

PHYTOMap is a scientific image-analysis pipeline for **multiplexed FISH (fluorescence in situ hybridization)** in whole-mount plant tissue. It enables single-cell and 3D spatial analysis of gene expression in a transgene-free, low-cost manner.

The pipeline has four stages:
1. **Image registration** — align images across sequential imaging rounds
2. **Spot detection and decoding** — find and identify individual RNA molecules
3. **Segmentation** — identify cell boundaries (handled externally; not in this repo)
4. **Cell-by-gene matrix** — assign decoded spots to cells for downstream analysis

## Repository Structure

```
PHYTOMap/
├── README.md                        # Project overview and pipeline description
├── resources/
│   ├── phytomap_principles.png      # Diagram of PHYTOMap methodology
│   └── phytomap_analysis_fig.png    # Analysis pipeline diagram
└── scripts/
    ├── phytomap_registration.py     # Importable Python module for image registration
    ├── global_registration.ipynb    # Notebook: batch registration across rounds
    └── starfish_analysis.ipynb      # Notebook: spot detection, decoding, cell assignment
```

## Technology Stack

- **Language**: Python
- **Execution environment**: Jupyter Notebooks (interactive, cell-by-cell)
- **No package manager files** (no requirements.txt, pyproject.toml, or setup.py)

### Key Dependencies

| Library | Purpose |
|---|---|
| `bigstream` | Large-scale image alignment (RANSAC affine, transform application) |
| `starfish` | Multiplexed FISH analysis (spot detection, decoding, cell assignment) |
| `zarr` | Chunked array storage for large 3D images |
| `dask` | Distributed array computation |
| `scikit-image` (skimage) | Image I/O (`io.imread`, `io.imsave`), image processing |
| `numpy` | Numerical computing |
| `numcodecs` | Compression codecs (Blosc/zstd) for zarr arrays |
| `matplotlib` | Visualization and result inspection plots |
| `xarray` | N-dimensional labeled arrays (used within starfish) |
| `napari` | Interactive multi-dimensional image viewer |
| `seaborn` | Statistical data visualization |

## Core Module: `phytomap_registration.py`

This is the only importable Python module. It exposes three public functions:

### `global_affine_reg(impath_fix, impath_mov, spacing, downsampling, slc, min_radius, max_radius, match_threshold)`

Computes a global affine alignment between a fixed (reference) image and a moving image using RANSAC-based feature matching via `bigstream.affine.ransac_affine`.

- Images are loaded as zarr arrays (chunked, compressed) then transposed from ZYX → XYZ
- Returns `(global_affine, mov_aligned)` — the affine matrix and the aligned image array
- Produces a matplotlib figure comparing fixed, aligned, and original moving images at a chosen z-slice

**Important**: Images are stored as ZYX on disk but the bigstream API works in XYZ. The `.transpose(2, 1, 0)` calls throughout the code perform this conversion.

### `ch_submit(channel, image_dir, image_prefix, out_dir, im_round, impath_fix_highres, slc, spacing, global_affine)`

Applies a pre-computed `global_affine` to a single imaging channel at full (high) resolution.

- Reads the moving image as `{image_dir}{image_prefix}CH{channel}.tif`
- Writes the registered output as `{out_dir}/registered_R{im_round}_Ch{channel}.tif`
- Transposes back to ZYX before saving (XYZ → ZYX for disk storage)

### `round_submit(imdir, im_round, image_prefix, channels, slc, impath_fix_highres, spacing, global_affine, out_main, img_dir)`

Orchestration function that applies `ch_submit` to all channels in a single imaging round.

- Creates output directory structure: `./{out_main}/R{im_round}/`
- Iterates over `channels` list and calls `ch_submit` for each

## Notebooks

### `global_registration.ipynb`

Batch-processes all imaging rounds against a fixed reference image. Typical usage:

1. Set imaging parameters: `spacing`, `downsampling`, `min_radius`, `max_radius`, `match_threshold`
2. Loop over rounds (e.g., rounds 1–7)
3. For each round: call `global_affine_reg()` to compute the affine, then `round_submit()` to apply it to all channels
4. Round 1 serves as the fixed reference (no registration needed for round 1 itself)

**Typical parameters used in experiments:**
- `spacing = [0.36, 0.36, 0.42]` — voxel size in microns (x, y, z)
- `slc = 150` — z-plane index used for visualization
- `min_radius = 6`, `max_radius = 30` — blob radius bounds in voxels
- `match_threshold = 0.75` — RANSAC correlation threshold

### `starfish_analysis.ipynb`

Performs spot detection, decoding, and single-cell quantification:

1. **Load data** — registered TIFF images loaded into starfish's SpaceTx experiment format
2. **Preprocess** — bandpass filter, Gaussian z-axis filter, percentile clip, max projection
3. **Detect spots** — `BlobDetector` (min_sigma=1, max_sigma=4, threshold=0.1)
4. **Decode** — `SimpleLookupDecoder` with exact-match trace builder against a codebook (JSON)
5. **Assign to cells** — load external segmentation mask TIFF, use `AssignTargets.Label()`
6. **Build matrix** — cell-by-gene count matrix, filter cells with <5 total transcripts
7. **Visualize** — napari viewer with spot overlays

## Data Conventions

- **Image format on disk**: TIFF, axis order **ZYX**
- **Image format in memory (bigstream)**: axis order **XYZ** (transposed on load, transposed back on save)
- **Zarr compression**: Blosc/zstd with BITSHUFFLE, chunk size `(40, 300, 150)`
- **Output file naming**: `registered_R{round}_Ch{channel}.tif`
- **Output directory structure**: `./{out_main}/R{round}/`
- **Channel identifiers**: passed as strings (e.g., `"1"`, `"2"`)
- **Round identifiers**: passed as strings (e.g., `"1"`, `"2"`)
- **Reference round**: Round 1 (the fixed image used for all registrations)
- **Reference channel**: Typically the cell boundary stain channel in round 1

## Development Workflow

This is a **research/analysis codebase** with no automated build system, test suite, or CI/CD pipeline. The intended workflow is:

1. **Copy notebooks** for a new experiment and update hardcoded file paths and parameters
2. **Run notebooks interactively** cell-by-cell in Jupyter (not as scripts)
3. **Inspect output plots** embedded in notebook output to verify registration quality
4. **No automated tests** — correctness is verified visually via matplotlib/napari output

There is no linting configuration, formatter config, or pre-commit hooks configured.

## Important Conventions

- **Do not refactor notebooks into scripts** — the interactive, cell-by-cell nature is intentional for a scientific workflow where intermediate visual inspection is required
- **Axis order is critical**: raw TIFF files are ZYX; bigstream expects XYZ; always verify transpose operations when modifying registration code
- **Parameters are experiment-specific** and hardcoded per-notebook — there is no central config file
- **Segmentation is external** — the pipeline expects pre-computed segmentation masks as TIFF input; no segmentation logic lives in this repo
- **Codebooks are JSON files** stored externally relative to the notebook working directory

## No Automated Commands

There are no standard `make`, `npm`, or `pytest` commands. The project has no:
- Test suite
- Linter configuration
- Build scripts
- CI/CD pipelines
- Dependency lockfiles

When helping with this repo, focus on the scientific correctness of image processing logic and data format conventions rather than software engineering tooling.
