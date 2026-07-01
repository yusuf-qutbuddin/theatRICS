# Workflow: ICS (Image Correlation Spectroscopy)

The **ICS** tab performs temporal Image Correlation Spectroscopy on TIFF image 
stacks, computing block-wise normalised correlation values to track changes in 
molecular density or aggregation state over time.

---

## Overview

The ICS workflow supports:

- **Single-file analysis** of one TIFF stack
- **Batch analysis** of a folder containing sample subfolders,
  each containing one or more TIFF stacks
- Block-wise temporal autocorrelation at a user-defined frame lag
- Intensity thresholding and masking to exclude background pixels
- Per-block export of diagnostic images
- Aggregation of results across multiple TIFFs in one sample folder
- Export of per-file CSVs, overview SVG figures, and a combined
  batch-level CSV and plot

---

## Input requirements

- A **TIFF file** containing a time-series image stack
- The stack must contain at least `block_length` frames
- For batch mode, the folder should contain sample subfolders,
  each containing one or more matching TIFF files

The ICS module tries to load files using **picasso** first
(handles large `TiffMultiMap` files), and falls back to **tifffile**
if picasso is not installed.

Install picasso for large file support:

```bash
pip install picasso
```

---

## Single-file analysis

1. Open the **ICS** tab.
2. Under **Single TIFF**, click **Browse** and select a TIFF file.
3. Set the analysis parameters:
   - **Block length (frames)**
   - **Frame skip (lag τ)**
   - **Bin frames**
   - **Threshold multiplier**
   - **Save block images**
4. Click **Run ICS**.

The result is displayed inside the GUI and exported to disk next to
the input file.

---

## Batch analysis

1. Under **Batch folder**, click **Browse** and select a parent folder.
2. Set a **File pattern**, for example:

   ```text
   *.tiff
   ```

3. Set analysis parameters as above.
4. Click **Run ICS**.

The software searches each immediate subfolder of the selected
parent folder for matching TIFF files. Each subfolder is treated as
one sample, and results are aggregated per sample.

After all files are processed:
- a combined CSV is written in the parent folder
- a combined overview plot is written in the parent folder

---

## Parameters

### Block length (frames)
Number of frames per temporal block.

Each block is treated as an independent time window.
The total stack is divided into non-overlapping blocks of this length.

Frames are discarded at the start of the stack if the remainder
would be exactly 1 (to avoid a degenerate single-frame block).

### Frame skip (lag τ)
The temporal lag used for the autocorrelation calculation within
each block.

A value of `1` means the correlation is computed between consecutive
frames. Increasing this value increases the effective lag time.

### Bin frames
If greater than 1, frames are averaged in groups of this size before
block analysis. This reduces noise at the cost of temporal resolution.

### Threshold multiplier
Controls the pixel inclusion mask.

Within each block:
1. The maximum intensity projection (MIP) is computed.
2. A Yen threshold is applied to the MIP.
3. Pixels with MIP intensity above `threshold_multiplier × Yen_threshold`
   are included in the correlation statistics.

Lower values include more pixels; higher values restrict to brighter regions.

### Save block images
If enabled, four diagnostic TIFF images are saved per block:

| File | Contents |
|------|----------|
| `*_b<n>_MIP.tiff` | Maximum intensity projection of the block |
| `*_b<n>_mean.tiff` | Time-averaged intensity of the block |
| `*_b<n>_mask.tiff` | Binary inclusion mask |
| `*_b<n>_corr.tiff` | Normalised G map |

---

## ICS computation

For each block, the temporal autocorrelation is computed as:

```
G(x, y) = <δF(x,y,t) · δF(x,y,t+τ)> / <F(x,y)>²
```

where:
- `δF = F - <F>` are the intensity fluctuations
- `<·>` denotes temporal averaging over the block
- `τ` is the frame lag set by **Frame skip**

This yields a per-pixel G map for each block. The mean and standard
deviation of G over the masked pixels give the block statistics.

---

## Normalisation

After all blocks are processed, the mean G per block is normalised
to the first valid (non-NaN) block:

```
Normalised G = mean_G(block n) / mean_G(block 0)
```

A value of 1.0 means no change relative to the first block.
Decreasing values may indicate reduction in molecular clustering,
photobleaching effects, or changes in labelling density.

---

## Display

The GUI display shows a 2×2 overview figure:

**Top-left**: Mean G ± SD vs block number
- error bars show the standard deviation over masked pixels

**Top-right**: Normalised G vs block number
- reference line at 1.0 (first block)

**Bottom-left**: Mean intensity image of the last block

**Bottom-right**: G map of the last block

---

## Outputs

### Per-file outputs

For an input file `sample.tiff`, the software writes:

- `sample_threshold<value>_corr.csv`  
  Per-block statistics table with columns:  
  `block`, `mean_G`, `sd_G`, `se_G`, `n_pixels`, `Normalized`

- `sample_ICS_overview.svg`  
  Vector export of the 2×2 overview figure

- (Optional) `sample_b<n>_MIP.tiff`, `sample_b<n>_mean.tiff`,  
  `sample_b<n>_mask.tiff`, `sample_b<n>_corr.tiff`  
  Block diagnostic images, if **Save block images** is enabled

### Batch outputs

After processing all files in a batch, the software writes
in the **parent selected folder**:

- `<timestamp>_allSamples_ICS.csv`  
  Combined normalised G table with mean and SD columns per sample

- `<timestamp>_allSamples_ICS.svg`  
  Combined plot: normalised G vs block number for all samples,
  with error bars

Per-sample aggregated CSVs are also written inside each sample subfolder:

- `<sample>_aggregated.csv`

---

## Interpretation

### Constant normalised G
A flat normalised G trace (≈ 1.0) indicates that the spatial
distribution of fluorescent molecules is stable over time.

### Decreasing normalised G
A decrease over time may indicate:
- reduction in molecular clustering
- loss of fluorescence (bleaching)
- changes in label density

### Increasing normalised G
An increase may indicate:
- aggregation of molecules over time
- redistribution into fewer, brighter clusters

### Large block-to-block variability
High SD within a block can indicate:
- heterogeneous sample
- insufficient frame number per block
- suboptimal mask threshold

---

## Troubleshooting

### "No TIFF files found in batch folder"
Check the **File pattern** field.

For TIFF files with `.tiff` extension:
```text
*.tiff
```

For `.tif` extension:
```text
*.tif
```

### Stack is too short
The stack must contain at least `block_length` frames. If
the stack is shorter, no blocks can be computed and the analysis
will produce no output.

Reduce the block length or use a longer acquisition.

### All G values are NaN
This usually means the mask is empty — no pixels passed the
threshold. Try:
- reducing the **Threshold multiplier**
- checking that the correct channel is loaded
- verifying that the TIFF contains intensity variation

### Block images are not saved
Ensure **Save block images** is checked before running the analysis,
and that the output directory is writable.

### Batch produces no combined CSV
This happens when no sample subfolders contain valid TIFF files
matching the pattern, or when all files fail processing.

Check the **Results & Logs** tab for error messages.

### picasso not found warning
If picasso is not installed, the software falls back to tifffile.
This is fine for standard TIFF files. Install picasso only if you
need support for very large multi-file TIFF stacks.