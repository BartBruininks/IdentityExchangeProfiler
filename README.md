# Identity Exchange Profiler (IEP)
Using segmentation data, like the leaflet identity per lipid over time, the lipid flip-flop can be profiled in an efficient and precise manner. The two labels of interest should be specified, in the case of lipid flip-flop, these would be the labels for the outer and inner leaflets. The segmentation can be obtained using [MDVoxelSegmentation](https://github.com/marrink-lab/MDVoxelSegmentation)).

## How to install
```
pip install git+https://github.com/BartBruininks/IdentityExchangeProfiler
```

## How to use
```python3
import identity_exchange_profiler as iep

# !mdvseg -f your.gro -x your.xtc -fs 0 # Create a clusters.npy for martini lipids

## All attributes and methods have proper documentation
help(iep.Profiler)

## Segments to check flips between
segment_A = 1
segment_B = 2

## Pre-filtering by certainty
# N frames to use for calculating fuzzy label assignment 
fuzzy_width = 5 
# The average relabeled frame annotation {-1, 0, 1} over the
# fuzzy width to use as threshold to set label
fuzzy_cutoff = .5 

## Pre-filtering for both labels present (useful for lipid flip-flop)
mask_valid_frames = True

## Post-processing to fill in
# Fills in the removed frames due to the fact that not both labels
# where present (respecting the total sum).
blib_processing = True

# Time indication
skip = 1 # Every nth frame to take into account
dt = 1.0 * skip # is expected to be in ps

# Set the segmentation data to use
selected_segmentation = np.copy(test_segmentation)

# Verbosity if said to True be loud and noisy
verbose = True

# Run analysis
analysis = iep.Profiler(
    segmentation=selected_segmentation[::skip],
    segment_A=segment_A,
    segment_B=segment_B,
    fuzzy_width=fuzzy_width, 
    fuzzy_cutoff=fuzzy_cutoff,
    mask_valid_frames=mask_valid_frames,
    blib_processing=blib_processing,
    dt=dt,
    verbose=verbose,
)

# Perform the actual analysis
analysis.compute()

# Plot the results
plot = analysis.plot(dt=dt)
```

<img width="1490" height="590" alt="Untitled" src="https://github.com/user-attachments/assets/d2571b2f-ef39-40ea-bdac-9c9ade5f401e" />


## Check examples
A more detailed [usecase](https://github.com/BartBruininks/IdentityExchangeProfiler/blob/main/examples/lipid_exchange.ipynb) is given in the `./examples`.
