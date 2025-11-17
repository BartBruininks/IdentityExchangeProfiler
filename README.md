# Identity Exchange Profiler (IEP)

## How to install
```
pip install git+https://github.com/BartBruininks/IdentityExchangeProfiler
```

## How to use
Using segmentation data, like the leaflet identity per lipid over time the lipid flip-flop can be profiled in an efficient and precise manner. The two labels of interest should be specified, in the case of lipid flip-flop this would be the label for the outer and inner leaflets.

```python3
import identity_exchange_profiler as iep

# All attributes and methods have proper documentation
help(iep.Profiler)

# Load the segmentation data
segmentation = np.load('clusters.npy')

# Set the bare minimum input values
segment_A = 1 # inner leaflet
segment_B = 2 # outer leaflet
dt = 1000.0 # in ps, we save a frame every 1 ns

# Prepare the analysis
analysis = iep.Profiler(
    segmentation=segmentation,
    segment_A=segment_A,
    segment_B=segment_B,
    dt = dt,
    verbose = True,
)

# Perform the actual analysis
analysis.compute()

# Plot the results
analysis.plot(dt=dt)
```

## Check examples
A more detailed usecase is given in the ./examples.
