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
plot = analysis.plot(dt=dt)
```

<img width="1490" height="590" alt="Untitled" src="https://github.com/user-attachments/assets/d2571b2f-ef39-40ea-bdac-9c9ade5f401e" />


## Check examples
A more detailed [usecase](https://github.com/BartBruininks/IdentityExchangeProfiler/blob/main/examples/lipid_exchange.ipynb) is given in the `./examples`.
