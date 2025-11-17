import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def process_traces_cython(cnp.ndarray[cnp.int8_t, ndim=2] traces, verbose):
    """
    Process traces to detect transitions between -1 and 1 with zero-runs in between.
    
    Args:
        traces: numpy array of shape (n_frames, n_traces) with values in {-1, 0, 1}
    
    Returns:
        List of lists, one per trace, containing tuples of (start, stop, duration, sign, trace_id)
        where sign is the destination value (-1 or 1)
    """
    cdef int n_frames = traces.shape[0]
    cdef int n_traces = traces.shape[1]
    cdef int i, j, val, duration
    cdef cnp.ndarray[cnp.int8_t, ndim=1] last_nonzero = np.zeros(n_traces, dtype=np.int8)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] zero_start = np.full(n_traces, -1, dtype=np.int32)
    
    # Results as a list of lists
    results = [[] for _ in range(n_traces)]
    
    for i in range(n_frames):
        if verbose: 
            if i % 10000 == 0:
                print(f'Frame: {i+1}/{n_frames}'.ljust(25), end='\r')
        
        for j in range(n_traces):
            val = traces[i, j]
            
            if val == 0:
                if last_nonzero[j] != 0 and zero_start[j] == -1:
                    zero_start[j] = i - 1
            
            elif val == -1 or val == 1:
                if last_nonzero[j] != 0 and last_nonzero[j] != val and zero_start[j] == -1:
                    results[j].append((i-1, i, 0, int(val), j))
                
                elif zero_start[j] != -1:
                    if last_nonzero[j] != val:
                        duration = i - zero_start[j] - 1
                        results[j].append((int(zero_start[j]), i, duration, int(val), j))
                    zero_start[j] = -1
                
                last_nonzero[j] = val
    if verbose: print(f'Frame: {i+1}/{n_frames}'.ljust(25))

    # Convert output to array shapde as start, stop, duration, sign, trace_id
    all_flips = []
    for flips in results:
        if len(flips):
            all_flips += flips
    return np.array(all_flips)