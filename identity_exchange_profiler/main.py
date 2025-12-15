import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .process_traces_cython import process_traces_cython   


class Profiler:
    """
    Analysis class for tracking exchange/flips in particle traces between segment ID A and B.

    This is a generalization of lipid flip-flops between two leaflets.
    
    Usage:
        analysis = ResidueExchangeAnalysis(segmentation, segment_A=1, segment_B=2)
        analysis.compute()
        analysis.plot()
    """
    
    def __init__(self, segmentation, segment_A=1, segment_B=2, 
                 mask_valid_frames=True, blib_processing=True, dt=1, verbose=True, ):
        """
        Initialize the analysis.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation data with shape (n_frames, n_traces)
        segment_A : int
            Segment ID for the first segment
        segment_B : int
            Segment ID for the second segment
        mask_valid_frames : bool
            Whether to mask frames where either segment A or B is missing (default=True)
        blib_processing : bool
            Whether to apply blib smearing to smooth out temporary segment losses (default=True)
        dt : int
            The amount of time in ps in between frames (default=1)
        """
        self._segmentation = segmentation
        self._segment_A = segment_A
        self._segment_B = segment_B
        self._mask_valid_frames = mask_valid_frames
        self._blib_processing = blib_processing
        self._dt = dt
        self._verbose = verbose
        
        # Results (populated by compute())
        self._masked_segmentation = None
        self._valid_mask = None
        self._flips = None
        self._flip_counts = None
        self._AB = None
        self._BA = None
        self._cumulative_AB = None
        self._cumulative_BA = None
        self._cumulative_net_AB = None
        self._computed = False

    @staticmethod
    def filter_close_events(data, threshold):
        """
        Vectorized filter for events that are close together in frames and have opposite directions.

        Parameters
        ----------
        data : np.ndarray
            Array with columns: [start, stop, dt, direction, trace_id]
        threshold : int
            Minimum time gap between events to pass filtering

        Returns
        -------
        np.ndarray
            Filtered array with close opposite-direction pairs removed
        """
        if data is None or len(data) == 0:
            return data

        sort_idx = np.lexsort((data[:, 0], data[:, 4]))
        sorted_data = data[sort_idx]

        same_trace = sorted_data[:-1, 4] == sorted_data[1:, 4]
        gaps = sorted_data[1:, 0] - sorted_data[:-1, 1]
        opposite_dir = sorted_data[:-1, 3] != sorted_data[1:, 3]

        pairs_to_remove = same_trace & (gaps < threshold) & opposite_dir

        keep_mask = np.ones(len(sorted_data), dtype=bool)
        keep_mask[:-1] &= ~pairs_to_remove
        keep_mask[1:] &= ~pairs_to_remove

        filtered_sorted = sorted_data[keep_mask]
        original_order = np.argsort(sort_idx[keep_mask])

        return filtered_sorted[original_order]

    @property
    def segmentation(self):
        """Input segmentation array"""
        return self._segmentation
    
    @property
    def segment_A(self):
        """Input segment label set to A (1)"""
        return self._segment_A

    @property
    def segment_B(self):
        """Input segment label set to B (-1)"""
        return self._segment_B
    
    @property
    def mask_valid_frames(self):
        """Input for masking valid frames with both segments being present"""
        return self._mask_valid_frames

    @property
    def blib_processing(self):
        """Input boolean, True if blib processing was used"""
        return self._blib_processing

    @property
    def dt(self):
        """Input timestep in ps (e.g. for MDAnalysis u.trajectory.dt)"""
        return self._dt
        
    @property
    def n_residues(self):
        """Number of residues in selection"""
        return self.segmentation.shape[1]
    
    @property
    def n_frames(self):
        """Number of frames in trajectory"""
        return self.segmentation.shape[0]

    @property
    def masked_segmentation(self):
        """
        Copy of segmentation array, where A=1, B=-1, Other=0 if A and B in a
        frame, else all values in the frame are set to 0
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._masked_segmentation

    @property
    def valid_mask(self):
        """
        Boolean array of size n_frames, True if segment A and B in frame 
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._valid_mask

    @property
    def flips(self):
        """
        Array of (:, [start, stop, duration, sign, trace_id])
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")
        
        return self._flips

    @property
    def flip_counts(self):
        """Array of (:, [trace_id, flip_count])"""
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._flip_counts
        
    @property
    def AB(self):
        """
        Array of the amount of flips A->B per frame over time 
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._AB

    @property
    def BA(self):
        """
        Array of the amount of flips B->A per frame over time
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._BA

    @property
    def cumulative_AB(self):
        """
        Array of the cumulative amount of flips A->B per frame over time
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._cumulative_AB

    @property
    def cumulative_BA(self):
        """
        Array of the cumulative amount of flips B->A per frame over time
        """
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._cumulative_BA

    @property
    def cumulative_net_AB(self):
        """Array of cumulativeAB - cumulativeBA"""
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")

        return self._cumulative_net_AB
    
    def _apply_segment_mask(self, segments):
        """
        Apply segment masking: convert segment IDs to 1 and -1, 
        everything else to 0.
        """
        if self._verbose: print("Applying segment masks...")
        segment_A_mask = segments == self.segment_A
        segment_B_mask = segments == self.segment_B
        
        masked = np.zeros_like(segments, dtype=np.int8)
        masked[segment_A_mask] = 1
        masked[segment_B_mask] = -1
        return masked

    def _mask_frames_without_segments(self, segments):
        """
        Set all residues to 0 in frames where either segment A or B
        does not appear at all (to avoid artifacts).

        Parameters
        ----------
        segments : np.ndarray
            Segmentation array with A=1, B=-1, Other=0
        
        Returns
        -------
        segments : np.ndarray
            Masked segments array
        """
        if self._verbose: print("Masking frames without both segments...")
        # Boolean arrays per frame
        has_A = np.any(segments == 1, axis=1)
        has_B = np.any(segments == -1, axis=1)
        valid = has_A & has_B
    
        # Frames where either A or B is missing → zero out
        segments[~valid, :] = 0
        return segments, valid
    
    def _calculate_flip_timeseries(self, raw_flips):
        """
        Convert raw flip data (start, stop, duration, sign, trace_id) into
        per-frame flip counts for AB and BA.
        
        Parameters
        ----------
        raw_flips : np.ndarray
            Array of shape (n_flips, 5) with columns:
            (start, stop, duration, sign, trace_id)
        
        Returns
        -------
        AB, BA : np.ndarray
            Per-frame flip counts (length = n_frames - 1)
        """
        if self._verbose: print("Calculating flip time series...")
        n_frames = self.n_frames
        
        # Initialize arrays (one less frame because flips happen between frames)
        AB = np.zeros(n_frames - 1, dtype=np.float64)
        BA = np.zeros(n_frames - 1, dtype=np.float64)
        
        if len(raw_flips) == 0:
            return AB, BA
        
        # Extract columns
        start_frames = raw_flips[:, 0].astype(int)
        stop_frames = raw_flips[:, 1].astype(int)
        durations = raw_flips[:, 2].astype(int)
        signs = raw_flips[:, 3].astype(int)
        
        for i in range(len(raw_flips)):
            start = start_frames[i]
            stop = stop_frames[i]
            duration = durations[i]
            sign = signs[i]
            
            if sign == -1:
                # Started from A (1) and ended at B (-1)
                AB[start] += 1
            elif sign == 1:
                # Started from B (-1) and ended at A (1)
                BA[start] += 1

        return AB, BA
    
    def _smear_blib_signals(self, AB, BA, valid_mask):
        """
        Smear flip signals over frames where segments are temporarily lost (blibs).
        
        The last non-blibbed flip value is smeared over the blib.
        Blib smearing preserves total flip count whilst making the time series continuous.
    
        Parameters
        ----------
        AB, BA : np.ndarray
            Instantaneous flip counts per frame.
    
        Returns
        -------
        AB_smooth, BA_smooth,  : np.ndarray
            Smoothed flip counts
        cumulative_AB, cumulative_BA : np.ndarray
            Cumulative flip counts
        cumulative_net_AB : np.ndarray
            Cumulative net AB flips
        """
        if self._verbose: print("Smearing blib signals...")
        
        def smear_signal(y):
            y_smooth = np.zeros_like(y, dtype=float)
            n = len(y)
            i = 0
            while i < n:
                if valid_mask[i]:
                    y_smooth[i] = y[i]
                    i += 1
                else:
                    start = i - 1
                    while i < n and not valid_mask[i]:
                        i += 1
                    end = i - 1
                    if end < n and start >= 0:
                        val = y[start]
                        blib_len = end - start + 1
                        smear_val = val / blib_len
                        y_smooth[start:end + 1] = smear_val
                    else:
                        y_smooth[start:] = 0
            return y_smooth
    
        AB_smooth = smear_signal(AB)
        BA_smooth = smear_signal(BA)
    
        cumulative_AB = np.cumsum(AB_smooth)
        cumulative_BA = np.cumsum(BA_smooth)
        cumulative_net_AB = np.cumsum(AB_smooth - BA_smooth)
    
        return (AB_smooth, BA_smooth,
                cumulative_AB, cumulative_BA,
                cumulative_net_AB)

    def compute(self):
        """
        Run the complete analysis pipeline.
        
        Returns
        -------
        self : ResidueExchangeAnalysis
            Returns self for method chaining
        """
        # Extract segments for all frames
        segments = self.segmentation.copy()
        
        # Apply segment masking (A->1, B->-1, Other->0)
        segments = self._apply_segment_mask(segments)

        # Only process frames where A and B exist
        if self.mask_valid_frames:
            segments, valid_mask = self._mask_frames_without_segments(segments)
        else :
            valid_mask = np.ones(segments.shape[0], dtype=bool) 
        self._masked_segmentation = segments.copy()
        self._valid_mask = valid_mask
        
        # Run the flip detection
        if self._verbose: print("Running flip detection...")
        raw_flips = process_traces_cython(segments, self._verbose)
        self._flips = raw_flips
        self._flip_counts =  np.array(np.unique(raw_flips[:,4].astype(int), return_counts=True)).T
        if self._verbose: print(f"Detected {len(raw_flips)} flips")
        
        
        # Convert raw flips to per-frame time series
        AB, BA = self._calculate_flip_timeseries(raw_flips)
        
        # Post process: smear blibs if requested
        if self.blib_processing:
            (AB, BA, cumulative_AB, cumulative_BA, cumulative_net_AB) = \
            self._smear_blib_signals(AB, BA, valid_mask[:-1])  # diff reduces frame count by 1
        else:
            cumulative_AB = np.cumsum(AB)
            cumulative_BA = np.cumsum(BA)
            cumulative_net_AB = np.cumsum(AB - BA)
        
        # Store results
        self._AB = AB
        self._BA = BA
        self._cumulative_AB = cumulative_AB
        self._cumulative_BA = cumulative_BA
        self._cumulative_net_AB = cumulative_net_AB
        self._computed = True
        
        if self._verbose: print("Analysis complete!")
        if self._verbose: self._print_summary()
        
        return self
    
    def _print_summary(self):
        """Print a summary of the analysis results."""
        if not self._computed:
            print("No results yet. Run .compute() first.")
            return
        
        total_AB = np.sum(self.AB)
        total_BA = np.sum(self.BA)
        net_exchange_AB = total_AB - total_BA
        
        print("\n" + "="*50)
        print("RESIDUE EXCHANGE SUMMARY")
        print("="*50)
        print(f"Total flips (segment {self.segment_A} → {self.segment_B}): {total_AB:.0f}")
        print(f"Total flips (segment {self.segment_B} → {self.segment_A}): {abs(total_BA):.0f}")
        print(f"Net exchange {self.segment_A} → {self.segment_B}: {net_exchange_AB:.0f}")
        print(f"Frames analyzed: {self.n_frames}")
        print(f"Residues tracked: {self.n_residues}")
        print(f"Total flip events detected: {len(self.flips)}")
        print("="*50 + "\n")

    def plot(self, figsize=(15, 6), save=None, running_average = 0, time_units = ('mus', 'mus')):
        """
        Plot AB/BA flips with dual y-axes and histogram:
        - Left panel: Time series with dual y-axes
        - Left y-axis: AB and BA (scaled to occupy half vertical range)
        - Right y-axis: Cumulative net AB (full symmetric range)
        - Right panel: Histogram of flip durations by direction
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        save : str, optional
            Path to save figure
        running_average: int
            Performs a running average over the AB and BA traces,
            but leaves the cumsum as is (default: 0)
        time_units: (str, str)
            The time units for the traces and the durations
            ['fs', 'ps', 'ns', 'mus', 'ms', s] (default: mus, mus)

        Returns
        -------
        fig  : matplotlib.figure.Figure
            The plotted figure
        """
        # We can only plot what is computed
        if not self._computed:
            raise RuntimeError("No results. Run .compute() first.")
    
        # Auto handle the units
        time_units = list(time_units)
        time_scalars = (1*10**3, 1*10**0, 1*10**-3, 1*10**-6, 1*10**-9, 1*10**-12)
        time_units_str = ('fs', 'ps', 'ns', 'mus', 'ms', 's')

        time_scalar_traces = time_scalars[time_units_str.index(time_units[0])]
        time_scalar_durations = time_scalars[time_units_str.index(time_units[1])]

        # Nicely format mus
        for idx in range(len(time_units)):
            if time_units[idx] == 'mus':
                time_unit = 'μs'
            else:
                time_unit = time_units[idx]
            time_units[idx] = time_unit
        time_unit_traces = time_units[0]
        time_unit_durations = time_units[1]

        # Get the flip time trace and average if demanded
        if running_average:
            AB = np.convolve(self.AB, np.ones(running_average), 'same') / running_average
            BA = np.convolve(self.BA, np.ones(running_average), 'same') / running_average
            #cumsum = np.convolve(self.cumulative_net_AB, np.ones(running_average), 'same') / running_average
        else:
            AB, BA = self.AB, self.BA
        cumsum = self.cumulative_net_AB

        dt = self.dt * time_scalar_traces
        time_values = np.arange(len(AB)) * dt
    
        # Determine scaling
        max_flip = max(abs(AB).max(), abs(BA).max())
        max_cumsum = abs(cumsum).max()
    
        # Left axis range (AB/BA occupy center half)
        flip_ylim = max_flip * 1.05
        # Right axis range (cumsum symmetrical)
        cumsum_ylim = max_cumsum * 1.05
    
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot(1, 2, 1)
        ax2 = ax1.twinx()  # secondary y-axis for cumulative data
        ax_hist = plt.subplot(1, 2, 2)

        # Create positive y ticks for both flip directions
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{abs(y):.02f}"))
    
        # --- Plot AB and BA on left axis ---
        ax1.plot(time_values, AB, color='tab:blue', 
                label=f'Inst. {self.segment_A}→{self.segment_B} (n={self.cumulative_AB[-1]:.0f})', 
                alpha=0.5, linewidth=1)
        ax1.plot(time_values, -BA, color='tab:orange', 
                label=f'Inst. {self.segment_B}→{self.segment_A} (n={abs(self.cumulative_BA[-1]):.0f})', 
                alpha=0.5, linewidth=1)
        ax1.set_ylim(-flip_ylim, flip_ylim)
        ax1.set_ylabel('Instantaneous flips', fontsize=12)
        ax1.hlines(0, time_values[0], time_values[-1], linewidth=1, linestyle='--', color='grey')
    
        # --- Plot cumulative net AB on right axis ---
        ax2.plot(time_values, cumsum, color='tab:green', 
                label=f'Net {self.segment_A}→{self.segment_B} ({self.cumulative_net_AB[-1]:.0f})', 
                alpha=0.8, linewidth=2)
        ax2.set_ylim(-cumsum_ylim, cumsum_ylim)
        ax2.set_ylabel('Cumulative net flips', fontsize=12)
    
        # --- Formatting ---
        ax1.set_xlabel(f'Time ({time_unit_traces})', fontsize=12)
        ax1.set_title('Residue Exchange Between Segments', fontsize=14)
    
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc='upper left')

        ax1.margins(x=0)
        ax2.margins(x=0)
        
        # --- Plot histogram if requested ---
        durations = self.flips[:, 2] * time_scalar_durations * self.dt
        signs = self.flips[:, 3]
        
        # Separate by direction
        BA_durations = durations[signs > 0]  # Positive sign: B→A
        AB_durations = durations[signs < 0]  # Negative sign: A→B
        
        # Define common bin edges
        all_durations = np.concatenate([BA_durations, AB_durations]) if len(BA_durations) > 0 and len(AB_durations) > 0 else (BA_durations if len(BA_durations) > 0 else AB_durations)
        bins = np.histogram_bin_edges(all_durations, bins=50)
        
        # Plot histograms
        if len(AB_durations) > 0:
            ax_hist.hist(AB_durations, bins=bins, alpha=0.6, 
                        label=f'{self.segment_A}→{self.segment_B} (n={len(AB_durations)})',
                        color='tab:blue', edgecolor='black', linewidth=0.5)
        if len(BA_durations) > 0:
            ax_hist.hist(BA_durations, bins=bins, alpha=0.6, 
                        label=f'{self.segment_B}→{self.segment_A} (n={len(BA_durations)})',
                        color='tab:orange', edgecolor='black', linewidth=0.5)
        
        ax_hist.set_xlabel(f'Flip duration ({time_unit_durations})', fontsize=12)
        ax_hist.set_ylabel('Count', fontsize=12)
        ax_hist.set_title('Flip Duration Distribution', fontsize=14)
        ax_hist.legend(fontsize=10)
        ax_hist.grid(alpha=0.3)
        
        plt.tight_layout()
    
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save}")
    
        return fig
