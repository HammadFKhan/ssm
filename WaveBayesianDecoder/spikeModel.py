import h5py
import numpy as np

mat_file_path = r"D:\BayesianWaveModel\Day4WavesIntan.mat"

with h5py.File(mat_file_path, 'r') as f:
    print(list(f.keys()))
    intan_behaviour = f['IntanBehaviour']

    # ---- lever traces (hit trials) ----
    hit = intan_behaviour['hitTrace']
    hit_trace = hit['trace']          # object references to each trial
    hit_refs = hit_trace[:].flatten()
    lever_traces = [f[ref][:] for ref in hit_refs]
    lever_traces = np.squeeze(np.stack(lever_traces, axis=0))  # (n_trials, T)

    print("lever_traces shape:", lever_traces.shape)

    # ---- PSTH spikes for hit trials ----
    spikes_hit = f['Spikes']['PSTH']['hit']['spks']
    
    # Each element is an object reference to spike data
    spk_refs = spikes_hit[:].flatten()
    spk_list = [f[ref][:] for ref in spk_refs]
    
    # Stack into array (n_trials, ...)
    spk_array = np.stack(spk_list, axis=0)
    
    print("spk_array shape:", spk_array.shape)
    
# Transpose from MATLAB's column-major (Fortran) order to row-major (C) order
spk_array = np.transpose(spk_array, (2, 1, 0))  # Reverse axes for MATLAB conversion

