
# %%
from pynwb import NWBHDF5IO
with NWBHDF5IO(filepath, mode="r") as io2:
    nwbfile = io2.read()
    nwbfile

# %%
nwbfile.experiment_description
nwbfile.experimenter
nwbfile.get_stimulus
# %%
nwbfile.ec_electrodes
# %%
from pynwb import NWBHDF5IO, NWBFile
filepath = r"C:\Users\khan332\Documents\GitHub\ssm\000363\sub-449141\sub-449141_ses-20190531T161406_behavior+ecephys+ogen.nwb"

with NWBHDF5IO(filepath, 'r') as io:
    nwbfile = io.read()
    a = nwbfile.acquisition['BehavioralEvents'].time_series['trialend_start_times']
    b = nwbfile.acquisition['BehavioralEvents'].time_series['trialend_stop_times']
    trial_start = a.timestamps[:]
    trial_end = b.timestamps[:]
    # Get raw spike times
    s = nwbfile.units.spike_times
    spike_times = s[:]
    # Get unit assignment
    i = nwbfile.units.spike_times_index
    spike_index = i[:]
    print(trial_start.shape)
    print(trial_end.shape)
    print(s.shape)
    print(i.shape)
    print(nwbfile.units.sampling_rate[:].shape)

# %% now loop through the spike index and build a vector array of times when the spike happens
n_trials = len(trial_start)
n_units = len(spike_index)

spikes_per_trial_unit = [
    [None for _ in range(n_units)] for _ in range(n_trials)
]

for unit in range(n_units):
    spikes = spike_index[unit]  # spike times for this unit

    for trial in range(n_trials):
        # Mask spikes that fall within this trial window
        trial_mask = (spikes >= trial_start[trial]) & (spikes < trial_end[trial])
        spikes_in_trial = spikes[trial_mask]
        spikes_per_trial_unit[trial][unit] = spikes_in_trial

# %%
