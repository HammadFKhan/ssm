# Load in TW decoder results and make some plots. 
# This is a refactored version of the original script, 
# with the same functionality but better organization and output management.
#%% Imports and setup
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# load in directory paths and file names of NPZ files
main_dir = r"D:\BayesianWaveModel\M1Waves\model_output"
# go through each folder in the directory and find the NPZ files
TCN_full_corr = []  # to store correlation values for all files
TCN_mean_corr = []  # to store correlation values for all files
TCN_shuff_corr = []
for folder in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.npz'):
                # load in .npz file and grab the full_tcn prediction
                lever_tcn_full = np.load(os.path.join(folder_path, file))['lever_tcn_full']
                lever_tcn_mean = np.load(os.path.join(folder_path, file))['lever_tcn_mean']
                try:
                    lever_tcn_shuff = np.load(os.path.join(folder_path, file))['full_corrs_shuf']
                except KeyError:
                    print(f"Warning: 'full_corrs_shuf' not found in {file}. Setting shuffled correlation to NaN.")
                    lever_tcn_shuff = 0
                lever_true = np.load(os.path.join(folder_path, file))['lever_true']
                # display size for both arrays
               # print(f"Loaded {file} with shapes: lever_tcn_full {lever_tcn_full.shape}, lever_true {lever_true.shape}")
                # create variable that calculate mean correlation between true and tcn across all time points
                full_correlation = np.corrcoef(lever_true.flatten(), lever_tcn_full.flatten())[0, 1]
                # store variable into array for later plotting
                TCN_full_corr.append(full_correlation)
                mean_correlation = np.corrcoef(lever_true.flatten(), lever_tcn_mean.flatten())[0, 1]
                TCN_mean_corr.append(mean_correlation)
                TCN_shuff_corr.append(np.mean(lever_tcn_shuff))
                print(f"Correlation of full, mean, and shuffled tcn prediction for {file}: {full_correlation:.4f}, {mean_correlation:.4f}, and {np.mean(lever_tcn_shuff):.4f}")

                

# %% Plot out correlation values of TCN full and mean
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(8, 6))
x_pos = np.arange(3)
plt.bar(x_pos, [np.mean(TCN_full_corr), np.mean(TCN_mean_corr), np.mean(TCN_shuff_corr)], 
    yerr=[np.std(TCN_full_corr), np.std(TCN_mean_corr), np.std(TCN_shuff_corr)],
    capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.xticks(x_pos, ['Full TCN', 'Mean TCN', 'Shuffled TCN'])
plt.ylabel('Correlation with True Lever')
plt.title('Correlation of TCN Predictions with True Lever')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'TCN_correlation_comparison.pdf'))  # save figure as PDF in main directory
# %% Display statistics of correlation values
print(f"Mean correlation of Full TCN: {np.mean(TCN_full_corr):.4f} ± {np.std(TCN_full_corr):.4f}")
print(f"Mean correlation of Mean TCN: {np.mean(TCN_mean_corr):.4f} ± {np.std(TCN_mean_corr):.4f}")
print(f"Mean correlation of Shuffled TCN: {np.mean(TCN_shuff_corr):.4f} ± {np.std(TCN_shuff_corr):.4f}")

# perform anova testing across full, mean, and shuffled TCN correlation values
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(TCN_full_corr, TCN_mean_corr, TCN_shuff_corr)
# Calculate degrees of freedom for ANOVA
k = 3  # number of groups
N = len(TCN_full_corr) + len(TCN_mean_corr) + len(TCN_shuff_corr)
df_between = k - 1
df_within = N - k
# Print ANOVA results with degrees of freedom

print(f"ANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4e}, df_between = {df_between}, df_within = {df_within}")
# if significant perform post-hoc testing using Tukey's HSD
if p_value < 0.05:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    data = TCN_full_corr + TCN_mean_corr + TCN_shuff_corr
    groups = (['Full TCN'] * len(TCN_full_corr)) + (['Mean TCN'] * len(TCN_mean_corr)) + (['Shuffled TCN'] * len(TCN_shuff_corr))
    tukey_results = pairwise_tukeyhsd(data, groups)
    print(tukey_results)
    # print p values for each comparison
    print(f"Post-hoc Tukey HSD results:\n{tukey_results.summary()}")

    

# %%
