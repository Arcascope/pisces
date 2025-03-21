from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from main import EXPERIMENT_RESULTS_CSV
from conv2d_net import TrainingResult

# Load the CSV file

df = pd.read_csv(EXPERIMENT_RESULTS_CSV, names=TrainingResult.get_header())
fig, axis = plt.subplots(figsize=(20, 10))

plt.rcParams.update({'font.size': 14})

ID_COL = TrainingResult.id_column()
EXPERIMENT_COL = TrainingResult.experiment_id_column()
WAKE_ACC_COL = TrainingResult.wake_acc_column()

wake_df = df[[ID_COL, EXPERIMENT_COL, WAKE_ACC_COL]]
wake_df = wake_df.dropna()


# Group by test_id and plot
for test_id, group in wake_df.groupby(ID_COL):
    axis.plot(group[EXPERIMENT_COL], 
              group[WAKE_ACC_COL], 
            #  color='tab:blue',
             alpha=0.5)

# Now plot the median trend line
median = wake_df[[EXPERIMENT_COL, WAKE_ACC_COL]].groupby(EXPERIMENT_COL).median()
axis.plot(median.index, median[WAKE_ACC_COL], label='Median', color='black', linewidth=4, linestyle='--')
axis.fill_between(median.index, median[WAKE_ACC_COL], color='gray', alpha=0.3)

mean = wake_df[[EXPERIMENT_COL, WAKE_ACC_COL]].groupby(EXPERIMENT_COL).mean()
axis.plot(mean.index, mean[WAKE_ACC_COL], 'x-', label='Mean', color='black', linewidth=2)

# Now plot the established values "to beat"
blur_wasa = 0.59
mo_wasa = 0.66

axis.axhline(y=blur_wasa, color='r', linestyle=':', linewidth=4, label='Blur')
axis.axhline(y=mo_wasa, color='g', linestyle=':', linewidth=4, label='MO')

axis.set_xlabel('Experiment ID')
axis.set_ylabel('wasa95')
axis.set_title('wasa95 grouped by test_id')
axis.legend()
axis.tick_params(axis='x', rotation=75)
axis.set_ylim(0.0, 1.0)
axis.grid(visible=True, axis='both')
fig.tight_layout(pad=0.1)
# plt.show()
fig.savefig(str(EXPERIMENT_RESULTS_CSV.resolve()).replace(".csv", ".png"), bbox_inches='tight', dpi=200)