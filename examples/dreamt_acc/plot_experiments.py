from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from main import EXPERIMENT_RESULTS_CSV

# Load the CSV file

df = pd.read_csv(EXPERIMENT_RESULTS_CSV)
plt.figure(figsize=(20, 10))

plt.rcParams.update({'font.size': 14})

# Group by test_id and plot
for test_id, group in df.groupby('test_id'):
    plt.plot(group['experiment_id'], group['wasa95'], 
            #  color='tab:blue',
             alpha=0.5)

# Now plot the median trend line
median = df.groupby('experiment_id').median()
plt.plot(median.index, median['wasa95'], label='Median', color='black', linewidth=4, linestyle='--')
plt.fill_between(median.index, median['wasa95'], color='gray', alpha=0.3)

mean = df.groupby('experiment_id').mean()
plt.plot(mean.index, mean['wasa95'], 'x-', label='Mean', color='black', linewidth=2)

# Now plot the established values "to beat"
blur_wasa = 0.59
mo_wasa = 0.66

plt.axhline(y=blur_wasa, color='r', linestyle=':', linewidth=4, label='Blur')
plt.axhline(y=mo_wasa, color='g', linestyle=':', linewidth=4, label='MO')

plt.xlabel('Experiment ID')
plt.ylabel('wasa95')
plt.title('wasa95 grouped by test_id')
plt.legend()
plt.xticks(rotation=90)
plt.grid(visible=True, axis='both')
plt.tight_layout(pad=0.1)
# plt.show()
plt.savefig(cwd / 'saved_outputs/wasa95.png', bbox_inches='tight', dpi=200)