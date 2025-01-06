from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
cwd = Path(__file__).resolve().parent

df = pd.read_csv(cwd / 'saved_outputs/wasa95.csv')
plt.figure(figsize=(20, 10))

plt.rcParams.update({'font.size': 14})

# Group by test_id and plot
for test_id, group in df.groupby('test_id'):
    plt.plot(group['experiment_id'], group['wasa95'], color='tab:blue', alpha=0.5)

# Now plot the median trend line
median = df.groupby('experiment_id').median()
plt.plot(median.index, median['wasa95'], label='Median', color='red', linewidth=3, linestyle='--')
plt.fill_between(median.index, median['wasa95'], color='gray', alpha=0.3)

mean = df.groupby('experiment_id').mean()
plt.plot(mean.index, mean['wasa95'], label='Mean', color='orange', linewidth=2, linestyle=':')

plt.xlabel('Experiment ID')
plt.ylabel('wasa95')
plt.title('wasa95 grouped by test_id')
plt.legend()
plt.xticks(rotation=90)
plt.grid(visible=True, axis='x')
plt.tight_layout(pad=0.1)
# plt.show()
plt.savefig(cwd / 'saved_outputs/wasa95.png', dpi=200)