from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
cwd = Path(__file__).resolve().parent

df = pd.read_csv(cwd / 'saved_outputs/wasa95.csv')

# Group by test_id and plot
for test_id, group in df.groupby('test_id'):
    plt.plot(group['experiment_id'], group['wasa95'], label=f'Test ID {test_id}')

plt.xlabel('Experiment ID')
plt.ylabel('wasa95')
plt.title('wasa95 grouped by test_id')
# plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
# plt.show()
plt.savefig(cwd / 'saved_outputs/wasa95.png')