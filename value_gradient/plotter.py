import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files into pandas dataframes
df1 = pd.read_csv('/home/daniel/Downloads/twolin_inv1.csv', dtype=float)
df2 = pd.read_csv('/home/daniel/Downloads/twolink_inv2.csv', dtype=float)
df3 = pd.read_csv('/home/daniel/Downloads/twolink_inv3.csv', dtype=float)

# Extract the loss values from the second column of each dataframe
losses1 = df1.iloc[:, 1].values
losses2 = df2.iloc[:, 1].values
losses3 = df3.iloc[:, 1].values
# Normalize all loss arrays to the maximum value of any of the three arrays
max_loss = max([np.max(losses1), np.max(losses2), np.max(losses3)])
losses1 /= max_loss
losses2 /= max_loss
losses3 /= max_loss

# Truncate all loss arrays to the length of the shortest array
min_length = min(len(losses1), len(losses2), len(losses3))
losses1 = losses1[:min_length]
losses2 = losses2[:min_length]
losses3 = losses3[:min_length]

# Define the x data
x = np.arange(0, len(losses1), 1)

# Calculate the mean and standard deviation of the losses at each x-value
y_mean = np.mean([losses1, losses2, losses3], axis=0)
y_std = np.std([losses1, losses2, losses3], axis=0)

# Define the upper and lower bounds of the shaded region as the mean plus/minus the standard deviation
upper_bound = y_mean + y_std
lower_bound = y_mean - y_std

# Create the plot
plt.plot(x, y_mean, label='Mean Loss')
plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, label='Loss Variance')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Normalized Loss')
plt.title('Reacher loss INV mode')

# Save the plot as a high-resolution PNG
plt.savefig('inv_reacher_loss.png', dpi=300)

plt.show()

