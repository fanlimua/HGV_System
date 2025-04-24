import pandas as pd
import numpy as np

# —————— 1. Read data ——————
# Replace 'your_file.xlsx' with your file name, and modify sheet_name according to the actual situation
df = pd.read_excel('./control_data/fuzzy_data_20250421_171631_0.3.xlsx', sheet_name='Sheet1')

# Extract coordinates
x = df['Position_X'].to_numpy()
y = df['Position_Y'].to_numpy()

# —————— 2. Circle fitting (Kasa method) ——————
# Construct linear equation A·[D, E, F]^T = b
A = np.column_stack([x, y, np.ones_like(x)])
b_vec = -(x**2 + y**2)

# Solve least squares: get D, E, F
D, E, F = np.linalg.lstsq(A, b_vec, rcond=None)[0]

# Calculate circle center (a, b0) and radius r from D, E, F
a = -D / 2
b0 = -E / 2
r = np.sqrt(a**2 + b0**2 - F)

# —————— 3. Error calculation ——————
# Distance from each point to the fitted circle center
distances = np.sqrt((x - a)**2 + (y - b0)**2)
# Absolute error from the fitted radius
errors = np.abs(distances - r)

# Maximum error
max_error = errors.max()
# Average error
mean_error = errors.mean()
# Standard deviation of error
std_error = errors.std()
# Roundness index (difference between maximum and minimum distance)
roundness = distances.max() - distances.min()

# —————— 4. Output results ——————
print(f'Fitted Circle Center:    ({a:.4f}, {b0:.4f})')
print(f'Fitted Circle Radius:    {r:.4f}')
print(f'Max Error:    {max_error:.4f}')
print(f'Average Error:    {mean_error:.4f}')
print(f'Standard Deviation:    {std_error:.4f}')
print(f'Roundness:    {roundness:.4f}')
