import numpy as np
import pandas as pd

# —————— 1. Read or Prepare Data ——————
# If you already have a DataFrame df, you can directly take the columns:
df = pd.read_excel('./control_data/fuzzy_data_20250422_193739_1.xlsx', sheet_name='Sheet1')
x = df['Position_X'].to_numpy()
y = df['Position_Y'].to_numpy()

# —————— 2. Calculate Discrete Curvature ——————
n = len(x)
kappa = np.zeros(n)  # Curvature array, the first and last points cannot be calculated, keep 0

for i in range(1, n-1):
    p_prev = np.array([x[i-1], y[i-1]])
    p      = np.array([x[i],   y[i]])
    p_next = np.array([x[i+1], y[i+1]])
    
    v1 = p - p_prev
    v2 = p_next - p
    # The scalar equivalent of the 2D cross product
    cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    norm3 = np.linalg.norm(p_next - p_prev)
    denom = norm1 * norm2 * norm3
    
    if denom > 0:
        kappa[i] = 2 * cross / denom
    else:
        kappa[i] = 0

# —————— 3. Curvature Statistics and Linear Judgment ——————
# Skip the first and last two invalid values
curvatures = kappa[1:-1]

max_k = curvatures.max()
mean_k = curvatures.mean()

# Threshold epsilon, adjust according to the dimension of the coordinates
epsilon = 1e-6

print(f"Max Local Curvature max(kappa) = {max_k:.3e}")
print(f"Average Local Curvature mean(kappa) = {mean_k:.3e}")

if max_k < epsilon:
    print("The trajectory is approximately linear within the numerical precision range")
else:
    print("The trajectory is obviously curved and not a straight line")
