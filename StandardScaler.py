from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([[10, 200],
                 [20, 300],
                 [30, 400]])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Standard Scaled Data:\n", scaled_data)
