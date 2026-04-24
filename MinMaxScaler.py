from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[10, 200],
                 [20, 300],
                 [30, 400]])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Min-Max Scaled Data:\n", scaled_data)
