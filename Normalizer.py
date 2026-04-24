from sklearn.preprocessing import Normalizer
import numpy as np

data = np.array([[10, 200],
                 [20, 300],
                 [30, 400]])

scaler = Normalizer()
normalized_data = scaler.fit_transform(data)

print("Normalized Data:\n", normalized_data)
