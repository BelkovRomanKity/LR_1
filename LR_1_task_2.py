from sklearn import preprocessing
import numpy as np

# Значення змінної input_data
input_data = np.array([1.3, -3.9, 6.5, -4.9, -2.2, 1.3, 2.2, 6.5, -6.1, -5.4, -1.4, 2.2, 1.1])

# Поріг бінаризації
threshold = 2.2

# Бінаризація даних
data_binarized = preprocessing.Binarizer(threshold=threshold).transform(input_data.reshape(-1, 1))
print("\nBinarized data:\n", data_binarized.flatten())

# Виведення середнього значення та стандартного відхилення
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Виключення середнього
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування Min-Max
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data.reshape(-1, 1))
print("\nMin-Max scaled data:\n", data_scaled_minmax.flatten())

# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data.reshape(-1, 1), norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data.reshape(-1, 1), norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1.flatten())
print("\nl2 normalized data:\n", data_normalized_l2.flatten())