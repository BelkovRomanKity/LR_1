import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


try:
    data = pd.read_csv("data_multivar_nb.txt", delimiter=",", header=None)  # Якщо дані розділені комами
except Exception as e:
    print("Помилка при зчитуванні файлу:", e)


try:
    data.iloc[:, -1] = data.iloc[:, -1].astype(float).astype(int)
except ValueError:
    print("Увага! Останній стовпець містить нечислові значення або некоректний формат.")
    print(data.iloc[:, -1].unique())

# Розділення даних на ознаки (X) і мітки класів (y)
X = data.iloc[:, :-1].astype(float)
y = data.iloc[:, -1].astype(int)

# Розділення даних на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація та навчання моделей
svm_model = SVC(kernel='linear', random_state=42)  # Машина опорних векторів
nb_model = GaussianNB()  # Наївний байєсівський класифікатор

# Навчання моделей
svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred_svm = svm_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)


print("Показники якості для SVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_svm, average='weighted'))
print(classification_report(y_test, y_pred_svm))


print("\nПоказники якості для Наївного Байєса:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_nb, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print(classification_report(y_test, y_pred_nb))

print("\nВисновок:")
if f1_score(y_test, y_pred_svm, average='weighted') > f1_score(y_test, y_pred_nb, average='weighted'):
    print("Модель SVM має вищі показники якості і рекомендується для вибору.")
else:
    print("Наївний байєсівський класифікатор має вищі показники якості і рекомендується для вибору.")
