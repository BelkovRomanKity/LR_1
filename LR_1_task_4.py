# Імпорт необхідних пакетів
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    # Налаштування області графіка
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Передбачення для кожної точки в сітці
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Візуалізація меж рішень
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Logistic Regression Classifier")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# Визначення зразка вхідних даних
X = np.array([[3.1, 7.2], [4.0, 6.7], [2.9, 8.0], [5.1, 4.5],
              [6.0, 5.0], [5.6, 5.0], [3.3, 0.4],
              [3.9, 0.9], [2.8, 1.0],
              [0.5, 3.4], [1.0, 4.0], [0.6, 4.9]])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Створення логістичного класифікатора
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Тренування класифікатора
classifier.fit(X, y)

# Візуалізація результатів роботи класифікатора
visualize_classifier(classifier, X, y)

# Показати графік
plt.show()

# Створення файлу та запис тексту
with open('data_multivar_nb.txt', 'w') as f:
    f.write("""2.18,0.57,0
4.13,5.12,1
9.87,1.95,2
4.02,-0.8,3
1.18,1.03,0
4.59,5.74,1
8.25,1.3,2
3.91,-0.68,3
0.55,1.26,0
5.64,6.67,1
9.22,1.46,2
4.36,-1.27,3
2.19,2.66,0
4.36,4.86,1
7.4,4.44,2
3.89,0.14,3
1.67,0.81,0
4.97,5.67,1
8.52,2.13,2
5.24,-0.76,3
1.06,2.07,0
5.69,6.08,1
9.52,2.98,2
4.73,-1.71,3
1.64,0.96,0
5.55,5.68,1
8.1,4.26,2
3.82,-1.23,3
1.09,1.4,0
4.83,6.16,1
7.81,1.86,2
4.91,-2.79,3
1.74,0.77,0
5.31,6.61,1
7.24,1.18,2
4.14,-1.36,3
1.12,2.73,0
5.76,4.67,1
9.04,1.63,2
3.89,-2.01,3
0.31,1.5,0
4.14,5.02,1
8.53,2.3,2
4.5,-0.74,3
1.69,1.72,0
4.99,5.01,1
7.93,3.39,2
3.84,-1.55,3
2.37,0.15,0
5.32,5.93,1
7.81,2.63,2
2.64,-0.69,3
0.59,0.67,0
5.54,5.44,1
9.62,4.59,2
3.6,-1.84,3
1.64,0.48,0
5.47,6.16,1
9.53,2.4,2
4.74,-1.74,3
1.96,1.99,0
5.92,6.03,1
7.88,1.71,2
5.39,-2.82,3
0.82,1.96,0
4.5,5.73,1
7.29,1.79,2
3.31,-1.45,3
0.94,2.09,0
5.2,6.15,1
7.98,1.61,2
4.47,-2.51,3
1.31,1.01,0
5.93,7.36,1
8.66,2.31,2
4.92,-1.34,3
-0.39,2.06,0
5.18,6.24,1
5.35,1.36,2
5.25,-0.93,3
1.65,2.26,0
5.74,6.22,1
8.04,1.6,2
4.43,-2.43,3
1.99,1.73,0
4.53,6.67,1
8.57,1.52,2
4.16,-0.22,3
1.5,1.31,0
6.13,6.32,1
10.62,1.67,2
4.92,-1.05,3
0.05,0.38,0
3.84,5.8,1
8.35,2.31,2
3.14,-1.23,3
1.81,1.33,0
2.56,5.91,1
9.02,2.76,2
4.6,-2.16,3
0.33,1.43,0
4.46,4.99,1
8.45,2.15,2
4.95,-2.04,3
0.87,1.53,0
4.81,6.54,1
9.79,2.24,2
3.27,-1.06,3
1.35,2.1,0
4.64,7.8,1
9.5,2.49,2
4.5,0.31,3
1.3,2.01,0
4.02,5.97,1
8.36,1.99,2
4.97,-2.08,3
0.62,1.85,0
5.38,5.19,1
9.05,3.26,2
4.72,-0.14,3
2.91,2.18,0
5.15,5.81,1
8.98,2.76,2
5.2,-1.74,3
0.29,2.09,0
5.73,6.34,1
9.78,2.65,2
4.6,-1.53,3
0.42,1.95,0
3.74,6.49,1
9.31,1.57,2
5.28,-0.33,3
2.2,1.72,0
4.55,5.14,1
7.77,2.17,2
4.43,-2.28,3
-0.04,0.47,0
4.1,6.79,1
8.61,1.64,2
4.22,-0.59,3
1.84,0.95,0
3.55,7.28,1
8.94,3.6,2
4.21,-2.44,3
1.24,3.41,0
5.16,7.36,1
8.93,1.25,2
4.09,-1.41,3
1.57,0.92,0
4.69,4.76,1
8.85,2.18,2
4.43,-0.82,3
0.59,2.13,0
5.84,5.02,1
8.9,3.61,2
4.04,0.71,3
0.01,1.43,0
5.22,6.49,1
8.37,2.41,2
4.95,-1.45,3
1.18,0.53,0
5.63,5.08,1
8.57,1.16,2
4.24,-1.03,3
0.94,1.44,0
5.92,5.07,1
8.85,4.45,2
2.64,-1.79,3
1.44,1.72,0
4.45,6.67,1
8.16,1.76,2
4.5,-2.56,3
1.9,2.23,0
4.92,6.01,1
8.67,2.18,2
4.51,-1.46,3
0.14,0.06,0
5.55,4.9,1
9.27,3.24,2
4.85,-1.07,3
1.52,2.25,0
6.17,5.23,1
7.55,1.92,2
4.72,-1.61,3
2.46,2.01,0
5.24,5.5,1
8.97,2.55,2
5.67,-1.36,3
2.08,3.0,0
5.6,5.93,1
7.48,3.28,2
3.58,-0.23,3
1.99,1.21,0
4.89,7.32,1
8.48,2.36,2
3.36,-0.88,3
1.49,1.44,0
4.97,6.23,1
8.64,2.56,2
4.79,-0.6,3
1.14,1.85,0
4.25,5.36,1
8.01,0.68,2
3.89,-1.53,3
1.31,1.78,0
3.51,6.48,1
7.57,1.73,2
4.38,-2.39,3
1.36,1.52,0
6.32,4.84,1
7.75,2.63,2
4.05,-0.61,3
0.85,0.72,0
3.81,5.92,1
8.46,2.43,2
5.56,-2.22,3
1.19,3.08,0
4.46,6.19,1
7.06,3.23,2
4.74,-1.1,3
2.02,1.86,0
3.66,7.55,1
9.31,2.03,2
5.12,0.46,3
0.16,0.9,0
5.55,6.47,1
7.8,2.97,2
3.26,-1.0,3
0.66,0.52,0
4.54,5.44,1
7.58,3.1,2
3.46,-2.51,3
1.58,1.81,0
5.17,7.57,1
8.27,1.36,2
4.55,-1.21,3
0.35,1.46,0
4.58,4.76,1
9.61,2.32,2
5.8,-1.06,3
1.29,1.56,0
3.8,5.42,1
8.78,1.29,2
4.6,-0.14,3
1.82,0.44,0
3.9,6.53,1
7.96,3.13,2
4.2,-0.62,3
0.98,1.41,0
6.29,5.48,1
9.31,2.93,2
4.1,-0.94,3
0.84,0.06,0
4.41,5.33,1
8.79,3.51,2
4.58,-0.38,3
0.92,1.98,0
5.64,6.38,1
9.24,3.5,2
5.0,-1.76,3
1.19,1.24,0
5.72,5.54,1
8.73,2.47,2
4.52,-1.57,3
1.04,2.23,0
4.4,6.56,1
7.16,2.26,2
4.36,-0.34,3
0.95,2.12,0
6.2,6.39,1
8.12,2.28,2
5.89,-0.95,3
1.3,0.3,0
4.79,6.8,1
9.22,1.75,2
4.46,-0.25,3
1.49,1.32,0
4.07,5.99,1
9.79,0.8,2
4.72,-2.27,3
1.29,0.8,0
3.89,5.92,1
8.11,2.7,2
5.05,-1.11,3
0.28,0.4,0
4.49,5.25,1
9.36,2.33,2
4.62,-1.49,3
1.85,0.23,0
4.34,5.57,1
8.03,1.73,2
4.41,-1.01,3
1.41,0.43,0
5.12,5.84,1
8.12,2.13,2
3.76,-0.81,3
0.32,1.33,0
4.93,6.04,1
7.79,2.02,2
3.88,-1.26,3
0.72,2.35,0
4.87,6.85,1
8.64,2.82,2
3.67,-0.63,3
0.63,1.5,0
5.1,5.33,1
8.01,3.83,2
4.29,-1.0,3
3.13,1.55,0
6.16,5.89,1
7.6,2.88,2
3.98,-0.87,3
1.35,-1.04,0
4.84,6.44,1
8.31,2.57,2
4.6,-2.11,3
1.55,2.11,0
5.86,6.72,1
7.7,2.5,2
4.19,-0.33,3
0.56,2.04,0
3.09,6.27,1
8.32,3.71,2
5.6,-1.68,3
2.21,1.84,0
3.85,7.08,1
7.19,3.13,2
4.82,-1.77,3
1.74,2.53,0
4.88,5.45,1
8.03,1.3,2
3.95,-2.81,3
1.53,2.37,0
5.52,4.55,1
7.97,1.66,2
4.53,-0.93,3
2.19,0.91,0
5.17,5.63,1
8.08,2.72,2
5.32,-0.75,3
1.41,2.47,0
6.33,5.31,1
9.16,2.25,2
4.97,-0.34,3
2.34,1.26,0
6.1,5.49,1
8.62,1.05,2
5.35,-0.09,3
1.77,2.37,0
5.88,6.19,1
9.51,1.51,2
4.68,-0.99,3
1.3,1.85,0
3.96,4.95,1
9.42,2.13,2
4.24,-1.84,3
-0.25,1.85,0
5.04,6.24,1
7.36,2.37,2
4.36,-2.54,3
0.95,2.12,0
4.61,7.16,1
7.89,2.58,2
4.71,-1.98,3
1.6,1.68,0
4.45,4.86,1
8.75,4.27,2
4.46,-0.21,3
0.68,1.89,0
4.5,5.32,1
8.46,3.73,2
4.68,-2.03,3
0.43,1.23,0
7.1,6.16,1
8.92,2.65,2
4.31,-1.96,3
0.01,0.19,0
5.55,3.53,1
7.71,1.47,2
4.63,-1.4,3
1.2,-0.17,0
5.16,6.5,1
9.04,2.81,2
4.17,-0.5,3
1.75,1.09,0
4.55,7.6,1
8.81,2.94,2
3.08,-0.92,3
1.22,1.07,0
5.4,6.44,1
8.09,2.02,2
6.2,-0.72,3
0.69,1.63,0
4.01,6.83,1
7.3,0.73,2
3.29,-0.3,3
0.55,0.7,0
4.4,6.75,1
8.71,3.38,2
6.12,-1.8,3""")

print("Файл 'data_multivar_nb.txt' створено.")