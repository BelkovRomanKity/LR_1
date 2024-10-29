import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'actual_label': np.random.randint(0, 2, size=100),
    'predicted_RF': np.random.randint(0, 2, size=100),
    'predicted_LR': np.random.randint(0, 2, size=100),
    'model_RF': np.random.rand(100),
    'model_LR': np.random.rand(100)
})

def belkov_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def belkov_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def belkov_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def belkov_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = belkov_find_TP(y_true, y_pred)
    FN = belkov_find_FN(y_true, y_pred)
    FP = belkov_find_FP(y_true, y_pred)
    TN = belkov_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def belkov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def belkov_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def belkov_recall_score(y_true, y_pred):
    TP, FN, _, _ = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def belkov_precision_score(y_true, y_pred):
    TP, _, FP, _ = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def belkov_f1_score(y_true, y_pred):
    precision = belkov_precision_score(y_true, y_pred)
    recall = belkov_recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Виведення результатів для порогу 0.5
print("Results with threshold = 0.5")
print("Accuracy RF: %.3f" % (belkov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print("Recall RF: %.3f" % (belkov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print("Precision RF: %.3f" % (belkov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print("F1 RF: %.3f" % (belkov_f1_score(df.actual_label.values, df.predicted_RF.values)))

print("\nResults with threshold = 0.25")
# Зміна порогу на 0.25 для класифікації
predicted_RF_025 = (df.model_RF >= 0.25).astype(int).values
print("Accuracy RF: %.3f" % (belkov_accuracy_score(df.actual_label.values, predicted_RF_025)))
print("Recall RF: %.3f" % (belkov_recall_score(df.actual_label.values, predicted_RF_025)))
print("Precision RF: %.3f" % (belkov_precision_score(df.actual_label.values, predicted_RF_025)))
print("F1 RF: %.3f" % (belkov_f1_score(df.actual_label.values, predicted_RF_025)))

print('Confusion Matrix for RF:')
print(belkov_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(belkov_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                       confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'belkov_confusion_matrix() is not correct for RF'

assert np.array_equal(belkov_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                       confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'belkov_confusion_matrix() is not correct for LR'

fpr_RF, tpr_RF, _ = roc_curve(df.actual_label, df.model_RF)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label, df.model_LR)

auc_RF = roc_auc_score(df.actual_label, df.model_RF)
auc_LR = roc_auc_score(df.actual_label, df.model_LR)

print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF (AUC = {auc_RF:.2f})')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR (AUC = {auc_LR:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='Perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for RF and LR Models')
plt.show()

if auc_RF > auc_LR:
    print("Модель RF має вищу AUC, що свідчить про кращу продуктивність порівняно з моделлю LR.")
else:
    print("Модель LR має вищу AUC, що свідчить про кращу продуктивність порівняно з моделлю RF.")
