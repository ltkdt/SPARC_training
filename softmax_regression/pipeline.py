import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


from model import SoftmaxRegression, SoftmaxRegressionWithMiniBatchSGD, one_hot_encode, softmax_stable

'''
Tạo dữ liệu cho bài toán hồi quy
1000 samples, 10 features, trong đó có 5 features có ảnh hưởng đến target (n_informative=5), và thêm một chút noise vào dữ liệu (noise=10).
random_state=42 đảm bảo rằng dữ liệu được tạo ra sẽ giống nhau mỗi khi chạy lại

noise: độ lệch chuẩn của Gauss noise được thêm vào output


'''

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=3, random_state=42)

print(X.shape)  # (1000, 10) // 1000 samples, 10 features
print(y.shape)  # (1000,) // 1000 target values

y_one_hot = one_hot_encode(y, num_classes=3)

# for i in range(10):
#     print(f"Sample {i+1}: Features: {X[i]}, Target: {y[i]}, One-Hot: {y_one_hot[i]}")


X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=.2, random_state=1)

X_fit = scaler.fit_transform(X_fit)
X_val = scaler.transform(X_val)

model = SoftmaxRegressionWithMiniBatchSGD(batch_size=32, n_classes=3, lr=0.01, n_iters=1000, reg_lambda=0.1, penalty='l2')
model.fit(X_fit, y_fit)
y_pred_train = model.predict(X_fit)

'''
TP: True Positives (số lượng mẫu được dự đoán đúng là lớp dương)
pd.crosstab: tạo bảng chéo giữa y_fit (giá trị thực tế) và y_pred_train (giá trị dự đoán) để tính toán số lượng TP, TN, FP, FN
rowname: tên của hàng (Actual), colnames: tên của cột (Predicted
conf_matrix: ma trận nhầm lẫn dưới dạng numpy array, kích cỡ là (n_classes, n_classes)
classes: danh sách các lớp (0, 1, 2)
'''

TP = pd.crosstab(y_fit, y_pred_train, rownames=['Actual'], colnames=['Predicted'])
'''
Nếu tại vị trí hàng 1, cột 2 có giá trị là 5, điều đó có nghĩa là có 5 mẫu thực tế thuộc lớp 1 nhưng mô hình lại dự đoán nhầm sang lớp 2
'''
conf_matrix = TP.values
classes = TP.index.tolist()

tp = np.diag(conf_matrix)
fp = conf_matrix.sum(axis=0) - tp
fn = conf_matrix.sum(axis=1) - tp
tn = conf_matrix.sum() - (tp + fp + fn)

# print("Confusion Matrix:")
# print(TP)

print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

with np.errstate(divide='ignore', invalid='ignore'):
    Recall = tp / (tp + fn)
    Precision = tp / (tp + fp)
    Recall = np.nan_to_num(Recall)
    Precision = np.nan_to_num(Precision)

F1_Score = np.zeros_like(Precision, dtype=float)
den = (Precision + Recall)
nonzero = den != 0
F1_Score[nonzero] = 2 * (Precision[nonzero] * Recall[nonzero]) / den[nonzero]

print(f"Recall: {np.round(100 * Recall, 2)}%")
print(f"Precision: {np.round(100 * Precision, 2)}%")
print(f"F1 Score: {np.round(100 * F1_Score, 2)}%")

# Save confusion matrix and per-class metrics to CSV
TP.to_csv('confusion_matrix.csv')
metrics_df = pd.DataFrame({
    'class': classes,
    'TP': tp,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'Recall': Recall,
    'Precision': Precision,
    'F1_Score': F1_Score
})
metrics_df.to_csv('classification_report.csv', index=False)
print('Saved confusion matrix to confusion_matrix.csv')
print('Saved classification report to classification_report.csv')