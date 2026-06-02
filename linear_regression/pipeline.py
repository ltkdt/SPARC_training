import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from model import LinearRegression, LinearRegressionWithMomentum

n_samples = 1000
n_features = 10


'''
Tạo dữ liệu cho bài toán hồi quy
1000 samples, 10 features, trong đó có 5 features có ảnh hưởng đến target (n_informative=5), và thêm một chút noise vào dữ liệu (noise=10).
random_state=42 đảm bảo rằng dữ liệu được tạo ra sẽ giống nhau mỗi khi chạy lại

noise: độ lệch chuẩn của Gauss noise được thêm vào output
'''

X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       n_informative=5, noise=10, random_state=42)

print(X.shape)  # (1000, 10) // 1000 samples, 10 features
print(y.shape)  # (1000,) // 1000 target values

'''
[n1, ..., n10] là [y_target ]
'''

'''
20% dữ liệu là tập validation
'''

X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=.2, random_state=1)


fig = plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], y, color="red", marker="o", s=5)
plt.savefig("test.png")

def mse(y, predictions):
    return np.mean(np.square(y - predictions))
'''
Model 1
'''
model = LinearRegression(lr=0.01, n_iters=1000, reg_lambda=0.1, penalty='l2')
model.fit(X_fit, y_fit)
train_predictions = model.predict(X_fit)


y_pred_train = model.predict(X_fit)

fig = plt.figure(figsize=(6,4))
plt.scatter(y_fit, y_pred_train, color="blue", s=5)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linewidth=2)
plt.xlabel("True values")
plt.ylabel("Predicted train values")
plt.savefig("train_predictions.png")

'''
Model 2: Linear Regression with Momentum

'''
model = LinearRegressionWithMomentum(lr=0.01, n_iters=1000, gamma=0.9, reg_lambda=0.1, penalty='l2')
model.fit(X_fit, y_fit)
y_pred_train_momentum = model.predict(X_fit)
fig = plt.figure(figsize=(6,4))
plt.scatter(y_fit, y_pred_train_momentum, color="green", s=5)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linewidth=2)
plt.xlabel("True values")
plt.ylabel("Predicted train values with Momentum")
plt.savefig("train_predictions_momentum.png")

y_pred_val = model.predict(X_val)
print("MSE on validation set:", mse(y_val, y_pred_val))