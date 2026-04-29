import numpy as np
from sympy import gamma

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, reg_lambda = 1, penalty = None):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.penalty = penalty
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ban đầu: weights được khởi tạo là 0, bias cũng là 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        tol = 1e-5
        prev_loss = 0
        for i in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            # Tính đạo hàm của loss function đối với weights và bias
            '''
            Y =  Xw+b
            L = (1/n) * Σ(Y - y)^2
            dL/dw = (1/n) * X^T(Y - y)
            dL/db = (1/n) * Σ(Y - y)
            '''
            
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)))
            # add regularization derivative term

            '''
            Regularization để ngăn overfit

            L1: L = (1/n) * Σ(Y - y)^2 + λ * Σ|w|
            dL/dw = (1/n) * X^T(Y - y) + λ * sign(w)
            sign(w) = 1 if w > 0, -1 if w < 0, 0 if w = 0, giúp tính đạo hàm trị tuyệt đối

            L1 khuyến khích sparsity (nhiều weights trở về 0), trong khi L2 khuyến khích weights nhỏ nhưng không nhất thiết phải bằng 0

            L2: L = (1/n) * Σ(Y - y)^2 + λ * Σw^2
            dL/dw = (1/n) * X^T(Y - y) + 2 * λ * w

            '''
            if self.penalty == 'l1':
                dw +=  self.reg_lambda * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += 2 * self.reg_lambda * self.weights

            db = (1/n_samples) * (np.sum(y_pred - y))

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
            current_loss = np.mean(np.square(y_pred - y))
            
            if abs(current_loss - prev_loss) < tol:
                break
                
            prev_loss = current_loss
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
class LinearRegressionWithMomentum:
    def __init__(self, lr=0.001, n_iters=1000, gamma=0.9, reg_lambda = 1, penalty = None):
        self.lr = lr
        self.n_iters = n_iters
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.penalty = penalty

    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ban đầu: weights được khởi tạo là 0, bias cũng là 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        
        ''' 
        Momentum: cập nhật weights dựa trên cả gradient hiện tại và gradient trước đó, giúp tăng tốc độ hội tụ và tránh bị kẹt ở local minima
        v: vận tốc (weight và bias là vị trí ) 
        gamma: hệ số momentum (thường là 0.9), xác định mức độ ảnh hưởng của gradient trước đó
        '''
        v_weights = np.zeros(n_features)
        v_bias = 0


        tol = 1e-5
        prev_loss = 0
        for i in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            # Tính đạo hàm của loss function đối với weights và bias
            '''
            Y =  Xw+b
            L = (1/n) * Σ(Y - y)^2
            dL/dw = (1/n) * X^T(Y - y)
            dL/db = (1/n) * Σ(Y - y)
            '''
            
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)))
            # add regularization derivative term

            if self.penalty == 'l1':
                dw += self.reg_lambda * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += 2 * self.reg_lambda * self.weights

            db = (1/n_samples) * (np.sum(y_pred - y))

            # 3. Cập nhật vận tốc (tích lũy gradient qua thời gian)
            v_weights = self.gamma * v_weights + self.lr * dw
            v_bias = self.gamma * v_bias + self.lr * db

            self.weights = self.weights - v_weights
            self.bias = self.bias - v_bias
            
            current_loss = np.mean(np.square(y_pred - y))
            
            if abs(current_loss - prev_loss) < tol:
                break
                
            prev_loss = current_loss
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
# https://viblo.asia/p/optimizer-hieu-sau-ve-cac-thuat-toan-toi-uu-gdsgdadam-Qbq5QQ9E5D8
# https://towardsdatascience.com/understanding-deep-learning-optimizers-momentum-adagrad-rmsprop-adam-e311e377e9c2/