import numpy as np

target = [0,4,2,1,2]
input = [[0.1, 0.2, 0.3, 0.4, 0.5],
         [0.5, 0.4, 0.3, 0.2, 0.1],
         [0.2, 0.3, 0.4, 0.5, 0.6],
         [0.6, 0.5, 0.4, 0.3, 0.2],
         [0.3, 0.4, 0.5, 0.6, 0.7]]

def cross_entropy_sparse(y, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(np.log(y_pred[np.arange(len(y)), y]))

print(cross_entropy_sparse(target, input))
#1.3815510557964275