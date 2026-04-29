import numpy as np


def softmax(Z):
    """
    Hàm softmax chuyển đổi một vector Z thành một vector xác suất A, trong đó mỗi phần tử của A nằm trong khoảng (0, 1) và tổng của tất cả phần tử trong A bằng 1.   
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    """
    Phiên bản ổn định của hàm softmax để tránh overflow khi tính exp(Z)
    
    Trừ max(Z) để đảm bảo rằng giá trị lớn nhất trong Z trở thành 0, giúp tránh overflow khi tính exp(Z)  
    """
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A

def one_hot_encode(y, num_classes):
    """
    Hàm one-hot encoding chuyển đổi một vector y chứa các nhãn lớp thành một ma trận one-hot encoding Y_one_hot, trong đó mỗi hàng tương ứng với một mẫu và mỗi cột tương ứng với một lớp. 
    Nếu mẫu thuộc lớp k, thì phần tử tại hàng đó và cột k sẽ có giá trị 1, ngược lại sẽ có giá trị 0.
    """
    m = y.shape[0] # số lượng mẫu
    Y_one_hot = np.zeros((m, num_classes))
    '''
    np.arange(m) tạo ra một mảng chứa các chỉ số từ 0 đến m-1, tương ứng với chỉ số của mỗi mẫu trong y.
    '''
    Y_one_hot[np.arange(m), y] = 1
    '''
    Tạo phần tử từ các chỉ số tương ứng của 2 mảng
    VD: np.arrange(m) trả về [0,1] và y trả về [2,0]
    thì Y_one_hot[0,2] = 1 và Y_one_hot[1,0] = 1
    '''
    return Y_one_hot

def create_batch(X, y, batch_size):
    """
    Hàm tạo batch chia dữ liệu thành các batch nhỏ hơn để huấn luyện mô hình theo từng batch, giúp giảm thiểu bộ nhớ và tăng tốc độ huấn luyện.
    """
    n_samples = X.shape[0]
    '''
    np.random.permutation(n_samples) tạo ra một mảng chứa các chỉ số từ 0 đến n_samples-1 được xáo trộn ngẫu nhiên, giúp đảm bảo rằng dữ liệu được trộn lẫn trước khi chia thành các batch.
    X có shape (1000, 10), y có shape (1000,)
    thì permutation X trả về ví dụ [312, 45, 123, ...] và permutation y trả về [312, 45, 123, ...]
    '''
    
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        yield X_batch, y_batch


class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.001, n_iters=1000, reg_lambda = 0.1, penalty = None):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.penalty = penalty
        self.n_classes = n_classes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = self.n_classes

        # Ban đầu: weights được khởi tạo là 0, bias cũng là 0
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        
        y_one_hot = one_hot_encode(y, num_classes=n_classes)
        
        prev_loss = float('inf')
        tol = 1e-6
        
        for _ in range(self.n_iters):
            
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = softmax_stable(linear_output.T).T
            
            '''
            Y =  XW + b
            L = - (1/n) * Σ(y_one_hot * log(y_pred))
            dL/dW = (1/n) * X^T(y_pred - y_one_hot)
            dL/db = (1/n) * Σ(y_pred - y_one_hot)
            '''
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_one_hot))

            if self.penalty == 'l1':
                dw += self.reg_lambda * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += 2 * self.reg_lambda * self.weights

            db = (1/n_samples) * (np.sum(y_pred - y_one_hot, axis=0))

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
            current_loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-15), axis=1))
            
            if abs(current_loss - prev_loss) < tol:
                break
                
            prev_loss = current_loss
            
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = softmax_stable(linear_output.T).T
        return np.argmax(y_pred, axis=1)
        
class SoftmaxRegressionWithMiniBatchSGD:
    def __init__(self, batch_size, n_classes, lr=0.01, n_iters=1000, reg_lambda=0.1, penalty=None):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.penalty = penalty
        self.n_classes = n_classes
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = self.n_classes

        # Khởi tạo trọng số
        self.weights = np.zeros((n_features, n_classes)) # shape (10, 3)
        self.bias = np.zeros(n_classes)
        
        # One-hot encode toàn bộ y trước khi chia batch để tối ưu tốc độ
        y_one_hot_full = one_hot_encode(y, num_classes=n_classes)
        
        prev_loss = float('inf')
        tol = 1e-6
        
        for epoch in range(self.n_iters):
            # Duyệt qua từng Mini-batch, y_one_hot_full có shape (1000, 3)
            for X_batch, y_batch_oh in create_batch(X, y_one_hot_full, self.batch_size):
                m_batch = X_batch.shape[0] # Số mẫu trong batch hiện tại

                # Forward pass cho batch
                linear_output = np.dot(X_batch, self.weights) + self.bias
                # (32, 10) dot (10, 3) = (32, 3)
                y_pred_batch = softmax_stable(linear_output.T).T
                
                # Tính đạo hàm trên batch
                error = y_pred_batch - y_batch_oh
                dw = (1/m_batch) * np.dot(X_batch.T, error)
                db = (1/m_batch) * np.sum(error, axis=0)

                # Áp dụng Regularization
                if self.penalty == 'l1':
                    dw += self.reg_lambda * np.sign(self.weights)
                elif self.penalty == 'l2':
                    dw += 2 * self.reg_lambda * self.weights

                # Cập nhật trọng số sau MỖI batch
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Tính loss trên toàn bộ dữ liệu sau mỗi epoch để kiểm tra hội tụ
            full_linear = np.dot(X, self.weights) + self.bias
            full_pred = softmax_stable(full_linear.T).T
            current_loss = -np.mean(np.sum(y_one_hot_full * np.log(full_pred + 1e-15), axis=1))
            
            if abs(current_loss - prev_loss) < tol:
                print(f"Hội tụ tại epoch {epoch}")
                break
                
            prev_loss = current_loss
            
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = softmax_stable(linear_output.T).T
        return np.argmax(y_pred, axis=1)