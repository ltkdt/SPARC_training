'''
Class sau mô phỏng cơ chế Automatic Gradient Calculation (tự động tính toán đạo hàm) trong deep learning.
Mỗi instance của class Scalar đại diện cho một giá trị số học (data) và gradient của nó (grad). 

Khi thực hiện các phép toán giữa các instance của Scalar, một DAG (Directed Acyclic Graph) được xây dựng để theo dõi các phép toán đã thực hiện và mối quan hệ giữa chúng.
Một đồ thị có hướng được gọi là một DAG khi và chỉ khi đồ thị tồn tại thứ tự tô pô.
'''


class Scalar:
    """ stores a single scalar Scalar and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        '''
        _backward là một hàm sẽ được gọi để tính toán gradient của các node trước đó trong graph khi backward() được gọi trên node cuối cùng.
        Ban đầu, _backward được gán là một hàm rỗng (lambda: None) vì khi khởi tạo một instance mới, nó chưa có thông tin về cách tính toán Gradient
        '''
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    
    def __add__(self, other):
        '''
        Phép cộng: 
        Y = a + b
        dY/da = 1
        dY/db = 1
        dL/da = dL/dY * dY/da = dL/dY * 1 = dL/dY
        dL/db = dL/dY * dY/db = dL/dY * 1 = dL/dY
        '''
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        '''
        Phép nhân:
        Y = a * b
        dY/da = b
        dY/db = a
        dL/da = dL/dY * dY/da = dL/dY * b
        dL/db = dL/dY * dY/db = dL/dY * a
        '''
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        '''
        Phép lũy thừa:
        Y = a^b
        dY/da = b * a^(b-1)
        dY/db = a^b * log(a)
        dL/da = dL/dY * dY/da = dL/dY * b * a^(b-1)
        dL/db = dL/dY * dY/db = dL/dY * a^b * log(a)

        Ở đây ta chỉ cần tính dL/da vì trong code chỉ hỗ trợ int/float powers thông qua lệnh assert, tức là b là hằng số, nên dY/db = 0
        '''
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        '''
        Phép relu:
        Y = relu(a) = max(0, a)
        dY/da = 1 if a > 0 else 0
        dL/da = dL/dY * dY/da = dL/dY * (1 if a > 0 else 0)
        '''
        out = Scalar(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        '''
        Build_topo: tạo một danh sách topo chứa tất cả các node trong graph theo thứ tự tô pô (topological order).
        Đầu vào của hàm là một node v. Nếu node v chưa được thăm (v not in visited), nó sẽ được thêm vào tập visited để đánh dấu là đã thăm. Sau đó, hàm sẽ đệ quy gọi build_topo cho tất cả các node con.
        Thuật toán duyệt đồ thị ở đây là DFS.
        Danh sách topo sẽ chứa các nút theo thứ tự từ Input lên đến Output. Ví dụ với biểu thức f=a+b, danh sách sẽ là [a, b, f]
        '''
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        '''
        Vì danh sách topo được xây dựng từ Input → Output, nên ta cần dùng reversed(topo) để đi từ Output → Input.
        Với mỗi nút v, ta gọi hàm v._backward()
        '''
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    '''
    __repr__ có tác dụng định nghĩa string của kết quả khi mà print ra, cái này thông dụng khi nói về class 
    '''
    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"