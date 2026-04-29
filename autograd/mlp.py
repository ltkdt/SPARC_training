import random
from autograd import Scalar

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Scalar(random.uniform(-1,1)) for _ in range(nin)]  ## nin là số input
        self.b = Scalar(0)
        self.nonlin = nonlin # nonlin = True thì sẽ có activation function là ReLU, còn nonlin = False thì sẽ là Linear
    '''
    hàm __call__ cho phép sử dụng một object giống như một hàm. 
    VD : n = Neuron(3) thì n(x) sẽ gọi hàm __call__ của class Neuron với x là arguement.
    '''
    def __call__(self, x):
        '''
        z = w1*x1 + w2*x2 + ... + wn*xn + b
        '''

        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    '''
    nin: số input của layer (bình thường là số neuron của layer trước)
    nout: số neuron của layer hiện tại
    '''
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    '''
    Kích hoạt tất cả các neuron trong layer với cùng một input x.
    Nếu layer chỉ có một neuron, trả về output của neuron đó. Nếu có nhiều neuron, trả về một list chứa output của tất cả các neuron.
    '''
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    '''
    nin: số features 
    nouts: danh sách số neuron cho từng layer
    nouts = [4, 4, 1] nghĩa là mạng có 3 tầng: 2 tầng ẩn (mỗi tầng 4 neuron) và 1 tầng đầu ra (1 neuron).
    '''
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"