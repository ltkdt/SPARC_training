import numpy as np


class ConvolutionLayer:
    def __init__(self, kernel_num, kernel_size):
        '''
        ConvolutionLayer sẽ khởi tạo các kernel ngẫu nhiên và số lượng kernel là kernum num. Kích cỡ vuông kernel_size x kernel_size (VD: kernel 3x3)
        '''
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        '''
        Đầu tiên, random.randn sẽ tạo ra một mảng có kích thước (kernel_num, kernel_size, kernel_size) với các giá trị ngẫu nhiên được lấy từ phân phối chuẩn. Với size = 3
        Hình dạng:
        [
            [ # Kernel đầu tiên
                [k11, k12, k13],
                [k21, k22, k23],
                [k31, k32, k33]
            ],
            [ # Kernel thứ hai
                [k11, k12, k13],
                [k21, k22, k23],
                [k31, k32, k33]
            ],
            ...
        ]
        Sau đó chia cho (kernel_size**2) (= 9) để chuẩn hóa các giá trị của kernel, giúp quá trình huấn luyện ổn định hơn. Việc chuẩn hóa này giúp tránh tình trạng giá trị của kernel quá lớn, dẫn đến việc mất mát trong quá trình lan truyền ngược (backpropagation) và giúp mô hình học hiệu quả hơn.
        '''

        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size**2)

    def patches_generator(self, image):
        """
        Chiều cao của ảnh là số hàng.
        Chiều rộng của ảnh là số cột.
        """
        image_h, image_w = image.shape
        self.image = image
        '''
        Các patch là các vùng nhỏ của ảnh mà kernel sẽ quét qua để thực hiện phép tích chập (convolution). Cụ thể, nếu ảnh có kích thước (h, w) và kernel có kích thước (f, f), thì số lượng patch sẽ là (h-f+1) x (w-f+1). Mỗi patch sẽ được sử dụng để tính toán giá trị đầu ra tại vị trí tương ứng trong output volume của lớp convolution.
        Tưởng tượng ảnh là 4x4 và kernel là 3x3. Coi như chiều dọc hay ngang có các phần từ 0,1,2,3 thì các theo chiều đó patch quét được [0,1,2] và [1,2,3], tương ứng với 4-3+1 = 2 patch theo mỗi chiều. Tổng cộng sẽ có 2x2 = 4 patch được tạo ra từ ảnh 4x4 với kernel 3x3.
        '''
        for h in range(image_h-self.kernel_size+1): 
            for w in range(image_w-self.kernel_size+1):
                patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
                yield patch, h, w
    
    def forward_prop(self, image):
        image_h, image_w = image.shape
        '''
        Ban đầu convo output được khởi tạo với tất cả giá trị là 0
        Shape của convolution_output là (image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num) vì sau khi quét kernel qua ảnh, kích thước của output sẽ giảm đi (kernel_size-1) ở mỗi chiều. Cụ thể, nếu ảnh có kích thước (h, w) và kernel có kích thước (f, f), thì kích thước của output sẽ là (h-f+1, w-f+1).
        Có bao nhiêu kernel num thì sẽ có bấy nhiêu kênh (channel) trong output, do đó chiều thứ ba của convolution_output là self.kernel_num.
        Lí do shape không phải là (kernel_num, image_h-self.kernel_size+1, image_w-self.kernel_size+1) là vì chúng ta muốn giữ nguyên thứ tự chiều của ảnh (height, width) trước khi đến chiều số lượng kernel. Điều này giúp việc xử lý và hiểu output dễ dàng hơn, đặc biệt khi kết hợp với các lớp khác như max pooling hoặc softmax sau này.
        
        Cho 1 kernel:
        [k11, k12, k13],
        [k21, k22, k23],
        [k31, k32, k33]

        và 1 ảnh 4x4
        [p11, p12, p13, p14],
        [p21, p22, p23, p24],
        [p31, p32, p33, p34],
        [p41, p42, p43, p44]

        Có thể nhận thấy các vị trí h, w  trả về là vị trí theo patch đầu tiên (h=0, w=0), (h=0, w=1), (h=1, w=0), (h=1, w=1) 
            Tại vị trí (h=0, w=0), patch sẽ là:
            [p11, p12, p13],
            [p21, p22, p23],
            [p31, p32, p33]

        Phép tích chập cho output ở  vị trí (h=0, w=0) sẽ là:
        output[0,0] = k11*p11 + k12*p12 + k13*p13 + k21*p21 + k22*p22 + k23*p23 + k31*p31 + k32*p32 + k33*p33
    
        '''
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        for patch, h, w in self.patches_generator(image):
            # Perform convolution for each patch
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        self.z = convolution_output
        return np.maximum(0, self.z) # ReLU activation
    
    def back_prop(self, dE_dY, alpha):
        """
        Ta update từng phần tử của kernel và gradient cũng có shape của kernel 
        dE_dY là gradient trả về từ lớp sau đó. (VD: conv -> max pooling layer thì dE_dY là gradient trả về từ max pooling layer)


        dE_dK = dE_dY * dY_dK
        Do dY_dK = patch vì ta thực hiện phép nhân patch với kernel 
        """

        dE_dY = dE_dY * (self.z > 0)

        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update kernels 
        self.kernels -= alpha*dE_dk
        return dE_dk

class MaxPoolingLayer:
    '''
    Đầu vào của lớp MaxPool là một volume (h, w, num_kernels) và đầu ra là một volume có kích thước nhỏ hơn (h//kernel_size, w//kernel_size, num_kernels). 
    Lớp MaxPooling sẽ chia ảnh thành các patch nhỏ có kích thước kernel_size x kernel_size và 
    lấy giá trị lớn nhất trong mỗi patch để tạo thành output mới. 
    Điều này giúp giảm kích thước của dữ liệu và giữ lại những đặc trưng quan trọng nhất.
    '''
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        '''
        Lấy ví dụ ảnh có kích thước 4x4 và kernel_size = 2, thì output sẽ có kích thước 2x2
        Nếu ảnh 5x5, kernel_size = 3 thì output sẽ có kích thước 1x1 vì chỉ có một patch 3x3 có thể được tạo ra từ ảnh 5x5. (nhớ là patch ko có overlap như convo)        '''

        output_h = image.shape[0] // self.kernel_size
        output_w = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(output_h):
            for w in range(output_w):
                patch = image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size), (w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                yield patch, h, w

    def forward_prop(self, image):
        image_h, image_w, num_kernels = image.shape
        max_pooling_output = np.zeros((image_h//self.kernel_size, image_w//self.kernel_size, num_kernels))
        for patch, h, w in self.patches_generator(image):
            '''
            Lấy theo trục 0,1 vì patch có shape (kernel_size, kernel_size, num_kernels) nên trục 0,1 là trục của chiều cao và chiều rộng của patch. Việc lấy max theo trục này sẽ trả về một mảng có shape (num_kernels,) chứa giá trị lớn nhất cho mỗi kernel trong patch đó.
            '''
            max_pooling_output[h,w] = np.amax(patch, axis=(0,1))

        return max_pooling_output

    def back_prop(self, dE_dY):
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically softmax.
        There are no weights to update, but the output is needed to update the weights of the convolutional layer.
        """
        dE_dk = np.zeros(self.image.shape)
        for patch,h,w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0,1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        '''
                        Vị trí max thì mới có thay đổi 
                        '''
                        if patch[idx_h,idx_w,idx_k] == max_val[idx_k]:
                            dE_dk[h*self.kernel_size+idx_h, w*self.kernel_size+idx_w, idx_k] = dE_dY[h,w,idx_k]
        return dE_dk

class SoftmaxLayer:
    """
    Input: Một volume có kích thước (h, w, num_kernels) từ lớp MaxPooling.
    Output: Một vector có kích thước (num_classes,) chứa xác suất dự đoán cho mỗi lớp. Trong trường hợp này, num_classes = 10 cho bài toán phân loại chữ
    Input sẽ flatten về 1 chiều và hoạt động tương tự MLP
    """

    def __init__(self, input_units, output_units):
        # Initiallize weights and biases
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward_prop(self, image):
        self.original_shape = image.shape # stored for backprop
        '''
        Flatten the input image
        '''
        image_flattened = image.flatten()
        self.flattened_input = image_flattened # stored for backprop
        # Perform matrix multiplication and add bias
        first_output = np.dot(image_flattened, self.weight) + self.bias # Y = XW + b
        self.output = first_output
        # Apply softmax activation
        softmax_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)
        return softmax_output

    def back_prop(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue
            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            # Compute gradients with respect to output (Z)
            dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
            dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

            # Compute gradients of output Z with respect to weight, bias, input
            dZ_dw = self.flattened_input
            dZ_db = 1
            dZ_dX = self.weight

            # Gradient of loss with respect ot output
            dE_dZ = gradient * dY_dZ

            # Gradient of loss with respect to weight, bias, input
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            # Update parameters
            self.weight -= alpha*dE_dw
            self.bias -= alpha*dE_db

            return dE_dX.reshape(self.original_shape)

def CNN_forward(image, label, layers):
    output = image/255.
    for layer in layers:
        output = layer.forward_prop(output)
    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label])
    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy

def CNN_backprop(gradient, layers, alpha=0.05):
    grad_back = gradient
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, SoftmaxLayer]:
            grad_back = layer.back_prop(grad_back, alpha)
        elif type(layer) == MaxPoolingLayer:
            grad_back = layer.back_prop(grad_back)
    return grad_back


def CNN_training(image, label, layers, alpha=0.05):
    # Forward step
    output, loss, accuracy = CNN_forward(image, label, layers)

    # Initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/output[label]

    # Backprop step
    gradient_back = CNN_backprop(gradient, layers, alpha)

    return loss, accuracy




"""
import pandas as pd
import matplotlib.pyplot as plt
#Test the convolutions with 1 image, to put in the article
# Test
df_train = pd.read_csv('train.csv')
img = df_train.iloc[40,:].values[1:]
img = np.reshape(img,(28,28))
plt.imshow(img, cmap='gray')
plt.show()
print(img.shape)
plt.savefig('images/original_image.png', format='png', dpi=1200)

# Test with a convolution of 16 filters of size 3x3
my_conv = ConvolutionLayer(32,3)
output = my_conv.forward_prop(img)
# See the dimensions of the output volume, they follow the usual formula
print(output.shape)

# Plot 16th volume after the convolution
plt.imshow(output[:,:,15], cmap='gray')
plt.show()
plt.savefig('images/image_convolved.png', format='png', dpi=1200)
"""