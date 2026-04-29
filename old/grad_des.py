import torch

# Example setup (imagine these are results from a model)
x = torch.tensor([2.0,1.0], requires_grad=True)
w = torch.tensor([3.0,4.0], requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Perform some operations (building the graph)
y = w * x + b  # y = [3*2 + 1, 4*1 + 1] = [7, 5]
loss = y * y   # loss giả làm ví dụ 
loss=loss.sum()  # Tổng loss để có một scalar

print(f"Loss: {loss.item()}") # Loss: 74.0 (7^2 + 5^2 = 49 + 25 = 74)
# Compute gradients
loss.backward()

# After backward pass, gradients are populated
print(f"Gradient for x after backward: {x.grad}") # d(loss)/dx = d(y^2)/dx = 2*y*(dy/dx) = 2*y*w = 2*[7,5]*[3,4] = [42,40]
print(f"Gradient for w after backward: {w.grad}") # d(loss)/dw = d(y^2)/dw = 2*y*(dy/dw) = 2*y*x = 2*[7,5]*[2,1] = [28,10]
print(f"Gradient for b after backward: {b.grad}") # d(loss)/db = Σd(y^2)/db = Σ2*y*(dy/db) = Σ2*y*1 = 2*(7+5) = 24
