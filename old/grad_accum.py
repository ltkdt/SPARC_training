import torch

# Create a tensor that requires gradients
x = torch.tensor([2.0], requires_grad=True)

# Perform some operations
y = x * x
z = y * 3 # z = 3 * x^2

# First backward pass
# dz/dx = 6*x = 6*2 = 12
z.backward(retain_graph=True) # retain_graph=True allows subsequent backward calls
print(f"After first backward pass, x.grad: {x.grad}")

z.backward() # Second backward pass
print(f"After second backward pass, x.grad: {x.grad}")

# Manually zero the gradient
x.grad.zero_()
print(f"After zeroing, x.grad: {x.grad}")