import torch
import torch.nn as nn

# Example: Expecting input features of size 20, producing output features of size 30
linear_layer = nn.Linear(in_features=20, out_features=30)

# Create a sample input tensor (batch size 64, 20 features)
input_tensor = torch.randn(64, 20)

# Pass the input through the layer
output_tensor = linear_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

# Inspect the layer's parameters (automatically created)
print(f"\nWeight shape: {linear_layer.weight.shape}")
print(f"Bias shape: {linear_layer.bias.shape}")
