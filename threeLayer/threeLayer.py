import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.fft import fft

# Input vectors (assuming they represent different sample points, not dimensions)
data = np.array([
    [95, 83, 75, 45, 86, 1],  # First sequence with indoor location
    [85, 83, 70, 45, 86, 0]   # Second sequence with outdoor location
])
# Resulting point spreads (assumed to be targets)
targets = np.array([5, -5])

# Normalize features
max_feature_value = 99  # Assuming this is the known maximum value in your features
data_normalized = data / max_feature_value

# Normalize targets to range [0, 1]
target_min, target_max = -22.5, 52.5  # Assuming these are the known min and max values of your targets
targets_normalized = (targets - target_min) / (target_max - target_min)

# Calculate the magnitude of DFT coefficients for weight initialization
dft_of_targets = fft(torch.tensor(targets_normalized, dtype=torch.float32))
dft_magnitudes = torch.abs(dft_of_targets)

# Convert normalized data and targets to PyTorch tensors
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
targets_tensor = torch.tensor(targets_normalized, dtype=torch.float32)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, dft_weights):
        super(NeuralNet, self).__init__()
        # First fully connected layer with DFT-initialized weights
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        with torch.no_grad():
            self.fc1.weight.data.copy_(dft_weights[:64].view_as(self.fc1.weight.data))  # Initialize weights with DFT
        
        # Additional fully connected layers
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)  # Assuming a single output for regression

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)  # Activation for first layer
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)  # Activation for second layer
        x = self.fc3(x)  # No activation, raw output for regression
        return x

# Determine the size of the input layer
input_size = data_tensor.shape[1]

# Instantiate the network
net = NeuralNet(input_size, dft_magnitudes)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # MSE for regression
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
max_epochs = 1000
for epoch in range(max_epochs):
    optimizer.zero_grad()                # Clear previous gradients
    output = net(data_tensor)            # Forward pass
    loss = criterion(output.squeeze(), targets_tensor)  # Compute loss, squeeze the output to match target dimensions
    loss.backward()                      # Backpropagate the loss
    optimizer.step()                     # Update weights
    
    # Print loss and show denormalized predictions every 100 epochs
    if epoch % 100 == 0:
        # Denormalize the network's predictions
        predicted_targets_denorm = (output.detach().numpy() * target_range) + target_min
        
        # Denormalize the actual target values for comparison
        actual_targets_denorm = (targets_tensor.numpy() * target_range) + target_min
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Predicted Targets (Denormalized): {predicted_targets_denorm.flatten()}')
        print(f'Actual Targets (Denormalized): {actual_targets_denorm.flatten()}')
# Save the trained model
torch.save(net.state_dict(), 'trained_model_3layer.pt')