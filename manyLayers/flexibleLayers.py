import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Input vectors (assuming they represent different sample points, not dimensions)
data = np.array([
    [95, 83, 75, 45, 86, 1],  # First sequence with indoor location
    [85, 83, 70, 45, 86, 0]   # Second sequence with outdoor location
])
# Resulting point spreads (assumed to be targets)
targets = np.array([5, -5])

#######Applying a regular Fouier transform on normalized values####
# Normalize features and targets
max_feature_value = 99.0
data_normalized = data / max_feature_value

# Assuming targets are bounded by [target_min, target_max]
target_min, target_max = -22.5, 52.5
# normalization process
targets_shifted = targets - target_min  # Shift to a [0, x] range
target_range = target_max - target_min
targets_normalized = targets_shifted / target_range  # Now in the range [0, 1]

# Calculate the magnitude of DFT coefficients for weight initialization
dft_of_normalized_targets = np.fft.fft(targets_normalized) # Compute the Discrete Fouier Transform
dft_magnitudes = np.abs(dft_of_normalized_targets)

# Normalizing targets and computing Discrete Fourier Transform (DFT)
# normalized_targets = np.fft.ifftshift(targets)  # Center zero frequency if needed

# Convert normalized data and targets to PyTorch tensors
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
targets_tensor = torch.tensor(targets_normalized, dtype=torch.float32).view(-1, 1)  # Reshape targets to match output shape [n_samples x 1]
dft_magnitudes_tensor = torch.tensor(dft_magnitudes, dtype=torch.float32)

# Complex tensors are supported in newer versions of PyTorch (e.g., PyTorch >= 1.8.0)
# dft_coeffs_tensor = torch.tensor(dft_of_targets, dtype=torch.complex64)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dft_weights):
        super(NeuralNet, self).__init__()
        
           # Initialize layers
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size)  # First layer
        ])

         # Initialize first layer weights using DFT coefficients
        with torch.no_grad():  # Disable gradient tracking for initialization
            self.layers[0].weight.data = dft_weights[:hidden_size].reshape(self.layers[0].weight.size())

       # Initialize additional hidden layers
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(self.layers[-1].out_features, hidden_size))
        
         # Initialize output layer
        self.layers.append(nn.Linear(self.layers[-1].out_features, 1))  # Output layer size to match the number of targets

    def forward(self, x):
        # Pass input through each layer except for the last one, with activation function
        for i in range(len(self.layers) - 1):
            x = F.leaky_relu(self.layers[i](x), negative_slope=0.01)  # Apply activation
            x = F.leaky_relu(self.layers[i](x))  # Apply activation
        # No activation for the final layer (regression problem)
        x = self.layers[-1](x)
        return x

# Instantiate the network
# Determine the size of the input layer (should match the number of features in the data)
input_size = data_tensor.shape[1]  # Assuming data is 2D with shape (num_samples, num_features)
hidden_size = 64  # Size for hidden layers
num_layers = 3    # Adjusted for simplicity, modify as needed
net = NeuralNet(input_size, hidden_size, num_layers, dft_magnitudes_tensor)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # MSE Loss for regression problems
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent, May need to adjust learning rate

# Training loop
max_epochs = 1000  # To be set based on the requirement
for epoch in range(max_epochs):
    optimizer.zero_grad()                # Clear previous gradients
    output = net(data_tensor)            # Obtain model predictions
    loss = criterion(output, targets_tensor)  # Compute loss Make sure dimensions match!!!!!!!!!!!!!!!!!!!!!!
    loss.backward()                      # Backpropagate the loss to compute gradients
    optimizer.step()                     # Update model parameters
    
   # Print loss and show denormalized predictions every 100 epochs
    if epoch % 100 == 0:
        # Denormalize the network's predictions
        predicted_targets_denorm = (output.detach().numpy() * target_range) + target_min
        
        # Denormalize the actual target values for comparison
        actual_targets_denorm = (targets_tensor.numpy() * target_range) + target_min
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Predicted Targets (Denormalized): {predicted_targets_denorm.flatten()}')
        print(f'Actual Targets (Denormalized): {actual_targets_denorm.flatten()}')

# Save the trained model to file
torch.save(net.state_dict(), 'trained_model_many_layer.pt')
# Training loop, optimization, and model saving as before...
