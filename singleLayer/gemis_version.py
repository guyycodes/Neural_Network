import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import fft

# Input vectors (assuming they represent different sample points, not dimensions)
data = np.array([
    [95, 83, 75, 45, 86, 1],  # First sequence with indoor location
    [85, 83, 70, 45, 86, 0]   # Second sequence with outdoor location
])

# Resulting point spreads (assumed to be targets)
targets = np.array([5, -5])

# Applying a regular Fourier transform on normalized values
# Step 1: Compute the DFT using ym values. (Discrete Fouier Transform) - Turns the vaues into frequencies
normalized_targets = np.fft.ifftshift(targets)  # Normalizing targets
dft_of_targets = np.fft.fft(normalized_targets)  # DFT of normalized targets

# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)

# Convert DFT coefficients to PyTorch tensors (float complex)
dft_coeffs = torch.tensor(dft_of_targets, dtype=torch.complex64)

# Define the neural network architecture with LeakyReLU activation
class NeuralNet(nn.Module):
    def __init__(self, weights):
        super(NeuralNet, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features=weights.shape[0], out_features=weights.shape[0])
        # Initialize weights using the absolute values of DFT coefficients
        self.fc1.weight.data = torch.abs(weights)

    def forward(self, x):
        # Pass data through the first fully connected layer
        x = self.fc1(x)
        # Apply the LeakyReLU activation function
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        return x

# Instantiate the network
net = NeuralNet(dft_coeffs)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Assume max_epochs is defined (e.g., max_epochs = 1000)
max_epochs = 1000

# Training loop
for epoch in range(max_epochs):
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save the trained model
torch.save(net.state_dict(), 'trained_model.pt') 

