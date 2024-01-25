import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Input vectors
data = np.array([
    [95, 83, 75, 45, 86, 1],  # First sequence with indoor location
    [85, 83, 70, 45, 86, 0]   # Second sequence with outdoor location
])

# Resulting point spreads (assumed to be targets)
targets = np.array([5, -5])

# The targets are normalized within a range determined by assumed minimum and maximum possible values, 
# so the output of the network will also need to be denormalized if you want to interpret it in the original scale of the targets.
# Normalize data
data_normalized = data / np.max(data) 

# Normalize targets
target_min, target_max = -10, 10  # Assume these are known min/max values for targets
targets_shifted = targets - target_min
target_range = target_max - target_min
targets_normalized = targets_shifted / target_range

# Compute the DFT of the normalized targets using NumPy
dft_of_targets_normalized = np.fft.fft(targets_normalized)
dft_magnitudes = np.abs(dft_of_targets_normalized)

# Convert normalized data and DFT magnitudes to PyTorch tensors
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
targets_tensor = torch.tensor(targets_normalized, dtype=torch.float32).view(-1, 1)

# Define the neural network architecture with LeakyReLU activation
class NeuralNet(nn.Module): # The input_size is determined by the number of features in your data.
    def __init__(self, input_size, dft_weights):    # dft_weights is a tensor that uses the magnitude of the DFT coefficients.
        super(NeuralNet, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features=input_size, out_features=1)    #The fc1 layer's weight is being reshaped to match the shape of the layer's weight (which should be [1, input_size] because we have one neuron in the output layer).
        # Initialize weights using the absolute values of DFT coefficients
        with torch.no_grad():
            self.fc1.weight.data = dft_weights[:input_size].view(-1, input_size)

    def forward(self, x):
        # Pass data through the first fully connected layer
        x = self.fc1(x)
        # Apply the LeakyReLU activation function
        x = F.leaky_relu(x, negative_slope=0.01)
        return x

# Instantiate the network
input_size = data_tensor.shape[1]
dft_weights = torch.tensor(dft_magnitudes, dtype=torch.float32)
net = NeuralNet(input_size, dft_weights)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
max_epochs = 1000
for epoch in range(max_epochs):
    optimizer.zero_grad()   # Zero the gradient buffers
    output = net(data_tensor)      # Forward pass
    loss = criterion(output, targets_tensor)  # Compute loss
    loss.backward()         # Backward pass
    optimizer.step()        # Update weights

    
    # Print loss and show denormalized predictions every 100 epochs
    if epoch % 100 == 0:
        # Denormalize the network's predictions
        predicted_targets_denorm = (output.detach().numpy() * target_range) + target_min
        
        # Denormalize the actual target values for comparison
        actual_targets_denorm = (targets_tensor.numpy() * target_range) + target_min
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Predicted Targets (Denormalized): {predicted_targets_denorm.flatten()}')
        print(f'Actual Targets (Denormalized): {actual_targets_denorm.flatten()}')

torch.save(net.state_dict(), 'trained_model_1layer.pt')
# After training, you can use 'net' to make predictions on new data.

#In this code:

# A NumPy array for input data and targets is first defined.
# The DFT of the targets is computed using NumPy's FFT functions.
# The data and DFT of the targets are converted to PyTorch tensors.
# We define a NeuralNet class where the weights of the fc1 layer are initialized with the absolute values of the DFT coefficients. LeakyReLU activation is applied in the forward pass.
# A training loop is implemented where the model is trained with the SGD optimizer and MSELoss criterion.
# Bear in mind that the shape of weights passed to the NeuralNet must match the expected number of features (6 in your input data). It's important to ensure that the output shape of the network also matches the expected shape of the targets for the loss calculation. If your design requires a different approach in terms of the dimensions or the number of layers, you'll need to adjust the code accordingly.
# By employing a training loop with backpropagation, each iteration of the loop updates the weights in the direction that reduces the loss between the predicted and target values, eventually converging to a solution. This workflow is standard in training neural networks and can be highly effective, avoiding the need for hand-calculating complex mathematical relations as in approximation (5).
        
# Can we just use the coefficients and perform forward passes and backprop until convergence and avoid the complicted calculations?
# Yes, that's a very common practice in training neural networks. We can initialize the weights with the coefficients obtained from the DFT of the target data. Then, we can optimize the weights using the forward pass and backpropagation during the training process, allowing the neural network to converge to a solution that minimally deviates from the expected output. This way, we avoid the direct manual calculation of the weights as discussed previously with the approximation (5) and rely on numerical optimization techniques to refine the weights.

# Here's a high-level outline of how you might approach this:

# Initialize Weights: Use the DFT coefficients directly as initial weights or as a base for initial weights.

# Forward Pass: Use standard neural network infrastructure to calculate predictions based on current weights and inputs.

# Loss Calculation: Use a loss function to quantify the difference between the network predictions and target outputs.

# Backpropagation: Utilize an optimization algorithm to perform backpropagation, automatically computing gradients and updating weights to minimize the loss.

# Iterate: Repeat the forward pass and backpropagation process for many epochs or until the network's predictions are satisfactorily close to the target values.

# This process can be done using libraries like TensorFlow or PyTorch, which provide auto-differentiation and a wealth of optimization algorithms out-of-the-box. Below is a simplified example of how this could be implemented using PyTorch:

# python
# Copy code