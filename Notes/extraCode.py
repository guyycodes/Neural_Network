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

##################Applying a regular Fouier transform and printing################
# Function to apply FFT and return the magnitude and phase of the frequency components
def apply_fft(data):
    # Apply Fast Fourier Transform
    fft_result = np.fft.fft(data)
    # Return the magnitude and phase of the frequency components
    return np.abs(fft_result), np.angle(fft_result)

# Apply Fourier Transform to each data vector
transformed_data = [apply_fft(vector) for vector in data]

# An example of extracting the first few magnitudes and phases of the frequency components
for index, (fft_magnitudes, fft_phases) in enumerate(transformed_data):
    print(f"Data Vector {index+1}:")
    print(f"Magnitudes: {fft_magnitudes[:3]}, Phases: {fft_phases[:3]}")
    print()
###################################################################