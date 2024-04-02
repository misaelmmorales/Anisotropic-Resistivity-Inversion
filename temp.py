import torch
import torch.nn as nn
import torch.optim as optim

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_static_features, num_time_features, 
                 hidden_dim=64, num_heads=8, num_encoder_layers=3, num_decoder_layers=3):
        super(TemporalFusionTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, num_static_features, num_time_features, 
                                          hidden_dim, num_heads, num_encoder_layers)
        self.decoder = TransformerDecoder(output_dim, hidden_dim, num_heads, num_decoder_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_data):
        encoder_output = self.encoder(input_data)
        decoder_output = self.decoder(encoder_output)
        output = self.linear(decoder_output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_static_features, num_time_features, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.static_embedding = nn.Linear(num_static_features, hidden_dim)
        self.time_embedding = nn.Linear(num_time_features, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, input_data):
        static_features = input_data[:, :, :num_static_features]
        time_features = input_data[:, :, num_static_features:num_static_features+num_time_features]
        input_embedding = self.static_embedding(static_features) + self.time_embedding(time_features)
        input_embedding = self.positional_encoding(input_embedding)
        encoder_output = self.transformer_encoder(input_embedding)
        return encoder_output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    
    def forward(self, encoder_output):
        decoder_output = self.transformer_decoder(encoder_output)
        return decoder_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

import numpy as np

# Assuming your data is stored in a variable named 'data'
data = np.random.rand(2500, 2)  # Example random data, replace with your actual data

# Reshape data into a suitable format for input into a temporal model
sequence_length = 2500  # Assuming you want to treat each depth point as a time step
num_features = 2  # Two columns representing the vertical and horizontal resistivity measurements

reshaped_data = data.reshape(sequence_length, num_features)

##################################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FourierNeuralOperator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3):
        super(FourierNeuralOperator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.linear_layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Apply Fourier transform
        x_freq = torch.fft.fft(torch.fft.fftshift(x), dim=-2)
        
        # Apply neural network operation in frequency domain
        for layer in self.linear_layers:
            x_freq = torch.relu(layer(x_freq))
        
        # Apply inverse Fourier transform
        x_out = torch.fft.ifft(torch.fft.ifftshift(x_freq), dim=-2)
        
        # Take real part as output
        output = x_out.real
        return output

# Example usage
input_dim = 2  # Assuming two input features (vertical and horizontal resistivity measurements)
output_dim = 2  # Assuming two output features (volumetric concentration of shale and sandstone resistivity)
fno_model = FourierNeuralOperator(input_dim, output_dim)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(fno_model.parameters(), lr=0.001)

# Example training loop (replace with your actual data)
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = fno_model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


##################################################################################################################
        
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pywt

class WaveletNeuralOperator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3, wavelet='db1'):
        super(WaveletNeuralOperator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.wavelet = wavelet
        
        self.wavelet_transform = nn.ModuleList([WaveletTransform(wavelet)])
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.linear_layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Apply wavelet transform
        x_wavelet = self.wavelet_transform(x)
        
        # Apply neural network operation
        for layer in self.linear_layers:
            x_wavelet = torch.relu(layer(x_wavelet))
        
        # Take the last output
        x_out = x_wavelet[-1]
        
        # Apply output layer
        output = self.output_layer(x_out)
        return output

class WaveletTransform(nn.Module):
    def __init__(self, wavelet):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        
    def forward(self, x):
        coeffs = pywt.wavedec(x, self.wavelet, mode='constant')
        return torch.tensor(np.concatenate(coeffs, axis=-1), dtype=torch.float32)

# Example usage
input_dim = 2  # Assuming two input features (vertical and horizontal resistivity measurements)
output_dim = 2  # Assuming two output features (volumetric concentration of shale and sandstone resistivity)
wno_model = WaveletNeuralOperator(input_dim, output_dim)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(wno_model.parameters(), lr=0.001)

# Example training loop (replace with your actual data)
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = wno_model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
