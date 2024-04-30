import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the DeepConvAutoencoder class
class DeepConvAutoencoder(nn.Module):
    def __init__(self):
        super(DeepConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)  # Adjusted stride to properly upsample
        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))  # output layer with sigmoid activation
                
        return x

# Load MNIST dataset and create data loaders
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])  # Resize images to match autoencoder input size

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the deep autoencoder model
deep_autoencoder = DeepConvAutoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(deep_autoencoder.parameters(), lr=0.001)

# Number of epochs
n_epochs = 10

# List to store spectrograms at different epochs
spectrograms = []

# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    running_loss = 0.0
    
    # Train the model
    for images, _ in train_loader:
        optimizer.zero_grad()
        outputs = deep_autoencoder(images)
        
        # Resize the reconstructed outputs to match the size of the input images
        outputs_resized = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = criterion(outputs_resized, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    # Compute the spectrogram of the latent space batch by batch
    with torch.no_grad():
        batch_spectrograms = []
        for images, _ in train_loader:
            latent_space = deep_autoencoder.conv3(deep_autoencoder.conv2(deep_autoencoder.conv1(images)))
            spectrogram = np.abs(np.fft.fftn(latent_space.numpy(), axes=(-2, -1)))**2
            batch_spectrogram = np.mean(spectrogram, axis=0)
            batch_spectrograms.append(batch_spectrogram)
        spectrogram = np.mean(batch_spectrograms, axis=0)
        spectrograms.append(spectrogram)
    
    # Print the average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch Loss: {epoch_loss:.4f}")

# Plot spectrograms at different epochs
plt.figure(figsize=(12, 8))
for i, spectrogram in enumerate(spectrograms):
    plt.subplot(3, 4, i+1)
    plt.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower')
    plt.xlabel('Frequency')
    plt.ylabel('Image Index')
    plt.title(f'Epoch {i+1}')
    plt.colorbar(label='Power')
plt.tight_layout()
plt.show()
