import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

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
reconstruction_losses = []

# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    running_loss = 0.0
    reconstruction_loss = 0.0
    
    # Train the model
    for batch_idx, (images, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = deep_autoencoder(images)
        
        # Resize the reconstructed outputs to match the size of the input images
        outputs_resized = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = criterion(outputs_resized, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        reconstruction_loss += loss.item()

        # Compute the spectrogram of the latent space for this batch
        with torch.no_grad():
            latent_space = deep_autoencoder.conv3(deep_autoencoder.conv2(deep_autoencoder.conv1(images)))
            # Reshape the latent space to (batch_size, channels, height, width)
            latent_space = latent_space.permute(0, 2, 3, 1)
            # Compute spectrogram
            freqs, times, Sxx = spectrogram(latent_space.numpy(), axis=-1)
            batch_spectrogram = np.mean(Sxx, axis=0)  # Average over batch
            spectrograms.append(batch_spectrogram)
    
    # Compute the average reconstruction loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    #avg_reconstruction_loss = reconstruction_loss / len(train_loader)
    print(f"Epoch Loss: {epoch_loss:.4f}")
    #reconstruction_losses.append(avg_reconstruction_loss)



# Save spectrograms
np.save('spectrograms.npy', np.array(spectrograms))

# Load the spectrogram data from the npy file
spectrogram_data = np.load('spectrograms.npy')

# Convert the numpy array to a PyTorch tensor
spectrogram_tensor = torch.tensor(spectrogram_data)





# Assuming mean_image is your PyTorch tensor
mean_image = torch.mean(spectrogram_tensor, dim=-1)  # Taking mean along the last dimension

# Reshape mean_image to a 2D array
mean_image_flat = mean_image.reshape(mean_image.shape[0], -1)

# Plot the mean spectrogram
plt.figure(figsize=(8, 6))
plt.imshow(mean_image_flat, aspect='auto', cmap='viridis', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Mean Spectrogram')
plt.colorbar(label='Power')
plt.show()
