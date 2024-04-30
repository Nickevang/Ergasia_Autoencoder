
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define your DeepConvAutoencoder class here

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
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

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
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize your deep autoencoder model
deep_autoencoder = DeepConvAutoencoder()

# Load trained model weights
# deep_autoencoder.load_state_dict(torch.load('path_to_trained_model_weights.pth'))

# Initialize an empty list to store the latent space representations
latent_space = []

# Set the number of images for inference
num_inference_images = 100

# Perform inference and collect latent space representations
with torch.no_grad():
    for i, (images, _) in enumerate(train_loader):
        outputs = deep_autoencoder(images)
        latent_space.append(deep_autoencoder.conv3(deep_autoencoder.conv2(deep_autoencoder.conv1(images))))
        if i == (num_inference_images // train_loader.batch_size):
            break

# Convert latent_space list to tensor
latent_space = torch.cat(latent_space, dim=0)

# Compute the spectrogram using FFT and IFFT
spectrogram = torch.fft.fftn(latent_space, dim=(-2, -1)).abs()**2
spectrogram = torch.mean(spectrogram, dim=0)

# Convert the spectrogram tensor to a NumPy array for plotting
spectrogram_np = spectrogram.numpy()

# Average across channels
spectrogram_mean = torch.mean(spectrogram, dim=0).numpy()

# Plot spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram_mean, aspect='auto', cmap='viridis', origin='lower')
plt.xlabel('Frequency')
plt.ylabel('Image Index')
plt.title('Spectrogram of Latent Space during Inference')
plt.colorbar(label='Power')
plt.show()