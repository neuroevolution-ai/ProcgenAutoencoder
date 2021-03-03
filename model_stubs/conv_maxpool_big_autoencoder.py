import torch.nn as nn
from model_stubs.BaseAutoencoder import BaseAutoencoder
from model_stubs.custom_layer import Reshape

'''
Autoencoder based of a bachelor-thesis. Basically an Autoencoder with Conv2D and Pooling Layers  with an bottleneck
layer.
Difference between conv_maxpool_autoencoder an this architecture is one more Linear Layer, also way more convolution
filters
'''

class Autoencoder(BaseAutoencoder):
    def __init__(self,latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6400, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 6400),
            nn.ReLU(),
            Reshape(-1, 64, 10, 10),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.encoder(input)
        input = self.decoder(input)
        return [input]

    def encode(self, input):
        return self.encoder(input)

    def loss_function(self, input, output, args):
        loss = nn.MSELoss()
        return loss(output, input)
