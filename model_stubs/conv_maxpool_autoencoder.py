from torch import nn

from model_stubs.BaseAutoencoder import BaseAutoencoder
from model_stubs.custom_layer import Reshape


'''
Autoencoder based of a bachelor-thesis. Basically an Autoencoder with Conv2D and Pooling Layers, with an bottleneck
layer 
'''

class ConvMaxpoolAutoencoder(BaseAutoencoder):
    def __init__(self,latent_dim=32):
        super(ConvMaxpoolAutoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Output shape [-1,4,16,16]
            nn.Flatten(),
            nn.Linear(1024,latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,1024),
            nn.ReLU(),
            # Reshape in same form as for the Flatten
            Reshape(-1, 4, 16, 16),
            nn.ConvTranspose2d(4,16,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,2,stride=2),
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
