import torch.nn as nn
from model_stubs.BaseAutoencoder import BaseAutoencoder

'''
Autoencoder based on Conv/MaxPooling Layers, without Bottleneck
'''

from torchinfo import summary


class Autoencoder(BaseAutoencoder):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 7, stride=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, input):
        input = self.encoder(input)
        input = self.decoder(input)
        return [input]

    def encode(self, input):
        return self.encoder(input)


    def loss_function(self, args, input):
        loss = nn.MSELoss()
        return loss(args[0], input)

