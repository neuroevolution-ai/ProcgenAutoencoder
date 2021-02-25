
import torch
from torchvision import transforms

from model_stubs.custom_layer import Conv2DBatchNormLeakyRelu, ConvTransposed2DBatchNormLeakyRelu
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.models
from model_stubs.BaseAutoencoder import BaseAutoencoder


'''
Oriented on https://arxiv.org/abs/2001.03444 - used LeakyReLU instead ReLU , otherwise same architecture
Instead of Retraining the net, both perceptual loss and MSE_Loss are used
'''


class Autoencoder(BaseAutoencoder):

    def __init__(self,
                 latent_dim=64,
                 in_channels=3,
                ):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        # Build Encoder
        modules = [Conv2DBatchNormLeakyRelu(in_channels=in_channels, n_filters=32, k_size=4, stride=2, padding=0),
                   Conv2DBatchNormLeakyRelu(in_channels=32, n_filters=64, k_size=4, stride=2, padding=0),
                   Conv2DBatchNormLeakyRelu(in_channels=64, n_filters=128, k_size=4, stride=2, padding=0),
                   Conv2DBatchNormLeakyRelu(in_channels=128, n_filters=256, k_size=4, stride=2, padding=0)]

        self.encoder = nn.Sequential(*modules)
        # Output Shape of Encoder is [256x2x2]
        output_shape_encoder = 256 * 2 * 2
        self.fc_mu = nn.Linear(output_shape_encoder, latent_dim)
        self.fc_var = nn.Linear(output_shape_encoder, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, output_shape_encoder)

        # Build Decoder
        modules.append(
            ConvTransposed2DBatchNormLeakyRelu(in_channels=1024, n_filters=128, k_size=5, stride=2, padding=0))
        modules.append(ConvTransposed2DBatchNormLeakyRelu(in_channels=128, n_filters=64, k_size=5, stride=2, padding=0))
        modules.append(ConvTransposed2DBatchNormLeakyRelu(in_channels=64, n_filters=32, k_size=6, stride=2, padding=0))
        #Last Layer
        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2, padding=0))
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

        self.alex =torchvision.models.alexnet(pretrained=True)
        self.alex = nn.Sequential(*list(self.alex.features.children())[0:5],nn.Sigmoid())

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 1024, 1, 1)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, args, input):
        recon_x, mu, logvar = args[0], args[1], args[2]

        # Needed because pretrained Alex-Net uses this normalization-values
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        normal_input =normalize(input*255)
        normal_reconx=normalize(recon_x*255)
        recons_loss = F.mse_loss(self.alex(normal_input), self.alex(normal_reconx), reduction='mean')
        real_loss = F.mse_loss(recon_x,input,reduction='mean')
        loss = recons_loss+real_loss
        return loss
