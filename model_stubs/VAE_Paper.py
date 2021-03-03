import torch

from torch import nn, Tensor
from model_stubs.BaseAutoencoder import BaseAutoencoder
from model_stubs.custom_layer import Conv2DBatchNormLeakyRelu, ConvTransposed2DBatchNormLeakyRelu

'''
Variational Autoencoder based on
https://arxiv.org/pdf/1803.10122.pdf

lamda_kl_loss set to 0.0001 otherwise it leads to "Posterior Collapse" - normally its 
lamda_kl_loss = Batch_size / Number of Samples
'''


class VAEPaperAutoencoder(BaseAutoencoder):

    def __init__(self,
                 latent_dim=22,
                 in_channels=3,
                 lamda_kl_loss=0.0001) -> None:
        super(VAEPaperAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.lamda_kld_loss=lamda_kl_loss
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
        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2, padding=0))
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

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

    def decode(self, z):
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

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, input, output, args):
        recon_x, mu, logvar = output, args[1], args[2]
        recons_loss = torch.mean((recon_x - input) ** 4)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), axis=-1), axis=0)
        loss = recons_loss + self.lamda_kld_loss * kld_loss
        return loss
