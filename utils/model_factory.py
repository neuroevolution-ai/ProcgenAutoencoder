
from model_stubs.VAE_Alex import VAEAlexAutoencoder
from model_stubs.VAE_Paper import VAEPaperAutoencoder
from model_stubs.conv_unpool import ConvUnpoolAutoencoder
from model_stubs.conv_maxpool_autoencoder import ConvMaxpoolAutoencoder
from model_stubs.conv_maxpool_big_autoencoder import ConvMaxPoolBigAutoencoder
from model_stubs.conv_no_bottleneck_autoencoder import ConvNoBottleneckAutoencoder



def factory(model):

    autoencoders = {'VAE_Alex':  VAEAlexAutoencoder(),
                    'VAE_Paper': VAEPaperAutoencoder(),
                    'Conv_Unpool': ConvUnpoolAutoencoder(),
                    'Conv_Maxpool':ConvMaxpoolAutoencoder(),
                    'Conv_Maxpool_big':ConvMaxPoolBigAutoencoder(),
                    'Conv_No_Bottleneck':ConvNoBottleneckAutoencoder()
                    }
    return autoencoders[model]
