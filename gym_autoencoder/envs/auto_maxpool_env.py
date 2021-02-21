
from gym_autoencoder.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderMaxPoolEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(AutoencoderMaxPoolEnv, self).__init__(path_stub='model_stubs.conv_maxpool_autoencoder',
                                                   path_model='models/Conv_Maxpool.pt',
                                                   shape=(32,))
