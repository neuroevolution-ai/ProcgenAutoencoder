
from gym_autoencoder.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderMaxPoolBigEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(AutoencoderMaxPoolBigEnv, self).__init__(path_stub='model_stubs.conv_maxpool_big_autoencoder',
                                                   path_model='./models/Conv_Maxpool_Big.pt',
                                                   shape=(32,))
