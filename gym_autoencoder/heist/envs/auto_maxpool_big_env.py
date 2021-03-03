
from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderMaxPoolBigEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(AutoencoderMaxPoolBigEnv, self).__init__(model_name='Conv_Maxpool_big',
                                                       path_model='./models/Conv_Maxpool_Big.pt',
                                                       shape=(32,))
