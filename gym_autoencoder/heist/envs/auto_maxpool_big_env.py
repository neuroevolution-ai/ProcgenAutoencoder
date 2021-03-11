
from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderMaxPoolBigEnv(AutoencoderBasicEnv):
    def __init__(self,use_gpu=True):
        super(AutoencoderMaxPoolBigEnv, self).__init__(model_name='Conv_Maxpool_big',
                                                       path_model='./models/Conv_Maxpool_Big.pt',
                                                       shape=(32,),
                                                       use_gpu=use_gpu)
