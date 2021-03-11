
from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderNoBottleneckEnv(AutoencoderBasicEnv):
    def __init__(self,use_gpu=True):
        super(AutoencoderNoBottleneckEnv, self).__init__(model_name='Conv_No_Bottleneck',
                                                         path_model='./models/Conv_No_Bottleneck.pt',
                                                         shape=(128, 1,1),
                                                         use_gpu=use_gpu)
