
from gym_autoencoder.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderNoBottleneckEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(AutoencoderNoBottleneckEnv, self).__init__(path_stub='model_stubs.conv_no_bottleneck_autoencoder',
                                                   path_model='models/Conv_No_Bottleneck.pt',
                                                   shape=(128, 1,1))
