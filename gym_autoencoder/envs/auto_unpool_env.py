from gym_autoencoder.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderUnpoolEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(AutoencoderUnpoolEnv, self).__init__(path_stub='model_stubs.conv_unpool',
              path_model='./models/Conv_Unpool.pt',
              shape=(32, 6, 6))
