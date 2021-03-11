from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderUnpoolEnv(AutoencoderBasicEnv):
    def __init__(self,use_gpu=True):
        super(AutoencoderUnpoolEnv, self).__init__(model_name='Conv_Unpool',
                                                   path_model='./models/Conv_Unpool.pt',
                                                   shape=(32, 6, 6),
                                                   use_gpu=use_gpu
                                                   )
