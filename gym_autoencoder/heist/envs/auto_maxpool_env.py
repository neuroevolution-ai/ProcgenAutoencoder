
from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class AutoencoderMaxPoolEnv(AutoencoderBasicEnv):
    def __init__(self,use_gpu=True):
        super(AutoencoderMaxPoolEnv, self).__init__(model_name='Conv_Maxpool',
                                                    path_model='./models/Conv_Maxpool.pt',
                                                    shape=(32,),
                                                    use_gpu=use_gpu)
