from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
class VaritionalAlexEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(VaritionalAlexEnv, self).__init__(path_stub='model_stubs.VAE_Alex',
                                                path_model='./models/VAE_Alex.pt',
                                                shape=(64,))
