from gym_autoencoder.envs.auto_basic_env import AutoencoderBasicEnv


class VaritionalAlexEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(VaritionalAlexEnv, self).__init__(path_stub='model_stubs.VAE_Alex',
                                                path_model='./models/VAE_Alex.pt',
                                                shape=(64,))
