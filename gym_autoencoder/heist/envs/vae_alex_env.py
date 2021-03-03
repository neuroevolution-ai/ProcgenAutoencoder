from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class VaritionalAlexEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(VaritionalAlexEnv, self).__init__(model_name='VAE_Alex',
                                                path_model='./models/VAE_Alex.pt',
                                                shape=(64,))
