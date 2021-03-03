from gym_autoencoder.heist.envs.auto_basic_env import AutoencoderBasicEnv


class VaritionalPaperEnv(AutoencoderBasicEnv):
    def __init__(self):
        super(VaritionalPaperEnv, self).__init__(model_name='VAE_Paper',
                                                 path_model='./models/VAE_Paper.pt',
                                                 shape=(22,))
