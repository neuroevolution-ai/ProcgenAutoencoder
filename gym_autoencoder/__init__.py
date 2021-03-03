import os

from gym.envs.registration import register


register(
    id='heist-auto-unpool-v0',
    entry_point='gym_autoencoder.heist.envs:AutoencoderUnpoolEnv',
)
register(
    id='heist-auto-maxpool-v0',
    entry_point='gym_autoencoder.heist.envs:AutoencoderMaxPoolEnv',
)
register(
    id='heist-auto-maxpool-big-v0',
    entry_point='gym_autoencoder.heist.envs:AutoencoderMaxPoolBigEnv',
)
register(
    id='heist-auto-no-bottleneck-v0',
    entry_point='gym_autoencoder.heist.envs:AutoencoderNoBottleneckEnv',
)
register(
    id='heist-vae-alex-v0',
    entry_point='gym_autoencoder.heist.envs:VaritionalAlexEnv',
)
register(
    id='heist-vae-paper-v0',
    entry_point='gym_autoencoder.heist.envs:VaritionalPaperEnv',
)



