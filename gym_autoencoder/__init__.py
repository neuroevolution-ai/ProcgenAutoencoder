import os

from gym.envs.registration import register


register(
    id='auto-unpool-v0',
    entry_point='gym_autoencoder.envs:AutoencoderUnpoolEnv',
)
register(
    id='auto-maxpool-v0',
    entry_point='gym_autoencoder.envs:AutoencoderMaxPoolEnv',
)
register(
    id='auto-maxpool-big-v0',
    entry_point='gym_autoencoder.envs:AutoencoderMaxPoolBigEnv',
)
register(
    id='auto-no-bottleneck-v0',
    entry_point='gym_autoencoder.envs:AutoencoderNoBottleneckEnv',
)
register(
    id='vae-alex-v0',
    entry_point='gym_autoencoder.envs:VaritionalAlexEnv',
)
register(
    id='vae-paper-v0',
    entry_point='gym_autoencoder.envs:VaritionalPaperEnv',
)



