import os

import gym
import numpy as np

"""
python script for generating training and test samples for the procgen game heist.
Hyperparameter: number of samples, split in test/training data , output directory
Generates 2 Files, File format .npy
"""


output_dir = "./data/heist/"

training_samples=[]
test_samples=[]

number_of_samples=2000
training_split = 10
number_of_random_moves=10



if __name__ == "__main__":
    for i in range(number_of_samples):
        env = gym.make("procgen:procgen-heist-v0", start_level=i, render_mode="rgb_array", use_backgrounds=False,
                       num_levels=0, distribution_mode="memory")
        observation = env.reset()
        for j in range(number_of_random_moves):
            observation, rew, done, info = env.step(env.action_space.sample())
            if done:
                break
        if(i%training_split==0):
            test_samples.append(observation)
        else:
            training_samples.append(observation)
        env.close()

    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir +"training_samples.npy", np.asarray(training_samples))
    np.save(output_dir +"test_samples.npy", np.asarray(test_samples))