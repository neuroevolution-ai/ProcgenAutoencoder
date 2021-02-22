import gym
import numpy as np
from PIL import Image
from data_generation import balance_data
import random

output_dir = "/data/heist"
import cv2 as cv

"""
python script for generating training and test samples for the procgen game heist.
This Script generated equals Number of Samples where Keys/Locks are seen and samples without locks/keys
Hyperparameter: number of samples, split in test/training data , output directory
Generates 2 Files, File format .nyp
"""

training_samples = []
test_samples = []

generated_samples = 0
number_of_samples = 60000
training_split = 20
next_function = 1
number_of_random_moves = 10


def random_next_function():
    global next_function
    next_function = random.randint(1, 14)


def has_element(img):
    '''
    Based on the value of next_function template Matching for a specific Template will be used
    :param img: The input image used for Template Matching
    :return: If a matching template is found True will be returned otherwise False.
    '''
    global next_function
    if next_function == 1:
        return balance_data.has_blue_locks(img)
    if next_function == 2:
        return balance_data.has_red_locks(img)
    if next_function == 3:
        return balance_data.has_green_locks(img)
    if next_function == 4:
        return balance_data.has_goal(img)
    if next_function == 5:
        return balance_data.has_blue_keys(img)
    if next_function == 6:
        return balance_data.has_green_keys(img)
    if next_function == 7:
        return balance_data.has_red_keys(img)
    if next_function >= 8:
        return balance_data.has_no_Keys_and_Doors(img)
    return False


if __name__ == "__main__":
    next_level = 1
    while generated_samples < number_of_samples:
        next_level = next_level + 1
        env = gym.make("procgen:procgen-heist-v0", start_level=next_level, render_mode="rgb_array",
                       use_backgrounds=False,
                       num_levels=0, distribution_mode="memory", )

        observation = env.reset()

        # Run Random
        for _ in range(number_of_random_moves):
            observation, rew, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                break
        observation = cv.cvtColor(observation, cv.COLOR_BGR2RGB)
        has = has_element(observation)
        if has:
            if (generated_samples % training_split == 0):
                test_samples.append(observation)
            else:
                training_samples.append(observation)
            generated_samples = generated_samples + 1
            random_next_function()
            bigger_template = cv.resize(observation, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)
            cv.imshow("Next Sample", bigger_template)
            cv.waitKey(1)
    env.close()

    np.save(output_dir + "training_samples_memory_noBack_balanced", np.asarray(training_samples))
    np.save(output_dir + "test_samples_memory_noBack_balanced", np.asarray(test_samples))
