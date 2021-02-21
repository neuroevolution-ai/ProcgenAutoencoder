import gym
import numpy as np
from PIL import Image
from data_generation import balance_data
import random
output_dir = "../"
import cv2 as cv


"""
python script for generating training and test samples for the procgen game heist.
Hyperparameter: number of samples, split in test/training data , output directory
Generates 2 Files, File format .nyp
"""

training_samples = []
test_samples = []

generated_samples =0
number_of_samples = 60000
training_split = 20
next_function=1




def nextSample():
    global next_function
    next_function = random.randint(1,14)

def use_function(img):
    global next_function
    if next_function ==1:
        return balance_data.has_blue_locks(img)
    if next_function ==2:
        return balance_data.has_red_locks(img)
    if next_function ==3:
        return balance_data.has_green_locks(img)
    if next_function ==4:
        return balance_data.has_goal(img)
    if next_function ==5:
        return balance_data.has_blue_keys(img)
    if next_function ==6:
        return balance_data.has_green_keys(img)
    if next_function ==7:
        return balance_data.has_red_keys(img)
    if next_function >= 8:
        return balance_data.has_no_Keys_and_Doors(img)
    return None




if __name__ == "__main__":
    i = 1
    while generated_samples < number_of_samples:
        i = i + 1

        env = gym.make("procgen:procgen-heist-v0", start_level=i, render_mode="rgb_array", use_backgrounds=False,
                       num_levels=0, distribution_mode="memory",)

        observation = env.reset()

        # Run Random
        obs = env.reset()
        for j in range(10):
            observation, rew, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                break


        bigger_template = cv.cvtColor(observation, cv.COLOR_BGR2RGB)
        has =use_function(bigger_template)
            #bigger_template = cv.resize(bigger_template, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)
            #cv.imshow("Hier",bigger_template)
            #cv.waitKey(1)
            #img, min_val, max_val, min_loc, max_loc, w, h =balance_data.find_template(observation,balance_data.template_blue_key[1],balance_data.template_blue_key[0])
        if has:
            if (generated_samples % training_split == 0):
                test_samples.append(observation)
            else:
                training_samples.append(observation)
                #im = Image.fromarray(bigger_template)
                #im.save("research4/" + str(generated_samples) + ".png")
                #cv.imwrite("research4/" + str(generated_samples) + ".png",bigger_template)
            generated_samples = generated_samples + 1
            nextSample()
            bigger_template =cv.resize(bigger_template, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)
            cv.imshow("Next Sample", bigger_template)
            cv.waitKey(1)
            if (generated_samples % 40 == 0):
                im = Image.fromarray(observation)
                im.save("research6/" + str(generated_samples) + ".png")
                print(str(generated_samples) + " Step")
                print("i is " + str(i))
    env.close()

    np.save(output_dir + "training_samples_memory_noBack_balanced", np.asarray(training_samples))
    np.save(output_dir + "test_samples_memory_noBack_balanced", np.asarray(test_samples))
