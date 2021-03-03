import importlib

import cv2
import gym
import numpy as np
import torch
from gym.spaces import Box


class AutoencoderBasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path_stub, path_model ,shape):
        self.env = gym.make("procgen:procgen-heist-v0", use_backgrounds=False, render_mode="rgb_array",
                            distribution_mode="memory")
        self.model = importlib.import_module(path_stub).Autoencoder()
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=shape,
                                     dtype=float)
        self.action_space = self.env.action_space
        self.last_observation=None

    def _transform_ob(self, ob):
        obs = self.preprocess_observation(ob)
        obs = self.model.encode(obs)
        obs = obs[0].cpu().detach().numpy()
        return obs

    def preprocess_observation(self, ob):
        '''
        Observation are in form [WxHxC],  pytorch needs them in [1xCxWxH] and normalized to (0,1) space
        therefore preprocessing is needed
        :param ob: the observation in form [WxHxC]
        :return: preprocessed observation form [1xCxWxH]
        '''
        obs = np.transpose(ob, (2, 0, 1))
        obs = obs.astype(np.float32)
        obs = obs / 255
        obs = torch.from_numpy(obs)
        obs = obs.unsqueeze(0)
        return obs

    def _transform_render(self, ob):
        obs = self.preprocess_observation(ob)
        obs = self.model(obs)
        obs = obs[0].cpu().detach().numpy() * 255
        obs = obs[0].astype(np.uint8)
        bigger = np.transpose(obs, (1, 2, 0))
        bigger = cv2.cvtColor(bigger, cv2.COLOR_BGR2RGB)
        bigger = cv2.resize(bigger, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)
        return bigger

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        info['original_ob']=ob
        self.last_observation=ob
        return self._transform_ob(ob), rew, done, info

    def reset(self):
        return self._transform_ob(self.env.reset())

    def render(self, mode='human'):
        obs = self._transform_render(self.last_observation)
        cv2.imshow("Result", obs)
        cv2.waitKey(1)
