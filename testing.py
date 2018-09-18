import gym
import numpy as np
import MyEnv


env = gym.make('GATuning-v0')
a = env.best_state(env.state)