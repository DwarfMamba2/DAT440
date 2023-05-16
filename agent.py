import random
import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.gamma = 0.95
        self.epsilon = 0.05


    def observe(self, observation, reward, done):
        #Add your code here

        pass
    def act(self, observation):
        #Add your code here
        if random.uniform(0,1) < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.q_table[observation,:])


        return np.random.randint(self.action_space)