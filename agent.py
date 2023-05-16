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
        self.alpha = 0.9
        


    def observe(self, observation, reward, done):
        #Add your code here
        delta = reward + self.gamma * np.max(self.q_table[observation, :])- self.q_table[self.previous_state, self.previous_action]
        self.q_table[self.previous_state, self.previous_action] = self.q_table[self.previous_state, self.previous_action] + self.alpha * delta
        #print(self.q_table[self.previous_state, self.previous_action])
        #print("delta: " + str(delta))
 
        pass
    def act(self, observation):
        #Add your code here
        if isinstance(observation, tuple):
            observation = observation[0]
        
        if random.uniform(0,1) < self.epsilon or np.all(self.q_table[observation, :]) == self.q_table[observation, 0]:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.q_table[observation,:])

        self.previous_action = action
        self.previous_state = observation
        return action