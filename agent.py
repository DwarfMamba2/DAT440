import random
import numpy as np


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

        self.gamma = 0.95
        self.epsilon = 0.0
        self.alpha = 0.2

        # q-learning, double-q-learning, sarsa or expected-sarsa
        #self.algorithm = "q-learning"
        #self.algorithm = "double-q-learning"
        self.algorithm = "sarsa"
        #self.algorithm = "expected-sarsa"

        if self.algorithm == "q-learning":
            self.alpha = 0.2
            self.q_table = np.zeros((self.state_space, self.action_space))
        elif (self.algorithm == "double-q-learning"):
            self.alpha = 0.2
            self.q1_table = np.zeros((self.state_space, self.action_space))
            self.q2_table = np.zeros((self.state_space, self.action_space))
        elif (self.algorithm == "sarsa"):
            self.q_table = np.zeros((self.state_space, self.action_space))
        elif (self.algorithm == "expected-sarsa"):
            self.q_table = np.zeros((self.state_space, self.action_space))
        else:
            print("Please specify algorithm used")

    def observe(self, observation, reward, done):
        # Add your code here
        if (self.algorithm == "q-learning"):
            delta = reward + self.gamma * \
                np.max(self.q_table[observation, :]) - \
                self.q_table[self.previous_state, self.previous_action]
            self.q_table[self.previous_state, self.previous_action] = self.q_table[self.previous_state,
                                                                                   self.previous_action] + self.alpha * delta

        elif (self.algorithm == "double-q-learning"):
            if self.q_used == 1:
                max_action = self.q1_table[observation, :].tolist().index(
                    np.max(self.q1_table[observation, :]))
                delta = reward + self.gamma * \
                    self.q2_table[observation, max_action] - \
                    self.q1_table[self.previous_state, self.previous_action]
                self.q1_table[self.previous_state, self.previous_action] = self.q1_table[self.previous_state,
                                                                                         self.previous_action] + self.alpha * delta
            else:
                max_action = self.q2_table[observation, :].tolist().index(
                    np.max(self.q2_table[observation, :]))
                delta = reward + self.gamma * \
                    self.q1_table[observation, max_action] - \
                    self.q2_table[self.previous_state, self.previous_action]
                self.q2_table[self.previous_state, self.previous_action] = self.q2_table[self.previous_state,
                                                                                         self.previous_action] + self.alpha * delta

        elif (self.algorithm == "sarsa"):
            next_action = self.act(observation)
            delta = reward + self.gamma * \
                self.q_table[observation, next_action] - \
                self.q_table[self.previous_state, self.previous_action]
            self.q_table[self.previous_state, self.previous_action] = self.q_table[self.previous_state,
                                                                                   self.previous_action] + self.alpha * delta

        elif (self.algorithm == "expected-sarsa"):
            next_action = self.act(observation)
            expected_value = np.sum(self.q_table[observation, :] * self.epsilon) + (
                1 - self.epsilon) * self.q_table[observation, next_action]
            delta = reward + self.gamma * expected_value - \
                self.q_table[self.previous_state, self.previous_action]
            self.q_table[self.previous_state, self.previous_action] = self.q_table[self.previous_state,
                                                                                   self.previous_action] + self.alpha * delta

        else:
            print("did not know what algorithm to use so I learnt nothing")

    def act(self, observation):
        # Add your code here
        if isinstance(observation, tuple):
            observation = observation[0]

        if self.algorithm == "double-q-learning":
            if random.randint(1, 2) == 1:
                self.q_table = self.q1_table
                self.q_used = 1
            else:
                self.q_table = self.q2_table
                self.q_used = 2

        if random.uniform(0, 1) < self.epsilon or np.all(self.q_table[observation, :]) == self.q_table[observation, 0]:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.q_table[observation, :])

        self.previous_action = action
        self.previous_state = observation
        return action
