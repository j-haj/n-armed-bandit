import numpy as np

def softmax(x):
    '''Computes softmax of array x'''
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

class StationaryQLearner():

    def __init__(self, n, tau=.1):
        '''Inits a Stationary Learner

        Args:
            epsilon: parameter that controls how much searching the learner does
            n: number of actions available to the learner
            tau: parameter that controls softmax distribution
        '''
        self.n = n
        self.tau = tau
        self.q_table = np.array([0 for i in range(n)])
        self.q_actions = np.array([0 for i in range(n)])

    def generate_action(self):
        p_vals = softmax(self.q_table/self.tau)
        action = np.random.choice(self.n, p=p_vals)
        self.q_actions[action] += 1
        return action

    def learn(self, action, reward):
        k = self.q_actions[action]
        self.q_table[action] = k * self.q_table[action] +  reward
        self.q_table[action] /= (k + 1)

    def average_reward(self):
        '''Returns the average reward
        '''
        reward = 0.
        total_rewards = 0
        for i in range(self.n):
            n_actions = self.q_actions[i]
            reward_total = self.q_table[i] * n_actions
            total_rewards += n_actions
            reward += reward_total
        return reward / total_rewards
