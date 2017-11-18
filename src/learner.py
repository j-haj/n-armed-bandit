import numpy as np

def softmax(x):
    '''Computes softmax of array x'''
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

class QLearner():
    '''Base Q-learner class
    '''

    def __init__(self, n):
        self.n = n
        self.q_table = np.array([0. for i in range(n)])
        self.q_actions = np.array([0 for i in range(n)])

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

class StationaryQLearner(QLearner):

    def __init__(self, n, tau=.1):
        '''Inits a Stationary Learner

        Args:
            epsilon: parameter that controls how much searching the learner does
            n: number of actions available to the learner
            tau: parameter that controls softmax distribution
        '''
        super().__init__(n)
        self.tau = tau

    def generate_action(self):
        p_vals = softmax(self.q_table/self.tau)
        action = np.random.choice(self.n, p=p_vals)
        self.q_actions[action] += 1
        return action

    def learn(self, action, reward):
        k = self.q_actions[action]
        self.q_table[action] = self.q_table[action] +\
                               (1./k) * (reward - self.q_table[action])


class MovingQLearner(QLearner):

    def __init__(self, n, tau=.1, alpha=.5):
        '''Inits moving learner

        Args:
            n: number of actions possible
            tau: parameter that shapes the action distribution
        '''
        super().__init__(n)
        self.tau = tau
        self.step_size = alpha

    def generate_action(self):
        p_vals = softmax(self.q_table/self.tau)
        action = np.random.choice(self.n, p=p_vals)
        self.q_actions[action] += 1
        return action

    def learn(self, action, reward):
        k = self.q_actions[action]
        self.q_table[action] = self.q_table[action] +\
                               self.step_size * (reward - self.q_table[action])

    
