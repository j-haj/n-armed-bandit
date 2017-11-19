import numpy as np
from collections import namedtuple

Distribution = namedtuple('Distribution', ['mean', 'stddev']) 

class Bandit():
    '''n-armed Bandit model

    There are `n` different normal distributions (default), each with a mean
    and standard deviation sampled from a uniform distribution.
    '''
    
    def __init__(self, n):
        '''Inits Bandit class with `n` arms'''
        self.n_arms = n
        self.distributions = [Distribution(np.random.uniform(low=1, high=10),
                                           np.random.uniform(low=1, high=5))
                              for i in range(n)]

    def draw(self, n):
        '''Draws from distribution `n`

        Args:
            n: the distribution to sample
        Returns:
            the value sampled from the distribution
        '''
        return np.random.normal(loc=self.distributions[n].mean,
                                scale=self.distributions[n].stddev)

class MovingBandit(Bandit):
    '''n-armed Bandit model that evolves over time

    There are `n` different normal distributions, similar to the `Bandit` class.
    However, the mean and variance of these distributions change over time. The
    `t` parameter determines how many draws it takes for the distributions to
    change. Each distribution will change after `t` draws from the given
    distribution.
    '''

    def __init__(self, n, t):
        '''Inits Bandit class with `n` arms and evolves each distribution
        after it has been drawn from `t` times.
        '''
        super().__init__(n)
        self.t = t
        self.draw_tracker = [0 for i in range(n)]

    def draw(self, n):
        '''Draws from distribution `n`

        This method also tracks the number of times a distribution has
        been drawn from and updates the distribution when it has been
        used `t` times.

        Args:
            n: distribution to draw from

        Returns:
            A draw from distribution `n`
        '''
        self.draw_tracker[n] += 1
        val = np.random.normal(loc=self.distributions[n].mean,
                               scale=self.distributions[n].stddev)
        if self.draw_tracker[n] % self.t == 0:
            self.distributions[n] = Distribution(
                np.random.uniform(low=1, high=10),
                np.random.uniform(low=1, high=5))
        return val
