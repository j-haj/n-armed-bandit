import numpy as np
from collections import namedtuple

Distribution = namedtuple('Distribution', ['mean', 'var']) 

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
                                scale=self.distributions[n].var)

