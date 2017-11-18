import learner
import bandit
import numpy as np

def run_bandit_experiment(n=10):
    '''Runs an n-armed bandit experiment.

    Args:
        n: number of bandits (default 10)
    '''
    bndt = bandit.Bandit(n)
    lrnr = learner.StationaryQLearner(n, 2)

    for i in range(10000):
        if i != 0 and i %100 == 0:
            reward = lrnr.average_reward()
            q_values = lrnr.q_table
            print('{}:\tReward: {}\n\tQ-vals: {}'.format(
                i, reward, q_values))
        action = lrnr.generate_action()
        response = bndt.draw(action)
        lrnr.learn(action, response)
    print('Bandit distributions:')
    for i in range(n):
        print('\t{}'.format(bndt.distributions[i]))
    
def main():
    run_bandit_experiment()
if __name__ == '__main__':
    main()
