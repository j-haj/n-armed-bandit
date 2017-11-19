import learner
import bandit
import numpy as np

def stationary_vs_moving_learners(n=10):
    '''Runs two learning algorithms, stationary and moving against and n-armed
    bandit
    '''

    # Set number of rounds
    n_rounds = 50
    n_steps_per_round = 10000
    n = 10

 
    rewards_stationary = []
    rewards_moving = []

    for _ in range(n_rounds):
        # Need to bandits
        bndt1 = bandit.MovingBandit(n, 50)
        bndt2 = bandit.MovingBandit(n, 50)

        # Build learners
        lrnr1 = learner.StationaryQLearner(n)
        lrnr2 = learner.MovingQLearner(n, alpha=.01)
        for _ in range(n_steps_per_round):
            action1 = lrnr1.generate_action()
            action2 = lrnr2.generate_action()
            reward1 = bndt1.draw(action1)
            reward2 = bndt2.draw(action2)
            lrnr1.learn(action1, reward1)
            lrnr2.learn(action2, reward2)
        rewards_stationary.append(lrnr1.average_reward())
        rewards_moving.append(lrnr2.average_reward())

    print('Ran {} experiments of {:,} steps each:'.format(n_rounds,
                                                          n_steps_per_round))
    print('\tStationary algorithm:\t{:.4f}'.format(np.average(rewards_stationary)))
    print('\tMoving algorithm:\t{:.4f}'.format(np.average(rewards_moving)))

def run_bandit_experiment(n=10):
    '''Runs an n-armed bandit experiment.

    Args:
        n: number of bandits (default 10)
    '''
    bndt = bandit.MovingBandit(n, 5)
    lrnr = learner.MovingQLearner(n, 2)

    for i in range(10000):
        if i != 0 and i %100 == 0:
            reward = lrnr.average_reward()
            q_values = lrnr.q_table
            q_val_str = ' '.join(['{:.2f}'.format(v) for v in q_values])
            print('{}: Reward: {:.4f}\tQ-vals: {}'.format(
                i, reward, q_val_str))
        action = lrnr.generate_action()
        response = bndt.draw(action)
        lrnr.learn(action, response)

    print('Bandit distributions:')
    for i in range(n):
        print('\t{}'.format(bndt.distributions[i]))
    
def main():
    #run_bandit_experiment()
    stationary_vs_moving_learners()

if __name__ == '__main__':
    main()
