#ifndef LEARNER_H__
#define LEARNER_H__

class QLearner {
  public:
    QLearner(size_t n);

    /**
     * Returns the average reward received, weighted by the actions taken
     */
    double average_reward() const noexcept;

  protected:
    /// Number of actions to choose from
    size_t n_;

    /// Q table - tracks rewards received
    std::vector<double> q_table_;

    /// Q actions - tracks how many time each action has been taken
    std::vector<int> q_actions_;
}; // class QLearner

class StationaryQLearner : protected QLearner {
  public:
    StationaryQLearner(size_t n, double tau = .1);

    /**
     * Generates an action for the learner to take based on the softmax function
     */
    int generate_action() const noexcept;

    void learn(int action, double reward) const noexcept;
    
  private:
    double tau_;
}; // class StationaryQLearner
#endif
