#ifndef LEARNER_H__
#define LEARNER_H__
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

class QLearner {
 public:
  QLearner(size_t n)
      : n_(n),
        q_table_(std::vector<double>(n)),
        q_actions_(std::vector<int>(n)) {}

  /**
   * Returns the average reward received, weighted by the actions taken
   */
  double average_reward() const noexcept;

  /// Getter for Q-table
  std::vector<double> q_table() const noexcept { return q_table_; };

  friend std::ostream& operator<<(std::ostream& os, const QLearner& o);

 protected:
  /// Number of actions to choose from
  size_t n_;

  /// Q table - tracks rewards received
  std::vector<double> q_table_;

  /// Q actions - tracks how many time each action has been taken
  std::vector<int> q_actions_;
};  // class QLearner

class StationaryQLearner : public QLearner {
 public:
  StationaryQLearner(size_t n, double tau = .1);

  /**
   * Generates an action for the learner to take based on the softmax function
   */
  int generate_action() noexcept;

  void learn(int action, double reward) noexcept;

  friend std::ostream& operator<<(std::ostream& os, const StationaryQLearner& o);

 private:
  double tau_;
};  // class StationaryQLearner
#endif
