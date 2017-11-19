#include <vector>

#include "learner.h"
#include "mathutil.h"

StationaryQLearner::StationaryQLearner(size_t n, double tau) : QLearner(n), tau_(tau) {}

int StationaryQLearner::generate_action() const {
  auto v {uniform()};
}

void StationaryQLearner::learn(int action, double reward) const {
  auto k = q_actions_[action];
  q_table_[action] = q_table_[action] + (1. / k) * (reward - q_table_[action]);
}
