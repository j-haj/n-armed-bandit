#include <algorithm>
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

double StationaryQLearner::average_reward() const {
  double prod_sum {0.0};
  double total_actions {0.0};
  for (size_t i = 0; i < h_prod.size(); ++i) {
    prod_sum += q_table_[i] * q_actions_[i];
    total_actions += q_actions_[i];
  }
  return prod_sum / total_actions;
}

