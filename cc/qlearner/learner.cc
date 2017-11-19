#include <algorithm>
#include <iostream>
#include <vector>

#include "learner.h"
#include "mathutil.h"

StationaryQLearner::StationaryQLearner(size_t n, double tau)
    : QLearner(n), tau_(tau) {}

void StationaryQLearner::learn(int action, double reward) noexcept {
  auto k = q_actions_[action];
  q_table_[action] = q_table_[action] + (1. / k) * (reward - q_table_[action]);
}

double QLearner::average_reward() const noexcept {
  double prod_sum{0.0};
  double total_actions{0.0};
  for (size_t i = 0; i < q_actions_.size(); ++i) {
    prod_sum += q_table_[i] * q_actions_[i];
    total_actions += q_actions_[i];
  }
  return prod_sum / total_actions;
}

int StationaryQLearner::generate_action() noexcept {
  auto p_vals = math::softmax(q_table_, tau_);
  auto action = math::sample_dist(p_vals);
  q_actions_[action] += 1;
  return action;
}

std::ostream& operator<<(std::ostream& os, const QLearner& o) {
  std::stringstream output;
  output << "Q-table: [ ";
  auto table = o.q_table();
  for (const auto& x : table) {
    output << std::setprecision(2) << x << ' ';
  }
  output << "]\n";
  os << output.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const StationaryQLearner& o) {
  os << static_cast<const QLearner&>(o);
  return os;
}
