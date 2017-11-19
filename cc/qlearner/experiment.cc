#include <iostream>

#include "learner.h"
#include "bandit.h"

int main(int argc, char* argv[]) {
  auto bandit (Bandit(10));
  std::cout << "Created bandit\n";
  auto learner (StationaryQLearner(10));
  std::cout << "Created stationary Q-learner\n";
  for (size_t i = 0; i < 50000; ++i) {
    auto action = learner.generate_action();
    auto reward = bandit.draw(action);
    learner.learn(action, reward);
  }
  std::cout << "Done.\n";
  std::cout << learner;
  std::cout << bandit;
  std::cout << "Success!\n";
  return 0;
}
