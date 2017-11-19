#include <random>

#include "bandit.h"
#include "mathutil.h"

Bandit::Bandit(size_t n) : n_(n) {
  distributions_ = std::vector<Distribution>(n);
  for (size_t i = 0; i < n; ++i) {
    distributions_[i] = Distribution{normal(10.0, 5.0), normal(5.0, 2.0)};
  }
}

double Bandit::draw(size_t n) {
  auto d = distributions_[n];
  return normal(d.mean, d.stddev);
}
