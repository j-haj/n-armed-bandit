#include <iomanip>
#include <random>
#include <sstream>

#include "bandit.h"
#include "mathutil.h"

Bandit::Bandit(size_t n) : n_(n) {
  distributions_ = std::vector<Distribution>(n);
  for (size_t i = 0; i < n; ++i) {
    distributions_[i] =
        Distribution{math::normal(10.0, 5.0), math::normal(5.0, 2.0)};
  }
}

double Bandit::draw(size_t n) {
  auto d = distributions_[n];
  return math::normal(d.mean, d.stddev);
}

std::ostream& operator<<(std::ostream& os, const Bandit& o) {
  std::stringstream output;
  output << "Distributions (mean, stddev):\n";
  auto len = o.distributions().size();
  for (auto i = 0; i < len; ++i) {
    auto d = o.distributions()[i];
    output << "\t" << std::setprecision(2) << d.mean << ", " << d.stddev << '\n';
  }
  output << '\n';
  os << output.str();
  return os;
}

