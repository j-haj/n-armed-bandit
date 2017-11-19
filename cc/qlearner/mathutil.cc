#include <cmath>
#include <random>
#include <vector>

#include "mathutil.h"

double math::uniform() {
  static std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

double math::normal(double mean, double stddev) {
  static std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(mean, stddev);
  return dist(gen);
}

int math::sample_dist(const std::vector<double>& dist) {
  double x{uniform()};
  int idx{0};
  size_t max_len {dist.size()};
  while (x > dist[idx]) {
    if (idx == max_len - 1) break;
    idx += 1;
  }
  return idx;
}

std::vector<double> math::softmax(const std::vector<double>& v, const double scale) {
  double total{0.0};
  std::vector<double> vals(v.size());

  for (size_t i = 0; i < vals.size(); ++i) {
    auto e = exp(v[i] / scale);
    vals[i] = e;
    total += e;
  }
  for (auto& x : vals) {
    x /= total;
  }
  return vals;
}
