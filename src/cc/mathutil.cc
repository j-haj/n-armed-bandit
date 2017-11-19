#include <cmath>
#include <random>
#include <vector>

#include "mathutil.h"

double uniform() {
  static std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

double normal(double mean, double stddev) {
  static std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(mean, stddev);
  return dist(gen);
}

int sample_dist(const std::vector<double>& dist) {
  double x{uniform()};
  int idx{0};
  while (x > dist[idx]) {
    idx += 1;
  }
  return idx;
}

std::vector<double> softmax(const std::vector<double>& v) {
  double total{0.0};
  std::vector<double> vals(v);
  for (size_t i = 0; i < vals.size(); ++i) {
    total += v[i];
  }
  for (auto& x : vals) {
    x /= total;
  }
  return vals;
}
