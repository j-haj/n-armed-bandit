#include <cmath>
#include <random>
#include <vector>

#include <iostream>

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

std::vector<double> math::convert_to_cdf(const std::vector<double>& dist) {
  std::vector<double> cdf(dist.size());
  double running_total {0.0};
  for (size_t i = 0; i < dist.size(); ++i) {
    cdf[i] = dist[i] + running_total;
    running_total += cdf[i];
  }
  return cdf;
}

int math::sample_dist(const std::vector<double>& dist) {
  const double x{uniform()};
  int idx{0};
  size_t max_len {dist.size()};
  auto cdf = convert_to_cdf(dist);
  std::cout << "Sampling...\n";
  while (x > cdf[idx]) {
    std::cout << "x: " << x << " cdf: " << cdf[idx] << '\n';
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
