#ifndef MATHUTIL_H__
#define MATHUTIL_H__

/**
 * struct to store distribution information
 */
struct Distribution {
  double mean;
  double stddev;
};  // struct Distribution

namespace math {
/**
 * Returns a value x ~ U(0,1), where x is a real.
 */
double uniform();

/**
 * Returns a random value drawn from a normal distribution with the given mean
 * and standard deviation.
 *
 * @param mean the mean of the gaussian
 * @param stddev the standard deviation of the gaussian
 *
 * @return a sample x ~ N(mean, stddev)
 */
double normal(double mean, double stddev);

/**
 * Returns a sample drawn uniformly from the given distribution.
 *
 * Assumes sum(dist) = 1
 *
 * @param dist vector of probabilities summing to 1
 *
 * @returns the index of the probability corresponding to a randomly sampled
 * point for `dist`
 */
int sample_dist(const std::vector<double>& dist);

/**
 * The softmax distribution for the given vector of means and standard
 * deviations
 *
 * @param v a vector of `Distribution` types
 *
 * @returns a vector of values summing to one, computed via the softmax function
 */
std::vector<double> softmax(const std::vector<double>& v, const double scale=1.0);
}  // namespace math
#endif  // MATHUTIL_H__
