#ifndef BANDIT_H__
#define BANDIT_H__

class Bandit {
  public:
    /**
     * Samples from the bandit's distribution
     *
     * @param n the distribution (0-indexed) to sample from
     *
     * @returns reward value from distribution
     */
    double draw(size_t n);

  private:
    /// Number of arms
    size_t n;
}; // class Bandit

class MovingBandit : Bandit {


}; // class MovingBandit

#endif // BANDIT_H__
