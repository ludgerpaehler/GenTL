#include <gentl/inference/mcmc.h>
#include <gentl/types.h>
#include <gentl/util/randutils.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std::chrono;

float logpdf(float x) {
    return -0.5 * (x * x);
}

int main(int argc, char *argv[]) {
  using std::cerr;
  using std::cout;
  using std::endl;

  // initialize RNG
  unsigned int seed = 314159;
  gentl::randutils::seed_seq_fe128 seed_seq{seed};
  std::mt19937 rng(seed_seq);

  auto num_iters = 10;
  auto nowAt = 0.1;
  auto mh_num_accepted = 0;
  float accept_reject = 0.0;
  float draw = 0.0;
  float nextMaybe = 0.0;
  std::ofstream timings("handcoded_timings.csv", std::ofstream::out);
  static std::uniform_real_distribution<float> uniform_real_dist(0.0, 1.0);

  // Micro timing benchmarks.
  for (size_t outer = 1; outer < 50000; outer++) {
    auto start = high_resolution_clock::now();
    for (size_t iter = 2; iter < num_iters; iter++) {
        draw = uniform_real_dist(rng);
        accept_reject = uniform_real_dist(rng);
        nextMaybe = nowAt + (draw - 0.5) / 2.0;
        if (logpdf(nextMaybe) - logpdf(nowAt) > logpdf(accept_reject)) {
            nowAt = nextMaybe;
            mh_num_accepted += 1;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<nanoseconds>(stop - start);
    timings << duration.count() << "," << endl;

  }
  return mh_num_accepted;
}
