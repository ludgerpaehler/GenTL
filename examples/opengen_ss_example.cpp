#include <gentl/inference/mcmc.h>
#include <gentl/types.h>
#include <gentl/util/randutils.h>

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std::chrono;

using gentl::GenerateOptions;
using gentl::SimulateOptions;
using gentl::UpdateOptions;

// ****************************
// *** Model implementation ***
// ****************************

typedef float mean_t;
typedef float cov_t;

// Selection types

class LatentsSelection {};

// Choice buffer types

class EmptyChoiceBuffer {};

typedef float latent_choices_t;

// return value change types

class RetvalChange {};

// learnable parameters

class GradientAccumulator {};

class ModelTrace;

class ModelParameters {};

class Model {
  typedef int return_type;
  friend class ModelTrace;

private:
  mean_t mean_;
  cov_t cov_;

public:
  template <typename RNGType>
  void exact_sample(latent_choices_t &latents, RNGType &rng) const {
    static std::normal_distribution<float> standard_normal_dist(0.0, 1.0);
    latents = standard_normal_dist(rng);
    return;
  }

  [[nodiscard]] float logpdf(const latent_choices_t &latents) const {
    static float logSqrt2Pi = 0.5 * std::log(2 * M_PI);
    auto w = -0.5 * (latents * latents) - logSqrt2Pi;
    return w;
  }

  template <typename RNGType>
  [[nodiscard]] std::pair<float, float>
  importance_sample(latent_choices_t &latents, RNGType &rng) const {
    exact_sample(latents, rng);
    float log_weight = 0.0;
    return {logpdf(latents), log_weight};
  }

public:
  Model(mean_t mean, cov_t cov)
      : mean_{std::move(mean)}, cov_{std::move(cov)} {}

  // simulate into a new trace object
  template <typename RNGType>
  std::unique_ptr<ModelTrace> simulate(RNGType &rng,
                                       ModelParameters &parameters,
                                       const SimulateOptions &) const;

  // simulate into an existing trace object (overwriting existing contents)
  template <typename RNGType>
  void simulate(RNGType &rng, ModelParameters &parameters,
                const SimulateOptions &, ModelTrace &trace) const;

  // generate into a new trace object
  template <typename RNGType>
  std::pair<std::unique_ptr<ModelTrace>, float>
  generate(const EmptyChoiceBuffer &constraints, RNGType &rng,
           ModelParameters &parameters, const GenerateOptions &) const;

  // generate into an existing trace object (overwriting existing contents)
  template <typename RNGType>
  float generate(ModelTrace &trace, const EmptyChoiceBuffer &constraints,
                 RNGType &rng, ModelParameters &parameters,
                 const GenerateOptions &) const;

  // equivalent to generate but without returning a trace
  template <typename RNG>
  std::pair<int, float> assess(RNG &, ModelParameters &,
                               const latent_choices_t &constraints) const;

  template <typename RNG>
  std::pair<int, float> assess(RNG &, ModelParameters &,
                               const EmptyChoiceBuffer &constraints) const;
};

class ModelTrace {
  friend class Model;

private:
  Model model_;
  float score_;
  latent_choices_t latents_;
  latent_choices_t alternate_latents_;
  latent_choices_t latent_gradient_;
  bool can_be_reverted_;

private:
  ModelTrace(Model model, float score, latent_choices_t &&latents)
      : model_{std::move(model)}, score_{score}, latents_{latents},
        can_be_reverted_{false} {}

public:
  ModelTrace() = delete;
  ModelTrace(const ModelTrace &other) = delete;
  ModelTrace(ModelTrace &&other) = delete;
  ModelTrace &operator=(const ModelTrace &other) = delete;
  ModelTrace &operator=(ModelTrace &&other) noexcept = delete;

  [[nodiscard]] float score() const;
  [[nodiscard]] const latent_choices_t &choices() const;
  [[nodiscard]] const latent_choices_t &
  choices(const LatentsSelection &selection) const;
  const latent_choices_t &choice_gradient(const LatentsSelection &selection);

  template <typename RNG>
  float update(RNG &, const gentl::change::NoChange &,
               const latent_choices_t &constraints,
               const UpdateOptions &options);
  const latent_choices_t &backward_constraints();

  void revert();
};

// ****************************
// *** Model implementation ***
// ****************************

template <typename RNGType>
std::unique_ptr<ModelTrace>
Model::simulate(RNGType &rng, ModelParameters &parameters,
                const SimulateOptions &options) const {
  latent_choices_t latents;
  exact_sample(latents, rng);
  auto log_density = logpdf(latents);
  // note: this copies the model
  return std::unique_ptr<ModelTrace>(
      new ModelTrace(*this, log_density, std::move(latents)));
}

template <typename RNGType>
void Model::simulate(RNGType &rng, ModelParameters &parameters,
                     const SimulateOptions &options, ModelTrace &trace) const {
  exact_sample(trace.latents_, rng);
  trace.score_ = logpdf(trace.latents_);
  trace.can_be_reverted_ = false;
}

template <typename RNGType>
std::pair<std::unique_ptr<ModelTrace>, float>
Model::generate(const EmptyChoiceBuffer &constraints, RNGType &rng,
                ModelParameters &parameters,
                const GenerateOptions &options) const {
  latent_choices_t latents;
  auto [log_density, log_weight] = importance_sample(latents, rng);
  std::unique_ptr<ModelTrace> trace = nullptr;
  trace = std::unique_ptr<ModelTrace>(
      new ModelTrace(*this, log_density, std::move(latents)));
  return {std::move(trace), log_weight};
}

template <typename RNGType>
float Model::generate(ModelTrace &trace, const EmptyChoiceBuffer &constraints,
                      RNGType &rng, ModelParameters &parameters,
                      const GenerateOptions &options) const {
  trace.model_ = *this;
  auto [log_density, log_weight] = importance_sample(trace.latents_, rng);
  trace.score_ = log_density;
  float score = logpdf(trace.latents_);
  trace.can_be_reverted_ = false;
  return log_weight;
}

template <typename RNG>
std::pair<int, float> Model::assess(RNG &, ModelParameters &parameters,
                                    const latent_choices_t &constraints) const {
  return {-1, logpdf(constraints)};
}

template <typename RNG>
std::pair<int, float>
Model::assess(RNG &, ModelParameters &parameters,
              const EmptyChoiceBuffer &constraints) const {
  return {-1, 0.0};
}

// ****************************
// *** Trace implementation ***
// ****************************

float ModelTrace::score() const { return score_; }

const latent_choices_t &
ModelTrace::choices(const LatentsSelection &selection) const {
  return latents_;
}

const latent_choices_t &ModelTrace::choices() const { return latents_; }

void ModelTrace::revert() {
  if (!can_be_reverted_)
    throw std::logic_error(
        "log_weight is only available between calls to update and revert");
  can_be_reverted_ = false;
  std::swap(latents_, alternate_latents_);
}

const latent_choices_t &ModelTrace::backward_constraints() {
  return alternate_latents_;
}

template <typename RNG>
float ModelTrace::update(RNG &, const gentl::change::NoChange &,
                         const latent_choices_t &latents,
                         const UpdateOptions &options) {
  if (options.save()) {
    std::swap(latents_, alternate_latents_);
    latents_ = latents; // copy assignment
    can_be_reverted_ = true;
  } else {
    latents_ = latents; // copy assignment
                        // can_be_reverted_ keeps its previous value
  };
  float new_log_density = model_.logpdf(latents_);
  float log_weight = new_log_density - score_;
  score_ = new_log_density;
  return log_weight;
}

// ****************************
// *** Proposal implementation ***
// ****************************

typedef float proposal_latent_choices_t;

class ProposalParameters {
public:
  float d_;
  ProposalParameters(float d) : d_{std::move(d)} {}
};

class ProposalTrace;

class Proposal {
  typedef int return_type;

  friend class ProposalTrace;

public:
  float current;

  template <typename RNGType>
  void exact_sample(float d, proposal_latent_choices_t &latents,
                    RNGType &rng) const {
    static std::uniform_real_distribution<float> uniform_real_dist(current - d,
                                                                   current + d);
    latents = uniform_real_dist(rng);
  }

  [[nodiscard]] float logpdf(const ProposalParameters &parameters, const proposal_latent_choices_t &latents) const {
      return 1 / (2.0 * parameters.d_);
  }

public:
  // simulate into a new trace object
  template <typename RNGType>
  std::unique_ptr<ProposalTrace> simulate(RNGType &rng,
                                          ProposalParameters &parameters,
                                          const SimulateOptions &) const;

  // simulate into an existing trace object (overwriting existing contents)
  template <typename RNGType>
  void simulate(RNGType &rng, ProposalParameters &parameters,
                const SimulateOptions &, ProposalTrace &trace) const;

  // equivalent to generate but without returning a trace
  template <typename RNG>
  std::pair<int, float>
  assess(RNG &, ProposalParameters &,
         const proposal_latent_choices_t &constraints) const;

  template <typename RNG>
  std::pair<int, float> assess(RNG &, ProposalParameters &,
                               const EmptyChoiceBuffer &constraints) const;
};

class ProposalTrace {
  friend class Proposal;

private:
  Proposal proposal_;
  ProposalParameters parameters_;
  float score_;
  proposal_latent_choices_t latents_;

private:
  ProposalTrace(Proposal proposal, ProposalParameters parameters, float score,
                proposal_latent_choices_t &&latents)
      : proposal_{std::move(proposal)},
        parameters_{std::move(parameters)}, score_{score}, latents_{latents} {}

public:
  ProposalTrace() = delete;
  ProposalTrace(const ProposalTrace &other) = delete;
  ProposalTrace(ProposalTrace &&other) = delete;
  ProposalTrace &operator=(const ProposalTrace &other) = delete;
  ProposalTrace &operator=(ProposalTrace &&other) noexcept = delete;

  [[nodiscard]] float score() const;
  [[nodiscard]] const proposal_latent_choices_t &choices() const;
  [[nodiscard]] const proposal_latent_choices_t &
  choices(const LatentsSelection &selection) const;

  const proposal_latent_choices_t &backward_constraints();

  void revert();
};

template <typename RNGType>
std::unique_ptr<ProposalTrace>
Proposal::simulate(RNGType &rng, ProposalParameters &parameters,
                   const SimulateOptions &options) const {
  proposal_latent_choices_t latents;
  exact_sample(parameters.d_, latents, rng);
  auto log_density = logpdf(parameters.d_, latents);
  return std::unique_ptr<ProposalTrace>(new ProposalTrace(
      *this, std::move(parameters), log_density, std::move(latents)));
}

template <typename RNGType>
void Proposal::simulate(RNGType &rng, ProposalParameters &parameters,
                        const SimulateOptions &options,
                        ProposalTrace &trace) const {
  exact_sample(parameters.d_, trace.latents_, rng);
  trace.score_ = logpdf(parameters.d_, trace.latents_);
}

template <typename RNG>
std::pair<int, float>
Proposal::assess(RNG &, ProposalParameters &parameters,
                 const proposal_latent_choices_t &constraints) const {
  return {-1, logpdf(parameters.d_, constraints)};
}

template <typename RNG>
std::pair<int, float>
Proposal::assess(RNG &, ProposalParameters &parameters,
                 const EmptyChoiceBuffer &constraints) const {
  return {-1, 0.0};
}

float ProposalTrace::score() const { return score_; }

const proposal_latent_choices_t &
ProposalTrace::choices(const LatentsSelection &selection) const {
  return latents_;
}

const proposal_latent_choices_t &ProposalTrace::choices() const {
  return latents_;
}

// *********************
// *** Example usage ***
// *********************

int main(int argc, char *argv[]) {
  using std::cerr;
  using std::cout;
  using std::endl;

  // initialize RNG
  unsigned int seed = 314159;
  gentl::randutils::seed_seq_fe128 seed_seq{seed};
  std::mt19937 rng(seed_seq);

  // define the model and proposal
  mean_t mean(0.0);
  cov_t target_covariance(1.0);
  ModelParameters unused{};
  Model model{mean, target_covariance};

  ProposalParameters proposal_parameters{0.25};
  Proposal proposal{};

  auto [trace, log_weight] =
      model.generate(EmptyChoiceBuffer{}, rng, unused, GenerateOptions());

  auto make_proposal = [&proposal](const ModelTrace &trace) {
    proposal.current = trace.choices();
    return proposal;
  };

  auto proposal_trace = make_proposal(*trace).simulate(rng, proposal_parameters,
                                                       SimulateOptions());

  auto num_iters = 10;
  auto mh_num_accepted = 0;
      
  std::ofstream timings("timings.csv",
                                std::ofstream::out);

  // Micro timing benchmarks.
  for (size_t outer = 1; outer < 1000; outer++) {
    auto start = high_resolution_clock::now();
    for (size_t iter = 2; iter < num_iters; iter++) {
      mh_num_accepted +=
          gentl::mcmc::mh(rng, *trace, make_proposal, proposal_parameters,
                          *proposal_trace, false);
    }
    auto stop = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<nanoseconds>(stop - start);
    timings << duration.count() << "," << endl;

  }
  return mh_num_accepted;
}
