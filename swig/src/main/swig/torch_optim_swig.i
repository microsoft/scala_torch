// optimization include for torch_swig

%{
using namespace torch::nn;
#include <torch/nn/utils.h>
%}

namespace torch {
namespace nn {
namespace utils {
inline double clip_grad_norm_(
  std::vector<torch::Tensor> parameters,
  double max_norm,
  double norm_type = 2.0);
} // namespace utils
} // namespace nn
} // namespace torch

namespace torch {
namespace optim {

  class TORCH_API OptimizerOptions {
  };

  // TODO handle param groups better.
  // A note about options: Torch has switched to using parameters groups, each with their own options
  // instead of having one set of parameters per Optimizer. There is backwards compatibility of sorts
  // for the one parameter setup: there is still a constructor for each optimizer that takes a single param
  // list and uses the default options. So, each optimizer implementation exposes its options
  // by calling through to the defaults.

  class TORCH_API OptimizerParamGroup {
   public:
    OptimizerParamGroup(const OptimizerParamGroup& param_group);
    OptimizerParamGroup(std::vector<torch::Tensor> params);

    bool has_options() const;
    OptimizerOptions& options();
    std::vector<torch::Tensor>& params();
  };

  struct Optimizer {
    Optimizer() = delete;
    virtual void step();
    virtual void zero_grad();
    // std::vector doesn't work out of the box, so we'll hack around it
    //const std::vector<OptimizerParamGroup>&  param_groups() const noexcept;
    %extend {
    int num_param_groups() const {
      return $self->param_groups().size();
    }
    OptimizerParamGroup& param_group(int i) {
      return $self->param_groups()[i];
    }
    }

    %extend {
    // Same as Optimizer::parameters, which is going to be deleted in 1.6
    // Remove if we handel param groups.
    const std::vector<torch::Tensor>& all_parameters() const noexcept {
      return $self->param_groups().at(0).params();
    }
    }
  };


struct SGDOptions : public OptimizerOptions {
  /* implicit */ SGDOptions(double lr);
  TORCH_ARG(SGDOptions, double, lr);
  TORCH_ARG(SGDOptions, double, momentum);
  TORCH_ARG(SGDOptions, double, dampening);
  TORCH_ARG(SGDOptions, double, weight_decay);
  TORCH_ARG(SGDOptions, bool, nesterov);
  %extend {
  // TODO make a macro or find some otherway of generalizing this casting
  static SGDOptions& cast(OptimizerOptions& opts) {
    return static_cast<torch::optim::SGDOptions&>(opts);
  }
  }
};

struct SGD: Optimizer {
  SGD(std::vector<torch::Tensor> parameters, const SGDOptions& options_);
  %extend {
  SGDOptions& getOptions() { return static_cast<torch::optim::SGDOptions&>($self->defaults()); }
  }
};

struct AdagradOptions : public OptimizerOptions {
  /* implicit */ AdagradOptions(double lr);
  TORCH_ARG(AdagradOptions, double, lr);
  TORCH_ARG(AdagradOptions, double, lr_decay);
  TORCH_ARG(AdagradOptions, double, weight_decay);
  %extend {
  // TODO make a macro or find some otherway of generalizing this casting
  static AdagradOptions& cast(OptimizerOptions& opts) {
    return static_cast<torch::optim::AdagradOptions&>(opts);
  }
  }
};

struct Adagrad: Optimizer {
  Adagrad(std::vector<torch::Tensor> parameters, const AdagradOptions& options_);
  %extend {
  AdagradOptions& getOptions() { return static_cast<torch::optim::AdagradOptions&>($self->defaults()); }
  }
};

struct AdamOptions : public OptimizerOptions {
  /* implicit */ AdamOptions(double lr);
  TORCH_ARG(AdamOptions, double, lr);
  %extend {
  double beta1() const { return std::get<0>($self->betas()); } // 0.9
  AdamOptions beta1(double v) { return $self->betas(std::make_tuple(v, std::get<1>($self->betas()))); }
  double beta2() const { return std::get<1>($self->betas()); } // 0.999
  AdamOptions beta2(double v) { return $self->betas(std::make_tuple(std::get<0>($self->betas()), v)); }
  }
  TORCH_ARG(AdamOptions, double, weight_decay); // 0
  TORCH_ARG(AdamOptions, double, eps); // 1E-8
  TORCH_ARG(AdamOptions, bool, amsgrad) // false;
  %extend {
  // TODO make a macro or find some otherway of generalizing this casting
  static AdamOptions& cast(OptimizerOptions& opts) {
    return static_cast<torch::optim::AdamOptions&>(opts);
  }
  }
};

struct Adam: Optimizer {
  Adam(std::vector<torch::Tensor> parameters, const AdamOptions& options_);
  %extend {
  AdamOptions& getOptions()  {
    return static_cast<torch::optim::AdamOptions&>($self->defaults());
  }
  }
};

struct RMSpropOptions : public OptimizerOptions {
  /* implicit */ RMSpropOptions(double lr);
  TORCH_ARG(RMSpropOptions, double, lr);
  TORCH_ARG(RMSpropOptions, double, alpha); // 0.99
  TORCH_ARG(RMSpropOptions, double, eps); // 1E-8
  TORCH_ARG(RMSpropOptions, double, weight_decay); // 0
  TORCH_ARG(RMSpropOptions, double, momentum); // 0
  TORCH_ARG(RMSpropOptions, bool, centered) // false;
  %extend {
  // TODO make a macro or find some otherway of generalizing this casting
  static RMSpropOptions& cast(OptimizerOptions& opts) {
    return static_cast<torch::optim::RMSpropOptions&>(opts);
  }
  }
};

struct RMSprop: Optimizer {
  RMSprop(std::vector<torch::Tensor> parameters, const RMSpropOptions& options_);
  %extend {
  RMSpropOptions& getOptions() {
    return static_cast<torch::optim::RMSpropOptions&>($self->defaults());
  }
  }
};

} // namespace  optim
}  // namespace torch
