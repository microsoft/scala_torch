#include <torch/csrc/autograd/profiler.h>
namespace torch {
namespace autograd {
namespace profiler {

struct TORCH_API RecordProfile {
  RecordProfile(const std::string& filename);

  ~RecordProfile();
};
} // namespace profiler
} // namespace autograd
} // namespace torch
