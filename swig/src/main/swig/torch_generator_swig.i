%{
#include <ATen/CUDAGeneratorImpl.h>
%}

%ignore at::Generator::mutex_;

struct TORCH_API Generator {
  Generator();


  Device device() const;
  torch::Tensor get_state() const;
  void set_state(const torch::Tensor& new_state);
  int64_t seed();

%extend {
  void manual_seed(int64_t seed) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock($self->mutex());
    $self->set_current_seed(seed);
  }

  int64_t initial_seed() const {
    return $self->current_seed();
  }
}
};

%extend Generator {
  explicit Generator(c10::Device device) {
    if (device.type() == at::kCPU) {
       return new Generator(c10::make_intrusive<at::CPUGeneratorImpl>(device.index()));
#ifdef USE_CUDA
     } else if (device.type() == at::kCUDA) {
       return new Generator(c10::make_intrusive<at::CUDAGeneratorImpl>(device.index()));
#endif
     } else {
       AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
                 " is not supported for torch.Generator() api.");
     }
  }
}

DEFINE_OPTIONAL(OptGenerator, Generator)

