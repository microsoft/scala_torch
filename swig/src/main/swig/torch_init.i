%javaconst(1) FanMode;
%inline %{
namespace FanMode {
enum FanMode {
  FanIn,
  FanOut
};
}
%}

typedef c10::variant<torch::enumtype::kFanIn, torch::enumtype::kFanOut, torch::enumtype::kSum> FanModeType;

%{
FanModeType to_fan_mode_t(unsigned r) {
  switch(r) {
    case 0: return torch::enumtype::kFanIn(); break;
    case 1: return torch::enumtype::kFanOut(); break;
    default: throw std::invalid_argument("Bad argument for FanMode");
  }
}

%}

VARIANT_ENUM(FanModeType, FanMode, to_fan_mode_t)


%javaconst(1) Nonlinearity;
%inline %{
namespace Nonlinearity {
enum Nonlinearity {
  Linear,
  Conv1D,
  Conv2D,
  Conv3D,
  ConvTranspose1D,
  ConvTranspose2D,
  ConvTranspose3D,
  Sigmoid,
  Tanh,
  ReLU,
  LeakyReLU
};
}
%}

typedef c10::variant<torch::enumtype::kLinear,
                     torch::enumtype::kConv1D,
                     torch::enumtype::kConv2D,
                     torch::enumtype::kConv3D,
                     torch::enumtype::kConvTranspose1D,
                     torch::enumtype::kConvTranspose2D,
                     torch::enumtype::kConvTranspose3D,
                     torch::enumtype::kSigmoid,
                     torch::enumtype::kTanh,
                     torch::enumtype::kReLU,
                     torch::enumtype::kLeakyReLU> NonlinearityType;

%{
NonlinearityType to_nonlinearity_t(unsigned r) {
  switch(r) {
  case 0: return torch::enumtype::kLinear(); break;
  case 1: return torch::enumtype::kConv1D(); break;
  case 2: return torch::enumtype::kConv2D(); break;
  case 3: return torch::enumtype::kConv3D(); break;
  case 4: return torch::enumtype::kConvTranspose1D(); break;
  case 5: return torch::enumtype::kConvTranspose2D(); break;
  case 6: return torch::enumtype::kConvTranspose3D(); break;
  case 7: return torch::enumtype::kSigmoid(); break;
  case 8: return torch::enumtype::kTanh(); break;
  case 9: return torch::enumtype::kReLU(); break;
  case 10: return torch::enumtype::kLeakyReLU(); break;
  default: throw std::invalid_argument("Bad argument for Nonlinearity");
  }
}

%}

VARIANT_ENUM(NonlinearityType, Nonlinearity, to_nonlinearity_t)



namespace torch {
namespace nn {
namespace init {
/// Return the recommended gain value for the given nonlinearity function.
TORCH_API double calculate_gain(NonlinearityType nonlinearity, double param = 0.01);

TORCH_API void constant_(torch::Tensor tensor, torch::Scalar value);

TORCH_API void dirac_(torch::Tensor tensor);

TORCH_API void eye_(torch::Tensor matrix);

TORCH_API void normal_(torch::Tensor tensor, double mean = 0, double std = 1);

TORCH_API void ones_(torch::Tensor tensor);

TORCH_API void orthogonal_(torch::Tensor tensor, double gain = 1.0);

TORCH_API void sparse_(torch::Tensor tensor, double sparsity, double std = 0.01);

TORCH_API void uniform_(torch::Tensor tensor, double low = 0, double high = 1);

TORCH_API void kaiming_normal_(
    torch::Tensor tensor,
    double a = 0,
    FanModeType mode = torch::kFanIn,
    NonlinearityType nonlinearity = torch::kLeakyReLU);

TORCH_API void kaiming_uniform_(
    torch::Tensor tensor,
    double a = 0,
    FanModeType mode = torch::kFanIn,
    NonlinearityType nonlinearity = torch::kLeakyReLU);

TORCH_API void xavier_normal_(torch::Tensor tensor, double gain = 1.0);

TORCH_API void xavier_uniform_(torch::Tensor tensor, double gain = 1.0);

TORCH_API void zeros_(torch::Tensor tensor);
} // namespace init
} // namespace nn
} // namespace torch
