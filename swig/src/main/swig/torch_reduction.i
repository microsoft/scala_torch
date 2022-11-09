%javaconst(1) Reduction;
namespace Reduction {
enum Reduction {
  None,             // Do not reduce
  Mean,             // (Possibly weighted) mean of losses
  Sum//,            // Sum losses
  //END
};
}

// torch is moving to this type, which is hard to map to Java, so we use the above enum in Java land and map it back
%naturalvar default_reduction_t;
%inline %{
typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> default_reduction_t;
%}


%{
default_reduction_t to_default_reduction_t(unsigned r) {
  switch(r) {
  case Reduction::None: return torch::kNone; break;
  case Reduction::Mean: return torch::kMean; break;
  case Reduction::Sum: return torch::kSum; break;
  default: throw std::invalid_argument("Bad argument for Reduction");
  }
}

%}

VARIANT_ENUM(default_reduction_t, Reduction, to_default_reduction_t)
