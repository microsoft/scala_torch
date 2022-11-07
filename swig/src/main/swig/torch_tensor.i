namespace torch {

%include "generated_tensor_bindings.i"
%include "generated_bindings.i"
TO_STRING_FROM_OSTREAM(Tensor);
}

// These need to be manually defined because of namespace collisions.
// I(adpauls) don't totally understand what's going on, but there are torch:: and at:: versions
// of all of the ATen functions. The torch:: ones should always take precedence, but in some cases
// there's a need to fallback to at:: Mysteriously, there are some clashes just for normal we can't
// invoke torch::normal explicitly
torch::Tensor normal(const torch::Tensor & mean, double std, c10::optional<Generator> generator);
torch::Tensor normal(double mean, const torch::Tensor & std, c10::optional<Generator> generator);
torch::Tensor normal(const torch::Tensor & mean, const torch::Tensor & std, c10::optional<Generator> generator);
torch::Tensor normal(double mean, double std, IntArrayRef size, c10::optional<Generator> generator, TensorOptions options);

