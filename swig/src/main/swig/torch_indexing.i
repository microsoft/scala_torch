namespace at {
namespace indexing {
struct TORCH_API EllipsisIndexType final { EllipsisIndexType(); };

struct TORCH_API Slice final {
  Slice(
    c10::optional<int64_t> start_index = c10::nullopt,
    c10::optional<int64_t> stop_index = c10::nullopt,
    c10::optional<int64_t> step_index = c10::nullopt);
};

struct TORCH_API TensorIndex final {

  // Case 1: `at::indexing::None`
  //TensorIndex(c10::nullopt_t);

  // Case 2: "..." / `at::indexing::Ellipsis`
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  TensorIndex(EllipsisIndexType);

  // Case 3: Integer value
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  TensorIndex(int64_t integer);

  // Case 4: Boolean value
  TensorIndex(bool boolean);

  // Case 5: Slice represented in `at::indexing::Slice` form
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  TensorIndex(Slice slice);

  // Case 6: Tensor value
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  TensorIndex(Tensor tensor);
};
}
}

ARRAY_REF_OF_OBJECT(TensorIndexArrayRef, at::indexing::TensorIndex)
