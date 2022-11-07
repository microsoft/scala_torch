
struct Tensor {
  Tensor();

  %rename(operator_equals) operator=;
  void operator=(Tensor& src);
  
  Tensor index(ArrayRef<at::indexing::TensorIndex> indices) const;
  void index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs);

  size_t nbytes() const;
  int64_t numel() const;
  size_t itemsize() const;

  Layout layout() const noexcept;
  TypeMeta dtype() const noexcept;
  Device device() const;

  void backward();
  Tensor& grad();

  bool requires_grad() const;

  bool defined() const;

  %extend {
    // Sometimes we need to explicitly return a (shallow) copy to get reference counting right.
    Tensor refCopy() const {
      return *($self);
    }
  }

  IntArrayRef sizes() const;

  template<class T> T item() const;
  %template(toFloat) item<float>;
// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT
// See build.sbt for details
void _backward(TensorList inputs, const c10::optional<Tensor> & gradient, c10::optional<bool> retain_graph, bool create_graph) const;
  void set_data(const Tensor & new_data) const;
  Tensor data() const;
  bool is_leaf() const;
  int64_t output_nr() const;
  int64_t _version() const;
  void requires_grad_(bool requires_grad);
  void retain_grad() const;
  bool retains_grad() const;
  Tensor _fw_primal(int64_t level) const;
  Tensor align_as(const Tensor & other) const;
  Tensor abs() const;
  void abs_();
  Tensor absolute() const;
  void absolute_();
  Tensor angle() const;
  Tensor sgn() const;
  void sgn_();
  Tensor _conj() const;
  Tensor conj() const;
  Tensor _conj_physical() const;
  Tensor conj_physical() const;
  void conj_physical_();
  Tensor resolve_conj() const;
  Tensor resolve_neg() const;
  Tensor _neg_view() const;
  Tensor acos() const;
  void acos_();
  Tensor arccos() const;
  void arccos_();
  Tensor add(const Tensor & other, const Scalar & alpha) const;
  void add_(const Tensor & other, const Scalar & alpha);
  Tensor add(const Scalar & other, const Scalar & alpha) const;
  void add_(const Scalar & other, const Scalar & alpha);
  Tensor addmv(const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) const;
  void addmv_(const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha);
  Tensor addr(const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) const;
  void addr_(const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha);
  Tensor all(int64_t dim, bool keepdim) const;
  bool allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const;
  Tensor any(int64_t dim, bool keepdim) const;
  Tensor argmax(c10::optional<int64_t> dim, bool keepdim) const;
  Tensor argmin(c10::optional<int64_t> dim, bool keepdim) const;
  Tensor acosh() const;
  void acosh_();
  Tensor arccosh() const;
  void arccosh_();
  Tensor asinh() const;
  void asinh_();
  Tensor arcsinh() const;
  void arcsinh_();
  Tensor atanh() const;
  void atanh_();
  Tensor arctanh() const;
  void arctanh_();
  Tensor as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const;
  void as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset);
  Tensor asin() const;
  void asin_();
  Tensor arcsin() const;
  void arcsin_();
  Tensor atan() const;
  void atan_();
  Tensor arctan() const;
  void arctan_();
  Tensor baddbmm(const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) const;
  void baddbmm_(const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);
  Tensor bernoulli(c10::optional<Generator> generator) const;
  void bernoulli_(const Tensor & p, c10::optional<Generator> generator);
  void bernoulli_(double p, c10::optional<Generator> generator);
  Tensor bernoulli(double p, c10::optional<Generator> generator) const;
  Tensor bincount(const c10::optional<Tensor> & weights, int64_t minlength) const;
  Tensor bitwise_not() const;
  void bitwise_not_();
  Tensor copysign(const Tensor & other) const;
  void copysign_(const Tensor & other);
  Tensor copysign(const Scalar & other) const;
  void copysign_(const Scalar & other);
  Tensor logical_not() const;
  void logical_not_();
  Tensor logical_xor(const Tensor & other) const;
  void logical_xor_(const Tensor & other);
  Tensor logical_and(const Tensor & other) const;
  void logical_and_(const Tensor & other);
  Tensor logical_or(const Tensor & other) const;
  void logical_or_(const Tensor & other);
  Tensor bmm(const Tensor & mat2) const;
  Tensor broadcast_to(IntArrayRef size) const;
  Tensor ceil() const;
  void ceil_();
  std::vector<Tensor> unsafe_chunk(int64_t chunks, int64_t dim) const;
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim) const;
  std::vector<Tensor> tensor_split(int64_t sections, int64_t dim) const;
  std::vector<Tensor> tensor_split(IntArrayRef indices, int64_t dim) const;
  std::vector<Tensor> tensor_split(const Tensor & tensor_indices_or_sections, int64_t dim) const;
  Tensor clamp(const c10::optional<Scalar> & min, const c10::optional<Scalar> & max) const;
  void clamp_(const c10::optional<Scalar> & min, const c10::optional<Scalar> & max);
  Tensor clamp_max(const Scalar & max) const;
  Tensor clamp_max(const Tensor & max) const;
  void clamp_max_(const Scalar & max);
  void clamp_max_(const Tensor & max);
  Tensor clamp_min(const Scalar & min) const;
  Tensor clamp_min(const Tensor & min) const;
  void clamp_min_(const Scalar & min);
  void clamp_min_(const Tensor & min);
  Tensor clip(const c10::optional<Scalar> & min, const c10::optional<Scalar> & max) const;
  void clip_(const c10::optional<Scalar> & min, const c10::optional<Scalar> & max);
  Tensor contiguous(MemoryFormat memory_format) const;
  void copy_(const Tensor & src, bool non_blocking);
  Tensor cos() const;
  void cos_();
  Tensor cosh() const;
  void cosh_();
  Tensor count_nonzero(IntArrayRef dim) const;
  Tensor count_nonzero(c10::optional<int64_t> dim) const;
  Tensor cov(int64_t correction, const c10::optional<Tensor> & fweights, const c10::optional<Tensor> & aweights) const;
  Tensor corrcoef() const;
  tuple2<Tensor, Tensor> cummax(int64_t dim) const;
  tuple2<Tensor, Tensor> cummin(int64_t dim) const;
  Tensor cumprod(int64_t dim, c10::optional<ScalarType> dtype) const;
  void cumprod_(int64_t dim, c10::optional<ScalarType> dtype);
  Tensor cumsum(int64_t dim, c10::optional<ScalarType> dtype) const;
  void cumsum_(int64_t dim, c10::optional<ScalarType> dtype);
  Tensor diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const;
  Tensor diagflat(int64_t offset) const;
  Tensor diagonal(int64_t offset, int64_t dim1, int64_t dim2) const;
  void fill_diagonal_(const Scalar & fill_value, bool wrap);
  Tensor diff(int64_t n, int64_t dim, const c10::optional<Tensor> & prepend, const c10::optional<Tensor> & append) const;
  Tensor div(const Tensor & other) const;
  void div_(const Tensor & other);
  Tensor div(const Tensor & other, c10::optional<c10::string_view> rounding_mode) const;
  void div_(const Tensor & other, c10::optional<c10::string_view> rounding_mode);
  Tensor div(const Scalar & other) const;
  void div_(const Scalar & other);
  Tensor div(const Scalar & other, c10::optional<c10::string_view> rounding_mode) const;
  void div_(const Scalar & other, c10::optional<c10::string_view> rounding_mode);
  Tensor divide(const Tensor & other) const;
  void divide_(const Tensor & other);
  Tensor divide(const Scalar & other) const;
  void divide_(const Scalar & other);
  Tensor divide(const Tensor & other, c10::optional<c10::string_view> rounding_mode) const;
  void divide_(const Tensor & other, c10::optional<c10::string_view> rounding_mode);
  Tensor divide(const Scalar & other, c10::optional<c10::string_view> rounding_mode) const;
  void divide_(const Scalar & other, c10::optional<c10::string_view> rounding_mode);
  Tensor true_divide(const Tensor & other) const;
  void true_divide_(const Tensor & other);
  Tensor true_divide(const Scalar & other) const;
  void true_divide_(const Scalar & other);
  Tensor dot(const Tensor & tensor) const;
  Tensor vdot(const Tensor & other) const;
  Tensor new_empty(IntArrayRef size, TensorOptions options) const;
  Tensor new_empty_strided(IntArrayRef size, IntArrayRef stride, TensorOptions options) const;
  Tensor new_full(IntArrayRef size, const Scalar & fill_value, TensorOptions options) const;
  Tensor new_zeros(IntArrayRef size, TensorOptions options) const;
  Tensor new_ones(IntArrayRef size, TensorOptions options) const;
  void resize_(IntArrayRef size, c10::optional<MemoryFormat> memory_format);
  Tensor erf() const;
  void erf_();
  Tensor erfc() const;
  void erfc_();
  Tensor exp() const;
  void exp_();
  Tensor exp2() const;
  void exp2_();
  Tensor expm1() const;
  void expm1_();
  Tensor expand(IntArrayRef size, bool implicit) const;
  Tensor expand_as(const Tensor & other) const;
  Tensor flatten(int64_t start_dim, int64_t end_dim) const;
  void fill_(const Scalar & value);
  void fill_(const Tensor & value);
  Tensor floor() const;
  void floor_();
  Tensor floor_divide(const Tensor & other) const;
  void floor_divide_(const Tensor & other);
  Tensor floor_divide(const Scalar & other) const;
  void floor_divide_(const Scalar & other);
  Tensor frac() const;
  void frac_();
  Tensor gcd(const Tensor & other) const;
  void gcd_(const Tensor & other);
  Tensor lcm(const Tensor & other) const;
  void lcm_(const Tensor & other);
  Tensor index(const c10::List<c10::optional<Tensor>> & indices) const;
  void index_copy_(int64_t dim, const Tensor & index, const Tensor & source);
  Tensor index_copy(int64_t dim, const Tensor & index, const Tensor & source) const;
  void index_put_(const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate);
  Tensor index_put(const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) const;
  Tensor inverse() const;
  Tensor isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const;
  Tensor isnan() const;
  bool is_distributed() const;
  bool is_floating_point() const;
  bool is_complex() const;
  bool is_conj() const;
  bool is_neg() const;
  Tensor isreal() const;
  bool is_nonzero() const;
  bool is_same_size(const Tensor & other) const;
  bool is_signed() const;
  bool is_inference() const;
  Tensor kron(const Tensor & other) const;
  tuple2<Tensor, Tensor> kthvalue(int64_t k, int64_t dim, bool keepdim) const;
  Tensor nan_to_num(c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) const;
  void nan_to_num_(c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf);
  Tensor ldexp(const Tensor & other) const;
  void ldexp_(const Tensor & other);
  Tensor log() const;
  void log_();
  Tensor log10() const;
  void log10_();
  Tensor log1p() const;
  void log1p_();
  Tensor log2() const;
  void log2_();
  Tensor logaddexp(const Tensor & other) const;
  Tensor logaddexp2(const Tensor & other) const;
  Tensor xlogy(const Tensor & other) const;
  Tensor xlogy(const Scalar & other) const;
  void xlogy_(const Tensor & other);
  void xlogy_(const Scalar & other);
  Tensor logdet() const;
  Tensor log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const;
  Tensor logcumsumexp(int64_t dim) const;
  Tensor logsumexp(IntArrayRef dim, bool keepdim) const;
  Tensor matmul(const Tensor & other) const;
  Tensor matrix_power(int64_t n) const;
  Tensor matrix_exp() const;
  tuple2<Tensor, Tensor> aminmax(c10::optional<int64_t> dim, bool keepdim) const;
  tuple2<Tensor, Tensor> max(int64_t dim, bool keepdim) const;
  Tensor amax(IntArrayRef dim, bool keepdim) const;
  Tensor mean(c10::optional<ScalarType> dtype) const;
  Tensor mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const;
  Tensor nanmean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const;
  Tensor median() const;
  tuple2<Tensor, Tensor> median(int64_t dim, bool keepdim) const;
  Tensor nanmedian() const;
  tuple2<Tensor, Tensor> nanmedian(int64_t dim, bool keepdim) const;
  tuple2<Tensor, Tensor> min(int64_t dim, bool keepdim) const;
  Tensor amin(IntArrayRef dim, bool keepdim) const;
  Tensor mm(const Tensor & mat2) const;
  tuple2<Tensor, Tensor> mode(int64_t dim, bool keepdim) const;
  Tensor mul(const Tensor & other) const;
  void mul_(const Tensor & other);
  Tensor mul(const Scalar & other) const;
  void mul_(const Scalar & other);
  Tensor multiply(const Tensor & other) const;
  void multiply_(const Tensor & other);
  Tensor multiply(const Scalar & other) const;
  void multiply_(const Scalar & other);
  Tensor mv(const Tensor & vec) const;
  Tensor mvlgamma(int64_t p) const;
  void mvlgamma_(int64_t p);
  Tensor narrow_copy(int64_t dim, int64_t start, int64_t length) const;
  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
  Tensor narrow(int64_t dim, const Tensor & start, int64_t length) const;
  Tensor permute(IntArrayRef dims) const;
  Tensor movedim(IntArrayRef source, IntArrayRef destination) const;
  Tensor movedim(int64_t source, int64_t destination) const;
  Tensor moveaxis(IntArrayRef source, IntArrayRef destination) const;
  Tensor moveaxis(int64_t source, int64_t destination) const;
  Tensor numpy_T() const;
  bool is_pinned(c10::optional<Device> device) const;
  Tensor pin_memory(c10::optional<Device> device) const;
  Tensor pinverse(double rcond) const;
  Tensor rad2deg() const;
  void rad2deg_();
  Tensor deg2rad() const;
  void deg2rad_();
  Tensor ravel() const;
  Tensor reciprocal() const;
  void reciprocal_();
  Tensor neg() const;
  void neg_();
  Tensor negative() const;
  void negative_();
  Tensor repeat(IntArrayRef repeats) const;
  Tensor repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) const;
  Tensor repeat_interleave(int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) const;
  Tensor reshape(IntArrayRef shape) const;
  Tensor _reshape_alias(IntArrayRef size, IntArrayRef stride) const;
  Tensor reshape_as(const Tensor & other) const;
  Tensor round() const;
  void round_();
  Tensor relu() const;
  void relu_();
  Tensor prelu(const Tensor & weight) const;
  tuple2<Tensor, Tensor> prelu_backward(const Tensor & grad_output, const Tensor & weight) const;
  Tensor hardshrink(const Scalar & lambd) const;
  Tensor hardshrink_backward(const Tensor & grad_out, const Scalar & lambd) const;
  Tensor rsqrt() const;
  void rsqrt_();
  Tensor select(int64_t dim, int64_t index) const;
  Tensor sigmoid() const;
  void sigmoid_();
  Tensor logit(c10::optional<double> eps) const;
  void logit_(c10::optional<double> eps);
  Tensor sin() const;
  void sin_();
  Tensor sinc() const;
  void sinc_();
  Tensor sinh() const;
  void sinh_();
  Tensor detach() const;
  void detach_();
  Tensor slice(int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) const;
  tuple2<Tensor, Tensor> slogdet() const;
  Tensor smm(const Tensor & mat2) const;
  Tensor softmax(int64_t dim, c10::optional<ScalarType> dtype) const;
  std::vector<Tensor> unsafe_split(int64_t split_size, int64_t dim) const;
  std::vector<Tensor> split(int64_t split_size, int64_t dim) const;
  std::vector<Tensor> unsafe_split_with_sizes(IntArrayRef split_sizes, int64_t dim) const;
  std::vector<Tensor> split_with_sizes(IntArrayRef split_sizes, int64_t dim) const;
  std::vector<Tensor> hsplit(int64_t sections) const;
  std::vector<Tensor> hsplit(IntArrayRef indices) const;
  std::vector<Tensor> vsplit(int64_t sections) const;
  std::vector<Tensor> vsplit(IntArrayRef indices) const;
  std::vector<Tensor> dsplit(int64_t sections) const;
  std::vector<Tensor> dsplit(IntArrayRef indices) const;
  Tensor squeeze() const;
  Tensor squeeze(int64_t dim) const;
  void squeeze_();
  void squeeze_(int64_t dim);
  Tensor sspaddmm(const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) const;
  Tensor stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) const;
  Tensor istft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) const;
  Tensor sum(c10::optional<ScalarType> dtype) const;
  Tensor sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const;
  Tensor nansum(c10::optional<ScalarType> dtype) const;
  Tensor nansum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const;
  Tensor sum_to_size(IntArrayRef size) const;
  Tensor sqrt() const;
  void sqrt_();
  Tensor square() const;
  void square_();
  Tensor std(bool unbiased) const;
  Tensor std(IntArrayRef dim, bool unbiased, bool keepdim) const;
  Tensor std(c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) const;
  Tensor prod(c10::optional<ScalarType> dtype) const;
  Tensor prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) const;
  Tensor t() const;
  void t_();
  Tensor tan() const;
  void tan_();
  Tensor tanh() const;
  void tanh_();
  Tensor tile(IntArrayRef dims) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  void transpose_(int64_t dim0, int64_t dim1);
  Tensor flip(IntArrayRef dims) const;
  Tensor fliplr() const;
  Tensor flipud() const;
  Tensor roll(IntArrayRef shifts, IntArrayRef dims) const;
  Tensor rot90(int64_t k, IntArrayRef dims) const;
  Tensor trunc() const;
  void trunc_();
  Tensor fix() const;
  void fix_();
  Tensor type_as(const Tensor & other) const;
  Tensor unsqueeze(int64_t dim) const;
  void unsqueeze_(int64_t dim);
  Tensor var(bool unbiased) const;
  Tensor var(IntArrayRef dim, bool unbiased, bool keepdim) const;
  Tensor var(c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) const;
  Tensor view_as(const Tensor & other) const;
  Tensor where(const Tensor & condition, const Tensor & other) const;
  Tensor norm(const c10::optional<Scalar> & p, ScalarType dtype) const;
  Tensor norm(const Scalar & p) const;
  Tensor norm(const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype) const;
  Tensor norm(const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim) const;
  tuple2<Tensor, Tensor> frexp() const;
  Tensor clone(c10::optional<MemoryFormat> memory_format) const;
  Tensor positive() const;
  void resize_as_(const Tensor & the_template, c10::optional<MemoryFormat> memory_format);
  void zero_();
  Tensor sub(const Tensor & other, const Scalar & alpha) const;
  void sub_(const Tensor & other, const Scalar & alpha);
  Tensor sub(const Scalar & other, const Scalar & alpha) const;
  void sub_(const Scalar & other, const Scalar & alpha);
  Tensor subtract(const Tensor & other, const Scalar & alpha) const;
  void subtract_(const Tensor & other, const Scalar & alpha);
  Tensor subtract(const Scalar & other, const Scalar & alpha) const;
  void subtract_(const Scalar & other, const Scalar & alpha);
  Tensor heaviside(const Tensor & values) const;
  void heaviside_(const Tensor & values);
  Tensor addmm(const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) const;
  void addmm_(const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha);
  void sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  void sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  Tensor sparse_mask(const Tensor & mask) const;
  Tensor to_dense(c10::optional<ScalarType> dtype) const;
  int64_t sparse_dim() const;
  int64_t _dimI() const;
  int64_t dense_dim() const;
  int64_t _dimV() const;
  int64_t _nnz() const;
  Tensor coalesce() const;
  bool is_coalesced() const;
  Tensor _indices() const;
  Tensor _values() const;
  void _coalesced_(bool coalesced);
  Tensor indices() const;
  Tensor values() const;
  Tensor crow_indices() const;
  Tensor col_indices() const;
  std::vector<Tensor> unbind(int64_t dim) const;
  Tensor to_sparse(int64_t sparse_dim) const;
  Tensor to_sparse() const;
  Tensor to_mkldnn(c10::optional<ScalarType> dtype) const;
  Tensor dequantize() const;
  double q_scale() const;
  int64_t q_zero_point() const;
  Tensor q_per_channel_scales() const;
  Tensor q_per_channel_zero_points() const;
  int64_t q_per_channel_axis() const;
  Tensor int_repr() const;
  QScheme qscheme() const;
  Tensor to(TensorOptions options, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const;
  Tensor to(Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const;
  Tensor to(ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const;
  Tensor to(const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) const;
  Scalar item() const;
  void set_(const Tensor & source);
  void set_();
  bool is_set_to(const Tensor & tensor) const;
  void masked_fill_(const Tensor & mask, const Scalar & value);
  Tensor masked_fill(const Tensor & mask, const Scalar & value) const;
  void masked_fill_(const Tensor & mask, const Tensor & value);
  Tensor masked_fill(const Tensor & mask, const Tensor & value) const;
  void masked_scatter_(const Tensor & mask, const Tensor & source);
  Tensor masked_scatter(const Tensor & mask, const Tensor & source) const;
  Tensor view(IntArrayRef size) const;
  Tensor view(ScalarType dtype) const;
  void put_(const Tensor & index, const Tensor & source, bool accumulate);
  Tensor put(const Tensor & index, const Tensor & source, bool accumulate) const;
  void index_add_(int64_t dim, const Tensor & index, const Tensor & source);
  void index_add_(int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha);
  Tensor index_add(int64_t dim, const Tensor & index, const Tensor & source) const;
  Tensor index_add(int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha) const;
  void index_fill_(int64_t dim, const Tensor & index, const Scalar & value);
  Tensor index_fill(int64_t dim, const Tensor & index, const Scalar & value) const;
  void index_fill_(int64_t dim, const Tensor & index, const Tensor & value);
  Tensor index_fill(int64_t dim, const Tensor & index, const Tensor & value) const;
  Tensor scatter(int64_t dim, const Tensor & index, const Tensor & src) const;
  void scatter_(int64_t dim, const Tensor & index, const Tensor & src);
  Tensor scatter(int64_t dim, const Tensor & index, const Scalar & value) const;
  void scatter_(int64_t dim, const Tensor & index, const Scalar & value);
  Tensor scatter(int64_t dim, const Tensor & index, const Tensor & src, c10::string_view reduce) const;
  void scatter_(int64_t dim, const Tensor & index, const Tensor & src, c10::string_view reduce);
  Tensor scatter(int64_t dim, const Tensor & index, const Scalar & value, c10::string_view reduce) const;
  void scatter_(int64_t dim, const Tensor & index, const Scalar & value, c10::string_view reduce);
  Tensor scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const;
  void scatter_add_(int64_t dim, const Tensor & index, const Tensor & src);
  void eq_(const Scalar & other);
  void eq_(const Tensor & other);
  Tensor bitwise_and(const Scalar & other) const;
  Tensor bitwise_and(const Tensor & other) const;
  void bitwise_and_(const Scalar & other);
  void bitwise_and_(const Tensor & other);
  Tensor __and__(const Scalar & other) const;
  Tensor __and__(const Tensor & other) const;
  void __iand__(const Scalar & other);
  void __iand__(const Tensor & other);
  Tensor bitwise_or(const Scalar & other) const;
  Tensor bitwise_or(const Tensor & other) const;
  void bitwise_or_(const Scalar & other);
  void bitwise_or_(const Tensor & other);
  Tensor __or__(const Scalar & other) const;
  Tensor __or__(const Tensor & other) const;
  void __ior__(const Scalar & other);
  void __ior__(const Tensor & other);
  Tensor bitwise_xor(const Scalar & other) const;
  Tensor bitwise_xor(const Tensor & other) const;
  void bitwise_xor_(const Scalar & other);
  void bitwise_xor_(const Tensor & other);
  Tensor __xor__(const Scalar & other) const;
  Tensor __xor__(const Tensor & other) const;
  void __ixor__(const Scalar & other);
  void __ixor__(const Tensor & other);
  Tensor __lshift__(const Scalar & other) const;
  Tensor __lshift__(const Tensor & other) const;
  void __ilshift__(const Scalar & other);
  void __ilshift__(const Tensor & other);
  Tensor bitwise_left_shift(const Tensor & other) const;
  void bitwise_left_shift_(const Tensor & other);
  Tensor bitwise_left_shift(const Scalar & other) const;
  void bitwise_left_shift_(const Scalar & other);
  Tensor __rshift__(const Scalar & other) const;
  Tensor __rshift__(const Tensor & other) const;
  void __irshift__(const Scalar & other);
  void __irshift__(const Tensor & other);
  Tensor bitwise_right_shift(const Tensor & other) const;
  void bitwise_right_shift_(const Tensor & other);
  Tensor bitwise_right_shift(const Scalar & other) const;
  void bitwise_right_shift_(const Scalar & other);
  void tril_(int64_t diagonal);
  void triu_(int64_t diagonal);
  void digamma_();
  void lerp_(const Tensor & end, const Scalar & weight);
  void lerp_(const Tensor & end, const Tensor & weight);
  void addbmm_(const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha);
  Tensor addbmm(const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) const;
  void random_(int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator);
  void random_(int64_t to, c10::optional<Generator> generator);
  void random_(c10::optional<Generator> generator);
  void uniform_(double from, double to, c10::optional<Generator> generator);
  void cauchy_(double median, double sigma, c10::optional<Generator> generator);
  void log_normal_(double mean, double std, c10::optional<Generator> generator);
  void exponential_(double lambd, c10::optional<Generator> generator);
  void geometric_(double p, c10::optional<Generator> generator);
  Tensor diag(int64_t diagonal) const;
  Tensor cross(const Tensor & other, c10::optional<int64_t> dim) const;
  Tensor triu(int64_t diagonal) const;
  Tensor tril(int64_t diagonal) const;
  Tensor trace() const;
  Tensor ne(const Scalar & other) const;
  Tensor ne(const Tensor & other) const;
  void ne_(const Scalar & other);
  void ne_(const Tensor & other);
  Tensor not_equal(const Scalar & other) const;
  Tensor not_equal(const Tensor & other) const;
  void not_equal_(const Scalar & other);
  void not_equal_(const Tensor & other);
  Tensor eq(const Scalar & other) const;
  Tensor eq(const Tensor & other) const;
  Tensor ge(const Scalar & other) const;
  Tensor ge(const Tensor & other) const;
  void ge_(const Scalar & other);
  void ge_(const Tensor & other);
  Tensor greater_equal(const Scalar & other) const;
  Tensor greater_equal(const Tensor & other) const;
  void greater_equal_(const Scalar & other);
  void greater_equal_(const Tensor & other);
  Tensor le(const Scalar & other) const;
  Tensor le(const Tensor & other) const;
  void le_(const Scalar & other);
  void le_(const Tensor & other);
  Tensor less_equal(const Scalar & other) const;
  Tensor less_equal(const Tensor & other) const;
  void less_equal_(const Scalar & other);
  void less_equal_(const Tensor & other);
  Tensor gt(const Scalar & other) const;
  Tensor gt(const Tensor & other) const;
  void gt_(const Scalar & other);
  void gt_(const Tensor & other);
  Tensor greater(const Scalar & other) const;
  Tensor greater(const Tensor & other) const;
  void greater_(const Scalar & other);
  void greater_(const Tensor & other);
  Tensor lt(const Scalar & other) const;
  Tensor lt(const Tensor & other) const;
  void lt_(const Scalar & other);
  void lt_(const Tensor & other);
  Tensor less(const Scalar & other) const;
  Tensor less(const Tensor & other) const;
  void less_(const Scalar & other);
  void less_(const Tensor & other);
  Tensor take(const Tensor & index) const;
  Tensor take_along_dim(const Tensor & indices, c10::optional<int64_t> dim) const;
  Tensor index_select(int64_t dim, const Tensor & index) const;
  Tensor masked_select(const Tensor & mask) const;
  Tensor nonzero() const;
  std::vector<Tensor> nonzero_numpy() const;
  Tensor gather(int64_t dim, const Tensor & index, bool sparse_grad) const;
  Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) const;
  void addcmul_(const Tensor & tensor1, const Tensor & tensor2, const Scalar & value);
  Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) const;
  void addcdiv_(const Tensor & tensor1, const Tensor & tensor2, const Scalar & value);
  tuple2<Tensor, Tensor> lstsq(const Tensor & A) const;
  tuple2<Tensor, Tensor> triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const;
  tuple2<Tensor, Tensor> symeig(bool eigenvectors, bool upper) const;
  tuple2<Tensor, Tensor> eig(bool eigenvectors) const;
  tuple3<Tensor, Tensor, Tensor> svd(bool some, bool compute_uv) const;
  Tensor swapaxes(int64_t axis0, int64_t axis1) const;
  void swapaxes_(int64_t axis0, int64_t axis1);
  Tensor swapdims(int64_t dim0, int64_t dim1) const;
  void swapdims_(int64_t dim0, int64_t dim1);
  Tensor cholesky(bool upper) const;
  Tensor cholesky_solve(const Tensor & input2, bool upper) const;
  tuple2<Tensor, Tensor> solve(const Tensor & A) const;
  Tensor cholesky_inverse(bool upper) const;
  tuple2<Tensor, Tensor> qr(bool some) const;
  tuple2<Tensor, Tensor> geqrf() const;
  Tensor orgqr(const Tensor & input2) const;
  Tensor ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const;
  Tensor lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const;
  Tensor multinomial(int64_t num_samples, bool replacement, c10::optional<Generator> generator) const;
  void lgamma_();
  Tensor lgamma() const;
  Tensor digamma() const;
  Tensor polygamma(int64_t n) const;
  void polygamma_(int64_t n);
  Tensor erfinv() const;
  void erfinv_();
  Tensor i0() const;
  void i0_();
  Tensor sign() const;
  void sign_();
  Tensor signbit() const;
  Tensor dist(const Tensor & other, const Scalar & p) const;
  void atan2_(const Tensor & other);
  Tensor atan2(const Tensor & other) const;
  Tensor lerp(const Tensor & end, const Scalar & weight) const;
  Tensor lerp(const Tensor & end, const Tensor & weight) const;
  Tensor histc(int64_t bins, const Scalar & min, const Scalar & max) const;
  tuple2<Tensor, Tensor> histogram(const Tensor & bins, const c10::optional<Tensor> & weight, bool density) const;
  tuple2<Tensor, Tensor> histogram(int64_t bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density) const;
  Tensor fmod(const Scalar & other) const;
  void fmod_(const Scalar & other);
  Tensor fmod(const Tensor & other) const;
  void fmod_(const Tensor & other);
  Tensor hypot(const Tensor & other) const;
  void hypot_(const Tensor & other);
  Tensor igamma(const Tensor & other) const;
  void igamma_(const Tensor & other);
  Tensor igammac(const Tensor & other) const;
  void igammac_(const Tensor & other);
  Tensor nextafter(const Tensor & other) const;
  void nextafter_(const Tensor & other);
  Tensor remainder(const Scalar & other) const;
  void remainder_(const Scalar & other);
  Tensor remainder(const Tensor & other) const;
  void remainder_(const Tensor & other);
  Tensor min() const;
  Tensor fmin(const Tensor & other) const;
  Tensor max() const;
  Tensor fmax(const Tensor & other) const;
  Tensor maximum(const Tensor & other) const;
  Tensor max(const Tensor & other) const;
  Tensor minimum(const Tensor & other) const;
  Tensor min(const Tensor & other) const;
  Tensor quantile(double q, c10::optional<int64_t> dim, bool keepdim) const;
  Tensor quantile(const Tensor & q, c10::optional<int64_t> dim, bool keepdim) const;
  Tensor nanquantile(double q, c10::optional<int64_t> dim, bool keepdim) const;
  Tensor nanquantile(const Tensor & q, c10::optional<int64_t> dim, bool keepdim) const;
  Tensor quantile(double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) const;
  Tensor quantile(const Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) const;
  Tensor nanquantile(double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) const;
  Tensor nanquantile(const Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) const;
  tuple2<Tensor, Tensor> sort(int64_t dim, bool descending) const;
  tuple2<Tensor, Tensor> sort(c10::optional<bool> stable, int64_t dim, bool descending) const;
  Tensor msort() const;
  Tensor argsort(int64_t dim, bool descending) const;
  tuple2<Tensor, Tensor> topk(int64_t k, int64_t dim, bool largest, bool sorted) const;
  Tensor all() const;
  Tensor any() const;
  Tensor renorm(const Scalar & p, int64_t dim, const Scalar & maxnorm) const;
  void renorm_(const Scalar & p, int64_t dim, const Scalar & maxnorm);
  Tensor unfold(int64_t dimension, int64_t size, int64_t step) const;
  bool equal(const Tensor & other) const;
  Tensor pow(const Tensor & exponent) const;
  Tensor pow(const Scalar & exponent) const;
  void pow_(const Scalar & exponent);
  void pow_(const Tensor & exponent);
  Tensor float_power(const Tensor & exponent) const;
  Tensor float_power(const Scalar & exponent) const;
  void float_power_(const Scalar & exponent);
  void float_power_(const Tensor & exponent);
  void normal_(double mean, double std, c10::optional<Generator> generator);
  Tensor alias() const;
  Tensor isfinite() const;
  Tensor isinf() const;
  Tensor isposinf() const;
  Tensor isneginf() const;
  Tensor special_polygamma(int64_t n) const;
  Tensor det() const;
  Tensor inner(const Tensor & other) const;
  Tensor outer(const Tensor & vec2) const;
  Tensor ger(const Tensor & vec2) const;
};
