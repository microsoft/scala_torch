%javaconst(1) Layout;
enum class Layout : int8_t { Strided, Sparse, Mkldnn };
DEFINE_OPTIONAL(OptLayout, Layout)

%javaconst(1) QScheme;
enum class QScheme : uint8_t {
  PER_TENSOR_AFFINE = 0,
  PER_CHANNEL_AFFINE = 1,
  PER_TENSOR_SYMMETRIC = 2,
  PER_CHANNEL_SYMMETRIC = 3,
  PER_CHANNEL_AFFINE_FLOAT_QPARAMS = 4,
  COMPILE_TIME_NUM_QSCHEMES = 5,
};

struct TypeMeta {
  TypeMeta() = delete; // this is actually available, but we don't need it.
  c10::string_view name() const;
  ScalarType toScalarType();
  size_t itemsize() const;

  static inline TypeMeta fromScalarType(ScalarType scalar_type);

};
EQUALS_FROM_EQ(TypeMeta)
DEFINE_OPTIONAL(OptTypeMeta, TypeMeta)

// this block defines a bunch of typemetas like kLongMeta to be used with TensorOptions
// cribbed from c10/core/ScalarType.h, but we can't %import it because of swig's limitations
#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(double, Double)

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                              \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(at::ComplexHalf, ComplexHalf) /* 8 */                \
  _(std::complex<float>, ComplexFloat) /* 9 */           \
  _(std::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */

%inline %{
#define TYPE_META_FOR(tpe, name) const TypeMeta k##name##Meta = caffe2::TypeMeta::Make<tpe>();

AT_FORALL_SCALAR_TYPES(TYPE_META_FOR)

#undef TYPE_META_FOR

%}

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};
DEFINE_OPTIONAL(OptScalarType, ScalarType)

namespace c10 {
bool isFloatingType(ScalarType t);
bool isSignedType(ScalarType t);
bool isComplexType(ScalarType t);
}

enum class MemoryFormat : int8_t { Contiguous, Preserve, ChannelsLast };
DEFINE_OPTIONAL(OptMemoryFormat, MemoryFormat)

%include <c10/core/DeviceType.h>
%include <c10/core/Device.h>
namespace c10 {
EQUALS_FROM_EQ(Device)
HASHCODE_FROM_STD_HASH(Device)
}

DEFINE_OPTIONAL(OptDevice, Device)
