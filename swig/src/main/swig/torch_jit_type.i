%{
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/api/compilation_unit.h>
using torch::jit::CompilationUnit;
%}

namespace c10 {
struct QualifiedName {
  QualifiedName(const std::string& name);
  explicit QualifiedName(std::vector<std::string> atoms);
  explicit QualifiedName(const QualifiedName& prefix, std::string name);
  const std::string& qualifiedName() const;
  const std::string& name() const;
  const std::string& prefix();
};
EQUALS_FROM_EQ(QualifiedName)
HASHCODE_FROM_STD_HASH(QualifiedName)
}


DEFINE_OPTIONAL(OptQualifiedName, c10::QualifiedName)
DEFINE_OPTIONAL(OptSizes, std::vector<int64_t>)

namespace std {
  %template(FunctionVector) vector<torch::jit::Function*>;
} // namepsace std {


namespace torch {
namespace jit {

struct CompilationUnit {

  torch::jit::Function* create_function(
        c10::QualifiedName name,
        std::shared_ptr<torch::jit::Graph> graph,
        bool shouldMangle = false);

  %extend {

    IValue run_method(const c10::QualifiedName& method_name, const std::vector<IValue>& args) {
      return $self->get_function(method_name)(args);
    }

    IValue run_method(const std::string& method_name, const std::vector<IValue>& args) {
      return $self->get_function(method_name)(args);
    }
  }
};
} // namespace jit
} // namespace torch

namespace c10 {
enum class TypeKind {
  AnyType,
  TensorType,
  TupleType,
  ListType,
  DictType,
  NumberType,
  FloatType,
  FutureType,
  IntType,
  NoneType,
  StringType,
  GeneratorType,
  BoolType,
  OptionalType,
  VarType,
  DeviceObjType,
  FunctionType,
  ClassType,
  CapsuleType,
  InterfaceType
};

struct CAFFE2_API Type {
  TypeKind kind() const;

  %extend {

    std::shared_ptr<c10::TensorType> expectTensor() {
      return $self->expect<TensorType>();
    }

    // TODO out of laziness we don't expose any C++ types other than Type and TupleType.
    // We could expose them all.

    // TODO more types
    static std::shared_ptr<c10::Type> createDict(std::shared_ptr<c10::Type> keyType, std::shared_ptr<Type> valueType) {
      return DictType::create(keyType, valueType);
    }

    static std::shared_ptr<c10::Type> createList(std::shared_ptr<c10::Type> elementType) {
      return ListType::create(elementType);
    }

    static std::shared_ptr<c10::Type> getString() {
      return StringType::get();
    }

    static std::shared_ptr<c10::Type> getFloat() {
      return FloatType::get();
    }

    static std::shared_ptr<c10::Type> getInt() {
      return IntType::get();
    }

    static std::shared_ptr<c10::Type> getBool() {
      return BoolType::get();
    }
  }
private:
  Type();
};
EQUALS_FROM_EQ(Type)

TO_STRING_FROM_OSTREAM(Type);

// TupleType actually inherits from NamedType (which inherits from Type) but swig is happy with this declaration.
struct CAFFE2_API TupleType: public Type {
  static std::shared_ptr<c10::TupleType> create(const std::vector<std::shared_ptr<c10::Type>>& types);
};

// ClassType actually inherits from NamedType (which inherits from Type) but swig is happy with this declaration.
struct CAFFE2_API ClassType: public Type {

  const c10::optional<c10::QualifiedName>& name() const;

  const std::vector<torch::jit::Function*>& methods() const;

  size_t addAttribute(const std::string& name, const std::shared_ptr<c10::Type>& type, bool is_parameter = false);
};

struct CAFFE2_API TensorType: public Type {
  // Dim/device/type unspecified
  static std::shared_ptr<c10::TensorType> get();
  c10::optional<at::Device> device();
%extend {

  static std::shared_ptr<c10::TensorType> createContiguous(TypeMeta typeMeta, DeviceType deviceType, IntArrayRef dim) {
    return TensorType::createContiguous(c10::typeMetaToScalarType(typeMeta), deviceType, dim);
  }

  // It seems that ScalarType is quasi-deprecated (https://pytorch.org/cppdocs/notes/tensor_creation.html),
  // so convert to a type meta here.
  c10::optional<TypeMeta> dtype() const {
    if ($self->scalarType()) {
      return c10::scalarTypeToTypeMeta(*($self->scalarType()));
    } else {
      return nullopt;
    }
  }

  c10::optional<std::vector<int64_t>> sizes() const {
    return $self->sizes().concrete_sizes();
  }
}
};

} // namespace c10
