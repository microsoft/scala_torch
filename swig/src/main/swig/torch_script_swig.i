
%{
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

using torch::jit::IValue;

using torch::jit::Function;

using torch::jit::ExtraFilesMap;
#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
%}

namespace c10 {

template<class Key, class Value>
class Dict final {
public:
  explicit Dict(std::shared_ptr<c10::Type> keyType, std::shared_ptr<c10::Type> valueType);
  // TODO more methods

  // TODO this should return an iterator
  void insert_or_assign(Key&& key, Value&& value) const;

private:
  explicit Dict();
};

} // namespace c10

using ExtraFilesMap = std::unordered_map<std::string, std::string>;

namespace std {
%template(ExtraFilesMap) unordered_map<std::string, std::string>;
%template(TensorTypeVector) vector<std::shared_ptr<c10::TensorType>>;
}

namespace c10 {
struct CAFFE2_API Symbol {
  static Symbol fromQualString(const std::string& s);
  const char * toDisplayString() const;
};
EQUALS_FROM_EQ(Symbol)
HASHCODE_FROM_STD_HASH(Symbol)
}

// https://pytorch.org/tutorials/advanced/cpp_export.html
struct IValue {
  // Constructs None
  IValue();
  ~IValue();

  std::shared_ptr<c10::Type> type() const;

  // Tensor
  IValue(torch::Tensor t);
  bool isTensor() const;
  torch::Tensor toTensor() &&;
  torch::Tensor toTensor() const &;

// TODO: figure out intrusive_ptrs
  // TODO? blobs
  // TODO? capsules


  IValue(c10::intrusive_ptr<ivalue::Tuple> v);

  // Double
  IValue(double t);
  bool isDouble() const;
  double toDouble();

  // TODO? futures

  // Int
  IValue(int64_t t);
  bool isInt() const;
  int64_t toInt();

  // Bool
  IValue(bool t);
  bool isBool() const;
  bool toBool();

  // IntList
  IValue(IntArrayRef v);
  bool isIntList() const { return Tag::IntList == tag; }
  // TODO: c10::List ?

  // ConstantString
  IValue(std::string v);
  bool isString() const;
  const std::string& toStringRef() const;


  // DoubleList
  // TODO c10::list
  IValue(std::vector<double> v);
  bool isDoubleList() const;

  //TensorList
  // TODO c10::list
  IValue(const std::vector<Tensor>& v);
  bool isTensorList() const;
  // TODO: fix swig output generation code for htis
  // TensorList toTensorListRef() const;

  // GenericList
  IValue(c10::List<IValue> v);

  // GenericDict
  IValue(c10::Dict<IValue, IValue> v);
  bool isGenericDict() const;
  c10::Dict<IValue, IValue> toGenericDict() const &;

  template<class Key, class Value>
  IValue(c10::Dict<Key, Value> v);

//   IValue(c10::intrusive_ptr<ivalue::Object> v);
  %extend {
    IValue(torch::jit::Module* v) {
      return new IValue(v->_ivalue());
    }
  }
  bool isModule() const;
  torch::jit::Module toModule() const;

  bool isNone() const;

  static IValue uninitialized();

  IValue(torch::Scalar s);
  bool isScalar() const;
  torch::Scalar toScalar() const;

  // perhaps counterintuitively, an IValue can represent a device too. Basically anything
  // that can be an argument to a torch function can be an IValue
  // Device
  IValue(Device s);
  bool isDevice() const;
  Device toDevice() const;
  // TODO: ScalarType?
  // TODO: Layout?
  // TODO: MemoryFormat?
  // TODO: QScheme?
  std::string tagKind() const;
  bool isSameIdentity(const IValue& rhs) const;
};

namespace c10 {

%template(IValueList) List<IValue>;
%template(IValueDict) Dict<IValue, IValue>;
%template(IValueArrayRef) ArrayRef<IValue>;

} // namespace c10

namespace std {
%template(IValueVector) vector<IValue>;
%template(NamedValueVector) vector<torch::jit::NamedValue>;
%template(TypeVector) vector<std::shared_ptr<c10::Type>>;
%template(NamedModuleVector) vector<torch::jit::Named<torch::jit::Module>>;
%template(NamedIValueVector) vector<torch::jit::Named<c10::IValue>>;
%template(NamedTensorVector) vector<torch::jit::Named<torch::Tensor>>;
}

namespace torch {
namespace jit {
struct TORCH_API Function {
  std::shared_ptr<torch::jit::Graph> graph() const;
  const std::string& name() const;
private:
 Function();
};

template <typename T>
struct Named {
  std::string name;
  %extend {
  // Exposing value like this instead of the raw member makes sure that swig
  // allocates a new T. Because all the types we use Named for are internally reference-counted pointers
  // (Module, Tensor, IValue), it's important that we do this to get reference-counting right.
  T value() {
    return $self->value;
  }
  }
};

%template(NamedModule) Named<torch::jit::Module>;
%template(NamedIValue) Named<c10::IValue>;
%template(NamedTensor) Named<torch::Tensor>;

// avoid clash with java.lang.Object
%rename(ScriptObject) Object;
struct Object {
  void setattr(const std::string& name, IValue v);

  IValue attr(const std::string& name) const;
  IValue attr(const std::string& name, IValue or_else) const;
  bool hasattr(const std::string& name) const;

  %extend {

    std::shared_ptr<torch::jit::CompilationUnit> compilation_unit() {
      return ($self)->_ivalue()->compilation_unit();
    }

    std::string name() const {
      return ($self)->_ivalue()->name();
    }

    std::shared_ptr<c10::Type> slot_type(const std::string& name) {
      size_t slot = ($self)->_ivalue()->type()->getAttributeSlot(name);
      return ($self)->_ivalue()->type()->getAttribute(slot);
    }

    const std::vector<torch::jit::Function*> get_method_functions() const {
      std::vector<torch::jit::Function*> result;
      for (const auto& m : ($self)->get_methods()) {
        result.push_back(&(m.function()));
      }
      return result;
    }

    IValue run_method(const std::string& method_name, std::vector<IValue> inputs) {
      return ($self)->get_method(method_name)(std::move(inputs));
    }
  }
};

struct Module: Object {

  explicit Module(c10::QualifiedName class_name);

  Module(
    c10::QualifiedName,
    std::shared_ptr<torch::jit::CompilationUnit> cu,
    bool shouldMangle = false
  );

  Module(std::shared_ptr<torch::jit::CompilationUnit> cu, std::shared_ptr<c10::ClassType> type);

  IValue forward(std::vector<IValue> inputs);

  std::shared_ptr<c10::ClassType> type() const;

  c10::IValue attr(const std::string& name) const;

  c10::IValue attr(const std::string& name, c10::IValue or_else) const;

  void setattr(const std::string& name, c10::IValue v);

  void register_parameter(const std::string& name, torch::Tensor v, bool is_buffer);
  void register_attribute(
        const std::string& name,
        const std::shared_ptr<c10::Type> t,
        IValue v,
        bool is_param = false,
        bool is_buffer = false);
        
  void register_module(const std::string& name, const Module& m);

  %extend {
  std::vector<torch::jit::Named<torch::jit::Module>> named_children() const {
    std::vector<torch::jit::Named<torch::jit::Module>> ret;
    ret.reserve($self->named_children().size());
    for (const auto& named_child: $self->named_children()) {
      ret.push_back(named_child);
    }
    return ret;
  }

  std::vector<torch::jit::Named<torch::Tensor>> named_parameters(bool recurse = true) const {
    const auto& params = $self->named_parameters(recurse);
    std::vector<torch::jit::Named<torch::Tensor>> ret;
    ret.reserve(params.size());
    for (const auto& named_param: params) {
      ret.push_back(named_param);
    }
    return ret;
  }
  }


  void define(const std::string& src);

  %extend {
  // TODO I (@adampauls) don't know if there's a better way to define a TorchScript "method"
  // (a function that is a member of a class) directly. Best I could come up with is calling
  // CompilationUnit.create_function and then passing the result to this method, which I pieced together
  // by reading torch/csrc/jit/script/compiler.cpp.
  void define_method(Function* fn) {
    const auto selfRef = torch::jit::SimpleSelf($self->type());
    selfRef.getClassType()->addMethod(fn);
  }

  bool has_method(const std::string& basename) const {
    return $self->find_method(basename).has_value();
  }

  torch::jit::Function* find_function(const std::string& basename) const {
    return &($self->find_method(basename)->function());
  }

  }

  void save(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  Module clone() const;

  void train(bool on = true);
  void eval();

  bool is_training();

  void to(Device device, bool non_blocking = false);
  void to(Device device, ScalarType dtype, bool non_blocking = false);
  void to(ScalarType dtype, bool non_blocking = false);
};

%rename(load_script_module) load;
TORCH_API Module load(
    const std::string& filename,
    c10::optional<c10::Device> device = c10::nullopt,
    ExtraFilesMap& extra_files = default_extra_files);

} // namespace jit
} // namespace torch

namespace torch {
namespace jit {
std::shared_ptr<torch::jit::CompilationUnit> compile(const std::string &source);
TORCH_API void runRequiredPasses(const std::shared_ptr<torch::jit::Graph>& g);
}
}

%inline {
// Copied verbatim from graph_executor.cpp. It is file-scoped unfortunately.
// // TODO try to expose this properly in torch.     
void runOptimization(std::shared_ptr<torch::jit::Graph> graph) {
  // Basic graph preprocessing to eliminate noise.
  EliminateDeadCode(graph);
  EliminateCommonSubexpression(graph);
  ConstantPooling(graph);

  PeepholeOptimize(graph);
  ConstantPropagation(graph);

  // Unroll small loops, and eliminate expressions that are the same at every
  // iteration.
  UnrollLoops(graph);
  EliminateCommonSubexpression(graph);

  CheckInplace(graph);
}

// Copied from a snippet ("Phase 2") inside compileSpec in graph_executor.cpp.
// TODO this is buggy -- it does the wrong thing for aten::matmul at least.
void runTensorShapePropagation(std::shared_ptr<torch::jit::Graph> opt_graph) {
  ConstantPropagation(opt_graph);
  PropagateInputShapes(opt_graph);
  PropagateRequiresGrad(opt_graph);
}

}

