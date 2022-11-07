%{
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/named_value.h>
#include <torch/csrc/jit/ir/ir.h>
using torch::jit::SourceRange;
using torch::jit::Source;
using torch::jit::Graph;
using torch::jit::Node;
using torch::jit::Value;
using torch::jit::NamedValue;
%}

DEFINE_OPTIONAL(OptIValue, torch::jit::IValue)

%shared_ptr(torch::jit::Source)

namespace std {
  %template(NodeVector) vector<torch::jit::Node*>;
  %template(JitValueVector) vector<torch::jit::Value*>;
} // namepsace std {

namespace torch {
namespace jit {

struct Source {
  Source(
        std::string text,
        c10::optional<std::string> filename,
        size_t starting_line_no
        );

private:
  Source();
};

struct SourceRange {
  SourceRange(
        std::shared_ptr<Source> source,
        size_t start,
        size_t end);

private:
  SourceRange();
};

struct Value {

  EQUALS_AND_HASH_CODE_FROM_PTR_EQUALITY(Value)

  torch::jit::Node* node();
  TORCH_API void replaceAllUsesWith(torch::jit::Value* newValue);
  Value* setType(std::shared_ptr<c10::Type> tpe);
  const std::shared_ptr<c10::Type>& type() const;
  std::string debugName() const;
%extend {

   c10::optional<IValue> maybeIValue() {
     return torch::jit::toIValue($self);
   }

  std::string toString() const {
    return $self->debugName();
  }

  std::vector<torch::jit::Node*> uses() {
    std::vector<torch::jit::Node*> ret;
    for (auto i: $self->uses()) {
      ret.push_back(i.user);
    }
    return ret;
  }
}
private:
  Value();
};

struct NamedValue {
  NamedValue(Value* value);
  NamedValue(const SourceRange& loc, Value* value);

  const std::string& name() const;

  %extend {
    Value* value(std::shared_ptr<torch::jit::Graph> g) const {
      return $self->value(*g);
    }
  }

  const SourceRange& loc() const;
};

struct TORCH_API Operator {
%extend {
  Symbol op() const {
    return Symbol::fromQualString($self->schema().name());
  }
}
private:
 Operator();

};

struct TORCH_API Node {
  EQUALS_AND_HASH_CODE_FROM_PTR_EQUALITY(Value)
  SourceRange sourceRange(); const
  void setSourceRange(SourceRange r);
  torch::jit::Value* output();
  TORCH_API void replaceInputWith(torch::jit::Value* from, torch::jit::Value* to);
  const Operator* maybeOperator() const;

  void moveBefore(Node* n);
  void moveAfter(Node* n);

  bool isBefore(const Node* n) const;

  bool isAfter(const Node* n) const;

  // Declaration has NodeKind but it's a typedef
  Symbol kind() const;

  %extend {

    size_t numOutputs() {
      return $self->outputs().size();
    }

    std::vector<torch::jit::Value*> outputs() {
      return std::vector<torch::jit::Value*>($self->outputs().begin(), $self->outputs().end());
    }

    std::vector<torch::jit::Value*> inputs() {
      return std::vector<torch::jit::Value*>($self->inputs().begin(), $self->inputs().end());
    }
  }

private:
  Node();
};

TO_STRING_FROM_OSTREAM(Node);

struct Graph {
 Graph();
 Value* addInput(std::string name = "");
 TORCH_API const std::string toString(bool print_source_locations = true) const;
 TORCH_API Value* insertConstant(
       const IValue& val,
       c10::optional<SourceRange> loc = c10::nullopt);
 TORCH_API Value* insertGetAttr(Value* obj, const std::string& field);

 %extend {
   std::vector<torch::jit::Value*> inputs() {
     return std::vector<torch::jit::Value*>($self->inputs().begin(), $self->inputs().end());
   }

   std::vector<torch::jit::Node*> nodes() {
     return std::vector<torch::jit::Node*>($self-> nodes().begin(), $self->nodes().end());
   }

   std::vector<torch::jit::Value*> outputs() {
     return std::vector<torch::jit::Value*>($self->outputs().begin(), $self->outputs().end());
   }

   std::vector<torch::jit::Value*> insertGraph(
     std::shared_ptr<torch::jit::Graph> callee,
     const std::vector<torch::jit::Value*>& inputs
     ) {
     return torch::jit::insertGraph(*($self), *callee, ArrayRef<torch::jit::Value*>(inputs));
   }

   Node* insertConstantChunk(Value* v, size_t size, int64_t dim) {
     auto* newNode = $self->create(prim::ConstantChunk, {v}, size);
     newNode->i_(attr::chunks, size);
     newNode->i_(attr::dim, dim);
     $self->insertNode(newNode);
     return newNode;
   }

   Value* insertObject(std::shared_ptr<c10::ClassType> type) {
     return $self->createObject(type)->output();
   }

   Value* insertList(const std::shared_ptr<c10::Type>& elem_type, const std::vector<torch::jit::Value*>& inputs) {
     auto* created = $self->createList(elem_type, ArrayRef<torch::jit::Value*>(inputs));
     return $self->insertNode(created)->output();
   }

   // This is the same as torch::jit::Graph::insertMethodCall, but avoids
   // taking a MatchedSchema and takes arguments and a return type directly.
   Value* insertMethodCall(
     std::string method_name,
     const std::vector<Value*>& arguments,
     const std::shared_ptr<c10::Type> returnType
   ) {
     Value* result = $self->insertNode($self->create(prim::CallMethod, arguments))
                         ->s_(attr::name, std::move(method_name))
                         ->output()
                         ->setType(returnType);
     return result;
   }
 }

 size_t registerOutput(Value* n);

 %extend {

   Value* insert(
       Symbol opname,
       const std::vector<torch::jit::NamedValue>& args,
       const SourceRange& range) {
     return $self->insert(opname, args, {}, range);
   }
 }

};

} // jit
} // torch
