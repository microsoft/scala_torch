"""A script for generating swig and Scala bindings based on Declarations.yaml. The swig bindings
are used to generate Java bindings (using swig) that the Scala bindings call through to.
"""

import re
import sys
import yaml
from pathlib import Path


# TODO need to handle (in decreasing order of importance
#  .*_out methods
#  Anything with Stream
#  Anything with Storage
#  Any overload or method using Dimnames
#  .*_backward methods
# returning (None, True) means an expected skip, i.e. we should never have to handle that case
def should_skip(fields, is_tensor_class):
  if fields['name'] == "normal" :
    # This overload conflicts with an overload in variable_factories.h. I don't know why it doesn't cause other
    # ambiguous overload errors.
    # These are provided manually for now.
    return ("ambiguous overload (libtorch's fault)", True)
  elif fields['name'].endswith("_out"):
    return ("out or backward", True)
  elif fields['name'] in ("_slow_conv2d_backward", "_slow_conv2d_forward"):
    # not sure why this one is missing
    return ("blocked function", True)
  elif "name" in fields['overload_name'] or fields['overload_name'].lower().startswith("out_") or fields['overload_name'].lower().endswith("_out")  or fields['overload_name'] in ("output", "out"):
    return ("blocked overload name", True)
  elif (is_tensor_class and 'Tensor' not in fields['method_of']) or ((not is_tensor_class) and 'namespace' not in fields['method_of']):
    return (None, True)
  elif (fields['name'], fields['overload_name']) in {("range", ""), ("clip", "Tensor"), ("clip_", "Tensor"), ("clamp", "Tensor"), ("clamp_", "Tensor")}:
    # These overloads don't work in Scala because of erasure
    return (None, True)
  elif any(convert_cpp_type(arg['type']) in ("DimnameList", "c10::optional<DimnameList>", "Storage", "Stream") for arg in fields['schema_order_arguments']):
    return ("blocked type", True)
  elif len(fields['returns']) > 1 and any(convert_cpp_type(ret['type']) in ("std::vector<Tensor>",) for ret in fields['returns']):
    return ("blocked return type", True)
  else:
    return (None, False)

def generate_swig(items, is_tensor_class):
  declarations = []
  for fields in items:
    (reason, skip) = should_skip(fields, is_tensor_class)
    if skip:
      if reason:
        print(f"rejecting {reason}: " + fields['schema_string'])
      continue

    returns = ""
    if len(fields['returns']) == 0:
      returns = "void"
    elif len(fields['returns']) == 1:
      if fields['inplace']:
        returns = 'void' # The SWIG C++ declaration lies a bit, but it's okay, it makes memory management easier
      else:
        returns = fields['returns'][0]['type']
      name = fields["name"]
    else:
      num = len(fields['returns'])
      returns = f"tuple{num}<" + ", ".join(r['type'] for r in fields['returns']) +  ">"
      name = fields["name"]


    args = []
    for arg in fields['arguments']:
      if is_tensor_class and arg['name'] == 'self':
        continue
      cpp_type = convert_cpp_type(arg['type'])
      cpp_string = cpp_type + " " + arg['name']
      # We don't add the defaults to swig because swig goes ahead and generates JNI calls
      # without defaults for each possible overload. We let Scala handle filling in the overloads.
      #if "default" in arg:
      #  cpp_string += " = " + str(arg["default"])
      args.append(cpp_string)
    maybe_const = ""
    if is_tensor_class and not fields['inplace']:
      maybe_const = " const"
    cpp_name = convert_cpp_type(returns) + " " + fields['name'] + "(" + ", ".join(args) + f"){maybe_const};"

    declarations.append(cpp_name)

  joiner = "\n  " if  is_tensor_class else "\n"
  swigFile = "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT\n// See build.sbt for details\n" + joiner.join(declarations)

  return swigFile


def convert_cpp_type(yamlType):
  yamlType = re.sub(r'at::', '', yamlType)
  yamlType = re.sub(r'::std', 'std', yamlType)
  return yamlType

SCALA_TYPE_MAP = {
  # CPP type to (public type, wrapper, unwrapper)
  "Tensor": ("Tensor", "Tensor", ".underlying"),
  "at::Tensor": ("Tensor", "Tensor", ".underlying"),
  "Tensor &": ("Tensor", "Tensor", ".underlying"),
  "at::Tensor &": ("Tensor", "Tensor", ".underlying"),
  "const Tensor &": ("Tensor", "Tensor", ".underlying"),
  "const at::Tensor &": ("Tensor", "Tensor", ".underlying"),
  "IntArrayRef": ("Array[Long]", "Array", None),
  "at::IntArrayRef": ("Array[Long]", "Array", None),
  "const c10::optional<Tensor> &": ("Option[Tensor]", "Option", ".map(_.underlying)"),
  "const c10::optional<at::Tensor> &": ("Option[Tensor]", "Option", ".map(_.underlying)"),
  "c10::optional<MemoryFormat>": ("Option[MemoryFormat]", "Option", None),
  "c10::optional<at::MemoryFormat>": ("Option[MemoryFormat]", "Option", None),
  "MemoryFormat": ("MemoryFormat", None, None),
  "c10::optional<ScalarType>": ("Option[dtype]", "Option", ".map(_.toScalarType)"),
  "c10::optional<at::ScalarType>": ("Option[dtype]", "Option", ".map(_.toScalarType)"),
  "ScalarType": ("dtype", "dtype", ".toScalarType"),
  "QScheme": ("QScheme", None, None),
  "at::ScalarType": ("dtype", "dtype", ".toScalarType"),
  "c10::optional<Generator>": ("Option[Generator]", "Generator", ".map(_.underlying)"),
  "c10::optional<at::Generator>": ("Option[Generator]", "Generator", ".map(_.underlying)"),
  "c10::optional<Layout>": ("Option[Layout]", "Option", None),
  "c10::optional<at::Layout>": ("Option[Layout]", "Option", None),
  "Device": ("Device", "Device", ".underlying"),
  "c10::optional<Device>": ("Option[Device]", "Option", ".map(_.underlying)"),
  "c10::optional<at::Device>": ("Option[Device]", "Option", ".map(_.underlying)"),
  "c10::optional<bool>": ("Option[Boolean]", "Option", ".map(java.lang.Boolean.valueOf)"),
  "c10::optional<int64_t>":  ("Option[Long]", "Option", ".asJavaLong"),
  "c10::optional<std::string>": ("Option[String]", "Option", None),
  "c10::optional<IntArrayRef>": ("Option[Array[Long]]", "Option", None),
  "c10::optional<ArrayRef<double>>": ("Option[Array[Double]]", "Option", None),
  "c10::optional<at::IntArrayRef>": ("Option[Array[Long]]", "Option", None),
  "tuple2<double, int64_t>": ("(Double, Long)", "wrapDoubleLongTuple2", ".asJava"),
  "tuple2<Tensor, Tensor>": ("(Tensor, Tensor)", "wrapTensorTuple2", ".map(_.underlying).asJava"),
  "tuple3<Tensor, Tensor, Tensor>": ("(Tensor, Tensor, Tensor)", "wrapTensorTuple3", ".map(_.underlying).asJava"),
  "tuple4<Tensor, Tensor, Tensor, Tensor>": ("(Tensor, Tensor, Tensor, Tensor)", "wrapTensorTuple4", ".map(_.underlying).asJava"),
  "tuple4<Tensor, Tensor, double, int64_t>": ("(Tensor, Tensor, Double, Long)", "wrapTensorTuple2DoubleLong", ".map(_.underlying).asJava"),
  "tuple5<Tensor, Tensor, Tensor, Tensor, Tensor>": ("(Tensor, Tensor, Tensor, Tensor, Tensor)", "wrapTensorTuple5", ".map(_.underlying).asJava"),
  "tuple5<Tensor, Tensor, Tensor, Tensor, int64_t>": ("(Tensor, Tensor, Tensor, Tensor, Long)", "wrapTensorTuple4Long", ".map(_.underlying).asJava"),
  "TensorList": ("Array[Tensor]", None, ".map(_.underlyingChecked)"),
  "at::TensorList": ("Array[Tensor]", None, ".map(_.underlyingChecked)"),
  "std::vector<Tensor>": ("Array[Tensor]", "tensorVectorToArray", ".map(_.underlyingChecked)"),
  "std::vector<at::Tensor>": ("Array[Tensor]", "tensorVectorToArray", ".map(_.underlyingChecked)"),
  "Scalar": ("Scalar", "Scalar", ".underlying"),
  "at::Scalar": ("Scalar", "Scalar", ".underlying"),
  "const Scalar &": ("Scalar", "Scalar", ".underlying"),
  "c10::optional<Scalar>": ("Option[Scalar]", "Option", ".map(_.underlying)"),
  "c10::optional<at::Scalar>": ("Option[Scalar]", "Option", ".map(_.underlying)"),
  "const c10::optional<at::Scalar> &": ("Option[Scalar]", "Option", ".map(_.underlying)"),
  "const c10::optional<Scalar> &": ("Option[Scalar]", "Option", ".map(_.underlying)"),
  "c10::string_view": ("String", None, None),
  "ArrayRef<Scalar>": ("Array[Scalar]", "Array", ".map(_.underlying)"),
  "ArrayRef<double>": ("Array[Double]", "Array", None),
  "std::array<bool,2>": ("Array[Boolean]", "Array", None),
  "std::array<bool,3>": ("Array[Boolean]", "Array", None),
  "std::array<bool,4>": ("Array[Boolean]", "Array", None),
  "c10::optional<c10::string_view>": ("Option[String]", "Option", None),
  "int64_t": ("Long", None, None),
  "double": ("Double", None, None),
  "c10::optional<double>": ("Option[Double]", "Option", ".asJavaDouble"),
  "TensorOptions": ("TensorOptions", "TensorOptions", ".underlying"),
  "bool": ("Boolean", None, None),
  "std::string": ("String", None, None),
  "const c10::List<c10::optional<Tensor>> &": ("Array[Option[Tensor]]", "Array", ".map(_.map(_.underlying))"),
  "c10::List<c10::optional<Tensor>>": ("Array[Option[Tensor]]", "Array", ".map(_.map(_.underlying))"),
  "void": ("Unit", None, None)
}

SCALA_BUILTIN_MAP = {
  ("bool", "True"): "true",
  ("bool", "False"): "false",
  ("const c10::optional<Tensor> &", "{}"): "None",
  ("c10::optional<TypeMeta>", "c10::nullopt"): "None",
  ("c10::optional<MemoryFormat>", "c10::nullopt"): "None",
  ("c10::optional<int64_t>", "9223372036854775807"): "None", # A bug in Declarations.yml
  ("MemoryFormat", "MemoryFormat::Contiguous"): "MemoryFormat.Contiguous",
  ("c10::optional<MemoryFormat>", "MemoryFormat::Contiguous"): "MemoryFormat.Contiguous",
  ("c10::optional<bool>", "False"): "false",
  ("c10::optional<bool>", "True"): "true",
  ("int64_t", "at::Reduction::Mean"): "internal.Reduction.Mean.swigValue()",
  ("c10::optional<ScalarType>", "at::kLong"): "int64",
}

# Manual parts of the Core Tensor class. Hopefully will shrink over time as we auto-gen more methods.
SWIG_TENSOR_BODY ='''
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
'''

SCALA_KEYWORDS = {"implicit", "val", "package", "trait", "abstract", "object", "case", "var"}
SCALA_OPERATORS = {"__and__": "&", "__or__": "|", "__ior__": "|=", "__iand__": "&=", "__lshift__": "<<",  "__rshift__": ">>", "__ilshift__": "<<=",  "__irshift__": ">>=", "__xor__": "^", "__ixor": "^="}

def generate_function_args_scala(fields, is_tensor_class, include_defaults):
  name = fields['name']
  args = []
  args_references = {} # Note: python keeps dictionary insertion order
  for arg in fields['schema_order_arguments']:
    tpe = convert_cpp_type(arg['type'])
    (arg_type, _, unwrapper) = SCALA_TYPE_MAP[tpe]
    is_this_argument = arg['name'] == 'self' and is_tensor_class
    if is_this_argument:
      arg_name = "this"
    else:
      arg_name = arg['name']
      # Might have to backtick other scala keywords here
      if arg_name in SCALA_KEYWORDS:
        arg_name = f'`{arg_name}`'
    # TODO Scala can't handle multiple overloaded functions with default arguments.
    #   we should explode out all non-default argument functions for anything other than fields['overload_name'] == ''
    # TODO need to make additional overloads for int-based versions of Scalar defaults since Scalar.fromInt
    #   doesn't compile because it needs a cg
    if include_defaults and "default" in arg:
      # TODO will need some work here
      (_, default_wrapper, _) = SCALA_TYPE_MAP[tpe]
      # Encode cpp literals like True as Scala literals
      cpp_value = str(arg["default"])
      default_literal: str
      if cpp_value == 'c10::nullopt':
        default_literal = "None"
      elif tpe in ("Scalar", "const Scalar &"):
        tpe = ""
        default_literal = cpp_value
        unwrapper = ".toInternalScalar"
        arg_type = "Double"
      elif default_wrapper == "Array" and len(cpp_value) > 0 and cpp_value[0] == "{" and cpp_value[-1] == "}":
        default_literal = "Array(" + cpp_value[1:-1] + ")"
      else:
        scala_value = SCALA_BUILTIN_MAP.get((tpe, cpp_value), cpp_value)
        # Apply any conversion we needed
        default_literal = f"{default_wrapper}({scala_value})" if default_wrapper and scala_value != "None" else scala_value
      scala_string = arg_name + ": " + arg_type + " = " + default_literal
    else:
      scala_string = arg_name + ": " + arg_type
    if not is_this_argument:
      args.append(scala_string)
    args_references[arg_name] = f"{arg_name}{unwrapper}" if unwrapper else arg_name

  return (args, args_references)

def format_direct_call(name, args_references, return_wrapper):
  args_pass = "(" + ", ".join(args_references) + ")"
  inner = f"swig.{name}{args_pass}"
  return f"{return_wrapper}({inner})" if return_wrapper else inner

def format_member_call(name, args_references, return_wrapper):
  this_reference_index = index_of_first(args_references, lambda s: s.startswith('this.'))
  this_reference = args_references[this_reference_index]
  del args_references[this_reference_index]
  args_pass = "(" + ", ".join(args_references) + ")"
  inner = f"{this_reference}.{name}{args_pass}"
  return f"{return_wrapper}({inner})" if return_wrapper else inner

def format_call(name, args_references, return_wrapper, tensor_options_pos, is_tensor_class):
  options_construct = ""
  final = ""
  if tensor_options_pos:
    options_construct = "TensorOptions(\n"
    if 'dtype' in args_references:
      options_construct += f"dtype=dtype,\n"
      del args_references['dtype']
    if 'device' in args_references:
      options_construct += f"device=device,\n"
      del args_references['device']
    if 'layout' in args_references:
      options_construct += f"layout=layout,\n"
      del args_references['layout']
    if 'requires_grad' in args_references:
      options_construct += f"requires_grad=requires_grad,\n"
      del args_references['requires_grad']
    if 'pin_memory' in args_references:
      options_construct += f"pinned_memory=pin_memory,\n"
      del args_references['pin_memory']
    options_construct += ").toInternal.apply { options => \n"
    final = "\n}\n"

  arg_exprs = [expr for _, expr in args_references.items()]
  if tensor_options_pos:
    arg_exprs.insert(tensor_options_pos, 'options')
  if is_tensor_class:
    call = format_member_call(name, arg_exprs, return_wrapper)
  else:
    call = format_direct_call(name, arg_exprs, return_wrapper)
  return options_construct + call + final

def format_function_scala(scala_name, name, args_decls, args_references, returns, tensor_options_pos, is_tensor_class, is_inplace):
  (return_type, return_wrapper, _) = SCALA_TYPE_MAP[returns]

  args_decl = "(" + ", ".join(args_decls) + ")"
  if is_inplace:
    return_wrapper = ""
  body = format_call(name, args_references, return_wrapper, tensor_options_pos, is_tensor_class)


  if is_inplace:
    this_reference = "this" if is_tensor_class else "self"
    scala_decl = f"  def {scala_name}{args_decl}(implicit rm: ReferenceManager): {this_reference}.type = NoGrad.noGrad {{\n    {body}\n    {this_reference}\n  }}"
  else:
    scala_decl = f"  def {scala_name}{args_decl}(implicit rm: ReferenceManager): {return_type} = {body}"
  return scala_decl


def scala_name_of(name):
  if name in SCALA_KEYWORDS:
    return f'`{name}`'
  else:
    return name

# Translates from the raw name to the name used in the API. This involves splitting off package prefixes (like fft_)
# and translating operators (like __and__ to &).
def to_api_name(name, python_module):
  if python_module in {"fft", "linalg", "special"}:
    assert name.startswith(f'{python_module}_'), f"methods in module {python_module} should all start with {python_module}_"
    name = name[(len(python_module) + 1):]
  if name in SCALA_OPERATORS:
    name = SCALA_OPERATORS[name]
  return name

def generate_functions_scala(fields, overloads, is_tensor_class, python_module):
  name = scala_name_of(fields['name'])

  returns = ""
  if len(fields['returns']) == 0:
    returns = "void"
  elif len(fields['returns']) == 1:
    returns = fields['returns'][0]['type']
  else:
    num = len(fields['returns'])
    returns = f"tuple{num}<" + ", ".join(r['type'] for r in fields['returns']) +  ">"
    name = fields["name"]
  is_inplace = fields['inplace']

  res = []
  # This is code that generates an overload like `exp$Tensor` in addition to the raw name. This is nice because
  # it allows us to workaround the limitation that only one overload can have defaults (and also work around
  # some issues with erasure), but it generates a lot of code and probably isn't worth it.
  #overload_options = ((name, True),) if len(overloads[fields['name']]) == 1 else ((name, fields['overload_name'] == overloads[fields['name']][0]), (fields['name']+"$"+fields['overload_name'], True))
  overload_options = ((to_api_name(name, python_module), fields['overload_name'] == overloads[fields['name']][0]),)
  for (scala_name, include_defaults) in overload_options:
    (args, args_references) = generate_function_args_scala(fields, is_tensor_class, include_defaults)
    tensor_options_pos = index_of_first(fields['arguments'], lambda d: convert_cpp_type(d['type']) == "TensorOptions")
    res.append(format_function_scala(scala_name, name, args, args_references, convert_cpp_type(returns), tensor_options_pos, is_tensor_class, is_inplace))
  return res


def index_of_first(lst, pred):
    for i,v in enumerate(lst):
        if pred(v):
            return i
    return None

def generate_package_scala(inputs, overloads, is_tensor_class, python_module):
  declarations = []

  for fields in inputs:
    if should_skip(fields, is_tensor_class)[1]:
      continue
    if not is_tensor_class and fields['python_module'] != python_module:
      continue
    scala_decls = generate_functions_scala(fields, overloads, is_tensor_class, python_module)
    declarations.extend(scala_decls)

  all = "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT\n// See swig/src/main/swig/build.sbt for details\n" + "\n".join(declarations)
  return all

def parse(declarationsFile):
  stream = open(declarationsFile, 'r')
  loaded = yaml.safe_load(stream)
  parsed = []
  for entry in loaded:
    parsed.append(entry)

  return parsed

def run(input, swig_dir, scalaDir):
  parsed = parse(input)

  with open(swig_dir / "generated_bindings.i", 'w') as swig_out:
    outputStr = generate_swig(parsed, is_tensor_class=False)

    swig_out.write(outputStr)

  with open(swig_dir / "generated_tensor_bindings.i", 'w') as swig_out:
    outputStr = generate_swig(parsed, is_tensor_class=True)
    swig_out.write(SWIG_TENSOR_BODY)
    swig_out.write(outputStr)
    swig_out.write("\n};\n")

  overloads = {}
  for fields in parsed:
    name = fields['name']
    if should_skip(fields, is_tensor_class=False)[1] and should_skip(fields, is_tensor_class=True)[1]:
      continue
    if not name in overloads:
      overloads[name] = []
    overloads[name].append(fields['overload_name'])

  def write_package_object_scala(dir, namespace):
    with open(dir / "package.scala", 'w') as package_out:
      package_out.write("// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to package.scala.in\n\n")
      for line in open(dir / "package.scala.in", 'r'):
        if line.strip() == "// @@@ bindgen.py inserts generated bindings here @@@":
          package_out.write(generate_package_scala(parsed, overloads, False, namespace))
        else:
          package_out.write(line)

  write_package_object_scala(scalaDir, "")
  write_package_object_scala(scalaDir / "special", "special")
  write_package_object_scala(scalaDir / "linalg", "linalg")
  write_package_object_scala(scalaDir / "nn" / "functional" , "nn")
  write_package_object_scala(scalaDir / "fft", "fft")

  with open(scalaDir / "Tensor.scala", 'w') as tensor_out:
    tensor_out.write("// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to Tensor.scala.in\n\n")
    for line in open(scalaDir / "Tensor.scala.in", 'r'):
      if line.strip() == "// @@@ bindgen.py inserts generated bindings here @@@":
        tensor_out.write(generate_package_scala(parsed, overloads, True, None))
      else:
        tensor_out.write(line)


if __name__ == "__main__":
  declarationsFile = sys.argv[1]
  swig_dir = sys.argv[2]
  scalaDir = sys.argv[3]
  run(declarationsFile, Path(swig_dir), Path(scalaDir))