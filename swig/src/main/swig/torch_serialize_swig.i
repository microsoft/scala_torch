%inline %{
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/import.h>
%}

// This set of macros defines the overloads for loading and saving using the torch serialization framework
%define SERIALIZATION_SET(NAME, TPE, ARG)
namespace torch {
void load(TPE & ARG, const std::string& file_name, c10::optional<torch::Device> device = c10::nullopt);
void load(TPE & ARG, const char* data, size_t size, c10::optional<torch::Device> device = c10::nullopt);
void save(const TPE & ARG, const std::string& file_name);
}

/** uses the above to write to a java byte array */
%native (save_##NAME##_to_byte_array) jbyteArray save_##NAME##_to_byte_array(const TPE & ARG);
/** uses the above to load from a java byte array */
%native (load_##NAME##_from_byte_array) void load_##NAME##_from_byte_array(TPE & ARG, jbyteArray arr);

%{
extern "C" {
  // TODO: it would probably be good to do this in a way that allows for writing to an OutputStream
SWIGEXPORT jbyteArray JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_save_1##NAME##_1to_1byte_1array(JNIEnv * env, jclass, jlong pA, jobject pA_) {
  std::ostringstream out;
  TPE* v = *(TPE **)&pA;
  torch::save(*v, out);
  // TODO: it would probably be better to do this in a way that allows for zero copy
  auto str = std::move(out.str());
  size_t size = str.size();
  jbyteArray result = env->NewByteArray(size);
  if (result == nullptr) {
      java_throw(env, "java/lang/OutOfMemoryError", "Unable to allocate new byte array");
      return nullptr;
  }
  env->SetByteArrayRegion(result, 0, size, (const jbyte*)str.c_str());
  return result;
}

SWIGEXPORT void JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_load_1##NAME##_1from_1byte_1array(JNIEnv * env, jclass, jlong pA, jobject pA_, jbyteArray arr) {
  TPE* v = *(TPE **)&pA;

  size_t len = env->GetArrayLength(arr);
  char* buf = static_cast<char*>(env->GetPrimitiveArrayCritical(arr, 0));
  if (buf == nullptr) {
      java_throw(env, "java/lang/OutOfMemoryError", "Unable to get JNI array");
      return;
  }
  std::istringstream in(std::string(buf, len));
  env->ReleasePrimitiveArrayCritical(arr, buf, 0);

  torch::load(*v, in);
  return;
}
}

%}


%enddef


SERIALIZATION_SET(TensorVector, std::vector<torch::Tensor>, tensor_vec)
SERIALIZATION_SET(Optimizer, torch::optim::Optimizer, opt)
SERIALIZATION_SET(Tensor, torch::Tensor, t)

%{
extern "C" {
  // TODO: it would probably be good to do this in a way that allows for writing to an OutputStream
SWIGEXPORT jbyteArray JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_save_1Module_1to_1byte_1array(JNIEnv * env, jclass, jlong pA, jobject pA_) {
  std::ostringstream out;
  torch::jit::Module* v = *(torch::jit::Module **)&pA;
  v->save(out);
  // TODO: it would probably be better to do this in a way that allows for zero copy
  auto str = std::move(out.str());
  size_t size = str.size();
  jbyteArray result = env->NewByteArray(size);
  if (result == nullptr) {
      java_throw(env, "java/lang/OutOfMemoryError", "Unable to allocate new byte array");
      return nullptr;
  }
  env->SetByteArrayRegion(result, 0, size, (const jbyte*)str.c_str());
  return result;
}

SWIGEXPORT jlong JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_load_1Module_1from_1byte_1array(JNIEnv * env, jclass, jbyteArray arr) {
  jlong jresult = 0;

  size_t len = env->GetArrayLength(arr);
  char* buf = static_cast<char*>(env->GetPrimitiveArrayCritical(arr, 0));
  if (buf == nullptr) {
      java_throw(env, "java/lang/OutOfMemoryError", "Unable to get JNI array");
      return jresult;
  }
  std::istringstream in(std::string(buf, len));
  env->ReleasePrimitiveArrayCritical(arr, buf, 0);

  *(torch::jit::Module **)&jresult = new torch::jit::Module(torch::jit::load(in));
  return jresult;
}
}

%}

/** uses the above to write to a java byte array */
%native (save_Module_to_byte_array) jbyteArray save_Module_to_byte_array(const torch::jit::Module & module);
/** uses the above to load from a java byte array */
%native (load_Module_from_byte_array) torch::jit::Module load_Module_from_byte_array(jbyteArray arr);

// IValue/pickles
namespace torch {
namespace jit {
std::vector<char> pickle_save(const IValue &ivalue);
}
}

%inline %{
IValue unpickle_from_file(const std::string& path) {
  std::vector<char> vec;
  std::ifstream file(path, std::ios::binary);
  if (!file.eof() && !file.fail()) {
      file.seekg(0, std::ios_base::end);
      std::streampos fileSize = file.tellg();
      vec.resize(fileSize);

      file.seekg(0, std::ios_base::beg);
      file.read(&vec[0], fileSize);
  }

  return torch::jit::unpickle(vec.data(), vec.size());
}

void pickle_save_to_file(const std::string& path, const IValue& ivalue) {
  std::vector<char> vec = torch::jit::pickle_save(ivalue);
  std::ofstream file(path, std::ios::binary);
  if (!file.eof() && !file.fail()) {
    file.write( &vec[0], vec.size() );
  }
}

%}
