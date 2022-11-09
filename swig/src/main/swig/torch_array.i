// Conversion to/from jvm arrays and tensors. creates methods {to,from}_{float}_array that we can use
%define ARRAY_INPUT_OUTPUT(ctype, jnitype, javatype, jniarraytype, capType, torchScalarType)
%native (to_##javatype##_array) jniarraytype to_##javatype##_array(const torch::Tensor& t);
%native (from_##javatype##_array) torch::Tensor from_##javatype##_array(jniarraytype data, jlongArray shape, const TensorOptions& options={});

%{

// the two jni methods in this block are implementations of the two functions above
extern "C" {
SWIGEXPORT jlong JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_from_1##javatype##_1array(JNIEnv * env, jclass, jniarraytype data, jlongArray shape, jlong pTensorOptions, jobject) {
  auto shapeElems = (env)->GetLongArrayElements(shape, nullptr);
  size_t len = (env)->GetArrayLength(shape);

  TensorOptions* opts = *(TensorOptions **)&pTensorOptions;
  TensorOptions withDType = opts->dtype(at::k##torchScalarType);

  auto tshape = IntArrayRef((int64_t*)shapeElems, len);

  auto result = torch::empty(tshape, withDType);

  auto tdata = (jnitype*)result.data_ptr<ctype>();
  (env)->Get##capType##ArrayRegion(data, 0, result.numel(), tdata);

  jlong jresult = 0;
  *(torch::Tensor **)&jresult = new torch::Tensor((const torch::Tensor &)result);
  (env)->ReleaseLongArrayElements(shape, shapeElems, 0);
  return jresult;
}

SWIGEXPORT jniarraytype JNICALL Java_com_microsoft_scalatorch_torch_internal_torch_1swigJNI_to_1##javatype##_1array(JNIEnv * env, jclass, jlong pT, jobject pT_) {
  torch::Tensor* v = *(torch::Tensor **)&pT;
  size_t size = v->numel();
  jniarraytype result = env->New##capType##Array(size);
  env->Set##capType##ArrayRegion(result, 0, size, (const jnitype*)v->data_ptr<ctype>());
  return result;
}
}
%}

%enddef

ARRAY_INPUT_OUTPUT(float, jfloat, float, jfloatArray, Float, Float)
ARRAY_INPUT_OUTPUT(int64_t, jlong, long, jlongArray, Long, Long)
ARRAY_INPUT_OUTPUT(double, jdouble, double, jdoubleArray, Double, Double)
ARRAY_INPUT_OUTPUT(signed char, jbyte, byte, jbyteArray, Byte, Char)
ARRAY_INPUT_OUTPUT(int32_t, jint, int, jintArray, Int, Int)
