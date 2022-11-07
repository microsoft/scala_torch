%typemap(in) TensorList {
  size_t len$1 = (jenv)->GetArrayLength($input);
  auto elems$1 = new torch::Tensor[len$1];
  for (size_t i = 0; i < len$1; ++i) {
     auto obj = (jenv)->GetObjectArrayElement($input, i);
     jclass cls = (jenv)->GetObjectClass(obj);
     auto fid = (jenv)->GetFieldID(cls, "swigCPtr", "J");
     (jenv)->DeleteLocalRef(cls);
     elems$1[i] = *(torch::Tensor *)(jenv)->GetLongField(obj, fid);
     (jenv)->DeleteLocalRef(obj);
  }
  $1 = TensorList(elems$1, len$1);
}

%typemap(freearg) TensorList {
  delete[] $1.data();
}

%typemap(jni) TensorList  "jobjectArray"
%typemap(jtype) TensorList "TorchTensor[]"
%typemap(jstype) TensorList  "TorchTensor[]"
%typemap(javain) TensorList  "$javainput"
