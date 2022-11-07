namespace c10 {

template <typename T>
class ArrayRef final {
private:
  constexpr ArrayRef();
};
}


// This makes e.g. IntArrayRef look like long[] to the jvm
%define ARRAY_REF(TYPE, SCALAR)
%typemap(in) TYPE {
  auto elems$1 = (SCALAR*)(jenv)->Get##$typemap(jboxtype, SCALAR)##ArrayElements($input, nullptr);
  size_t len$1 = (jenv)->GetArrayLength($input);
  $1 = TYPE(elems$1, len$1);
}

%typemap(freearg) TYPE {
  (jenv)->Release##$typemap(jboxtype, SCALAR)##ArrayElements($input, ($typemap(jni, SCALAR)*)$1.data(), 0);
}

%typemap(out) TYPE {
  $result = (jenv)->New##$typemap(jboxtype, SCALAR)##Array($1.size());
  (jenv)->Set##$typemap(jboxtype, SCALAR)##ArrayRegion($result, 0, $1.size(), (const $typemap(jni, SCALAR)*)$1.data());
}

%typemap(jni) TYPE  "$typemap(jni, SCALAR)""Array"
%typemap(jtype) TYPE "$typemap(jtype, SCALAR)[]"
%typemap(jstype) TYPE  "$typemap(jtype, SCALAR)[]"
%typemap(javain) TYPE  "$javainput"
%typemap(javaout) TYPE  {
  return $jnicall;
}

%enddef

ARRAY_REF(IntArrayRef, int64_t)
ARRAY_REF(ArrayRef<double>, double)

DEFINE_OPTIONAL(OptDoubleArrayRef, c10::ArrayRef<double>)
DEFINE_OPTIONAL(OptIntArrayRef, IntArrayRef)

%define ARRAY_REF_OF_OBJECT(ListT, T)

%template(ListT) c10::ArrayRef< T >;
%naturalvar c10::ArrayRef< T >;

%typemap(jni) c10::ArrayRef< T >  "jlongArray" // Use jlongArray for CPtrs, really these are objects
%typemap(jstype) c10::ArrayRef< T > "$typemap(jboxtype, T)[]"
%typemap(jtype) c10::ArrayRef< T > "long[]"
%typemap(javain) c10::ArrayRef< T >  "$javainput"

%typemap(javain,
         pre="    long[] cptrs$javainput = new long[$javainput.length]; for (int i = 0; i < $javainput.length; ++i) { cptrs$javainput[i] = $typemap(jboxtype, T).getCPtr($javainput[i]); }",
         //post="     opt$javainput.delete();",
         pgcppname="cptrs$javainput")
         c10::ArrayRef< T > "cptrs$javainput"

%typemap(javaout) c10::ArrayRef< T >  {
  throw new java.lang.IllegalStateException("There should never be a need to return an ArrayRef of objects. ");
}

%typemap(in) c10::ArrayRef< T > {

  size_t len$1 = (jenv)->GetArrayLength($input);
  // https://stackoverflow.com/questions/4754763/object-array-initialization-without-default-constructor
  void *raw_memory = operator new[](len$1 * sizeof(T));
  T* array$1 = static_cast<T*>(raw_memory);
  jlong* elems$1 = (jenv)->GetLongArrayElements($input, nullptr);
  for (size_t i = 0; i < len$1; ++i) {
    new(&array$1[i]) T(*(T*)elems$1[i]);
  }
  (jenv)->ReleaseLongArrayElements($input, elems$1, 0);
  $1 = c10::ArrayRef<T>(array$1, len$1);
}

%typemap(freearg) c10::ArrayRef< T > {
  delete (T*)$1.data();
}

%enddef
