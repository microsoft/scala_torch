%define STD_ARRAY(SCALAR,SIZE)
%typemap(in) std::array<SCALAR,SIZE> {
  auto elems$1 = (SCALAR*)(jenv)->Get##$typemap(jboxtype, SCALAR)##ArrayElements($input, nullptr);
  size_t len$1 = (jenv)->GetArrayLength($input);
  if (len$1 != SIZE) {
    throw std::invalid_argument("Wrong size for fixed size array");
  }
  for (int i = 0; i < SIZE; ++i) {
    $1[i] = elems$1[i];
  }
  (jenv)->Release##$typemap(jboxtype, SCALAR)##ArrayElements($input, ($typemap(jni, SCALAR)*)elems$1, 0);
}


%typemap(jni) std::array<SCALAR,SIZE>  "$typemap(jni, SCALAR)""Array"
%typemap(jtype) std::array<SCALAR,SIZE> "$typemap(jtype, SCALAR)[]"
%typemap(jstype) std::array<SCALAR,SIZE>  "$typemap(jtype, SCALAR)[]"
%typemap(javain) std::array<SCALAR,SIZE>  "$javainput"

%enddef

STD_ARRAY(bool, 2)
STD_ARRAY(bool, 3)
STD_ARRAY(bool, 4)
