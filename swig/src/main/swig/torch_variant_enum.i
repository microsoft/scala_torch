%define VARIANT_ENUM(ENUM_T, ENUM_CLASS, CONVERT)

%typemap(jni) const ENUM_T&, ENUM_T "jint"
%typemap(jtype) const ENUM_T&, ENUM_T "int"
%typemap(jstype) const ENUM_T&, ENUM_T  "ENUM_CLASS"
%typemap(javain) const ENUM_T&, ENUM_T  "$javainput.swigValue()"
%typemap(javaout) const ENUM_T&, ENUM_T  { return ENUM_CLASS.swigToEnum($jnicall); }
%typemap(in) const ENUM_T&, ENUM_T %{
   $1 = CONVERT($input);
%}

%enddef
