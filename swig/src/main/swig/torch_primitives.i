// this is garbage under linux gcc. In particular, it decides that int64_t is long, when it's
// actually long long and gcc complains about various coercions.
// see https://github.com/swig/swig/issues/568 which is closed but it's still busted.
// So rather than use it, we use enough typemaps (below) to make swig happy
//%include <stdint.i>

typedef jint int32_t;
typedef jshort int16_t;

%typemap(jboxtype) int64_t, const int64_t &               "Long"
%typemap(jni) int64_t, const int64_t &               "jlong"
%typemap(javaout) int64_t, const int64_t & { return $jnicall; }
%typemap(javain) int64_t, const int64_t & "$javainput"
%typemap(jstype) int64_t, const int64_t & "long"
%typemap(jtype) int64_t, const int64_t & "long"

%typemap(in) int64_t %{$1 = ($1_ltype)$input; %}
%typemap(out) int64_t %{ $result = (jlong)$1; %}
// Reference types get treated like pointers, so we need to take addresses and dereference.
%typemap(in) const int64_t & %{$1 = ($1_ltype)(&$input); %}
%typemap(out) const int64_t & %{ $result = (jlong)(*$1); %}

%typemap(jboxtype) int8_t, const int8_t &               "Byte"
%typemap(jni) int8_t, const int8_t &               "jbyte"
%typemap(javaout) int8_t, const int8_t & { return $jnicall; }
%typemap(javain) int8_t, const int8_t & "$javainput"
%typemap(jstype) int8_t, const int8_t & "byte"
%typemap(jtype) int8_t, const int8_t & "byte"

%typemap(in) int8_t %{$1 = ($1_ltype)$input; %}
%typemap(out) int8_t %{ $result = (jbyte)$1; %}
// Reference types get treated like pointers, so we need to take addresses and dereference.
%typemap(in) const int8_t & %{$1 = ($1_ltype)(&$input); %}
%typemap(out) const int8_t & %{ $result = (jbyte)(*$1); %}
