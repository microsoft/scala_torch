%{
#include <c10/util/string_view.h>
%}

%typemap(jni) c10::string_view "jstring"
%typemap(jtype) c10::string_view               "String"
%typemap(jstype) c10::string_view               "String"
%typemap(out) c10::string_view %{ $result = jenv->NewStringUTF($1.data()); %}

%typemap(javain) c10::string_view "$javainput"
%typemap(javaout) c10::string_view { return $jnicall; }