// based on https://gist.github.com/vadz/7ba2bfb73d04483e2f6254b05feb7e1f

// Typemap that makes the c10::optional type in swig appear as scala.Option

namespace c10 {
template<typename T>
struct optional {
  optional();
  optional(T value);
  bool has_value() const noexcept;
  T value();
};
}

%define DEFINE_OPTIONAL(OptT, T)

// Use reference, not pointer, typemaps for member variables of this type.
%naturalvar c10::optional< T >;

%template(OptT) c10::optional< T >;

// Note the use of "jboxtype" instead of just "jstype": for primitive types,
// such as "double", they're different and we must use the former as
// Optional can only be used with reference types in Java.
%typemap(jstype) c10::optional< T >, const c10::optional< T >& "scala.Option<$typemap(jboxtype, T)>"

// This typemap is used for function arguments of this type.
%typemap(javain,
         pre="    OptT opt$javainput = $javainput.isDefined() ? new OptT($javainput.get()) : new OptT();",
         post="     opt$javainput.delete();",
         pgcppname="opt$javainput")
         c10::optional< T >, const c10::optional< T >& "$javaclassname.getCPtr(opt$javainput)"

// This typemap is for functions returning objects of this type.
%typemap(javaout) c10::optional< T >, const c10::optional< T >& {
    OptT optValue = new OptT($jnicall, $owner);
    if (optValue.has_value()) {
      scala.Option<$typemap(jboxtype, T)> someValue = new scala.Some<$typemap(jboxtype, T)>(optValue.value());
      optValue.delete();
      return someValue;
    } else {
      return scala.Option.apply(null);
    }
  }

%enddef

%define PRIMITIVE_OPTIONAL(OptT, T)

// Use reference, not pointer, typemaps for member variables of this type.
%naturalvar c10::optional< T >;

%template(OptT) c10::optional< T >;

// Note the use of "jboxtype" instead of just "jstype": for primitive types,
// such as "double", they're different and we must use the former as
// Optional can only be used with reference types in Java.
%typemap(jstype) c10::optional< T >, const c10::optional< T >& "java.util.Optional""$typemap(jboxtype, T)"

// This typemap is used for function arguments of this type.
%typemap(javain,
         pre= "    OptT opt$javainput = $javainput.isPresent() ? new OptT($javainput.getAs$typemap(jboxtype, T)()) : new OptT();",
         post="      opt$javainput.delete();",
         pgcppname="opt$javainput")
         c10::optional< T >, const c10::optional< T >& "$javaclassname.getCPtr(opt$javainput)"

// This typemap is for functions returning objects of this type.
%typemap(javaout) c10::optional< T >, const c10::optional< T >& {
    OptT optValue = new OptT($jnicall, $owner);
    if (optValue.has_value()) {
      java.util.Optional$typemap(jboxtype, T) someValue = java.util.Optional$typemap(jboxtype, T).of(optValue.value());
      optValue.delete();
      return someValue;
    } else {
      return java.util.Optional$typemap(jboxtype, T).empty();
    }
  }

%enddef

DEFINE_OPTIONAL(OptBool, bool)
DEFINE_OPTIONAL(OptFloat, float)
PRIMITIVE_OPTIONAL(OptDouble, double)
PRIMITIVE_OPTIONAL(OptLong, int64_t)
PRIMITIVE_OPTIONAL(OptInt, int32_t)
DEFINE_OPTIONAL(OptString, std::string)
