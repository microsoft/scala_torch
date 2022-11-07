%{
#include <ATen/core/List.h>
%}

namespace c10 {
template<class T>
class List final {
public:
  //List(std::shared_ptr<c10::Type> elementType);
  void push_back(const T& value) const;
  void reserve(size_t new_cap) const;
  %extend{

    // This is present in master but not in the version (1.4) that we're pinned to.
    std::vector<T> vec() const {
      std::vector<T> result(($self)->begin(), ($self)->end());
      return result;
    }
  }

private:
   explicit List();
};
}

%define DEFINE_LIST_OF_OPTIONAL(ListT, T)

%template(ListT) c10::List< c10::optional< T > >;
%naturalvar c10::List< T >;

%typemap(jni) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > &  "jlongArray" // Use jlongArray for CPtrs, really these are objects
%typemap(jstype) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > & "scala.Option<$typemap(jboxtype, T)>[]"
%typemap(jtype) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > & "long[]"
%typemap(javain) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > &  "$javainput"

// Here, we pass in an array of long representing pointers to negative objects. We use -1 to indicate an optional
// value not present. I am pretty sure that points must be positive and so -1 is safe, but we're in
// for a vanishngly rare but really bad time if that ends up not being true.
%typemap(javain,
         pre="    long[] cptrs$javainput = new long[$javainput.length]; for (int i = 0; i < $javainput.length; ++i) { if ($javainput[i].isEmpty()) { cptrs$javainput[i] = -1; } else { cptrs$javainput[i] = $typemap(jboxtype, T).getCPtr($javainput[i].get()); } }",
         pgcppname="cptrs$javainput")
         c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > & "cptrs$javainput"

%typemap(in) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > & {

  $1 = new c10::List< c10::optional< T > >();
  auto elems$1 = (jenv)->GetLongArrayElements($input, nullptr);
  size_t len$1 = (jenv)->GetArrayLength($input);
  $1->reserve(len$1);
  for (size_t i = 0; i < len$1; ++i) {
    if (elems$1[i] == -1) {
      $1->push_back(c10::nullopt);
    } else {
      $1->push_back(*(T*)elems$1[i]);
    }
  }
  (jenv)->ReleaseLongArrayElements($input, elems$1, 0);
}

%typemap(freearg) c10::List< c10::optional< T > >, const c10::List< c10::optional< T > > & {
  delete $1;
}

%enddef
