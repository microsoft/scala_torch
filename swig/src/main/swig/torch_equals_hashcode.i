// make a nice toString method using the ostream stuff they use everywhere
%define TO_STRING_FROM_OSTREAM(T)
%extend T {
  std::string toString() const {
    std::ostringstream ss;
    ss << *($self);
    return ss.str();
  }
}
%enddef

%define EQUALS_FROM_EQ(T)

%extend T {
  bool equalTo(const T& o) const {
      return (*$self) == o;
  }


  %proxycode %{
   @Override public boolean equals(Object o) {
       if (o instanceof $javaclassname) {
           return equalTo(($javaclassname)o);
       } else {
           return false;
       }
   }
  %}
}
%enddef

// Unlike other macros in this file, you should use this one inside the class declaration
%define EQUALS_AND_HASH_CODE_FROM_PTR_EQUALITY(T)

%proxycode %{
  @Override public boolean equals(Object obj) {
    boolean equal = false;
    if (obj instanceof $javaclassname) {
      return ((($javaclassname)obj).swigCPtr == this.swigCPtr);
    } else {
      return false;
    }
  }
  @Override public int hashCode() {
     return (int)this.swigCPtr;
  }
%}
%enddef

%define HASHCODE_FROM_STD_HASH(T)

%extend T {
  size_t hash() const {
    return std::hash<T>()(*($self));
  }


  %proxycode %{
    @Override public int hashCode() {
      return (int)hash();
    }
  %}
}
%enddef
