/* -----------------------------------------------------------------------------
 * Like std_pair.i, but specialized for std::tuples with two arguments
 * Swig can't handle variadic templated types like std::tuple. So, instead,
 * we make an individual class for each arity. Importantly, each std::tuple
 * is implicitly convertible (https://en.cppreference.com/w/cpp/language/implicit_conversion)
 * to each tupleN declared here. This means that, somewhat sneakily,
 * if a C++ function has a signature like
 * std::tuple<A, B> func();
 * then you can write the following declaration in a swig file
 * tuple2<A, B> func();
 * and the swig-generated code will still compile because the implicit conversion
 * from std::tuple -> tuple2.
 *
 * ----------------------------------------------------------------------------- */

%inline {
template<class T, class U> struct tuple2 {
  T first;
  U second;

  tuple2(const T& f, const U& s):first(f), second(s) {}

  // can be used for implicit conversion from std::tuple<,>
  tuple2(const std::tuple<T, U> &other):tuple2(std::get<0>(other), std::get<1>(other)) {}
};
}

// For concrete instantation tuple2<A, B>, you will need to call this macr.
// e.g. DEFINE_TUPLE2(TensorBoolTuple, Tensor, bool)
%define DEFINE_TUPLE_2(JTuple, T1, T2)

%template(JTuple) tuple2< T1, T2 >;

// Wrap with an actual Scala tuple
%typemap(jstype) tuple2< T1, T2 >, const tuple2< T1, T2 >& "scala.Tuple2<$typemap(jboxtype, T1), $typemap(jboxtype, T2)>"
%typemap(javaout) tuple2< T1, T2 >, const tuple2< T1, T2 >& {
    JTuple jTuple = new JTuple($jnicall, $owner);
    try {
      return new scala.Tuple2<$typemap(jboxtype, T1), $typemap(jboxtype, T2)>(jTuple.getFirst(), jTuple.getSecond());
    } finally {
      jTuple.delete();
    }
  }

%enddef

%inline {
template<class T, class U, class V> struct tuple3 {
  T first;
  U second;
  V third;

  tuple3(const T& f, const U& s, const V& v):first(f), second(s), third(v) {}
  tuple3(const std::tuple<T, U, V> &other):tuple3(std::get<0>(other), std::get<1>(other), std::get<2>(other)) {}
};
}

%define DEFINE_TUPLE_3(JTuple, T1, T2, T3)

%template(JTuple) tuple3< T1, T2, T3 >;

// Wrap with an actual Scala tuple
%typemap(jstype) tuple3< T1, T2, T3 >, const tuple3< T1, T2, T3 >& "scala.Tuple3<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3)>"
%typemap(javaout) tuple3< T1, T2, T3 >, const tuple3< T1, T2, T3 >& {
    JTuple jTuple = new JTuple($jnicall, $owner);
    try {
      return new scala.Tuple3<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3)>(jTuple.getFirst(), jTuple.getSecond(), jTuple.getThird());
    } finally {
      jTuple.delete();
    }
  }

%enddef

%inline {
template<class T, class U, class V, class W> struct tuple4 {
  T first;
  U second;
  V third;
  W fourth;

  tuple4(const T& f, const U& s, const V& v, const W& w):first(f), second(s), third(v), fourth(w) {}
  tuple4(const std::tuple<T, U, V, W> &other):tuple4(std::get<0>(other), std::get<1>(other), std::get<2>(other), std::get<3>(other)) {}
};
}

%define DEFINE_TUPLE_4(JTuple, T1, T2, T3, T4)

%template(JTuple) tuple4< T1, T2, T3, T4 >;

// Wrap with an actual Scala tuple
%typemap(jstype) tuple4< T1, T2, T3, T4 >, const tuple4< T1, T2, T3, T4 >& "scala.Tuple4<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3), $typemap(jboxtype, T4)>"
%typemap(javaout) tuple4< T1, T2, T3, T4 >, const tuple4< T1, T2, T3, T4 >& {
    JTuple jTuple = new JTuple($jnicall, $owner);
    try {
      return new scala.Tuple4<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3), $typemap(jboxtype, T4)>(jTuple.getFirst(), jTuple.getSecond(), jTuple.getThird(), jTuple.getFourth());
    } finally {
      jTuple.delete();
    }
  }

%enddef

%inline {
template<class T, class U, class V, class W, class X> struct tuple5 {
  T first;
  U second;
  V third;
  W fourth;
  X fifth;

  tuple5(const T& f, const U& s, const V& v, const W& w, const X&x ):first(f), second(s), third(v), fourth(w), fifth(x) {}
  tuple5(const std::tuple<T, U, V, W, X> &other):tuple5(std::get<0>(other), std::get<1>(other), std::get<2>(other), std::get<3>(other), std::get<4>(other)) {}
};
}

%define DEFINE_TUPLE_5(JTuple, T1, T2, T3, T4, T5)

%template(JTuple) tuple5< T1, T2, T3, T4, T5 >;

// Wrap with an actual Scala tuple
%typemap(jstype) tuple5< T1, T2, T3, T4, T5 >, const tuple5< T1, T2, T3, T4, T5 >& "scala.Tuple5<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3), $typemap(jboxtype, T4), $typemap(jboxtype, T5)>"
%typemap(javaout) tuple5< T1, T2, T3, T4, T5 >, const tuple5< T1, T2, T3, T4, T5 >& {
    JTuple jTuple = new JTuple($jnicall, $owner);
    try {
      return new scala.Tuple5<$typemap(jboxtype, T1), $typemap(jboxtype, T2), $typemap(jboxtype, T3), $typemap(jboxtype, T4), $typemap(jboxtype, T5)>(jTuple.getFirst(), jTuple.getSecond(), jTuple.getThird(), jTuple.getFourth(), jTuple.getFifth());
    } finally {
      jTuple.delete();
    }
  }

%enddef
