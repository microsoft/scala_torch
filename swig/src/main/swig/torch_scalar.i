namespace torch {

struct Scalar {
  Scalar(float value);
  Scalar(double value);
  Scalar(int value);
  Scalar(short value);
  Scalar(int64_t value);
  Scalar(bool value);

  template<class T> T to() const;
  %template(toFloat) to<float>;
  %template(toDouble) to<double>;
  %template(toInt) to<int>;
  %template(toLong) to<int64_t>;
  %template(toBoolean) to<bool>;

  ScalarType type() const;
  %rename(unary_minus) operator-;
  Scalar operator-() const;
  Scalar conj() const;
  Scalar log() const;

  bool isFloatingPoint() const;
  bool isIntegral(bool includeBool) const;
  bool isComplex() const;
  bool isBoolean() const;
};
TO_STRING_FROM_OSTREAM(Scalar);
}

ARRAY_REF_OF_OBJECT(ScalarList, torch::Scalar)
