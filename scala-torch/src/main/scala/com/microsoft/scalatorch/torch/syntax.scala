package com.microsoft.scalatorch.torch

import scala.reflect.ClassTag

/** Various implicits to provide syntactic sugar that mirrors the PyTorch API as much as one can do with Scala syntax.
  * Use with {{{import com.microsoft.scalatorch.torch.syntax._}}} */
object syntax {


  /** Since Python has such nice syntax for list literals with [], it's nice to have nearly as short syntax
    * in PyTorch. It's not very idiomatic Scala to do this though, you should really just write `Array`.
    * You are welcome to exclude this particular sugar with
    * {{{import com.microsoft.scalatorch.torch.syntax.{$ => _, _}}}}*/
  def $[T: ClassTag](ts: T*): Array[T] = Array(ts: _*)

  // We could have a generic some[T], but we prefer to limit the use of implicits to just those necessary
  // for pytorch syntax.
  implicit def someTensor(tensor: Tensor): Option[Tensor] = Some(tensor)
  implicit def someBoolean(b: Boolean): Option[Boolean] = Some(b)
  implicit def someDouble(x: Double): Option[Double] = Some(x)
  implicit def someGenerator(x: Generator): Option[Generator] = Some(x)
  implicit def someLayout(x: Layout): Option[Layout] = Some(x)
  implicit def somedtype(x: dtype): Option[dtype] = Some(x)
  implicit def someDevice(x: Device): Option[Device] = Some(x)

  implicit def stringToDevice(s: String): Device = device(s)
  implicit def stringToOptionDevice(s: String): Option[Device] = Some(device(s))

  implicit def anyToTensor(a: Any)(implicit rm: ReferenceManager): Tensor = tensor(a)

  implicit def intToArray(int: Int): Array[Long] = Array(int)
  implicit def intTupleToArray2(intTuple: (Int, Int)): Array[Long] = Array(intTuple._1, intTuple._2)
  implicit def intTupleToArray3(intTuple: (Int, Int, Int)): Array[Long] = Array(intTuple._1, intTuple._2, intTuple._3)
  implicit def intTupleToArray4(intTuple: (Int, Int, Int, Int)): Array[Long] =
    Array(intTuple._1, intTuple._2, intTuple._3, intTuple._4)
  implicit def intTupleToArray5(intTuple: (Int, Int, Int, Int, Int)): Array[Long] =
    Array(intTuple._1, intTuple._2, intTuple._3, intTuple._4, intTuple._5)

  implicit def boolTupleToArray2(boolTuple: (Boolean, Boolean)): Array[Boolean] = Array(boolTuple._1, boolTuple._2)
  implicit def boolTupleToArray3(boolTuple: (Boolean, Boolean, Boolean)): Array[Boolean] =
    Array(boolTuple._1, boolTuple._2, boolTuple._3)
  implicit def boolTupleToArray4(boolTuple: (Boolean, Boolean, Boolean, Boolean)): Array[Boolean] =
    Array(boolTuple._1, boolTuple._2, boolTuple._3, boolTuple._4)

  implicit def reductionToLong(reduction: Reduction): Long = reduction.swigValue()

  /** Supports the indexing documents in [[Tensor.apply(syntax.Indexer*)(ReferenceManager)]]
    */
  sealed trait Indexer

  object Indexer {

    /** Used across varies indexing syntaxes, see use cases below.
      */
    private[torch] case class RangeStepIndexer(
        bottom: java.util.OptionalLong,
        top: java.util.OptionalLong,
        step: java.util.OptionalLong,
    ) extends Indexer

    /** Allows for the syntax {{{x(1 -> 2)}}}. Python syntax ({{{x[1:2]}}}) is not possible, both because of
      * of the different meaning of square and round parens, and also because : is not available operator in Scala.
      * Note that {{{x(1::2)}}} and {{{x(::)}}} are both possible and match the meaning in Python.
      */
    implicit def intPairToIndexer(pair: (Int, Int)): Indexer = RangeStepIndexer(
      java.util.OptionalLong.of(pair._1),
      java.util.OptionalLong.of(pair._2),
      java.util.OptionalLong.empty(),
    )

    /** Allows for the syntax {{{x(1 -> 2 -> 3)}}}, matching Python's {{{x[1:2:3]}}}
      */
    implicit def intTripleToIndexer(triple: ((Int, Int), Int)): Indexer = RangeStepIndexer(
      java.util.OptionalLong.of(triple._1._1),
      java.util.OptionalLong.of(triple._1._2),
      java.util.OptionalLong.of(triple._2),
    )

    /** Allow for {{{foo(1)}}}
      */
    implicit def intToElemIndexer(elem: Int): Indexer = ElemIndexer(elem)
    private[torch] case class ElemIndexer(elem: Int) extends Indexer

    /** Allow for {{{foo(true)}}}
      */
    implicit def boolToBoolIndexer(bool: Boolean): Indexer = BoolIndexer(bool)
    private[torch] case class BoolIndexer(bool: Boolean) extends Indexer

    /** Allow for {{{foo(None)}}}
      */
    implicit def noneToRangeIndexer(none: None.type): Indexer =
      RangeStepIndexer(java.util.OptionalLong.empty(), java.util.OptionalLong.empty(), java.util.OptionalLong.empty())
  }

  /** Ellipsis (...) in python
    */
  case object --- extends Indexer

  /** Range (colon) in python
    */
  val :: : Indexer = None

  /** Allow for {{{foo(1.::)}}} and {{{foo(1.::(2)}}}
    */
  implicit class RichInteger(val bottom: Int) extends AnyVal {
    def ::(step: Int): Indexer = {
      // Note that despite the names, :: reverses the operators, that is a :: b calls b.::(a)
      // So step and bottom are reversed here
      Indexer.RangeStepIndexer(
        java.util.OptionalLong.of(step),
        java.util.OptionalLong.empty(),
        java.util.OptionalLong.of(bottom),
      )
    }

    def :: : Indexer =
      Indexer.RangeStepIndexer(
        java.util.OptionalLong.of(bottom),
        java.util.OptionalLong.empty(),
        java.util.OptionalLong.empty(),
      )
  }
}
