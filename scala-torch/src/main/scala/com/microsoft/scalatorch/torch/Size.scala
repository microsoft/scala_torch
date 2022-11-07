package com.microsoft.scalatorch.torch

import scala.collection.compat._
import scala.collection.JavaConverters._

import com.microsoft.scalatorch.torch.internal.LongVector

/** Same as torch.Size. We don't use a simple alias for an Array[Long] because
  * we want a nice toString and structural equality.
  */
class Size(val underlying: immutable.ArraySeq.ofLong) extends AnyVal {
  def rank: Int = underlying.unsafeArray.length

  def sizes: Array[Long] = underlying.unsafeArray

  def numel(): Long = underlying.unsafeArray.product
  override def toString(): String = underlying.mkString("Size(", ", ", ")")

  def apply(i: Int): Long = underlying(i)

}
object Size {
  def apply(array: Array[Long]): Size = new Size(new immutable.ArraySeq.ofLong(array))
  def apply(size: Long*): Size = apply(size.toArray)
  def apply(dims: LongVector): Size = apply(dims.asScala.map(_.toLong).toArray)

  implicit def unwrapSizeToArray(size: Size): Array[Long] = size.sizes
  implicit def unwrapSizeToSeq(size: Size): Seq[Long] = size.underlying
}
