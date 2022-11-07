package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal

/** Scalar represents a 0-dimensional tensor which contains a single element.
  * Wraps the C++ class of the same name.
  *
  * @see https://github.com/pytorch/pytorch/tree/master/c10/core/Scalar.h
  */
class Scalar private (protected[torch] val underlying: internal.Scalar) extends TorchReference[internal.Scalar] {
  def toFloat: Float = underlying.toFloat
  def toDouble: Double = underlying.toDouble
  def toInt: Int = underlying.toInt
  def toLong: Long = underlying.toLong
  def toBoolean: Boolean = underlying.toBoolean

  override def toString: String = underlying.toString

  def `type`()(implicit rm: ReferenceManager): dtype = dtype(underlying.`type`())
  def unary_-(implicit rm: ReferenceManager): Scalar = Scalar(underlying.unary_minus())
  def conj()(implicit rm: ReferenceManager): Scalar = Scalar(underlying.conj())
  def log()(implicit rm: ReferenceManager): Scalar = Scalar(underlying.log())

  def isFloatingPoint(): Boolean = underlying.isFloatingPoint()

  def isIntegral(includeBool: Boolean): Boolean = underlying.isIntegral(includeBool)

  def isComplex(): Boolean = underlying.isComplex()

  def isBoolean(): Boolean = underlying.isBoolean()

  override protected def delete(): Unit = underlying.delete()
}

object Scalar {
  private[torch] def apply(underlying: internal.Scalar)(implicit manager: ReferenceManager): Scalar = {
    manager.addReference(new Scalar(underlying))
  }

  implicit def fromFloat(f: Float)(implicit manager: ReferenceManager): Scalar = Scalar(new internal.Scalar(f))
  implicit def fromInt(f: Int)(implicit manager: ReferenceManager): Scalar = Scalar(new internal.Scalar(f))
  implicit def fromBoolean(f: Boolean)(implicit manager: ReferenceManager): Scalar = Scalar(new internal.Scalar(f))
  implicit def fromDouble(f: Double)(implicit manager: ReferenceManager): Scalar = Scalar(new internal.Scalar(f))
  implicit def fromLong(f: Long)(implicit manager: ReferenceManager): Scalar = Scalar(new internal.Scalar(f))
}
