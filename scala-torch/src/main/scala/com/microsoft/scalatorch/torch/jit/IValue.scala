package com.microsoft.scalatorch.torch.jit

import java.io.File
import java.lang

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.torch_swig
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch._
import com.microsoft.scalatorch.torch.internal.{ torch_swig, IValueVector }

/** An IValue in C++ land is a "generic tagged union used by the interpreter to hold
  * all value types."
  *
  * They're used by [[Module]], which are sort of necessarily dynamically typed.
  */
class IValue private[torch] (protected[torch] val underlying: internal.IValue)(rm: ReferenceManager)
    extends TorchReference[internal.IValue] {
  def scriptType: Type = Type(underlying.`type`())

  rm.addReference(this)
  override protected def delete(): Unit = underlying.delete()

  def asTensor(implicit rm: ReferenceManager): Tensor = Tensor(underlyingChecked.toTensor)
  def asDouble: Double = underlyingChecked.toDouble
  def asLong: Long = underlyingChecked.toInt
  def asBoolean: Boolean = underlyingChecked.toBool
  def asString: String = underlyingChecked.toStringRef
  def asScalar(implicit rm: ReferenceManager): Scalar = Scalar(underlyingChecked.toScalar)
}

object IValue {
  def apply(underlying: internal.IValue)(implicit rm: ReferenceManager): IValue = new IValue(underlying)(rm)
  def fromPickle(f: File)(implicit rm: ReferenceManager): IValue = {
    new IValue(torch_swig.unpickle_from_file(f.toString))(rm)
  }

  def none(implicit rm: ReferenceManager): IValue = IValue(new internal.IValue())(rm)

  implicit def fromTensor(t: Tensor)(implicit rm: ReferenceManager): IValue = {
    new IValue(new internal.IValue(t.underlyingChecked))(rm)
  }

  implicit def fromModule(m: Module)(implicit rm: ReferenceManager): IValue =
    IValue(new internal.IValue(m.underlyingChecked))(rm)

  implicit def fromDouble(t: Double)(implicit rm: ReferenceManager): IValue = IValue(new internal.IValue(t))(rm)

  implicit def fromLong(t: Long)(implicit rm: ReferenceManager): IValue = IValue(new internal.IValue(t))(rm)

  implicit def fromBoolean(t: Boolean)(implicit rm: ReferenceManager): IValue = IValue(new internal.IValue(t))(rm)

  implicit def fromLongs(t: Array[Long])(
      implicit rm: ReferenceManager,
  ): IValue = IValue(new internal.IValue(t))(rm)

  implicit def fromString(t: String)(implicit rm: ReferenceManager): IValue = IValue(new internal.IValue(t))(rm)

  // TODO: fromDoubles
  // TODO: TensorList
  implicit def fromScalar(t: Scalar)(implicit rm: ReferenceManager): IValue = {
    IValue(new internal.IValue(t.underlyingChecked))(rm)
  }
}
