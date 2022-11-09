package com.microsoft.scalatorch.torch.jit

import com.microsoft.scalatorch.torch.{ internal, Size }
import com.microsoft.scalatorch.torch.internal.{ DeviceType, TypeMeta, TypeVector }
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.{ internal, Size }
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.{ float, internal, Size }
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.syntax._
import com.microsoft.scalatorch.torch.internal

/** A wrapper around c10::Type at https://github.com/pytorch/pytorch/blob/v1.4.0/aten/src/ATen/core/jit_type.h#L65. */
class Type protected (protected[torch] val underlying: internal.Type) {
  override def toString = underlying.toString()
  override def equals(o: Any): Boolean = o match {
    case t: Type => underlying == t.underlying
    case _       => false
  }

  // TODO figure out a hash code that matches with equals if we ever use this is a key in a HashMap.
  // Currently we only support hashing on [[TensorType]].
  override def hashCode: Int = ???
}

object Type {
  private[torch] def apply(underlying: internal.Type): Type = {
    Disposer.add(new Type(underlying), () => underlying.delete())
  }

  def createDict(keyType: Type, valueType: Type): Type = {
    Type(internal.Type.createDict(keyType.underlying, valueType.underlying))
  }

  def createList(elementType: Type): Type = {
    Type(internal.Type.createList(elementType.underlying))
  }

  val string: Type = Type(internal.Type.getString)
  val float: Type = Type(internal.Type.getFloat)
  val bool: Type = Type(internal.Type.getBool)
  val int: Type = Type(internal.Type.getInt)
  val tensor: TensorType = TensorType(internal.TensorType.get())
}

// TODO more static types mirroring TorchScript's typesystem. It's inconsistent that we only have a static
//   wrapper for TupleType.
class TupleType(override protected[torch] val underlying: internal.TupleType) extends Type(underlying)

object TupleType {
  private[torch] def apply(underlying: internal.TupleType): TupleType = {
    Disposer.add(new TupleType(underlying), () => underlying.delete())
  }

  def create(fieldTypes: Seq[Type]): TupleType = {
    val typeVector = new TypeVector(fieldTypes.map(_.underlying).toArray[internal.Type])
    try TupleType(internal.TupleType.create(typeVector))
    finally typeVector.delete()
  }
}

class ClassType private (override protected[torch] val underlying: internal.ClassType) extends Type(underlying) {
  lazy val qualifiedName: Option[String] = {
    underlying.name().map { name =>
      try name.qualifiedName()
      finally name.delete()
    }
  }
}

object ClassType {
  private[torch] def apply(underlying: internal.ClassType): ClassType = {
    Disposer.add(new ClassType(underlying), () => underlying.delete())
  }
}

class TensorType private (
    override protected[torch] val underlying: internal.TensorType,
) extends Type(underlying) {
  def shape: Option[Size] = {
    val sizes = underlying.sizes()
    try sizes.map(Size(_))
    finally sizes.foreach(_.delete())
  }

  override def hashCode: Int = {
    (underlying.sizes(), underlying.dtype(), underlying.device()).hashCode
  }
}

object TensorType {
  private[torch] def apply(underlying: internal.TensorType): TensorType = {
    Disposer.add(new TensorType(underlying), () => underlying.delete())
  }

  def create(
      shape: Size,
      typeMeta: TypeMeta = float.underlying,
      deviceType: DeviceType = DeviceType.CPU,
  ): TensorType = {
    TensorType(internal.TensorType.createContiguous(typeMeta, deviceType, shape.sizes))
  }
}
