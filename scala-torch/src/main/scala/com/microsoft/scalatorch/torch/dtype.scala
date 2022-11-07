package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, ScalarType, TypeMeta }
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, ScalarType, TypeMeta }

/** @see https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
  */
class dtype private[torch] (private[torch] val underlying: TypeMeta) {

  def is_complex: Boolean = {
    torch_swig.isComplexType(underlying.toScalarType)
  }

  def is_floating_point: Boolean = {
    torch_swig.isFloatingType(underlying.toScalarType)
  }

  def is_signed: Boolean = {
    internal.torch_swig.isSignedType(underlying.toScalarType)
  }

  override def toString: String = underlying.name

  override def equals(o: Any): Boolean = o match {
    case that: dtype => underlying.equalTo(that.underlying)
  }

  def toScalarType: ScalarType = underlying.toScalarType
}

object dtype {
  private[torch] def apply(underlying: ScalarType)(implicit cg: ReferenceManager): dtype = {
    cg.addReference(new dtype(TypeMeta.fromScalarType(underlying)))
  }
}
