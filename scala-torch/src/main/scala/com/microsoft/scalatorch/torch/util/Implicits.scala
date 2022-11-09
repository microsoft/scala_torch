package com.microsoft.scalatorch.torch.util

import java.util.{ OptionalDouble, OptionalLong }

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.{ internal, ReferenceManager, Scalar }

private[torch] object Implicits {
  implicit class RichOptionDouble(private val option: Option[Double]) extends AnyVal {
    def asJavaDouble: OptionalDouble = {
      option.fold(OptionalDouble.empty())(d => OptionalDouble.of(d))
    }
  }

  implicit class RichOptionLong(private val option: Option[Long]) extends AnyVal {
    def asJavaLong: OptionalLong = {
      option.fold(OptionalLong.empty())(l => OptionalLong.of(l))
    }
  }

  implicit class RichDouble(private val d: Double) extends AnyVal {
    def toInternalScalar(implicit cg: ReferenceManager): internal.Scalar = {
      Scalar.fromDouble(d).underlying
    }
  }
}
