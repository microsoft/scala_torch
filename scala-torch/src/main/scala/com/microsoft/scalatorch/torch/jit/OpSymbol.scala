package com.microsoft.scalatorch.torch.jit

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.Symbol
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal

/** Names a Torch op. */
case class OpSymbol(private[torch] val underlying: Symbol) {
  override def toString: String = underlying.toDisplayString
}

object OpSymbol {
  // TODO think about whether we need to do anything with the "qualified" part of QualifiedName.
  def apply(s: String) = new OpSymbol(internal.Symbol.fromQualString(s))
}
