package com.microsoft.scalatorch.torch.jit

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch._
import com.microsoft.scalatorch.torch.util.Disposer

class CompilationUnit private (protected[torch] val underlying: internal.CompilationUnit)
    extends TorchReference[internal.CompilationUnit] {
  override protected def delete(): Unit = underlying.delete()
}

object CompilationUnit {

  private[torch] def apply(underlying: internal.CompilationUnit): CompilationUnit = {
    Disposer.add(new CompilationUnit(underlying), () => underlying.delete())
  }
}
