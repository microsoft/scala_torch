package com.microsoft.scalatorch.torch.util

import com.microsoft.scalatorch.torch.internal.GradMode

/** Like PyTorch's no_grad. Sets a threadlocal variable, then unsets at the end (unless the bit was
  * already enabled at invocation).
  *
  * Useful for directly manipulating parameters. Generally you should avoid this unless you know what you're doing.
  * https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch
  */
object NoGrad {
  def noGrad[T](body: => T): T = {
    val wasEnabled = GradMode.is_enabled()
    GradMode.set_enabled(false)

    val x =
      try {
        body
      } finally {
        if (GradMode.is_enabled()) {
          throw new IllegalStateException("Inconsistent state with GradMode, someone turned it on that shouldn't have.")
        }
        GradMode.set_enabled(wasEnabled)
      }
    x
  }
}
