package com.microsoft.scalatorch.torch.util

import com.microsoft.scalatorch.torch.internal.RecordProfile

/** Profiles `body` and dumps the output to `file`. File can be viewed in chrome://tracing.
  *
  * Copied from profiler.h:
  * NOTE: changing profiler modes is **NOT THREAD SAFE**. You should ensure that
  * there no autograd functions are being executed when these function are used.
  */
object Profiler {
  def profile[T](file: String)(body: => T): T = {
    resource.makeManagedResource(new RecordProfile(file))(_.delete())(List.empty).apply(_ => body)
  }
}
