package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ Layout, TypeMeta }
import com.microsoft.scalatorch.torch.internal
import resource.ManagedResource
import com.microsoft.scalatorch.torch.syntax._

// TODO: devices
/** Class for configuring the underlying properties of a tensor. Most notably the [[dtype]]
  * and (eventually) device.
  *
  * @see https://pytorch.org/cppdocs/notes/tensor_creation.html#configuring-properties-of-the-tensor
  */
case class TensorOptions(
    dtype: Option[dtype] = None,
    device: Option[Device] = None,
    layout: Option[Layout] = None,
    requires_grad: Option[Boolean] = None,
    pinned_memory: Option[Boolean] = None,
) {
  private[torch] def toInternal: ManagedResource[internal.TensorOptions] = {
    def man(o: => internal.TensorOptions) = resource.makeManagedResource(o)(_.delete())(List.empty)
    // TODO will this be a perf problem?
    for {
      orig <- man(new internal.TensorOptions())
      afterDtype <- man(orig.dtype(dtype.map(_.underlying)))
      afterDevice <- man(afterDtype.device(device.map(_.underlying)))
      afterLayout <- man(afterDevice.layout(layout))
      afterRequiredsGrad <- man(afterLayout.requires_grad(requires_grad.map(java.lang.Boolean.valueOf)))
      afterPinnedMemory <- man(afterRequiredsGrad.pinned_memory(pinned_memory.map(java.lang.Boolean.valueOf)))
    } yield afterPinnedMemory
  }
}

object TensorOptions {
  // implicits to more or less mirror the torch api
  implicit def fromMeta(meta: TypeMeta): TensorOptions = TensorOptions(Some(new dtype(meta)))

  val default: TensorOptions = TensorOptions()
}
