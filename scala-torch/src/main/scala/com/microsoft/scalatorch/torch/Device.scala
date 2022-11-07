package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.jit.CompilationUnit
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.internal

case class Device private[torch] (protected[torch] val underlying: internal.Device)
    extends TorchReference[internal.Device] {
  override protected def delete(): Unit = underlying.delete()

  def tpe: internal.DeviceType = underlyingChecked.`type`()
  def index: Short = underlying.index()
  def isCPU: Boolean = underlyingChecked.is_cpu()
  def isCUDA: Boolean = underlyingChecked.is_cuda()

  override def toString: String = underlyingChecked.str()
}

object Device {
  private[torch] def fromString(str: String): Device = {
    val underlying = new internal.Device(str)
    Disposer.add(new Device(underlying), () => underlying.delete())
  }

  private[torch] def apply(`type`: internal.DeviceType, index: Int): Device = {
    val underlying = new internal.Device(`type`, index.toByte)
    Disposer.add(new Device(underlying), () => underlying.delete())
  }
}
