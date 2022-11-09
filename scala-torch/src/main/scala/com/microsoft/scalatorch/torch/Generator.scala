package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.internal

/** @see https://pytorch.org/docs/stable/generated/torch.Generator.html
  */
class Generator private (private[torch] val underlying: internal.Generator) {
  def device: device = new Device(underlying.device())
  def get_state(implicit rm: ReferenceManager): Tensor = Tensor(underlying.get_state())
  def set_state(state: Tensor): Generator = {
    underlying.set_state(state.underlying)
    this
  }
  def initial_seed: Long = underlying.initial_seed()
  def manual_seed(seed: Long): Unit = underlying.manual_seed(seed)
  def seed(): Long = underlying.seed()
}

object Generator {
  private[torch] def apply(underlying: internal.Generator): Generator = {
    Disposer.add(new Generator(underlying), () => underlying.delete())
  }
}
