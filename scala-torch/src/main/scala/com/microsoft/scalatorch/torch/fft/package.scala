// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to package.scala.in

package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch._
import com.microsoft.scalatorch.torch.util.Implicits._
import com.microsoft.scalatorch.torch.internal.{ TensorIndex, TensorVector, TorchTensor, LongVector, torch_swig => swig }
import com.microsoft.scalatorch.torch.util.NoGrad

package object fft {
// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT
// See swig/src/main/swig/build.sbt for details
  def fft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_fft(self.underlying, n.asJavaLong, dim, norm))
  def ifft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_ifft(self.underlying, n.asJavaLong, dim, norm))
  def rfft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_rfft(self.underlying, n.asJavaLong, dim, norm))
  def irfft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_irfft(self.underlying, n.asJavaLong, dim, norm))
  def hfft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_hfft(self.underlying, n.asJavaLong, dim, norm))
  def ihfft(self: Tensor, n: Option[Long] = None, dim: Long = -1, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_ihfft(self.underlying, n.asJavaLong, dim, norm))
  def fft2(self: Tensor, s: Option[Array[Long]] = None, dim: Array[Long] = Array(-2,-1), norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_fft2(self.underlying, s, dim, norm))
  def ifft2(self: Tensor, s: Option[Array[Long]] = None, dim: Array[Long] = Array(-2,-1), norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_ifft2(self.underlying, s, dim, norm))
  def rfft2(self: Tensor, s: Option[Array[Long]] = None, dim: Array[Long] = Array(-2,-1), norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_rfft2(self.underlying, s, dim, norm))
  def irfft2(self: Tensor, s: Option[Array[Long]] = None, dim: Array[Long] = Array(-2,-1), norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_irfft2(self.underlying, s, dim, norm))
  def fftn(self: Tensor, s: Option[Array[Long]] = None, dim: Option[Array[Long]] = None, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_fftn(self.underlying, s, dim, norm))
  def ifftn(self: Tensor, s: Option[Array[Long]] = None, dim: Option[Array[Long]] = None, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_ifftn(self.underlying, s, dim, norm))
  def rfftn(self: Tensor, s: Option[Array[Long]] = None, dim: Option[Array[Long]] = None, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_rfftn(self.underlying, s, dim, norm))
  def irfftn(self: Tensor, s: Option[Array[Long]] = None, dim: Option[Array[Long]] = None, norm: Option[String] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_irfftn(self.underlying, s, dim, norm))
  def fftfreq(n: Long, d: Double = 1.0, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.fft_fftfreq(n, d, options))
}

  def rfftfreq(n: Long, d: Double = 1.0, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.fft_rfftfreq(n, d, options))
}

  def fftshift(self: Tensor, dim: Option[Array[Long]] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_fftshift(self.underlying, dim))
  def ifftshift(self: Tensor, dim: Option[Array[Long]] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.fft_ifftshift(self.underlying, dim))}