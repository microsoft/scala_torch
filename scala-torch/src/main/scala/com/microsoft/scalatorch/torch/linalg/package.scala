// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to package.scala.in

package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch._
import com.microsoft.scalatorch.torch.util.Implicits._
import com.microsoft.scalatorch.torch.internal.{ TensorIndex, TensorVector, TorchTensor, LongVector, torch_swig => swig }
import com.microsoft.scalatorch.torch.util.NoGrad

package object linalg {
// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT
// See swig/src/main/swig/build.sbt for details
  def cholesky_ex(self: Tensor, upper: Boolean = false, check_errors: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_cholesky_ex(self.underlying, upper, check_errors))
  def cholesky(self: Tensor, upper: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_cholesky(self.underlying, upper))
  def det(self: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_det(self.underlying))
  def lstsq(self: Tensor, b: Tensor, rcond: Option[Double] = None, driver: Option[String] = None)(implicit rm: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig.linalg_lstsq(self.underlying, b.underlying, rcond.asJavaDouble, driver))
  def matmul(self: Tensor, other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matmul(self.underlying, other.underlying))
  def slogdet(self: Tensor)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_slogdet(self.underlying))
  def eig(self: Tensor)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_eig(self.underlying))
  def eigvals(self: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_eigvals(self.underlying))
  def eigh(self: Tensor, UPLO: String = "L")(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_eigh(self.underlying, UPLO))
  def eigvalsh(self: Tensor, UPLO: String = "L")(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_eigvalsh(self.underlying, UPLO))
  def householder_product(input: Tensor, tau: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_householder_product(input.underlying, tau.underlying))
  def inv_ex(self: Tensor, check_errors: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_inv_ex(self.underlying, check_errors))
  def inv(self: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_inv(self.underlying))
  def norm(self: Tensor, ord: Option[Scalar] = None, dim: Option[Array[Long]] = None, keepdim: Boolean = false, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_norm(self.underlying, ord.map(_.underlying), dim, keepdim, dtype.map(_.toScalarType)))
  def norm(self: Tensor, ord: String, dim: Option[Array[Long]], keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_norm(self.underlying, ord, dim, keepdim, dtype.map(_.toScalarType)))
  def vector_norm(self: Tensor, ord: Double = 2, dim: Option[Array[Long]] = None, keepdim: Boolean = false, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_vector_norm(self.underlying, ord.toInternalScalar, dim, keepdim, dtype.map(_.toScalarType)))
  def matrix_norm(self: Tensor, ord: Scalar, dim: Array[Long] = Array(-2,-1), keepdim: Boolean = false, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matrix_norm(self.underlying, ord.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def matrix_norm(self: Tensor, ord: String, dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matrix_norm(self.underlying, ord, dim, keepdim, dtype.map(_.toScalarType)))
  def svd(self: Tensor, full_matrices: Boolean = true)(implicit rm: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.linalg_svd(self.underlying, full_matrices))
  def svdvals(input: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_svdvals(input.underlying))
  def cond(self: Tensor, p: Option[Scalar] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_cond(self.underlying, p.map(_.underlying)))
  def cond(self: Tensor, p: String)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_cond(self.underlying, p))
  def pinv(self: Tensor, rcond: Double = 1e-15, hermitian: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_pinv(self.underlying, rcond, hermitian))
  def pinv(self: Tensor, rcond: Tensor, hermitian: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_pinv(self.underlying, rcond.underlying, hermitian))
  def solve(input: Tensor, other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_solve(input.underlying, other.underlying))
  def tensorinv(self: Tensor, ind: Long = 2)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_tensorinv(self.underlying, ind))
  def tensorsolve(self: Tensor, other: Tensor, dims: Option[Array[Long]] = None)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_tensorsolve(self.underlying, other.underlying, dims))
  def qr(self: Tensor, mode: String = "reduced")(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.linalg_qr(self.underlying, mode))
  def matrix_power(self: Tensor, n: Long)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matrix_power(self.underlying, n))
  def matrix_rank(self: Tensor, tol: Option[Double] = None, hermitian: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matrix_rank(self.underlying, tol.asJavaDouble, hermitian))
  def matrix_rank(input: Tensor, tol: Tensor, hermitian: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_matrix_rank(input.underlying, tol.underlying, hermitian))
  def multi_dot(tensors: Array[Tensor])(implicit rm: ReferenceManager): Tensor = Tensor(swig.linalg_multi_dot(tensors.map(_.underlyingChecked)))}
