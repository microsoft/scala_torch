// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to Tensor.scala.in

package com.microsoft.scalatorch.torch

import java.io.File

import scala.collection.JavaConverters._

import com.microsoft.scalatorch.torch.internal.{ TensorIndex, Slice, TensorVector, TorchTensor, torch_swig => swig }
import com.microsoft.scalatorch.torch.jit.TensorType
import com.microsoft.scalatorch.torch.syntax.{anyToTensor =>_, _}
import com.microsoft.scalatorch.torch.util.Implicits._
import com.microsoft.scalatorch.torch.util.NoGrad

/** The central class in Torch. Represents either a parameter, a constant, or an intermediate
  * expression. Stores both value and gradient.
  *
  * @see https://pytorch.org/cppdocs/notes/tensor_basics.html
  * @see https://pytorch.org/cppdocs/notes/tensor_creation.html for creation
  * @see https://pytorch.org/docs/stable/tensors.html for the more complete python docs
  * @see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/TensorBody.h
  *      for the C++ header
  */
class Tensor private (
    val underlying: internal.TorchTensor,
) extends TorchReference[internal.TorchTensor] {

  /** Careful: Java arrays do not have the right equality semantics. You probably want
    * [[sizes]] instead unless you really know what you're doing.
    */
  private def rawSizes: Array[Long] = underlying.sizes()

  def size(): Size = shape

  def shape: Size = Size(rawSizes)

  def device: device = new Device(underlying.device())

  private[torch] def tensorInfo: TensorInfo = {
    TensorType.create(shape, underlying.dtype(), underlying.device().`type`())
  }

  def numel(): Long = shape.numel()

  def T(implicit manager: ReferenceManager): Tensor = this.t()

  def requires_grad(): Boolean = underlying.requires_grad()

  private def indexersToInternal(indexers: Array[syntax.Indexer]): Array[internal.TensorIndex] = {
    indexers.map {
      case syntax.---            => new TensorIndex(InternalEllipsis)
      case Indexer.ElemIndexer(i) => new TensorIndex(i)
      case Indexer.BoolIndexer(b) => new TensorIndex(b)
      case Indexer.RangeStepIndexer(bottom, top, step) =>
        val slice = new internal.Slice(bottom, top, step)
        try new TensorIndex(slice)
        finally slice.delete()
    }
  }

  /** Allows for indexing and slicing syntax similar to what's available in Python. The syntax here
    * attempts to match Python as much as possible within the limits of Scala syntax.
    * | Scala         | Python      |  Notes |
    * | :---:         | :---:       | ---    |
    * | `x(1)`        | `x[1]`      |        |
    * | `x(1,2)`      | `x[1,2]`    |        |
    * | `x(1->2)`     | `x[1:2]`    |        |
    * | `x(1->2->3)`  | `x[1:2:3]`  |        |
    * | `x(::)`       | `x[::]`     |        |
    * | `x(1::2)`     | `x[1::2]`   |        |
    * | `x(1.::)`     | `x[1::]`    |        |
    * | `x(1::)`      | `x[1::]`    | might need to `import scala.language.postfixOps` depending on Scala version     |
    * | `x(None)`     | `x[None]`   |        |
    * | `x(0::)`      | `x[::]`     | No special syntax in Scala for omitting 0 |
    * | `x(0->3)`     | `x[:3]`     | No special syntax in Scala for omitting 0 |
    * | `x(0->3->5)`  | `x[:3:5]`   | No special syntax in Scala for omitting 0 |
    * | `x(1,---)`    | `x[1,...]`  |        |
    * | `x(---,1)`    | `x[...,1]`  |        |
    */
  def apply(indexers: syntax.Indexer*)(implicit manager: ReferenceManager): Tensor = {
    val indexerArray = indexersToInternal(indexers.toArray)
    try Tensor(this.underlying.index(indexerArray))
    finally indexerArray.foreach(_.delete())
  }

  /** Allows for assignment syntax like {{{x(1, 2 -> 3, ::) = someTensor}}}.
    * Unfortunately, Scala doesn't allow for variadic args on update, so we have to do so manually overloading
    * up to a fixed arity. If you need an arity > 5, you can use {{{x(Array(1, 2 -> 3, ....) = someTensor}}}
    * See comment on [[apply]] detailing Tensor slicing/indexing syntax.
    */
  def update(indexer1: syntax.Indexer, rhs: Tensor)(implicit manager: ReferenceManager): this.type = update(Array(indexer1), rhs)
  def update(indexer1: syntax.Indexer, indexer2: syntax.Indexer, rhs: Tensor)(implicit manager: ReferenceManager): this.type = update(Array(indexer1, indexer2), rhs)
  def update(indexer1: syntax.Indexer, indexer2: syntax.Indexer, indexer3: syntax.Indexer, rhs: Tensor)(implicit manager: ReferenceManager): this.type = update(Array(indexer1, indexer3, indexer3), rhs)
  def update(indexer1: syntax.Indexer, indexer2: syntax.Indexer, indexer3: syntax.Indexer, indexer4: syntax.Indexer, rhs: Tensor)(implicit manager: ReferenceManager): this.type = update(Array(indexer1, indexer3, indexer3, indexer4), rhs)
  def update(indexer1: syntax.Indexer, indexer2: syntax.Indexer, indexer3: syntax.Indexer, indexer4: syntax.Indexer, indexer5: syntax.Indexer, rhs: Tensor)(implicit manager: ReferenceManager): this.type = update(Array(indexer1, indexer3, indexer3, indexer4, indexer5), rhs)

  def update(indexers: Array[syntax.Indexer], rhs: Tensor)(implicit manager: ReferenceManager): this.type = {
    val indexerArray = indexersToInternal(indexers)
    try {
      this.underlying.index_put_(indexerArray, rhs.underlying)
      this
    } finally {
      indexerArray.foreach(_.delete())
    }
  }

  // Start of main api

  def backward(): Unit = assertOpen(underlying.backward())

  def toFloat: Float = assertOpen {
    require(numel() == 1)
    underlying.toFloat
  }

  def real(implicit manager: ReferenceManager): Tensor = com.microsoft.scalatorch.torch.real(this)
  def imag(implicit manager: ReferenceManager): Tensor = com.microsoft.scalatorch.torch.imag(this)

  def toArray[T](implicit manifest: Manifest[T] = Manifest.Float): Array[T] = {
    manifest match {
      case Manifest.Float  => swig.to_float_array(underlyingChecked)
      case Manifest.Long   => swig.to_long_array(underlyingChecked)
      case Manifest.Double => swig.to_double_array(underlyingChecked)
      case Manifest.Int    => swig.to_int_array(underlyingChecked)
      case Manifest.Byte   => swig.to_byte_array(underlyingChecked)
    }
  }.asInstanceOf[Array[T]]

  def grad(implicit manager: ReferenceManager): Tensor = Tensor(underlying.grad())

  def unary_-(implicit manager: ReferenceManager): Tensor = unOp(swig.neg)

  def copy()(implicit manager: ReferenceManager): Tensor = {
    val copy = Tensor.empty(shape, TensorOptions(requires_grad = Some(this.requires_grad())))
    copy.copy_(this)
    copy
  }

  def defined(): Boolean = underlying.defined()

  def save(f: File): Unit = com.microsoft.scalatorch.torch.save(this, f.toString)
  def serialize(): Array[Byte] = com.microsoft.scalatorch.torch.serialize(this)

// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT
// See swig/src/main/swig/build.sbt for details
  def _backward(inputs: Array[Tensor], gradient: Option[Tensor] = None, retain_graph: Option[Boolean] = None, create_graph: Boolean = false)(implicit rm: ReferenceManager): Unit = this.underlying._backward(inputs.map(_.underlyingChecked), gradient.map(_.underlying), retain_graph.map(java.lang.Boolean.valueOf), create_graph)
  def set_data(new_data: Tensor)(implicit rm: ReferenceManager): Unit = this.underlying.set_data(new_data.underlying)
  def data()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.data())
  def is_leaf()(implicit rm: ReferenceManager): Boolean = this.underlying.is_leaf()
  def output_nr()(implicit rm: ReferenceManager): Long = this.underlying.output_nr()
  def _version()(implicit rm: ReferenceManager): Long = this.underlying._version()
  def requires_grad_(requires_grad: Boolean = true)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.requires_grad_(requires_grad)
    this
  }
  def retain_grad()(implicit rm: ReferenceManager): Unit = this.underlying.retain_grad()
  def retains_grad()(implicit rm: ReferenceManager): Boolean = this.underlying.retains_grad()
  def _fw_primal(level: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._fw_primal(level))
  def align_as(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.align_as(other.underlying))
  def abs()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.abs())
  def abs_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.abs_()
    this
  }
  def absolute()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.absolute())
  def absolute_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.absolute_()
    this
  }
  def angle()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.angle())
  def sgn()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sgn())
  def sgn_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sgn_()
    this
  }
  def _conj()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._conj())
  def conj()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.conj())
  def _conj_physical()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._conj_physical())
  def conj_physical()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.conj_physical())
  def conj_physical_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.conj_physical_()
    this
  }
  def resolve_conj()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.resolve_conj())
  def resolve_neg()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.resolve_neg())
  def _neg_view()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._neg_view())
  def acos()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.acos())
  def acos_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.acos_()
    this
  }
  def arccos()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arccos())
  def arccos_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arccos_()
    this
  }
  def add(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.add(other.underlying, alpha.toInternalScalar))
  def add_(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.add_(other.underlying, alpha.toInternalScalar)
    this
  }
  def add(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.add(other.underlying, alpha.underlying))
  def add_(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.add_(other.underlying, alpha.underlying)
    this
  }
  def addmv(mat: Tensor, vec: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addmv(mat.underlying, vec.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def addmv_(mat: Tensor, vec: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addmv_(mat.underlying, vec.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    this
  }
  def addr(vec1: Tensor, vec2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addr(vec1.underlying, vec2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def addr_(vec1: Tensor, vec2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addr_(vec1.underlying, vec2.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    this
  }
  def all(dim: Long, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.all(dim, keepdim))
  def allclose(other: Tensor, rtol: Double = 1e-05, atol: Double = 1e-08, equal_nan: Boolean = false)(implicit rm: ReferenceManager): Boolean = this.underlying.allclose(other.underlying, rtol, atol, equal_nan)
  def any(dim: Long, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.any(dim, keepdim))
  def argmax(dim: Option[Long] = None, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.argmax(dim.asJavaLong, keepdim))
  def argmin(dim: Option[Long] = None, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.argmin(dim.asJavaLong, keepdim))
  def acosh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.acosh())
  def acosh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.acosh_()
    this
  }
  def arccosh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arccosh())
  def arccosh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arccosh_()
    this
  }
  def asinh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.asinh())
  def asinh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.asinh_()
    this
  }
  def arcsinh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arcsinh())
  def arcsinh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arcsinh_()
    this
  }
  def atanh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.atanh())
  def atanh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.atanh_()
    this
  }
  def arctanh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arctanh())
  def arctanh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arctanh_()
    this
  }
  def as_strided(size: Array[Long], stride: Array[Long], storage_offset: Option[Long] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.as_strided(size, stride, storage_offset.asJavaLong))
  def as_strided_(size: Array[Long], stride: Array[Long], storage_offset: Option[Long] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.as_strided_(size, stride, storage_offset.asJavaLong)
    this
  }
  def asin()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.asin())
  def asin_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.asin_()
    this
  }
  def arcsin()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arcsin())
  def arcsin_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arcsin_()
    this
  }
  def atan()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.atan())
  def atan_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.atan_()
    this
  }
  def arctan()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.arctan())
  def arctan_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.arctan_()
    this
  }
  def baddbmm(batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.baddbmm(batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def baddbmm_(batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.baddbmm_(batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    this
  }
  def bernoulli(generator: Option[Generator] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bernoulli(generator.map(_.underlying)))
  def bernoulli_(p: Tensor, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bernoulli_(p.underlying, generator.map(_.underlying))
    this
  }
  def bernoulli_(p: Double, generator: Option[Generator])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bernoulli_(p, generator.map(_.underlying))
    this
  }
  def bernoulli(p: Double, generator: Option[Generator])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bernoulli(p, generator.map(_.underlying)))
  def bincount(weights: Option[Tensor] = None, minlength: Long = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bincount(weights.map(_.underlying), minlength))
  def bitwise_not()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_not())
  def bitwise_not_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_not_()
    this
  }
  def copysign(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.copysign(other.underlying))
  def copysign_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.copysign_(other.underlying)
    this
  }
  def copysign(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.copysign(other.underlying))
  def copysign_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.copysign_(other.underlying)
    this
  }
  def logical_not()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logical_not())
  def logical_not_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.logical_not_()
    this
  }
  def logical_xor(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logical_xor(other.underlying))
  def logical_xor_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.logical_xor_(other.underlying)
    this
  }
  def logical_and(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logical_and(other.underlying))
  def logical_and_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.logical_and_(other.underlying)
    this
  }
  def logical_or(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logical_or(other.underlying))
  def logical_or_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.logical_or_(other.underlying)
    this
  }
  def bmm(mat2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bmm(mat2.underlying))
  def broadcast_to(size: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.broadcast_to(size))
  def ceil()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ceil())
  def ceil_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ceil_()
    this
  }
  def unsafe_chunk(chunks: Long, dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.unsafe_chunk(chunks, dim))
  def chunk(chunks: Long, dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.chunk(chunks, dim))
  def tensor_split(sections: Long, dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.tensor_split(sections, dim))
  def tensor_split(indices: Array[Long], dim: Long)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.tensor_split(indices, dim))
  def tensor_split(tensor_indices_or_sections: Tensor, dim: Long)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.tensor_split(tensor_indices_or_sections.underlying, dim))
  def clamp(min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clamp(min.map(_.underlying), max.map(_.underlying)))
  def clamp_(min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clamp_(min.map(_.underlying), max.map(_.underlying))
    this
  }
  def clamp_max(max: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clamp_max(max.underlying))
  def clamp_max(max: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clamp_max(max.underlying))
  def clamp_max_(max: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clamp_max_(max.underlying)
    this
  }
  def clamp_max_(max: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clamp_max_(max.underlying)
    this
  }
  def clamp_min(min: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clamp_min(min.underlying))
  def clamp_min(min: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clamp_min(min.underlying))
  def clamp_min_(min: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clamp_min_(min.underlying)
    this
  }
  def clamp_min_(min: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clamp_min_(min.underlying)
    this
  }
  def clip(min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clip(min.map(_.underlying), max.map(_.underlying)))
  def clip_(min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.clip_(min.map(_.underlying), max.map(_.underlying))
    this
  }
  def contiguous(memory_format: MemoryFormat = MemoryFormat.Contiguous)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.contiguous(memory_format))
  def copy_(src: Tensor, non_blocking: Boolean = false)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.copy_(src.underlying, non_blocking)
    this
  }
  def cos()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cos())
  def cos_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.cos_()
    this
  }
  def cosh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cosh())
  def cosh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.cosh_()
    this
  }
  def count_nonzero(dim: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.count_nonzero(dim))
  def count_nonzero(dim: Option[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.count_nonzero(dim.asJavaLong))
  def cov(correction: Long = 1, fweights: Option[Tensor] = None, aweights: Option[Tensor] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cov(correction, fweights.map(_.underlying), aweights.map(_.underlying)))
  def corrcoef()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.corrcoef())
  def cummax(dim: Long)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.cummax(dim))
  def cummin(dim: Long)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.cummin(dim))
  def cumprod(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cumprod(dim, dtype.map(_.toScalarType)))
  def cumprod_(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.cumprod_(dim, dtype.map(_.toScalarType))
    this
  }
  def cumsum(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cumsum(dim, dtype.map(_.toScalarType)))
  def cumsum_(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.cumsum_(dim, dtype.map(_.toScalarType))
    this
  }
  def diag_embed(offset: Long = 0, dim1: Long = -2, dim2: Long = -1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.diag_embed(offset, dim1, dim2))
  def diagflat(offset: Long = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.diagflat(offset))
  def diagonal(offset: Long = 0, dim1: Long = 0, dim2: Long = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.diagonal(offset, dim1, dim2))
  def fill_diagonal_(fill_value: Scalar, wrap: Boolean = false)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fill_diagonal_(fill_value.underlying, wrap)
    this
  }
  def diff(n: Long = 1, dim: Long = -1, prepend: Option[Tensor] = None, append: Option[Tensor] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.diff(n, dim, prepend.map(_.underlying), append.map(_.underlying)))
  def div(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.div(other.underlying))
  def div_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.div_(other.underlying)
    this
  }
  def div(other: Tensor, rounding_mode: Option[String])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.div(other.underlying, rounding_mode))
  def div_(other: Tensor, rounding_mode: Option[String])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.div_(other.underlying, rounding_mode)
    this
  }
  def div(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.div(other.underlying))
  def div_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.div_(other.underlying)
    this
  }
  def div(other: Scalar, rounding_mode: Option[String])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.div(other.underlying, rounding_mode))
  def div_(other: Scalar, rounding_mode: Option[String])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.div_(other.underlying, rounding_mode)
    this
  }
  def divide(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.divide(other.underlying))
  def divide_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.divide_(other.underlying)
    this
  }
  def divide(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.divide(other.underlying))
  def divide_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.divide_(other.underlying)
    this
  }
  def divide(other: Tensor, rounding_mode: Option[String])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.divide(other.underlying, rounding_mode))
  def divide_(other: Tensor, rounding_mode: Option[String])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.divide_(other.underlying, rounding_mode)
    this
  }
  def divide(other: Scalar, rounding_mode: Option[String])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.divide(other.underlying, rounding_mode))
  def divide_(other: Scalar, rounding_mode: Option[String])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.divide_(other.underlying, rounding_mode)
    this
  }
  def true_divide(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.true_divide(other.underlying))
  def true_divide_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.true_divide_(other.underlying)
    this
  }
  def true_divide(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.true_divide(other.underlying))
  def true_divide_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.true_divide_(other.underlying)
    this
  }
  def dot(tensor: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.dot(tensor.underlying))
  def vdot(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.vdot(other.underlying))
  def new_empty(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.new_empty(size, options))
}

  def new_empty_strided(size: Array[Long], stride: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.new_empty_strided(size, stride, options))
}

  def new_full(size: Array[Long], fill_value: Scalar, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.new_full(size, fill_value.underlying, options))
}

  def new_zeros(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.new_zeros(size, options))
}

  def new_ones(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.new_ones(size, options))
}

  def resize_(size: Array[Long], memory_format: Option[MemoryFormat] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.resize_(size, memory_format)
    this
  }
  def erf()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.erf())
  def erf_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.erf_()
    this
  }
  def erfc()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.erfc())
  def erfc_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.erfc_()
    this
  }
  def exp()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.exp())
  def exp_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.exp_()
    this
  }
  def exp2()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.exp2())
  def exp2_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.exp2_()
    this
  }
  def expm1()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.expm1())
  def expm1_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.expm1_()
    this
  }
  def expand(size: Array[Long], `implicit`: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.expand(size, `implicit`))
  def expand_as(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.expand_as(other.underlying))
  def flatten(start_dim: Long = 0, end_dim: Long = -1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.flatten(start_dim, end_dim))
  def fill_(value: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fill_(value.underlying)
    this
  }
  def fill_(value: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fill_(value.underlying)
    this
  }
  def floor()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.floor())
  def floor_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.floor_()
    this
  }
  def floor_divide(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.floor_divide(other.underlying))
  def floor_divide_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.floor_divide_(other.underlying)
    this
  }
  def floor_divide(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.floor_divide(other.underlying))
  def floor_divide_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.floor_divide_(other.underlying)
    this
  }
  def frac()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.frac())
  def frac_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.frac_()
    this
  }
  def gcd(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.gcd(other.underlying))
  def gcd_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.gcd_(other.underlying)
    this
  }
  def lcm(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lcm(other.underlying))
  def lcm_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lcm_(other.underlying)
    this
  }
  def index(indices: Array[Option[Tensor]])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index(indices.map(_.map(_.underlying))))
  def index_copy_(dim: Long, index: Tensor, source: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_copy_(dim, index.underlying, source.underlying)
    this
  }
  def index_copy(dim: Long, index: Tensor, source: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_copy(dim, index.underlying, source.underlying))
  def index_put_(indices: Array[Option[Tensor]], values: Tensor, accumulate: Boolean = false)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_put_(indices.map(_.map(_.underlying)), values.underlying, accumulate)
    this
  }
  def index_put(indices: Array[Option[Tensor]], values: Tensor, accumulate: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_put(indices.map(_.map(_.underlying)), values.underlying, accumulate))
  def inverse()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.inverse())
  def isclose(other: Tensor, rtol: Double = 1e-05, atol: Double = 1e-08, equal_nan: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isclose(other.underlying, rtol, atol, equal_nan))
  def isnan()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isnan())
  def is_distributed()(implicit rm: ReferenceManager): Boolean = this.underlying.is_distributed()
  def is_floating_point()(implicit rm: ReferenceManager): Boolean = this.underlying.is_floating_point()
  def is_complex()(implicit rm: ReferenceManager): Boolean = this.underlying.is_complex()
  def is_conj()(implicit rm: ReferenceManager): Boolean = this.underlying.is_conj()
  def is_neg()(implicit rm: ReferenceManager): Boolean = this.underlying.is_neg()
  def isreal()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isreal())
  def is_nonzero()(implicit rm: ReferenceManager): Boolean = this.underlying.is_nonzero()
  def is_same_size(other: Tensor)(implicit rm: ReferenceManager): Boolean = this.underlying.is_same_size(other.underlying)
  def is_signed()(implicit rm: ReferenceManager): Boolean = this.underlying.is_signed()
  def is_inference()(implicit rm: ReferenceManager): Boolean = this.underlying.is_inference()
  def kron(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.kron(other.underlying))
  def kthvalue(k: Long, dim: Long = -1, keepdim: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.kthvalue(k, dim, keepdim))
  def nan_to_num(nan: Option[Double] = None, posinf: Option[Double] = None, neginf: Option[Double] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nan_to_num(nan.asJavaDouble, posinf.asJavaDouble, neginf.asJavaDouble))
  def nan_to_num_(nan: Option[Double] = None, posinf: Option[Double] = None, neginf: Option[Double] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.nan_to_num_(nan.asJavaDouble, posinf.asJavaDouble, neginf.asJavaDouble)
    this
  }
  def ldexp(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ldexp(other.underlying))
  def ldexp_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ldexp_(other.underlying)
    this
  }
  def log()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.log())
  def log_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.log_()
    this
  }
  def log10()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.log10())
  def log10_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.log10_()
    this
  }
  def log1p()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.log1p())
  def log1p_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.log1p_()
    this
  }
  def log2()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.log2())
  def log2_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.log2_()
    this
  }
  def logaddexp(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logaddexp(other.underlying))
  def logaddexp2(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logaddexp2(other.underlying))
  def xlogy(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.xlogy(other.underlying))
  def xlogy(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.xlogy(other.underlying))
  def xlogy_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.xlogy_(other.underlying)
    this
  }
  def xlogy_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.xlogy_(other.underlying)
    this
  }
  def logdet()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logdet())
  def log_softmax(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.log_softmax(dim, dtype.map(_.toScalarType)))
  def logcumsumexp(dim: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logcumsumexp(dim))
  def logsumexp(dim: Array[Long], keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logsumexp(dim, keepdim))
  def matmul(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.matmul(other.underlying))
  def matrix_power(n: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.matrix_power(n))
  def matrix_exp()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.matrix_exp())
  def aminmax(dim: Option[Long] = None, keepdim: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.aminmax(dim.asJavaLong, keepdim))
  def max(dim: Long, keepdim: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.max(dim, keepdim))
  def amax(dim: Array[Long] = Array(), keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.amax(dim, keepdim))
  def mean(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mean(dtype.map(_.toScalarType)))
  def mean(dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mean(dim, keepdim, dtype.map(_.toScalarType)))
  def nanmean(dim: Array[Long] = Array(), keepdim: Boolean = false, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanmean(dim, keepdim, dtype.map(_.toScalarType)))
  def median()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.median())
  def median(dim: Long, keepdim: Boolean)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.median(dim, keepdim))
  def nanmedian()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanmedian())
  def nanmedian(dim: Long, keepdim: Boolean)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.nanmedian(dim, keepdim))
  def min(dim: Long, keepdim: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.min(dim, keepdim))
  def amin(dim: Array[Long] = Array(), keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.amin(dim, keepdim))
  def mm(mat2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mm(mat2.underlying))
  def mode(dim: Long = -1, keepdim: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.mode(dim, keepdim))
  def mul(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mul(other.underlying))
  def mul_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.mul_(other.underlying)
    this
  }
  def mul(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mul(other.underlying))
  def mul_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.mul_(other.underlying)
    this
  }
  def multiply(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.multiply(other.underlying))
  def multiply_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.multiply_(other.underlying)
    this
  }
  def multiply(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.multiply(other.underlying))
  def multiply_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.multiply_(other.underlying)
    this
  }
  def mv(vec: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mv(vec.underlying))
  def mvlgamma(p: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.mvlgamma(p))
  def mvlgamma_(p: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.mvlgamma_(p)
    this
  }
  def narrow_copy(dim: Long, start: Long, length: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.narrow_copy(dim, start, length))
  def narrow(dim: Long, start: Long, length: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.narrow(dim, start, length))
  def narrow(dim: Long, start: Tensor, length: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.narrow(dim, start.underlying, length))
  def permute(dims: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.permute(dims))
  def movedim(source: Array[Long], destination: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.movedim(source, destination))
  def movedim(source: Long, destination: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.movedim(source, destination))
  def moveaxis(source: Array[Long], destination: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.moveaxis(source, destination))
  def moveaxis(source: Long, destination: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.moveaxis(source, destination))
  def numpy_T()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.numpy_T())
  def is_pinned(device: Option[Device] = None)(implicit rm: ReferenceManager): Boolean = this.underlying.is_pinned(device.map(_.underlying))
  def pin_memory(device: Option[Device] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.pin_memory(device.map(_.underlying)))
  def pinverse(rcond: Double = 1e-15)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.pinverse(rcond))
  def rad2deg()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.rad2deg())
  def rad2deg_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.rad2deg_()
    this
  }
  def deg2rad()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.deg2rad())
  def deg2rad_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.deg2rad_()
    this
  }
  def ravel()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ravel())
  def reciprocal()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.reciprocal())
  def reciprocal_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.reciprocal_()
    this
  }
  def neg()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.neg())
  def neg_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.neg_()
    this
  }
  def negative()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.negative())
  def negative_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.negative_()
    this
  }
  def repeat(repeats: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.repeat(repeats))
  def repeat_interleave(repeats: Tensor, dim: Option[Long], output_size: Option[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.repeat_interleave(repeats.underlying, dim.asJavaLong, output_size.asJavaLong))
  def repeat_interleave(repeats: Long, dim: Option[Long], output_size: Option[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.repeat_interleave(repeats, dim.asJavaLong, output_size.asJavaLong))
  def reshape(shape: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.reshape(shape))
  def _reshape_alias(size: Array[Long], stride: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._reshape_alias(size, stride))
  def reshape_as(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.reshape_as(other.underlying))
  def round()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.round())
  def round_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.round_()
    this
  }
  def relu()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.relu())
  def relu_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.relu_()
    this
  }
  def prelu(weight: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.prelu(weight.underlying))
  def prelu_backward(grad_output: Tensor, weight: Tensor)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.prelu_backward(grad_output.underlying, weight.underlying))
  def hardshrink(lambd: Double = 0.5)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.hardshrink(lambd.toInternalScalar))
  def hardshrink_backward(grad_out: Tensor, lambd: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.hardshrink_backward(grad_out.underlying, lambd.underlying))
  def rsqrt()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.rsqrt())
  def rsqrt_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.rsqrt_()
    this
  }
  def select(dim: Long, index: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.select(dim, index))
  def sigmoid()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sigmoid())
  def sigmoid_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sigmoid_()
    this
  }
  def logit(eps: Option[Double] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.logit(eps.asJavaDouble))
  def logit_(eps: Option[Double] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.logit_(eps.asJavaDouble)
    this
  }
  def sin()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sin())
  def sin_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sin_()
    this
  }
  def sinc()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sinc())
  def sinc_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sinc_()
    this
  }
  def sinh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sinh())
  def sinh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sinh_()
    this
  }
  def detach()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.detach())
  def detach_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.detach_()
    this
  }
  def slice(dim: Long = 0, start: Option[Long] = None, end: Option[Long] = None, step: Long = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.slice(dim, start.asJavaLong, end.asJavaLong, step))
  def slogdet()(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.slogdet())
  def smm(mat2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.smm(mat2.underlying))
  def softmax(dim: Long, dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.softmax(dim, dtype.map(_.toScalarType)))
  def unsafe_split(split_size: Long, dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.unsafe_split(split_size, dim))
  def split(split_size: Long, dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.split(split_size, dim))
  def unsafe_split_with_sizes(split_sizes: Array[Long], dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.unsafe_split_with_sizes(split_sizes, dim))
  def split_with_sizes(split_sizes: Array[Long], dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.split_with_sizes(split_sizes, dim))
  def hsplit(sections: Long)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.hsplit(sections))
  def hsplit(indices: Array[Long])(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.hsplit(indices))
  def vsplit(sections: Long)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.vsplit(sections))
  def vsplit(indices: Array[Long])(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.vsplit(indices))
  def dsplit(sections: Long)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.dsplit(sections))
  def dsplit(indices: Array[Long])(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.dsplit(indices))
  def squeeze()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.squeeze())
  def squeeze(dim: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.squeeze(dim))
  def squeeze_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.squeeze_()
    this
  }
  def squeeze_(dim: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.squeeze_(dim)
    this
  }
  def sspaddmm(mat1: Tensor, mat2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sspaddmm(mat1.underlying, mat2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def stft(n_fft: Long, hop_length: Option[Long] = None, win_length: Option[Long] = None, window: Option[Tensor] = None, normalized: Boolean = false, onesided: Option[Boolean] = None, return_complex: Option[Boolean] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.stft(n_fft, hop_length.asJavaLong, win_length.asJavaLong, window.map(_.underlying), normalized, onesided.map(java.lang.Boolean.valueOf), return_complex.map(java.lang.Boolean.valueOf)))
  def istft(n_fft: Long, hop_length: Option[Long] = None, win_length: Option[Long] = None, window: Option[Tensor] = None, center: Boolean = true, normalized: Boolean = false, onesided: Option[Boolean] = None, length: Option[Long] = None, return_complex: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.istft(n_fft, hop_length.asJavaLong, win_length.asJavaLong, window.map(_.underlying), center, normalized, onesided.map(java.lang.Boolean.valueOf), length.asJavaLong, return_complex))
  def sum(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sum(dtype.map(_.toScalarType)))
  def sum(dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sum(dim, keepdim, dtype.map(_.toScalarType)))
  def nansum(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nansum(dtype.map(_.toScalarType)))
  def nansum(dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nansum(dim, keepdim, dtype.map(_.toScalarType)))
  def sum_to_size(size: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sum_to_size(size))
  def sqrt()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sqrt())
  def sqrt_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sqrt_()
    this
  }
  def square()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.square())
  def square_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.square_()
    this
  }
  def std(unbiased: Boolean = true)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.std(unbiased))
  def std(dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.std(dim, unbiased, keepdim))
  def std(dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.std(dim, correction.asJavaLong, keepdim))
  def prod(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.prod(dtype.map(_.toScalarType)))
  def prod(dim: Long, keepdim: Boolean, dtype: Option[dtype])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.prod(dim, keepdim, dtype.map(_.toScalarType)))
  def t()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.t())
  def t_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.t_()
    this
  }
  def tan()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.tan())
  def tan_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.tan_()
    this
  }
  def tanh()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.tanh())
  def tanh_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.tanh_()
    this
  }
  def tile(dims: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.tile(dims))
  def transpose(dim0: Long, dim1: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.transpose(dim0, dim1))
  def transpose_(dim0: Long, dim1: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.transpose_(dim0, dim1)
    this
  }
  def flip(dims: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.flip(dims))
  def fliplr()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fliplr())
  def flipud()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.flipud())
  def roll(shifts: Array[Long], dims: Array[Long] = Array())(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.roll(shifts, dims))
  def rot90(k: Long = 1, dims: Array[Long] = Array(0,1))(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.rot90(k, dims))
  def trunc()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.trunc())
  def trunc_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.trunc_()
    this
  }
  def fix()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fix())
  def fix_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fix_()
    this
  }
  def type_as(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.type_as(other.underlying))
  def unsqueeze(dim: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.unsqueeze(dim))
  def unsqueeze_(dim: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.unsqueeze_(dim)
    this
  }
  def `var`(unbiased: Boolean = true)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.`var`(unbiased))
  def `var`(dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.`var`(dim, unbiased, keepdim))
  def `var`(dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.`var`(dim, correction.asJavaLong, keepdim))
  def view_as(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.view_as(other.underlying))
  def where(condition: Tensor, other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.where(condition.underlying, other.underlying))
  def norm(p: Option[Scalar], dtype: dtype)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.norm(p.map(_.underlying), dtype.toScalarType))
  def norm(p: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.norm(p.underlying))
  def norm(p: Option[Scalar], dim: Array[Long], keepdim: Boolean, dtype: dtype)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.norm(p.map(_.underlying), dim, keepdim, dtype.toScalarType))
  def norm(p: Option[Scalar], dim: Array[Long], keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.norm(p.map(_.underlying), dim, keepdim))
  def frexp()(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.frexp())
  def clone(memory_format: Option[MemoryFormat] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.clone(memory_format))
  def positive()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.positive())
  def resize_as_(the_template: Tensor, memory_format: Option[MemoryFormat] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.resize_as_(the_template.underlying, memory_format)
    this
  }
  def zero_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.zero_()
    this
  }
  def sub(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sub(other.underlying, alpha.toInternalScalar))
  def sub_(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sub_(other.underlying, alpha.toInternalScalar)
    this
  }
  def sub(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sub(other.underlying, alpha.underlying))
  def sub_(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sub_(other.underlying, alpha.underlying)
    this
  }
  def subtract(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.subtract(other.underlying, alpha.toInternalScalar))
  def subtract_(other: Tensor, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.subtract_(other.underlying, alpha.toInternalScalar)
    this
  }
  def subtract(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.subtract(other.underlying, alpha.underlying))
  def subtract_(other: Scalar, alpha: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.subtract_(other.underlying, alpha.underlying)
    this
  }
  def heaviside(values: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.heaviside(values.underlying))
  def heaviside_(values: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.heaviside_(values.underlying)
    this
  }
  def addmm(mat1: Tensor, mat2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addmm(mat1.underlying, mat2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def addmm_(mat1: Tensor, mat2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addmm_(mat1.underlying, mat2.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    this
  }
  def sparse_resize_(size: Array[Long], sparse_dim: Long, dense_dim: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sparse_resize_(size, sparse_dim, dense_dim)
    this
  }
  def sparse_resize_and_clear_(size: Array[Long], sparse_dim: Long, dense_dim: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sparse_resize_and_clear_(size, sparse_dim, dense_dim)
    this
  }
  def sparse_mask(mask: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sparse_mask(mask.underlying))
  def to_dense(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to_dense(dtype.map(_.toScalarType)))
  def sparse_dim()(implicit rm: ReferenceManager): Long = this.underlying.sparse_dim()
  def _dimI()(implicit rm: ReferenceManager): Long = this.underlying._dimI()
  def dense_dim()(implicit rm: ReferenceManager): Long = this.underlying.dense_dim()
  def _dimV()(implicit rm: ReferenceManager): Long = this.underlying._dimV()
  def _nnz()(implicit rm: ReferenceManager): Long = this.underlying._nnz()
  def coalesce()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.coalesce())
  def is_coalesced()(implicit rm: ReferenceManager): Boolean = this.underlying.is_coalesced()
  def _indices()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._indices())
  def _values()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying._values())
  def _coalesced_(coalesced: Boolean)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying._coalesced_(coalesced)
    this
  }
  def indices()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.indices())
  def values()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.values())
  def crow_indices()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.crow_indices())
  def col_indices()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.col_indices())
  def unbind(dim: Long = 0)(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.unbind(dim))
  def to_sparse(sparse_dim: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to_sparse(sparse_dim))
  def to_sparse()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to_sparse())
  def to_mkldnn(dtype: Option[dtype] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to_mkldnn(dtype.map(_.toScalarType)))
  def dequantize()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.dequantize())
  def q_scale()(implicit rm: ReferenceManager): Double = this.underlying.q_scale()
  def q_zero_point()(implicit rm: ReferenceManager): Long = this.underlying.q_zero_point()
  def q_per_channel_scales()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.q_per_channel_scales())
  def q_per_channel_zero_points()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.q_per_channel_zero_points())
  def q_per_channel_axis()(implicit rm: ReferenceManager): Long = this.underlying.q_per_channel_axis()
  def int_repr()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.int_repr())
  def qscheme()(implicit rm: ReferenceManager): QScheme = this.underlying.qscheme()
  def to(dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, non_blocking: Boolean = false, copy: Boolean = false, memory_format: Option[MemoryFormat] = None)(implicit rm: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(this.underlying.to(options, non_blocking, copy, memory_format))
}

  def to(device: Device, dtype: dtype, non_blocking: Boolean, copy: Boolean, memory_format: Option[MemoryFormat])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to(device.underlying, dtype.toScalarType, non_blocking, copy, memory_format))
  def to(dtype: dtype, non_blocking: Boolean, copy: Boolean, memory_format: Option[MemoryFormat])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to(dtype.toScalarType, non_blocking, copy, memory_format))
  def to(other: Tensor, non_blocking: Boolean, copy: Boolean, memory_format: Option[MemoryFormat])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.to(other.underlying, non_blocking, copy, memory_format))
  def item()(implicit rm: ReferenceManager): Scalar = Scalar(this.underlying.item())
  def set_(source: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.set_(source.underlying)
    this
  }
  def set_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.set_()
    this
  }
  def is_set_to(tensor: Tensor)(implicit rm: ReferenceManager): Boolean = this.underlying.is_set_to(tensor.underlying)
  def masked_fill_(mask: Tensor, value: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.masked_fill_(mask.underlying, value.underlying)
    this
  }
  def masked_fill(mask: Tensor, value: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.masked_fill(mask.underlying, value.underlying))
  def masked_fill_(mask: Tensor, value: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.masked_fill_(mask.underlying, value.underlying)
    this
  }
  def masked_fill(mask: Tensor, value: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.masked_fill(mask.underlying, value.underlying))
  def masked_scatter_(mask: Tensor, source: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.masked_scatter_(mask.underlying, source.underlying)
    this
  }
  def masked_scatter(mask: Tensor, source: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.masked_scatter(mask.underlying, source.underlying))
  def view(size: Array[Long])(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.view(size))
  def view(dtype: dtype)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.view(dtype.toScalarType))
  def put_(index: Tensor, source: Tensor, accumulate: Boolean = false)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.put_(index.underlying, source.underlying, accumulate)
    this
  }
  def put(index: Tensor, source: Tensor, accumulate: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.put(index.underlying, source.underlying, accumulate))
  def index_add_(dim: Long, index: Tensor, source: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_add_(dim, index.underlying, source.underlying)
    this
  }
  def index_add_(dim: Long, index: Tensor, source: Tensor, alpha: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_add_(dim, index.underlying, source.underlying, alpha.underlying)
    this
  }
  def index_add(dim: Long, index: Tensor, source: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_add(dim, index.underlying, source.underlying))
  def index_add(dim: Long, index: Tensor, source: Tensor, alpha: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_add(dim, index.underlying, source.underlying, alpha.underlying))
  def index_fill_(dim: Long, index: Tensor, value: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_fill_(dim, index.underlying, value.underlying)
    this
  }
  def index_fill(dim: Long, index: Tensor, value: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_fill(dim, index.underlying, value.underlying))
  def index_fill_(dim: Long, index: Tensor, value: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.index_fill_(dim, index.underlying, value.underlying)
    this
  }
  def index_fill(dim: Long, index: Tensor, value: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_fill(dim, index.underlying, value.underlying))
  def scatter(dim: Long, index: Tensor, src: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.scatter(dim, index.underlying, src.underlying))
  def scatter_(dim: Long, index: Tensor, src: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.scatter_(dim, index.underlying, src.underlying)
    this
  }
  def scatter(dim: Long, index: Tensor, value: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.scatter(dim, index.underlying, value.underlying))
  def scatter_(dim: Long, index: Tensor, value: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.scatter_(dim, index.underlying, value.underlying)
    this
  }
  def scatter(dim: Long, index: Tensor, src: Tensor, reduce: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.scatter(dim, index.underlying, src.underlying, reduce))
  def scatter_(dim: Long, index: Tensor, src: Tensor, reduce: String)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.scatter_(dim, index.underlying, src.underlying, reduce)
    this
  }
  def scatter(dim: Long, index: Tensor, value: Scalar, reduce: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.scatter(dim, index.underlying, value.underlying, reduce))
  def scatter_(dim: Long, index: Tensor, value: Scalar, reduce: String)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.scatter_(dim, index.underlying, value.underlying, reduce)
    this
  }
  def scatter_add(dim: Long, index: Tensor, src: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.scatter_add(dim, index.underlying, src.underlying))
  def scatter_add_(dim: Long, index: Tensor, src: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.scatter_add_(dim, index.underlying, src.underlying)
    this
  }
  def eq_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.eq_(other.underlying)
    this
  }
  def eq_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.eq_(other.underlying)
    this
  }
  def bitwise_and(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_and(other.underlying))
  def bitwise_and(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_and(other.underlying))
  def bitwise_and_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_and_(other.underlying)
    this
  }
  def bitwise_and_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_and_(other.underlying)
    this
  }
  def &(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__and__(other.underlying))
  def &(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__and__(other.underlying))
  def &=(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__iand__(other.underlying)
    this
  }
  def &=(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__iand__(other.underlying)
    this
  }
  def bitwise_or(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_or(other.underlying))
  def bitwise_or(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_or(other.underlying))
  def bitwise_or_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_or_(other.underlying)
    this
  }
  def bitwise_or_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_or_(other.underlying)
    this
  }
  def |(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__or__(other.underlying))
  def |(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__or__(other.underlying))
  def |=(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ior__(other.underlying)
    this
  }
  def |=(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ior__(other.underlying)
    this
  }
  def bitwise_xor(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_xor(other.underlying))
  def bitwise_xor(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_xor(other.underlying))
  def bitwise_xor_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_xor_(other.underlying)
    this
  }
  def bitwise_xor_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_xor_(other.underlying)
    this
  }
  def ^(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__xor__(other.underlying))
  def ^(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__xor__(other.underlying))
  def __ixor__(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ixor__(other.underlying)
    this
  }
  def __ixor__(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ixor__(other.underlying)
    this
  }
  def <<(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__lshift__(other.underlying))
  def <<(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__lshift__(other.underlying))
  def <<=(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ilshift__(other.underlying)
    this
  }
  def <<=(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__ilshift__(other.underlying)
    this
  }
  def bitwise_left_shift(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_left_shift(other.underlying))
  def bitwise_left_shift_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_left_shift_(other.underlying)
    this
  }
  def bitwise_left_shift(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_left_shift(other.underlying))
  def bitwise_left_shift_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_left_shift_(other.underlying)
    this
  }
  def >>(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__rshift__(other.underlying))
  def >>(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.__rshift__(other.underlying))
  def >>=(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__irshift__(other.underlying)
    this
  }
  def >>=(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.__irshift__(other.underlying)
    this
  }
  def bitwise_right_shift(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_right_shift(other.underlying))
  def bitwise_right_shift_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_right_shift_(other.underlying)
    this
  }
  def bitwise_right_shift(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.bitwise_right_shift(other.underlying))
  def bitwise_right_shift_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.bitwise_right_shift_(other.underlying)
    this
  }
  def tril_(diagonal: Long = 0)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.tril_(diagonal)
    this
  }
  def triu_(diagonal: Long = 0)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.triu_(diagonal)
    this
  }
  def digamma_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.digamma_()
    this
  }
  def lerp_(end: Tensor, weight: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lerp_(end.underlying, weight.underlying)
    this
  }
  def lerp_(end: Tensor, weight: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lerp_(end.underlying, weight.underlying)
    this
  }
  def addbmm_(batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addbmm_(batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    this
  }
  def addbmm(batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addbmm(batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def random_(from: Long, to: Option[Long], generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.random_(from, to.asJavaLong, generator.map(_.underlying))
    this
  }
  def random_(to: Long, generator: Option[Generator])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.random_(to, generator.map(_.underlying))
    this
  }
  def random_(generator: Option[Generator])(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.random_(generator.map(_.underlying))
    this
  }
  def uniform_(from: Double = 0, to: Double = 1, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.uniform_(from, to, generator.map(_.underlying))
    this
  }
  def cauchy_(median: Double = 0, sigma: Double = 1, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.cauchy_(median, sigma, generator.map(_.underlying))
    this
  }
  def log_normal_(mean: Double = 1, std: Double = 2, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.log_normal_(mean, std, generator.map(_.underlying))
    this
  }
  def exponential_(lambd: Double = 1, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.exponential_(lambd, generator.map(_.underlying))
    this
  }
  def geometric_(p: Double, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.geometric_(p, generator.map(_.underlying))
    this
  }
  def diag(diagonal: Long = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.diag(diagonal))
  def cross(other: Tensor, dim: Option[Long] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cross(other.underlying, dim.asJavaLong))
  def triu(diagonal: Long = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.triu(diagonal))
  def tril(diagonal: Long = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.tril(diagonal))
  def trace()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.trace())
  def ne(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ne(other.underlying))
  def ne(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ne(other.underlying))
  def ne_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ne_(other.underlying)
    this
  }
  def ne_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ne_(other.underlying)
    this
  }
  def not_equal(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.not_equal(other.underlying))
  def not_equal(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.not_equal(other.underlying))
  def not_equal_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.not_equal_(other.underlying)
    this
  }
  def not_equal_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.not_equal_(other.underlying)
    this
  }
  def eq(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.eq(other.underlying))
  def eq(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.eq(other.underlying))
  def ge(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ge(other.underlying))
  def ge(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ge(other.underlying))
  def ge_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ge_(other.underlying)
    this
  }
  def ge_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.ge_(other.underlying)
    this
  }
  def greater_equal(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.greater_equal(other.underlying))
  def greater_equal(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.greater_equal(other.underlying))
  def greater_equal_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.greater_equal_(other.underlying)
    this
  }
  def greater_equal_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.greater_equal_(other.underlying)
    this
  }
  def le(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.le(other.underlying))
  def le(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.le(other.underlying))
  def le_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.le_(other.underlying)
    this
  }
  def le_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.le_(other.underlying)
    this
  }
  def less_equal(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.less_equal(other.underlying))
  def less_equal(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.less_equal(other.underlying))
  def less_equal_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.less_equal_(other.underlying)
    this
  }
  def less_equal_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.less_equal_(other.underlying)
    this
  }
  def gt(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.gt(other.underlying))
  def gt(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.gt(other.underlying))
  def gt_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.gt_(other.underlying)
    this
  }
  def gt_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.gt_(other.underlying)
    this
  }
  def greater(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.greater(other.underlying))
  def greater(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.greater(other.underlying))
  def greater_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.greater_(other.underlying)
    this
  }
  def greater_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.greater_(other.underlying)
    this
  }
  def lt(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lt(other.underlying))
  def lt(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lt(other.underlying))
  def lt_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lt_(other.underlying)
    this
  }
  def lt_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lt_(other.underlying)
    this
  }
  def less(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.less(other.underlying))
  def less(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.less(other.underlying))
  def less_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.less_(other.underlying)
    this
  }
  def less_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.less_(other.underlying)
    this
  }
  def take(index: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.take(index.underlying))
  def take_along_dim(indices: Tensor, dim: Option[Long] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.take_along_dim(indices.underlying, dim.asJavaLong))
  def index_select(dim: Long, index: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.index_select(dim, index.underlying))
  def masked_select(mask: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.masked_select(mask.underlying))
  def nonzero()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nonzero())
  def nonzero_numpy()(implicit rm: ReferenceManager): Array[Tensor] = tensorVectorToArray(this.underlying.nonzero_numpy())
  def gather(dim: Long, index: Tensor, sparse_grad: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.gather(dim, index.underlying, sparse_grad))
  def addcmul(tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addcmul(tensor1.underlying, tensor2.underlying, value.toInternalScalar))
  def addcmul_(tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addcmul_(tensor1.underlying, tensor2.underlying, value.toInternalScalar)
    this
  }
  def addcdiv(tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.addcdiv(tensor1.underlying, tensor2.underlying, value.toInternalScalar))
  def addcdiv_(tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.addcdiv_(tensor1.underlying, tensor2.underlying, value.toInternalScalar)
    this
  }
  def lstsq(A: Tensor)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.lstsq(A.underlying))
  def triangular_solve(A: Tensor, upper: Boolean = true, transpose: Boolean = false, unitriangular: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.triangular_solve(A.underlying, upper, transpose, unitriangular))
  def symeig(eigenvectors: Boolean = false, upper: Boolean = true)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.symeig(eigenvectors, upper))
  def eig(eigenvectors: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.eig(eigenvectors))
  def svd(some: Boolean = true, compute_uv: Boolean = true)(implicit rm: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(this.underlying.svd(some, compute_uv))
  def swapaxes(axis0: Long, axis1: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.swapaxes(axis0, axis1))
  def swapaxes_(axis0: Long, axis1: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.swapaxes_(axis0, axis1)
    this
  }
  def swapdims(dim0: Long, dim1: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.swapdims(dim0, dim1))
  def swapdims_(dim0: Long, dim1: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.swapdims_(dim0, dim1)
    this
  }
  def cholesky(upper: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cholesky(upper))
  def cholesky_solve(input2: Tensor, upper: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cholesky_solve(input2.underlying, upper))
  def solve(A: Tensor)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.solve(A.underlying))
  def cholesky_inverse(upper: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.cholesky_inverse(upper))
  def qr(some: Boolean = true)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.qr(some))
  def geqrf()(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.geqrf())
  def orgqr(input2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.orgqr(input2.underlying))
  def ormqr(input2: Tensor, input3: Tensor, left: Boolean = true, transpose: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ormqr(input2.underlying, input3.underlying, left, transpose))
  def lu_solve(LU_data: Tensor, LU_pivots: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lu_solve(LU_data.underlying, LU_pivots.underlying))
  def multinomial(num_samples: Long, replacement: Boolean = false, generator: Option[Generator] = None)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.multinomial(num_samples, replacement, generator.map(_.underlying)))
  def lgamma_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.lgamma_()
    this
  }
  def lgamma()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lgamma())
  def digamma()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.digamma())
  def polygamma(n: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.polygamma(n))
  def polygamma_(n: Long)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.polygamma_(n)
    this
  }
  def erfinv()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.erfinv())
  def erfinv_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.erfinv_()
    this
  }
  def i0()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.i0())
  def i0_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.i0_()
    this
  }
  def sign()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.sign())
  def sign_()(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.sign_()
    this
  }
  def signbit()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.signbit())
  def dist(other: Tensor, p: Double = 2)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.dist(other.underlying, p.toInternalScalar))
  def atan2_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.atan2_(other.underlying)
    this
  }
  def atan2(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.atan2(other.underlying))
  def lerp(end: Tensor, weight: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lerp(end.underlying, weight.underlying))
  def lerp(end: Tensor, weight: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.lerp(end.underlying, weight.underlying))
  def histc(bins: Long = 100, min: Double = 0, max: Double = 0)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.histc(bins, min.toInternalScalar, max.toInternalScalar))
  def histogram(bins: Tensor, weight: Option[Tensor] = None, density: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.histogram(bins.underlying, weight.map(_.underlying), density))
  def histogram(bins: Long, range: Option[Array[Double]], weight: Option[Tensor], density: Boolean)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.histogram(bins, range, weight.map(_.underlying), density))
  def fmod(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fmod(other.underlying))
  def fmod_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fmod_(other.underlying)
    this
  }
  def fmod(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fmod(other.underlying))
  def fmod_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.fmod_(other.underlying)
    this
  }
  def hypot(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.hypot(other.underlying))
  def hypot_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.hypot_(other.underlying)
    this
  }
  def igamma(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.igamma(other.underlying))
  def igamma_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.igamma_(other.underlying)
    this
  }
  def igammac(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.igammac(other.underlying))
  def igammac_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.igammac_(other.underlying)
    this
  }
  def nextafter(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nextafter(other.underlying))
  def nextafter_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.nextafter_(other.underlying)
    this
  }
  def remainder(other: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.remainder(other.underlying))
  def remainder_(other: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.remainder_(other.underlying)
    this
  }
  def remainder(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.remainder(other.underlying))
  def remainder_(other: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.remainder_(other.underlying)
    this
  }
  def min()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.min())
  def fmin(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fmin(other.underlying))
  def max()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.max())
  def fmax(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.fmax(other.underlying))
  def maximum(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.maximum(other.underlying))
  def max(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.max(other.underlying))
  def minimum(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.minimum(other.underlying))
  def min(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.min(other.underlying))
  def quantile(q: Double, dim: Option[Long] = None, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.quantile(q, dim.asJavaLong, keepdim))
  def quantile(q: Tensor, dim: Option[Long], keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.quantile(q.underlying, dim.asJavaLong, keepdim))
  def nanquantile(q: Double, dim: Option[Long] = None, keepdim: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanquantile(q, dim.asJavaLong, keepdim))
  def nanquantile(q: Tensor, dim: Option[Long], keepdim: Boolean)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanquantile(q.underlying, dim.asJavaLong, keepdim))
  def quantile(q: Double, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.quantile(q, dim.asJavaLong, keepdim, interpolation))
  def quantile(q: Tensor, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.quantile(q.underlying, dim.asJavaLong, keepdim, interpolation))
  def nanquantile(q: Double, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanquantile(q, dim.asJavaLong, keepdim, interpolation))
  def nanquantile(q: Tensor, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.nanquantile(q.underlying, dim.asJavaLong, keepdim, interpolation))
  def sort(dim: Long = -1, descending: Boolean = false)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.sort(dim, descending))
  def sort(stable: Option[Boolean], dim: Long, descending: Boolean)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.sort(stable.map(java.lang.Boolean.valueOf), dim, descending))
  def msort()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.msort())
  def argsort(dim: Long = -1, descending: Boolean = false)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.argsort(dim, descending))
  def topk(k: Long, dim: Long = -1, largest: Boolean = true, sorted: Boolean = true)(implicit rm: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(this.underlying.topk(k, dim, largest, sorted))
  def all()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.all())
  def any()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.any())
  def renorm(p: Scalar, dim: Long, maxnorm: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.renorm(p.underlying, dim, maxnorm.underlying))
  def renorm_(p: Scalar, dim: Long, maxnorm: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.renorm_(p.underlying, dim, maxnorm.underlying)
    this
  }
  def unfold(dimension: Long, size: Long, step: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.unfold(dimension, size, step))
  def equal(other: Tensor)(implicit rm: ReferenceManager): Boolean = this.underlying.equal(other.underlying)
  def pow(exponent: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.pow(exponent.underlying))
  def pow(exponent: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.pow(exponent.underlying))
  def pow_(exponent: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.pow_(exponent.underlying)
    this
  }
  def pow_(exponent: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.pow_(exponent.underlying)
    this
  }
  def float_power(exponent: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.float_power(exponent.underlying))
  def float_power(exponent: Scalar)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.float_power(exponent.underlying))
  def float_power_(exponent: Scalar)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.float_power_(exponent.underlying)
    this
  }
  def float_power_(exponent: Tensor)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.float_power_(exponent.underlying)
    this
  }
  def normal_(mean: Double = 0, std: Double = 1, generator: Option[Generator] = None)(implicit rm: ReferenceManager): this.type = NoGrad.noGrad {
    this.underlying.normal_(mean, std, generator.map(_.underlying))
    this
  }
  def alias()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.alias())
  def isfinite()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isfinite())
  def isinf()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isinf())
  def isposinf()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isposinf())
  def isneginf()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.isneginf())
  def special_polygamma(n: Long)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.special_polygamma(n))
  def det()(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.det())
  def inner(other: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.inner(other.underlying))
  def outer(vec2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.outer(vec2.underlying))
  def ger(vec2: Tensor)(implicit rm: ReferenceManager): Tensor = Tensor(this.underlying.ger(vec2.underlying))

  def dtype: dtype = new dtype(underlyingChecked.dtype())
  def layout: Layout = underlyingChecked.layout()

  def +(e2: Tensor)(implicit manager: ReferenceManager): Tensor = this.add(e2, 1)
  def +=(e2: Tensor)(implicit manager: ReferenceManager): this.type = this.add_(e2, 1)

  /** Note that this is scalar (pointwise) multiply in Torch.
    * [[*@*]] is matrix multiplication.
    */
  def *(e2: Tensor)(implicit manager: ReferenceManager): Tensor = this.mul(e2)
  def *=(e2: Tensor)(implicit manager: ReferenceManager): this.type = this.mul_(e2)
  def -(e2: Tensor)(implicit manager: ReferenceManager): Tensor = this.sub(e2, 1)
  def -=(e2: Tensor)(implicit manager: ReferenceManager): this.type = this.sub_(e2, 1)
  def /(e2: Tensor)(implicit manager: ReferenceManager): Tensor = this.div(e2)
  def /=(e2: Tensor)(implicit manager: ReferenceManager): this.type = this.div_(e2)

  /** Matrix multiplication, alias for [[matmul]] */
  def *@*(e2: Tensor)(implicit manager: ReferenceManager) = this.matmul(e2)

  def +(e2: Scalar)(implicit manager: ReferenceManager): Tensor = this.add(e2, 1)
  def +=(e2: Scalar)(implicit manager: ReferenceManager): this.type = this.add_(e2, 1)
  def *(e2: Scalar)(implicit manager: ReferenceManager): Tensor = this.mul(e2)
  def *=(e2: Scalar)(implicit manager: ReferenceManager): this.type = this.mul_(e2)
  def -(e2: Scalar)(implicit manager: ReferenceManager): Tensor = this.sub(e2, 1)
  def -=(e2: Scalar)(implicit manager: ReferenceManager): this.type = this.sub_(e2, 1)
  def /(e2: Scalar)(implicit manager: ReferenceManager): Tensor = this.div(e2)
  def /=(e2: Scalar)(implicit manager: ReferenceManager): this.type = this.div_(e2)

  override def equals(other: Any): Boolean = other match {
    case x: Tensor => underlying.equal(x.underlying)
    case _         => false
  }

  override def hashCode = ???

  override def toString: String = underlying.toString

  override protected def delete(): Unit = {
    underlying.delete()
  }

  /** Zero out the gradient. Should only be used on [[Tensor]]s that represents parameters in a [[Model]]. */
  private[torch] def zeroGradient_(): this.type = NoGrad.noGrad {
    Tensor.ensureGradDefined(underlying)
    underlying.grad.zero_()
    this
  }

  /** Add the gradient from `from` in place.
    * Should only be used on [[Tensor]]s that represents parameters in a [[Model]].
    */
  private[torch] def accumulateGradient_(from: Tensor): this.type = NoGrad.noGrad {
    Tensor.ensureGradDefined(this.underlying)
    Tensor.ensureGradDefined(from.underlying)
    val alpha = new internal.Scalar(1)
    try {
      underlying.grad.add_(from.underlying.grad, alpha)
      this
    } finally {
      alpha.delete()
    }

  }

  /** Add (the values of) `from` in place.
    * Should only be used on [[Tensor]]s that represents parameters in a [[Model]].
    */
  private[torch] def accumulate_(from: Tensor): this.type = NoGrad.noGrad {
    val alpha = new internal.Scalar(1)
    try {
      underlying.add_(from.underlying, alpha)
      this
    } finally {
      alpha.delete()
    }
    this
  }

  private[torch] def binOp[Underlying2](
      other: TorchReference[Underlying2],
  )(
      f: (internal.TorchTensor, Underlying2) => internal.TorchTensor,
  )(
      implicit manager: ReferenceManager,
  ): Tensor = {
    TorchReference.assertOpen(this, other)(Tensor(f(this.underlying, other.underlying)))
  }

  private[torch] def unOp(
      f: (internal.TorchTensor) => internal.TorchTensor,
  )(
      implicit manager: ReferenceManager,
  ): Tensor = {
    TorchReference.assertOpen(this)(Tensor(f(underlying)))
  }

  private[torch] def ternOp[Underlying2, Underlying3](
      b: TorchReference[Underlying2],
      c: TorchReference[Underlying3],
  )(
      f: (internal.TorchTensor, Underlying2, Underlying3) => internal.TorchTensor,
  )(
      implicit manager: ReferenceManager,
  ): Tensor = {
    TorchReference.assertOpen(this, b, c)(Tensor(f(this.underlying, b.underlying, c.underlying)))
  }
}

object Tensor {

  private[torch] def apply(internal: TorchTensor)(implicit manager: ReferenceManager): Tensor = {
    manager.addReference(new Tensor(internal))
  }

  private[torch] def apply()(implicit manager: ReferenceManager): Tensor = {
    manager.addReference(new Tensor(new internal.TorchTensor()))
  }

  /** Torch tries to lazily initialize the gradient vector. The way we do things doesn't
    * seem to hit all the right initialization paths, so we explicitly force the gradient
    * to zero in some places.
    */
  private[torch] def ensureGradDefined(underlying: TorchTensor): Unit = {
    if (!underlying.grad.defined()) {
      TensorOptions(requires_grad = Some(true)).toInternal.apply { opts =>
        val empty = swig.empty(underlying.sizes(), opts, None)
        try {
          underlying.grad.operator_equals(empty)
        } finally {
          empty.delete()
        }
      }
    }
  }

  /** note that empty does *not* initialize the memory */
  def empty(
      dim: Size,
      options: TensorOptions = TensorOptions(),
  )(implicit manager: ReferenceManager): Tensor = {
    options.toInternal.apply { opts => Tensor(swig.empty(dim.sizes, opts, None)) }
  }

  def zeros(
      dim: Size,
      options: TensorOptions = TensorOptions(),
  )(implicit manager: ReferenceManager): Tensor = {
    options.toInternal.apply { opts => Tensor(swig.zeros(dim.sizes, opts)) }
  }

  def ones(
      dim: Size,
      options: TensorOptions = TensorOptions(),
  )(implicit manager: ReferenceManager): Tensor = {
    options.toInternal.apply { opts => Tensor(swig.ones(dim.sizes, opts)) }
  }

  def full(dim: Size, scalar: Scalar, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    options.toInternal.apply { opts => Tensor(swig.full(dim.sizes, scalar.underlyingChecked, opts)) }
  }

  def sparseCooTensor(indices: Tensor, values: Tensor, shape: Size, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
     options.toInternal.apply { opts => Tensor(swig.sparse_coo_tensor(indices.underlyingChecked, values.underlyingChecked, shape.sizes, opts)) }
  }

  def randomNormal(dim: Size, mean: Float = 0.0f, std: Float = 1.0f, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    init.normal_(empty(dim, options), mean, std)
  }

  def randomNormalLike(t: Tensor, mean: Float = 0.0f, std: Float = 1.0f, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    init.normal_(empty(t.shape, options), mean, std)
  }

  def randomUniform(
      shape: Size,
      low: Float = 0.0f,
      high: Float = 1.0f,
      options: TensorOptions = TensorOptions(),
  )(
      implicit manager: ReferenceManager,
  ): Tensor = {
    init.uniform_(empty(shape, options), low, high)
  }

  def randomBernoulli(shape: Size, p: Float, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    full(shape, 1.0f, options).bernoulli(p, None)
  }

  // TODO: different scalar types
  // All methods below implicitly have requiresGrad = false.

  /** @param data row-major
    */
  def fromFloatArray(data: Array[Float], shape: Size, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    require(shape.numel() == data.length, s"Shape ${shape} (numel = ${shape.numel()}) requested but ${data.length} elements provided")
    options.toInternal.apply { opts => Tensor(swig.from_float_array(data, shape.sizes, opts)) }
  }

  def fromLongArray(data: Array[Long], shape: Size, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    require(shape.numel() == data.length, s"Shape ${shape} (numel = ${shape.numel()}) requested but ${data.length} elements provided")
    options.toInternal.apply { opts => Tensor(swig.from_long_array(data, shape.sizes, opts)) }
  }

  def fromIntArray(data: Array[Int], shape: Size, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    require(shape.numel() == data.length, s"Shape ${shape} (numel = ${shape.numel()}) requested but ${data.length} elements provided")
    options.toInternal.apply { opts => Tensor(swig.from_int_array(data, shape.sizes, opts)) }
  }

  def fromByteArray(data: Array[Byte], shape: Size, options: TensorOptions = TensorOptions())(
      implicit manager: ReferenceManager,
  ): Tensor = {
    require(shape.numel() == data.length, s"Shape ${shape} (numel = ${shape.numel()}) requested but ${data.length} elements provided")
    options.toInternal.apply { opts => Tensor(swig.from_byte_array(data, shape.sizes, opts)) }
  }

  def fromDoubleArray(data: Array[Double], shape: Size, options: TensorOptions = TensorOptions())(
        implicit manager: ReferenceManager,
  ): Tensor = {
    require(shape.numel() == data.length, s"Shape ${shape} (numel = ${shape.numel()}) requested but ${data.length} elements provided")
    options.toInternal.apply { opts => Tensor(swig.from_double_array(data, shape.sizes, opts)) }
  }

  def apply(values: Float*)(implicit manager: ReferenceManager): Tensor = {
    fromFloatArray(values.toArray, Size(values.length))
  }

  def fromLongs(values: Long*)(implicit manager: ReferenceManager): Tensor = {
    fromLongArray(values.toArray, Size(values.length))
  }

  implicit class OpsForScalarable[T](f: T)(implicit toScalar: T => Scalar) {
    def +(other: Tensor)(implicit manager: ReferenceManager): Tensor = other + f
    def *(other: Tensor)(implicit manager: ReferenceManager): Tensor = other * f
    // TODO: +, *
  }

  def load(f: File, device: Option[Device] = None)(implicit rm: ReferenceManager): Tensor = {
    val v = Tensor(new TorchTensor())
    swig.load(v.underlying, f.toString, device.map(_.underlying))
    v
  }

  def loadArray(f: File, device: Option[Device] = None)(implicit rm: ReferenceManager): Array[Tensor] = {
    val v = new TensorVector()
    try {
      swig.load(v, f.toString, device.map(_.underlying))
      v.asScala.map(t => Tensor(t)).toArray
    } finally {
      v.delete()
    }
  }

}
