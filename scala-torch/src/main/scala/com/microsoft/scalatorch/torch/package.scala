// THIS FILE IS AUTO-GENERATED, DO NOT EDIT. Changes should be made to package.scala.in

package com.microsoft.scalatorch

import java.io.File

import com.microsoft.scalatorch.torch.internal.{ TensorIndex, TensorVector, TorchTensor, LongVector, EllipsisIndexType, torch_swig => swig }
import com.microsoft.scalatorch.torch.jit.{ Module, TensorType }
import com.microsoft.scalatorch.torch.syntax._
import com.microsoft.scalatorch.torch.util.NoGrad
import com.microsoft.scalatorch.torch.InternalEllipsis
import scala.collection.JavaConverters._
import scala.collection.compat._
import scala.reflect.ClassTag

/** Holds most "top level" functions exposed by torchlib.
  *
  * @see https://pytorch.org/cppdocs/api/namespace_at.html#functions for most of them
  */
package object torch {
  import com.microsoft.scalatorch.torch.util.Implicits._

  private[torch] val InternalEllipsis = new EllipsisIndexType()
  private[torch] def wrapTensorTuple2(tensorTuple: (TorchTensor, TorchTensor))(implicit cg: ReferenceManager): (Tensor, Tensor) = {
    (Tensor(tensorTuple._1), Tensor(tensorTuple._2))
  }

  private[torch] def wrapTensorTuple3(tensorTuple: (TorchTensor, TorchTensor, TorchTensor))(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = {
    (Tensor(tensorTuple._1), Tensor(tensorTuple._2), Tensor(tensorTuple._3))
  }

  private[torch] def wrapTensorTuple4(tensorTuple: (TorchTensor, TorchTensor, TorchTensor, TorchTensor))(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = {
    (Tensor(tensorTuple._1), Tensor(tensorTuple._2), Tensor(tensorTuple._3), Tensor(tensorTuple._4))
  }

  private[torch] def wrapTensorTuple5(tensorTuple: (TorchTensor, TorchTensor, TorchTensor, TorchTensor, TorchTensor))(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = {
    (Tensor(tensorTuple._1), Tensor(tensorTuple._2), Tensor(tensorTuple._3), Tensor(tensorTuple._4), Tensor(tensorTuple._5))
  }

  private[torch] def wrapTensorTuple4Long(tuple: (TorchTensor, TorchTensor, TorchTensor, TorchTensor, java.lang.Long))(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Long) = {
    (Tensor(tuple._1), Tensor(tuple._2), Tensor(tuple._3), Tensor(tuple._4), tuple._5.longValue())
  }

  private[torch] def wrapTensorTuple2DoubleLong(tuple: (TorchTensor, TorchTensor, java.lang.Double, java.lang.Long))(implicit cg: ReferenceManager): (Tensor, Tensor, Double, Long) = {
    (Tensor(tuple._1), Tensor(tuple._2), tuple._3.doubleValue(), tuple._4.longValue())
  }


  private[torch] def wrapDoubleLongTuple2(tuple: (java.lang.Double, java.lang.Long))(implicit cg: ReferenceManager): (Double, Long) = {
    (tuple._1.doubleValue(), tuple._2.longValue())
  }

  type MemoryFormat = internal.MemoryFormat
  // TODO add pass-through constants for all
  object MemoryFormat {
    val Contiguous = internal.MemoryFormat.Contiguous
    val Preserve = internal.MemoryFormat.Preserve
    val ChannelsLast = internal.MemoryFormat.ChannelsLast
  }
  type QScheme = internal.QScheme
  object QScheme {
    val PER_TENSOR_AFFINE =  internal.QScheme.PER_TENSOR_AFFINE
    val PER_CHANNEL_AFFINE = internal.QScheme.PER_CHANNEL_AFFINE
    val PER_TENSOR_SYMMETRIC = internal.QScheme.PER_TENSOR_SYMMETRIC
    val PER_CHANNEL_SYMMETRIC = internal.QScheme.PER_CHANNEL_SYMMETRIC
    val PER_CHANNEL_AFFINE_FLOAT_QPARAMS = internal.QScheme.PER_CHANNEL_AFFINE_FLOAT_QPARAMS
  }

  type Layout = internal.Layout
  object Layout {
    val Strided = internal.Layout.Strided
    val Sparse = internal.Layout.Sparse
    val Mkldnn = internal.Layout.Mkldnn
  }

  type Reduction = internal.Reduction
  object Reduction {
    val None = internal.Reduction.None
    val Mean = internal.Reduction.Mean
    val Sum = internal.Reduction.Sum
  }

  type device = Device

  type DeviceType = internal.DeviceType
  object DeviceType {
    val CPU = internal.DeviceType.CPU
    val CUDA = internal.DeviceType.CUDA
    val MKLDNN = internal.DeviceType.MKLDNN
    val OPENGL = internal.DeviceType.OPENGL
    val OPENCL = internal.DeviceType.OPENCL
    val IDEEP = internal.DeviceType.IDEEP
    val HIP = internal.DeviceType.HIP
    val FPGA = internal.DeviceType.FPGA
    val XLA = internal.DeviceType.XLA
    val Vulkan = internal.DeviceType.Vulkan
    val Metal = internal.DeviceType.Metal
    val XPU = internal.DeviceType.XPU
    val ORT = internal.DeviceType.ORT
    val MLC = internal.DeviceType.MLC
    val Meta = internal.DeviceType.Meta
    val HPU = internal.DeviceType.HPU
    val VE = internal.DeviceType.VE
    val Lazy = internal.DeviceType.Lazy
  }

  type TensorInfo = TensorType

  // From https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype

  val float32: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Float))
  val float: dtype = float32

  val float64: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Double))
  val double: dtype = float64

  val complex64: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.ComplexFloat))
  val cfloat: dtype = complex64

  val complex128: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.ComplexDouble))
  val cdouble: dtype = complex128

  val float16: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Half))
  val half: dtype = float16

  val fbloat16: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.BFloat16))

  val uint8: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Byte))

  val int8: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Char))

  val int16: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Short))

  val int32: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Int))

  val int64: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Long))
  val long: dtype = int64

  val bool: dtype = new dtype(internal.TypeMeta.fromScalarType(internal.ScalarType.Bool))

  // end dtypes

  def device(str: String): Device = Device.fromString(str)
  def device(`type`: DeviceType, index: Int): Device = Device(`type`, index)

  def manual_seed(seed: Int): Unit = {
    swig.manual_seed_fixed(seed)
  }
  // TODO these should be packages not objects
  object cuda {
    def is_available(): Boolean = swig.is_cuda_available()
    def device_count(): Int = swig.cuda_device_count()
  }
  object backend {
    object cudnn {
      def is_available(): Boolean = swig.cudnn_is_available()
    }
  }

  // manually added for now
  def normal(mean: Tensor, std: Double, generator: Option[Generator])(implicit cg: ReferenceManager): Tensor = Tensor(swig.normal(mean.underlying, std, generator.map(_.underlying)))
    def normal(mean: Double, std: Tensor, generator: Option[Generator])(implicit cg: ReferenceManager): Tensor = Tensor(swig.normal(mean, std.underlying, generator.map(_.underlying)))
    def normal(mean: Tensor, std: Tensor, generator: Option[Generator])(implicit cg: ReferenceManager): Tensor = Tensor(swig.normal(mean.underlying, std.underlying, generator.map(_.underlying)))
    def normal(mean: Double, std: Double, size: Array[Long], generator: Option[Generator] = None, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
  dtype=dtype,
  device=device,
  layout=layout,
  pinned_memory=pin_memory,
  ).toInternal.apply { options =>
    Tensor(swig.normal(mean, std, size, generator.map(_.underlying), options))
  }

  def save(value: Array[Tensor], file: String): Unit = swig.save(new TensorVector(value.map(_.underlying)), file)
  def serialize(value: Array[Tensor]): Array[Byte] = {
    val vs = new TensorVector(value.map(_.underlying))
    try {
      swig.save_TensorVector_to_byte_array(vs)
    } finally {
      vs.delete()
    }
  }
  def save(value: Tensor, file: String): Unit = swig.save(value.underlying, file)
  def serialize(value: Tensor): Array[Byte] = swig.save_Tensor_to_byte_array(value.underlying)
  def save(value: optim.Optimizer, file: String): Unit = swig.save(value.optimizer, file)
  def serialize(value: optim.Optimizer): Array[Byte] = swig.save_Optimizer_to_byte_array(value.optimizer)
  def save(value: Module, file: String): Unit = value.save(new File(file))
  def serialize(value: Module): Array[Byte] = value.serialize()

  def tensor(data: Any, options: TensorOptions = TensorOptions())(
      implicit cg: ReferenceManager,
  ): Tensor = data match {
    case d: Int    => Tensor.fromIntArray(Array(d), Size(), options)
    case d: Long   => Tensor.fromLongArray(Array(d), Size(), options)
    case d: Float  => Tensor.fromFloatArray(Array(d), Size(), options)
    case d: Byte   => Tensor.fromByteArray(Array(d), Size(), options)
    case d: Double => Tensor.fromDoubleArray(Array(d), Size(), options)
    case d: Array[_] =>
      type I = X forSome { type X }
      def findInnerType(t: Any): ClassTag[I] = (t match {
        case Array() =>
          throw new IllegalArgumentException("Empty array can't be used because the dtype cannot be inferred")
        case Array(head, tail@_*) => findInnerType(head)
        case d: Int              => ClassTag.Int
        case d: Long             => ClassTag.Long
        case d: Float            => ClassTag.Float
        case d: Byte             => ClassTag.Byte
        case d: Double           => ClassTag.Double
      }).asInstanceOf[ClassTag[I]]
      val innerClassTag = findInnerType(d)
      def flatten(a: Array[_]): (Array[I], List[Long]) = {
        if (a.length == 0) (innerClassTag.newArray(0), Nil)
        else if (a.head.getClass.isArray) {
          val elemLength = a.head.asInstanceOf[Array[_]].length
          require(
            a.forall(_.asInstanceOf[Array[I]].length == elemLength),
            s"Expected square array as input, but got ${java.util.Arrays.deepToString(a.asInstanceOf[Array[Object]])}",
          )
          val recursed = a.map(x => flatten(x.asInstanceOf[Array[I]]))
          (
            recursed.toSeq.flatMap(_._1).toArray(innerClassTag),
            a.length :: recursed.head._2,
          )
        } else (a.asInstanceOf[Array[I]], List(a.length))
      }
      val (flattened, dims) = flatten(d)
      innerClassTag match {
        case x if x == ClassTag.Int => Tensor.fromIntArray(flattened.asInstanceOf[Array[Int]], Size(dims: _*), options)
        case x if x == ClassTag.Float =>
          Tensor.fromFloatArray(flattened.asInstanceOf[Array[Float]], Size(dims: _*), options)
        case x if x == ClassTag.Byte =>
          Tensor.fromByteArray(flattened.asInstanceOf[Array[Byte]], Size(dims: _*), options)
        case x if x == ClassTag.Double =>
          Tensor.fromDoubleArray(flattened.asInstanceOf[Array[Double]], Size(dims: _*), options)
        case x if x == ClassTag.Long =>
          Tensor.fromLongArray(flattened.asInstanceOf[Array[Long]], Size(dims: _*), options)
      }
  }
  // start of auto-generated API

// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT
// See swig/src/main/swig/build.sbt for details
  def _cast_Byte(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Byte(self.underlying, non_blocking))
  def _cast_Char(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Char(self.underlying, non_blocking))
  def _cast_Double(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Double(self.underlying, non_blocking))
  def _cast_Float(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Float(self.underlying, non_blocking))
  def _cast_Int(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Int(self.underlying, non_blocking))
  def _cast_Long(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Long(self.underlying, non_blocking))
  def _cast_Short(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Short(self.underlying, non_blocking))
  def _cast_Half(self: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cast_Half(self.underlying, non_blocking))
  def _make_dual(primal: Tensor, tangent: Tensor, level: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._make_dual(primal.underlying, tangent.underlying, level))
  def _unpack_dual(dual: Tensor, level: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._unpack_dual(dual.underlying, level))
  def align_tensors(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.align_tensors(tensors.map(_.underlyingChecked)))
  def _assert_async(self: Tensor)(implicit cg: ReferenceManager): Unit = swig._assert_async(self.underlying)
  def _use_cudnn_ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Array[Long], target_lengths: Array[Long], blank: Long)(implicit cg: ReferenceManager): Boolean = swig._use_cudnn_ctc_loss(log_probs.underlying, targets.underlying, input_lengths, target_lengths, blank)
  def _cudnn_ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Array[Long], target_lengths: Array[Long], blank: Long, deterministic: Boolean, zero_infinity: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._cudnn_ctc_loss(log_probs.underlying, targets.underlying, input_lengths, target_lengths, blank, deterministic, zero_infinity))
  def _use_cudnn_rnn_flatten_weight()(implicit cg: ReferenceManager): Boolean = swig._use_cudnn_rnn_flatten_weight()
  def _cudnn_rnn_flatten_weight(weight_arr: Array[Tensor], weight_stride0: Long, input_size: Long, mode: Long, hidden_size: Long, proj_size: Long, num_layers: Long, batch_first: Boolean, bidirectional: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cudnn_rnn_flatten_weight(weight_arr.map(_.underlyingChecked), weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional))
  def _cudnn_rnn(input: Tensor, weight: Array[Tensor], weight_stride0: Long, weight_buf: Option[Tensor], hx: Tensor, cx: Option[Tensor], mode: Long, hidden_size: Long, proj_size: Long, num_layers: Long, batch_first: Boolean, dropout: Double, train: Boolean, bidirectional: Boolean, batch_sizes: Array[Long], dropout_state: Option[Tensor])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig._cudnn_rnn(input.underlying, weight.map(_.underlyingChecked), weight_stride0, weight_buf.map(_.underlying), hx.underlying, cx.map(_.underlying), mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state.map(_.underlying)))
  def _cudnn_init_dropout_state(dropout: Double, train: Boolean, dropout_seed: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = Option(false))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._cudnn_init_dropout_state(dropout, train, dropout_seed, options))
}

  def _debug_has_internal_overlap(self: Tensor)(implicit cg: ReferenceManager): Long = swig._debug_has_internal_overlap(self.underlying)
  def _fused_dropout(self: Tensor, p: Double, generator: Option[Generator] = None)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._fused_dropout(self.underlying, p, generator.map(_.underlying)))
  def _masked_scale(self: Tensor, mask: Tensor, scale: Double)(implicit cg: ReferenceManager): Tensor = Tensor(swig._masked_scale(self.underlying, mask.underlying, scale))
  def _sobol_engine_draw(quasi: Tensor, n: Long, sobolstate: Tensor, dimension: Long, num_generated: Long, dtype: Option[dtype])(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._sobol_engine_draw(quasi.underlying, n, sobolstate.underlying, dimension, num_generated, dtype.map(_.toScalarType)))
  def _sobol_engine_ff_(self: Tensor, n: Long, sobolstate: Tensor, dimension: Long, num_generated: Long)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._sobol_engine_ff_(self.underlying, n, sobolstate.underlying, dimension, num_generated)
    self
  }
  def _sobol_engine_scramble_(self: Tensor, ltm: Tensor, dimension: Long)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._sobol_engine_scramble_(self.underlying, ltm.underlying, dimension)
    self
  }
  def _sobol_engine_initialize_state_(self: Tensor, dimension: Long)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._sobol_engine_initialize_state_(self.underlying, dimension)
    self
  }
  def _reshape_from_tensor(self: Tensor, shape: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._reshape_from_tensor(self.underlying, shape.underlying))
  def _shape_as_tensor(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._shape_as_tensor(self.underlying))
  def dropout(input: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.dropout(input.underlying, p, train))
  def dropout_(self: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.dropout_(self.underlying, p, train)
    self
  }
  def feature_dropout(input: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.feature_dropout(input.underlying, p, train))
  def feature_dropout_(self: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.feature_dropout_(self.underlying, p, train)
    self
  }
  def alpha_dropout(input: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.alpha_dropout(input.underlying, p, train))
  def alpha_dropout_(self: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.alpha_dropout_(self.underlying, p, train)
    self
  }
  def feature_alpha_dropout(input: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.feature_alpha_dropout(input.underlying, p, train))
  def feature_alpha_dropout_(self: Tensor, p: Double, train: Boolean)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.feature_alpha_dropout_(self.underlying, p, train)
    self
  }
  def abs(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.abs(self.underlying))
  def abs_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.abs_(self.underlying)
    self
  }
  def absolute(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.absolute(self.underlying))
  def angle(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.angle(self.underlying))
  def view_as_real(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.view_as_real(self.underlying))
  def view_as_complex(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.view_as_complex(self.underlying))
  def sgn(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sgn(self.underlying))
  def real(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.real(self.underlying))
  def imag(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.imag(self.underlying))
  def _conj(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._conj(self.underlying))
  def conj(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conj(self.underlying))
  def _conj_physical(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._conj_physical(self.underlying))
  def conj_physical(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conj_physical(self.underlying))
  def conj_physical_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.conj_physical_(self.underlying)
    self
  }
  def resolve_conj(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.resolve_conj(self.underlying))
  def resolve_neg(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.resolve_neg(self.underlying))
  def _neg_view(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._neg_view(self.underlying))
  def acos(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.acos(self.underlying))
  def acos_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.acos_(self.underlying)
    self
  }
  def arccos(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arccos(self.underlying))
  def arccos_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arccos_(self.underlying)
    self
  }
  def avg_pool1d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), ceil_mode: Boolean = false, count_include_pad: Boolean = true)(implicit cg: ReferenceManager): Tensor = Tensor(swig.avg_pool1d(self.underlying, kernel_size, stride, padding, ceil_mode, count_include_pad))
  def adaptive_avg_pool1d(self: Tensor, output_size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.adaptive_avg_pool1d(self.underlying, output_size))
  def adaptive_max_pool1d(self: Tensor, output_size: Array[Long])(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.adaptive_max_pool1d(self.underlying, output_size))
  def add(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.add(self.underlying, other.underlying, alpha.toInternalScalar))
  def _add_relu(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._add_relu(self.underlying, other.underlying, alpha.toInternalScalar))
  def _add_relu_(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._add_relu_(self.underlying, other.underlying, alpha.toInternalScalar)
    self
  }
  def _add_relu(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig._add_relu(self.underlying, other.underlying, alpha.underlying))
  def _add_relu_(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._add_relu_(self.underlying, other.underlying, alpha.underlying)
    self
  }
  def add(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.add(self.underlying, other.underlying, alpha.underlying))
  def addmv(self: Tensor, mat: Tensor, vec: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addmv(self.underlying, mat.underlying, vec.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def addmv_(self: Tensor, mat: Tensor, vec: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.addmv_(self.underlying, mat.underlying, vec.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    self
  }
  def addr(self: Tensor, vec1: Tensor, vec2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addr(self.underlying, vec1.underlying, vec2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def affine_grid_generator(theta: Tensor, size: Array[Long], align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.affine_grid_generator(theta.underlying, size, align_corners))
  def affine_grid_generator_backward(grad: Tensor, size: Array[Long], align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.affine_grid_generator_backward(grad.underlying, size, align_corners))
  def all(self: Tensor, dim: Long, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.all(self.underlying, dim, keepdim))
  def allclose(self: Tensor, other: Tensor, rtol: Double = 1e-05, atol: Double = 1e-08, equal_nan: Boolean = false)(implicit cg: ReferenceManager): Boolean = swig.allclose(self.underlying, other.underlying, rtol, atol, equal_nan)
  def any(self: Tensor, dim: Long, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.any(self.underlying, dim, keepdim))
  def arange(end: Scalar, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.arange(end.underlying, options))
}

  def arange(start: Scalar, end: Scalar, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.arange(start.underlying, end.underlying, options))
}

  def arange(start: Scalar, end: Scalar, step: Scalar, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.arange(start.underlying, end.underlying, step.underlying, options))
}

  def _dim_arange(like: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._dim_arange(like.underlying, dim))
  def argmax(self: Tensor, dim: Option[Long] = None, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.argmax(self.underlying, dim.asJavaLong, keepdim))
  def argmin(self: Tensor, dim: Option[Long] = None, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.argmin(self.underlying, dim.asJavaLong, keepdim))
  def acosh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.acosh(self.underlying))
  def acosh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.acosh_(self.underlying)
    self
  }
  def arccosh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arccosh(self.underlying))
  def arccosh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arccosh_(self.underlying)
    self
  }
  def asinh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.asinh(self.underlying))
  def asinh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.asinh_(self.underlying)
    self
  }
  def arcsinh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arcsinh(self.underlying))
  def arcsinh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arcsinh_(self.underlying)
    self
  }
  def atanh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atanh(self.underlying))
  def atanh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.atanh_(self.underlying)
    self
  }
  def arctanh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arctanh(self.underlying))
  def arctanh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arctanh_(self.underlying)
    self
  }
  def as_strided(self: Tensor, size: Array[Long], stride: Array[Long], storage_offset: Option[Long] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.as_strided(self.underlying, size, stride, storage_offset.asJavaLong))
  def as_strided_(self: Tensor, size: Array[Long], stride: Array[Long], storage_offset: Option[Long] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.as_strided_(self.underlying, size, stride, storage_offset.asJavaLong)
    self
  }
  def asin(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.asin(self.underlying))
  def asin_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.asin_(self.underlying)
    self
  }
  def arcsin(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arcsin(self.underlying))
  def arcsin_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arcsin_(self.underlying)
    self
  }
  def atan(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atan(self.underlying))
  def atan_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.atan_(self.underlying)
    self
  }
  def arctan(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.arctan(self.underlying))
  def arctan_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.arctan_(self.underlying)
    self
  }
  def atleast_1d(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atleast_1d(self.underlying))
  def atleast_1d(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.atleast_1d(tensors.map(_.underlyingChecked)))
  def atleast_2d(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atleast_2d(self.underlying))
  def atleast_2d(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.atleast_2d(tensors.map(_.underlyingChecked)))
  def atleast_3d(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atleast_3d(self.underlying))
  def atleast_3d(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.atleast_3d(tensors.map(_.underlyingChecked)))
  def baddbmm(self: Tensor, batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.baddbmm(self.underlying, batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def _baddbmm_mkl_(self: Tensor, batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._baddbmm_mkl_(self.underlying, batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar)
    self
  }
  def bartlett_window(window_length: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.bartlett_window(window_length, options))
}

  def bartlett_window(window_length: Long, periodic: Boolean, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.bartlett_window(window_length, periodic, options))
}

  def batch_norm(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], training: Boolean, momentum: Double, eps: Double, cudnn_enabled: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.batch_norm(input.underlying, weight.map(_.underlying), bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), training, momentum, eps, cudnn_enabled))
  def quantized_batch_norm(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], mean: Tensor, `var`: Tensor, eps: Double, output_scale: Double, output_zero_point: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_batch_norm(input.underlying, weight.map(_.underlying), bias.map(_.underlying), mean.underlying, `var`.underlying, eps, output_scale, output_zero_point))
  def _batch_norm_impl_index(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], training: Boolean, momentum: Double, eps: Double, cudnn_enabled: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Long) = wrapTensorTuple4Long(swig._batch_norm_impl_index(input.underlying, weight.map(_.underlying), bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), training, momentum, eps, cudnn_enabled))
  def _batch_norm_impl_index_backward(impl_index: Long, input: Tensor, grad_output: Tensor, weight: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], save_mean: Option[Tensor], save_var_transform: Option[Tensor], train: Boolean, eps: Double, output_mask: Array[Boolean], reservedSpace: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._batch_norm_impl_index_backward(impl_index, input.underlying, grad_output.underlying, weight.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), save_mean.map(_.underlying), save_var_transform.map(_.underlying), train, eps, output_mask, reservedSpace.underlying))
  def bernoulli(self: Tensor, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bernoulli(self.underlying, generator.map(_.underlying)))
  def bernoulli(self: Tensor, p: Double, generator: Option[Generator])(implicit cg: ReferenceManager): Tensor = Tensor(swig.bernoulli(self.underlying, p, generator.map(_.underlying)))
  def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Option[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.bilinear(input1.underlying, input2.underlying, weight.underlying, bias.map(_.underlying)))
  def binary_cross_entropy_with_logits(self: Tensor, target: Tensor, weight: Option[Tensor] = None, pos_weight: Option[Tensor] = None, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.binary_cross_entropy_with_logits(self.underlying, target.underlying, weight.map(_.underlying), pos_weight.map(_.underlying), reduction))
  def binary_cross_entropy_with_logits_backward(grad_output: Tensor, self: Tensor, target: Tensor, weight: Option[Tensor] = None, pos_weight: Option[Tensor] = None, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.binary_cross_entropy_with_logits_backward(grad_output.underlying, self.underlying, target.underlying, weight.map(_.underlying), pos_weight.map(_.underlying), reduction))
  def bincount(self: Tensor, weights: Option[Tensor] = None, minlength: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bincount(self.underlying, weights.map(_.underlying), minlength))
  def bitwise_not(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_not(self.underlying))
  def copysign(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.copysign(self.underlying, other.underlying))
  def copysign(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.copysign(self.underlying, other.underlying))
  def logical_not(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logical_not(self.underlying))
  def logical_xor(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logical_xor(self.underlying, other.underlying))
  def logical_and(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logical_and(self.underlying, other.underlying))
  def logical_or(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logical_or(self.underlying, other.underlying))
  def blackman_window(window_length: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.blackman_window(window_length, options))
}

  def blackman_window(window_length: Long, periodic: Boolean, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.blackman_window(window_length, periodic, options))
}

  def bmm(self: Tensor, mat2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bmm(self.underlying, mat2.underlying))
  def broadcast_tensors(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.broadcast_tensors(tensors.map(_.underlyingChecked)))
  def broadcast_to(self: Tensor, size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.broadcast_to(self.underlying, size))
  def cat(tensors: Array[Tensor], dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cat(tensors.map(_.underlyingChecked), dim))
  def concat(tensors: Array[Tensor], dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.concat(tensors.map(_.underlyingChecked), dim))
  def block_diag(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.block_diag(tensors.map(_.underlyingChecked)))
  def ceil(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ceil(self.underlying))
  def ceil_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.ceil_(self.underlying)
    self
  }
  def chain_matmul(matrices: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.chain_matmul(matrices.map(_.underlyingChecked)))
  def unsafe_chunk(self: Tensor, chunks: Long, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.unsafe_chunk(self.underlying, chunks, dim))
  def chunk(self: Tensor, chunks: Long, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.chunk(self.underlying, chunks, dim))
  def tensor_split(self: Tensor, sections: Long, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.tensor_split(self.underlying, sections, dim))
  def tensor_split(self: Tensor, indices: Array[Long], dim: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.tensor_split(self.underlying, indices, dim))
  def tensor_split(self: Tensor, tensor_indices_or_sections: Tensor, dim: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.tensor_split(self.underlying, tensor_indices_or_sections.underlying, dim))
  def clamp(self: Tensor, min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clamp(self.underlying, min.map(_.underlying), max.map(_.underlying)))
  def clamp_(self: Tensor, min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clamp_(self.underlying, min.map(_.underlying), max.map(_.underlying))
    self
  }
  def clamp_max(self: Tensor, max: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clamp_max(self.underlying, max.underlying))
  def clamp_max(self: Tensor, max: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clamp_max(self.underlying, max.underlying))
  def clamp_max_(self: Tensor, max: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clamp_max_(self.underlying, max.underlying)
    self
  }
  def clamp_max_(self: Tensor, max: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clamp_max_(self.underlying, max.underlying)
    self
  }
  def clamp_min(self: Tensor, min: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clamp_min(self.underlying, min.underlying))
  def clamp_min(self: Tensor, min: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clamp_min(self.underlying, min.underlying))
  def clamp_min_(self: Tensor, min: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clamp_min_(self.underlying, min.underlying)
    self
  }
  def clamp_min_(self: Tensor, min: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clamp_min_(self.underlying, min.underlying)
    self
  }
  def clip(self: Tensor, min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clip(self.underlying, min.map(_.underlying), max.map(_.underlying)))
  def clip_(self: Tensor, min: Option[Scalar] = None, max: Option[Scalar] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.clip_(self.underlying, min.map(_.underlying), max.map(_.underlying))
    self
  }
  def cudnn_is_acceptable(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.cudnn_is_acceptable(self.underlying)
  def complex(real: Tensor, imag: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.complex(real.underlying, imag.underlying))
  def polar(abs: Tensor, angle: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.polar(abs.underlying, angle.underlying))
  def constant_pad_nd(self: Tensor, pad: Array[Long], value: Double = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.constant_pad_nd(self.underlying, pad, value.toInternalScalar))
  def convolution(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.convolution(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, transposed, output_padding, groups))
  def convolution_overrideable(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.convolution_overrideable(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, transposed, output_padding, groups))
  def convolution_backward_overrideable(grad_output: Tensor, input: Tensor, weight: Tensor, stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.convolution_backward_overrideable(grad_output.underlying, input.underlying, weight.underlying, stride, padding, dilation, transposed, output_padding, groups, output_mask))
  def _convolution(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, cudnn_enabled: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._convolution(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32))
  def _convolution(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, cudnn_enabled: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._convolution(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled))
  def _convolution_mode(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: String, dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._convolution_mode(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def _convolution_nogroup(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._convolution_nogroup(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, transposed, output_padding))
  def _convolution_double_backward(ggI: Option[Tensor], ggW: Option[Tensor], ggb: Option[Tensor], gO: Tensor, weight: Tensor, self: Tensor, stride: Array[Long], padding: Array[Long], dilation: Array[Long], transposed: Boolean, output_padding: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, cudnn_enabled: Boolean, allow_tf32: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._convolution_double_backward(ggI.map(_.underlying), ggW.map(_.underlying), ggb.map(_.underlying), gO.underlying, weight.underlying, self.underlying, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask))
  def conv1d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), groups: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv1d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv2d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), groups: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv2d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv3d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), groups: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv3d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv1d(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: String, dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv1d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv2d(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: String, dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv2d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv3d(input: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: String, dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv3d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def conv_tbc(self: Tensor, weight: Tensor, bias: Tensor, pad: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv_tbc(self.underlying, weight.underlying, bias.underlying, pad))
  def conv_tbc_backward(self: Tensor, input: Tensor, weight: Tensor, bias: Tensor, pad: Long)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.conv_tbc_backward(self.underlying, input.underlying, weight.underlying, bias.underlying, pad))
  def conv_transpose1d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), output_padding: Array[Long] = Array(0), groups: Long = 1, dilation: Array[Long] = Array(1))(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv_transpose1d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, output_padding, groups, dilation))
  def conv_transpose2d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), output_padding: Array[Long] = Array(0), groups: Long = 1, dilation: Array[Long] = Array(1))(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv_transpose2d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, output_padding, groups, dilation))
  def conv_transpose3d(input: Tensor, weight: Tensor, bias: Option[Tensor] = None, stride: Array[Long] = Array(1), padding: Array[Long] = Array(0), output_padding: Array[Long] = Array(0), groups: Long = 1, dilation: Array[Long] = Array(1))(implicit cg: ReferenceManager): Tensor = Tensor(swig.conv_transpose3d(input.underlying, weight.underlying, bias.map(_.underlying), stride, padding, output_padding, groups, dilation))
  def _copy_from(self: Tensor, dst: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._copy_from(self.underlying, dst.underlying, non_blocking))
  def _copy_from_and_resize(self: Tensor, dst: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._copy_from_and_resize(self.underlying, dst.underlying))
  def cos(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cos(self.underlying))
  def cos_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.cos_(self.underlying)
    self
  }
  def cosh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cosh(self.underlying))
  def cosh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.cosh_(self.underlying)
    self
  }
  def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: Double = 0.0, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.cosine_embedding_loss(input1.underlying, input2.underlying, target.underlying, margin, reduction))
  def count_nonzero(self: Tensor, dim: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.count_nonzero(self.underlying, dim))
  def count_nonzero(self: Tensor, dim: Option[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.count_nonzero(self.underlying, dim.asJavaLong))
  def cov(self: Tensor, correction: Long = 1, fweights: Option[Tensor] = None, aweights: Option[Tensor] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cov(self.underlying, correction, fweights.map(_.underlying), aweights.map(_.underlying)))
  def corrcoef(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.corrcoef(self.underlying))
  def cudnn_affine_grid_generator(theta: Tensor, N: Long, C: Long, H: Long, W: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_affine_grid_generator(theta.underlying, N, C, H, W))
  def cudnn_affine_grid_generator_backward(grad: Tensor, N: Long, C: Long, H: Long, W: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_affine_grid_generator_backward(grad.underlying, N, C, H, W))
  def cudnn_batch_norm(input: Tensor, weight: Tensor, bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], training: Boolean, exponential_average_factor: Double, epsilon: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig.cudnn_batch_norm(input.underlying, weight.underlying, bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), training, exponential_average_factor, epsilon))
  def cudnn_batch_norm_backward(input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Option[Tensor], running_var: Option[Tensor], save_mean: Option[Tensor], save_var: Option[Tensor], epsilon: Double, reserveSpace: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.cudnn_batch_norm_backward(input.underlying, grad_output.underlying, weight.underlying, running_mean.map(_.underlying), running_var.map(_.underlying), save_mean.map(_.underlying), save_var.map(_.underlying), epsilon, reserveSpace.underlying))
  def cudnn_convolution(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution(self.underlying, weight.underlying, bias.map(_.underlying), padding, stride, dilation, groups, benchmark, deterministic))
  def cudnn_convolution(self: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution(self.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def cudnn_convolution(self: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution(self.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_backward_input(self_size: Array[Long], grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_backward_input(self_size, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.cudnn_convolution_backward(self.underlying, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask))
  def cudnn_convolution_backward_weight(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_backward_weight(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_transpose(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_transpose(self.underlying, weight.underlying, bias.map(_.underlying), padding, output_padding, stride, dilation, groups, benchmark, deterministic))
  def cudnn_convolution_transpose(self: Tensor, weight: Tensor, padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_transpose(self.underlying, weight.underlying, padding, output_padding, stride, dilation, groups, benchmark, deterministic))
  def cudnn_convolution_transpose(self: Tensor, weight: Tensor, padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_transpose(self.underlying, weight.underlying, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_transpose_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.cudnn_convolution_transpose_backward(self.underlying, grad_output.underlying, weight.underlying, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask))
  def cudnn_convolution_transpose_backward_input(grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_transpose_backward_input(grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_transpose_backward_weight(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, allow_tf32: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_transpose_backward_weight(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32))
  def cudnn_convolution_relu(self: Tensor, weight: Tensor, bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_relu(self.underlying, weight.underlying, bias.map(_.underlying), stride, padding, dilation, groups))
  def cudnn_convolution_add_relu(self: Tensor, weight: Tensor, z: Tensor, alpha: Option[Scalar], bias: Option[Tensor], stride: Array[Long], padding: Array[Long], dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_convolution_add_relu(self.underlying, weight.underlying, z.underlying, alpha.map(_.underlying), bias.map(_.underlying), stride, padding, dilation, groups))
  def cudnn_grid_sampler(self: Tensor, grid: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cudnn_grid_sampler(self.underlying, grid.underlying))
  def cudnn_grid_sampler_backward(self: Tensor, grid: Tensor, grad_output: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.cudnn_grid_sampler_backward(self.underlying, grid.underlying, grad_output.underlying))
  def cummax(self: Tensor, dim: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.cummax(self.underlying, dim))
  def _cummax_helper(self: Tensor, values: Tensor, indices: Tensor, dim: Long)(implicit cg: ReferenceManager): Unit = swig._cummax_helper(self.underlying, values.underlying, indices.underlying, dim)
  def cummin(self: Tensor, dim: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.cummin(self.underlying, dim))
  def _cummin_helper(self: Tensor, values: Tensor, indices: Tensor, dim: Long)(implicit cg: ReferenceManager): Unit = swig._cummin_helper(self.underlying, values.underlying, indices.underlying, dim)
  def cummaxmin_backward(grad: Tensor, input: Tensor, indices: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cummaxmin_backward(grad.underlying, input.underlying, indices.underlying, dim))
  def cumprod(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cumprod(self.underlying, dim, dtype.map(_.toScalarType)))
  def cumprod_backward(grad: Tensor, input: Tensor, dim: Long, output: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cumprod_backward(grad.underlying, input.underlying, dim, output.underlying))
  def cumsum(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cumsum(self.underlying, dim, dtype.map(_.toScalarType)))
  def cumulative_trapezoid(y: Tensor, x: Tensor, dim: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cumulative_trapezoid(y.underlying, x.underlying, dim))
  def cumulative_trapezoid(y: Tensor, dx: Scalar, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cumulative_trapezoid(y.underlying, dx.underlying, dim))
  def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Array[Long], target_lengths: Array[Long], blank: Long = 0, reduction: Long = internal.Reduction.Mean.swigValue(), zero_infinity: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ctc_loss(log_probs.underlying, targets.underlying, input_lengths, target_lengths, blank, reduction, zero_infinity))
  def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: Long, reduction: Long, zero_infinity: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ctc_loss(log_probs.underlying, targets.underlying, input_lengths.underlying, target_lengths.underlying, blank, reduction, zero_infinity))
  def _ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Array[Long], target_lengths: Array[Long], blank: Long = 0, zero_infinity: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._ctc_loss(log_probs.underlying, targets.underlying, input_lengths, target_lengths, blank, zero_infinity))
  def _ctc_loss_backward(grad: Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Array[Long], target_lengths: Array[Long], neg_log_likelihood: Tensor, log_alpha: Tensor, blank: Long, zero_infinity: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._ctc_loss_backward(grad.underlying, log_probs.underlying, targets.underlying, input_lengths, target_lengths, neg_log_likelihood.underlying, log_alpha.underlying, blank, zero_infinity))
  def diag_embed(self: Tensor, offset: Long = 0, dim1: Long = -2, dim2: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diag_embed(self.underlying, offset, dim1, dim2))
  def diagflat(self: Tensor, offset: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diagflat(self.underlying, offset))
  def diagonal(self: Tensor, offset: Long = 0, dim1: Long = 0, dim2: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diagonal(self.underlying, offset, dim1, dim2))
  def diagonal_backward(grad_output: Tensor, input_sizes: Array[Long], offset: Long, dim1: Long, dim2: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diagonal_backward(grad_output.underlying, input_sizes, offset, dim1, dim2))
  def diff(self: Tensor, n: Long = 1, dim: Long = -1, prepend: Option[Tensor] = None, append: Option[Tensor] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diff(self.underlying, n, dim, prepend.map(_.underlying), append.map(_.underlying)))
  def gradient(self: Tensor, spacing: Option[Scalar] = None, dim: Option[Long] = None, edge_order: Long = 1)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.map(_.underlying), dim.asJavaLong, edge_order))
  def gradient(self: Tensor, spacing: Scalar, dim: Array[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.underlying, dim, edge_order))
  def gradient(self: Tensor, dim: Array[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, dim, edge_order))
  def gradient(self: Tensor, spacing: Array[Scalar], dim: Option[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.map(_.underlying), dim.asJavaLong, edge_order))
  def gradient(self: Tensor, spacing: Array[Scalar], dim: Array[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.map(_.underlying), dim, edge_order))
  def gradient(self: Tensor, spacing: Array[Tensor], dim: Option[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.map(_.underlyingChecked), dim.asJavaLong, edge_order))
  def gradient(self: Tensor, spacing: Array[Tensor], dim: Array[Long], edge_order: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.gradient(self.underlying, spacing.map(_.underlyingChecked), dim, edge_order))
  def div(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.div(self.underlying, other.underlying))
  def div(self: Tensor, other: Tensor, rounding_mode: Option[String])(implicit cg: ReferenceManager): Tensor = Tensor(swig.div(self.underlying, other.underlying, rounding_mode))
  def div(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.div(self.underlying, other.underlying))
  def div(self: Tensor, other: Scalar, rounding_mode: Option[String])(implicit cg: ReferenceManager): Tensor = Tensor(swig.div(self.underlying, other.underlying, rounding_mode))
  def divide(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.divide(self.underlying, other.underlying))
  def divide(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.divide(self.underlying, other.underlying))
  def divide(self: Tensor, other: Tensor, rounding_mode: Option[String])(implicit cg: ReferenceManager): Tensor = Tensor(swig.divide(self.underlying, other.underlying, rounding_mode))
  def divide(self: Tensor, other: Scalar, rounding_mode: Option[String])(implicit cg: ReferenceManager): Tensor = Tensor(swig.divide(self.underlying, other.underlying, rounding_mode))
  def true_divide(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.true_divide(self.underlying, other.underlying))
  def true_divide(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.true_divide(self.underlying, other.underlying))
  def dot(self: Tensor, tensor: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.dot(self.underlying, tensor.underlying))
  def vdot(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.vdot(self.underlying, other.underlying))
  def einsum(equation: String, tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.einsum(equation, tensors.map(_.underlyingChecked)))
  def embedding(weight: Tensor, indices: Tensor, padding_idx: Long = -1, scale_grad_by_freq: Boolean = false, sparse: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.embedding(weight.underlying, indices.underlying, padding_idx, scale_grad_by_freq, sparse))
  def embedding_backward(grad: Tensor, indices: Tensor, num_weights: Long, padding_idx: Long, scale_grad_by_freq: Boolean, sparse: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.embedding_backward(grad.underlying, indices.underlying, num_weights, padding_idx, scale_grad_by_freq, sparse))
  def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: Long, padding_idx: Long, scale_grad_by_freq: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.embedding_dense_backward(grad_output.underlying, indices.underlying, num_weights, padding_idx, scale_grad_by_freq))
  def embedding_renorm_(self: Tensor, indices: Tensor, max_norm: Double, norm_type: Double)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.embedding_renorm_(self.underlying, indices.underlying, max_norm, norm_type)
    self
  }
  def embedding_sparse_backward(grad: Tensor, indices: Tensor, num_weights: Long, padding_idx: Long, scale_grad_by_freq: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.embedding_sparse_backward(grad.underlying, indices.underlying, num_weights, padding_idx, scale_grad_by_freq))
  def _embedding_bag_forward_only(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Boolean = false, mode: Long = 0, sparse: Boolean = false, per_sample_weights: Option[Tensor] = None, include_last_offset: Boolean = false, padding_idx: Long = -1)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig._embedding_bag_forward_only(weight.underlying, indices.underlying, offsets.underlying, scale_grad_by_freq, mode, sparse, per_sample_weights.map(_.underlying), include_last_offset, padding_idx))
  def _rowwise_prune(weight: Tensor, mask: Tensor, compressed_indices_dtype: dtype)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._rowwise_prune(weight.underlying, mask.underlying, compressed_indices_dtype.toScalarType))
  def row_stack(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.row_stack(tensors.map(_.underlyingChecked)))
  def embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Boolean = false, mode: Long = 0, sparse: Boolean = false, per_sample_weights: Option[Tensor] = None, include_last_offset: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig.embedding_bag(weight.underlying, indices.underlying, offsets.underlying, scale_grad_by_freq, mode, sparse, per_sample_weights.map(_.underlying), include_last_offset))
  def embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Boolean, mode: Long, sparse: Boolean, per_sample_weights: Option[Tensor], include_last_offset: Boolean, padding_idx: Option[Long])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig.embedding_bag(weight.underlying, indices.underlying, offsets.underlying, scale_grad_by_freq, mode, sparse, per_sample_weights.map(_.underlying), include_last_offset, padding_idx.asJavaLong))
  def _embedding_bag(weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: Boolean = false, mode: Long = 0, sparse: Boolean = false, per_sample_weights: Option[Tensor] = None, include_last_offset: Boolean = false, padding_idx: Long = -1)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig._embedding_bag(weight.underlying, indices.underlying, offsets.underlying, scale_grad_by_freq, mode, sparse, per_sample_weights.map(_.underlying), include_last_offset, padding_idx))
  def _embedding_bag_backward(grad: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, bag_size: Tensor, maximum_indices: Tensor, num_weights: Long, scale_grad_by_freq: Boolean, mode: Long, sparse: Boolean, per_sample_weights: Option[Tensor], padding_idx: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._embedding_bag_backward(grad.underlying, indices.underlying, offsets.underlying, offset2bag.underlying, bag_size.underlying, maximum_indices.underlying, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights.map(_.underlying), padding_idx))
  def _embedding_bag_sparse_backward(grad: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, bag_size: Tensor, num_weights: Long, scale_grad_by_freq: Boolean, mode: Long, per_sample_weights: Option[Tensor], padding_idx: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._embedding_bag_sparse_backward(grad.underlying, indices.underlying, offsets.underlying, offset2bag.underlying, bag_size.underlying, num_weights, scale_grad_by_freq, mode, per_sample_weights.map(_.underlying), padding_idx))
  def _embedding_bag_dense_backward(grad: Tensor, indices: Tensor, offset2bag: Tensor, bag_size: Tensor, maximum_indices: Tensor, num_weights: Long, scale_grad_by_freq: Boolean, mode: Long, per_sample_weights: Option[Tensor], padding_idx: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._embedding_bag_dense_backward(grad.underlying, indices.underlying, offset2bag.underlying, bag_size.underlying, maximum_indices.underlying, num_weights, scale_grad_by_freq, mode, per_sample_weights.map(_.underlying), padding_idx))
  def _embedding_bag_per_sample_weights_backward(grad: Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, mode: Long, padding_idx: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._embedding_bag_per_sample_weights_backward(grad.underlying, weight.underlying, indices.underlying, offsets.underlying, offset2bag.underlying, mode, padding_idx))
  def empty(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.empty(size, options, memory_format))
}

  def _empty_affine_quantized(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, scale: Double = 1, zero_point: Long = 0, memory_format: Option[MemoryFormat] = Option(MemoryFormat.Contiguous))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._empty_affine_quantized(size, options, scale, zero_point, memory_format))
}

  def _empty_per_channel_affine_quantized(size: Array[Long], scales: Tensor, zero_points: Tensor, axis: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = Option(MemoryFormat.Contiguous))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._empty_per_channel_affine_quantized(size, scales.underlying, zero_points.underlying, axis, options, memory_format))
}

  def empty_quantized(size: Array[Long], qtensor: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.empty_quantized(size, qtensor.underlying, options, memory_format))
}

  def empty_like(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.empty_like(self.underlying, options, memory_format))
}

  def empty_strided(size: Array[Long], stride: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.empty_strided(size, stride, options))
}

  def erf(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.erf(self.underlying))
  def erf_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.erf_(self.underlying)
    self
  }
  def erfc(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.erfc(self.underlying))
  def erfc_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.erfc_(self.underlying)
    self
  }
  def exp(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.exp(self.underlying))
  def exp_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.exp_(self.underlying)
    self
  }
  def exp2(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.exp2(self.underlying))
  def exp2_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.exp2_(self.underlying)
    self
  }
  def expm1(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.expm1(self.underlying))
  def expm1_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.expm1_(self.underlying)
    self
  }
  def eye(n: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.eye(n, options))
}

  def eye(n: Long, m: Long, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.eye(n, m, options))
}

  def flatten(self: Tensor, start_dim: Long = 0, end_dim: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.flatten(self.underlying, start_dim, end_dim))
  def fill_(self: Tensor, value: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.fill_(self.underlying, value.underlying)
    self
  }
  def fill_(self: Tensor, value: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.fill_(self.underlying, value.underlying)
    self
  }
  def floor(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.floor(self.underlying))
  def floor_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.floor_(self.underlying)
    self
  }
  def floor_divide(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.floor_divide(self.underlying, other.underlying))
  def floor_divide(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.floor_divide(self.underlying, other.underlying))
  def frac(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.frac(self.underlying))
  def frac_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.frac_(self.underlying)
    self
  }
  def full(size: Array[Long], fill_value: Scalar, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.full(size, fill_value.underlying, options))
}

  def full_like(self: Tensor, fill_value: Scalar, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.full_like(self.underlying, fill_value.underlying, options, memory_format))
}

  def from_file(filename: String, shared: Option[Boolean] = None, size: Option[Long] = Option(0), dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.from_file(filename, shared.map(java.lang.Boolean.valueOf), size.asJavaLong, options))
}

  def gcd(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gcd(self.underlying, other.underlying))
  def gcd_(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.gcd_(self.underlying, other.underlying)
    self
  }
  def lcm(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lcm(self.underlying, other.underlying))
  def lcm_(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.lcm_(self.underlying, other.underlying)
    self
  }
  def grid_sampler(input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.grid_sampler(input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def grid_sampler_2d(input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.grid_sampler_2d(input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def grid_sampler_2d_backward(grad_output: Tensor, input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.grid_sampler_2d_backward(grad_output.underlying, input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def _grid_sampler_2d_cpu_fallback(input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._grid_sampler_2d_cpu_fallback(input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def _grid_sampler_2d_cpu_fallback_backward(grad_output: Tensor, input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._grid_sampler_2d_cpu_fallback_backward(grad_output.underlying, input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def grid_sampler_3d(input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.grid_sampler_3d(input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def grid_sampler_3d_backward(grad_output: Tensor, input: Tensor, grid: Tensor, interpolation_mode: Long, padding_mode: Long, align_corners: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.grid_sampler_3d_backward(grad_output.underlying, input.underlying, grid.underlying, interpolation_mode, padding_mode, align_corners))
  def hann_window(window_length: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hann_window(window_length, options))
}

  def hann_window(window_length: Long, periodic: Boolean, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hann_window(window_length, periodic, options))
}

  def hamming_window(window_length: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hamming_window(window_length, options))
}

  def hamming_window(window_length: Long, periodic: Boolean, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hamming_window(window_length, periodic, options))
}

  def hamming_window(window_length: Long, periodic: Boolean, alpha: Double, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hamming_window(window_length, periodic, alpha, options))
}

  def hamming_window(window_length: Long, periodic: Boolean, alpha: Double, beta: Double, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.hamming_window(window_length, periodic, alpha, beta, options))
}

  def kaiser_window(window_length: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.kaiser_window(window_length, options))
}

  def kaiser_window(window_length: Long, periodic: Boolean, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.kaiser_window(window_length, periodic, options))
}

  def kaiser_window(window_length: Long, periodic: Boolean, beta: Double, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.kaiser_window(window_length, periodic, beta, options))
}

  def hinge_embedding_loss(self: Tensor, target: Tensor, margin: Double = 1.0, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.hinge_embedding_loss(self.underlying, target.underlying, margin, reduction))
  def group_norm(input: Tensor, num_groups: Long, weight: Option[Tensor] = None, bias: Option[Tensor] = None, eps: Double = 1e-05, cudnn_enabled: Boolean = true)(implicit cg: ReferenceManager): Tensor = Tensor(swig.group_norm(input.underlying, num_groups, weight.map(_.underlying), bias.map(_.underlying), eps, cudnn_enabled))
  def native_group_norm(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], N: Long, C: Long, HxW: Long, group: Long, eps: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_group_norm(input.underlying, weight.map(_.underlying), bias.map(_.underlying), N, C, HxW, group, eps))
  def native_group_norm_backward(grad_out: Tensor, input: Tensor, mean: Tensor, rstd: Tensor, weight: Option[Tensor], N: Long, C: Long, HxW: Long, group: Long, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_group_norm_backward(grad_out.underlying, input.underlying, mean.underlying, rstd.underlying, weight.map(_.underlying), N, C, HxW, group, output_mask))
  def _fft_r2c(self: Tensor, dim: Array[Long], normalization: Long, onesided: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._fft_r2c(self.underlying, dim, normalization, onesided))
  def _fft_c2r(self: Tensor, dim: Array[Long], normalization: Long, last_dim_size: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._fft_c2r(self.underlying, dim, normalization, last_dim_size))
  def _fft_c2c(self: Tensor, dim: Array[Long], normalization: Long, forward: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._fft_c2c(self.underlying, dim, normalization, forward))
  def _cufft_get_plan_cache_size(device_index: Long)(implicit cg: ReferenceManager): Long = swig._cufft_get_plan_cache_size(device_index)
  def _cufft_get_plan_cache_max_size(device_index: Long)(implicit cg: ReferenceManager): Long = swig._cufft_get_plan_cache_max_size(device_index)
  def _cufft_set_plan_cache_max_size(device_index: Long, max_size: Long)(implicit cg: ReferenceManager): Unit = swig._cufft_set_plan_cache_max_size(device_index, max_size)
  def _cufft_clear_plan_cache(device_index: Long)(implicit cg: ReferenceManager): Unit = swig._cufft_clear_plan_cache(device_index)
  def index(self: Tensor, indices: Array[Option[Tensor]])(implicit cg: ReferenceManager): Tensor = Tensor(swig.index(self.underlying, indices.map(_.map(_.underlying))))
  def index_copy(self: Tensor, dim: Long, index: Tensor, source: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_copy(self.underlying, dim, index.underlying, source.underlying))
  def index_put_(self: Tensor, indices: Array[Option[Tensor]], values: Tensor, accumulate: Boolean = false)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.index_put_(self.underlying, indices.map(_.map(_.underlying)), values.underlying, accumulate)
    self
  }
  def index_put(self: Tensor, indices: Array[Option[Tensor]], values: Tensor, accumulate: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_put(self.underlying, indices.map(_.map(_.underlying)), values.underlying, accumulate))
  def _index_put_impl_(self: Tensor, indices: Array[Option[Tensor]], values: Tensor, accumulate: Boolean = false, unsafe: Boolean = false)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._index_put_impl_(self.underlying, indices.map(_.map(_.underlying)), values.underlying, accumulate, unsafe)
    self
  }
  def instance_norm(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], use_input_stats: Boolean, momentum: Double, eps: Double, cudnn_enabled: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.instance_norm(input.underlying, weight.map(_.underlying), bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), use_input_stats, momentum, eps, cudnn_enabled))
  def inverse(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.inverse(self.underlying))
  def _inverse_helper(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._inverse_helper(self.underlying))
  def isclose(self: Tensor, other: Tensor, rtol: Double = 1e-05, atol: Double = 1e-08, equal_nan: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isclose(self.underlying, other.underlying, rtol, atol, equal_nan))
  def isin(elements: Tensor, test_elements: Tensor, assume_unique: Boolean = false, invert: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isin(elements.underlying, test_elements.underlying, assume_unique, invert))
  def isin(elements: Tensor, test_element: Scalar, assume_unique: Boolean, invert: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isin(elements.underlying, test_element.underlying, assume_unique, invert))
  def isin(element: Scalar, test_elements: Tensor, assume_unique: Boolean, invert: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isin(element.underlying, test_elements.underlying, assume_unique, invert))
  def isnan(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isnan(self.underlying))
  def is_distributed(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_distributed(self.underlying)
  def is_floating_point(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_floating_point(self.underlying)
  def is_complex(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_complex(self.underlying)
  def is_conj(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_conj(self.underlying)
  def is_neg(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_neg(self.underlying)
  def isreal(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isreal(self.underlying))
  def is_nonzero(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_nonzero(self.underlying)
  def is_same_size(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_same_size(self.underlying, other.underlying)
  def is_signed(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_signed(self.underlying)
  def is_inference(self: Tensor)(implicit cg: ReferenceManager): Boolean = swig.is_inference(self.underlying)
  def kl_div(self: Tensor, target: Tensor, reduction: Long = internal.Reduction.Mean.swigValue(), log_target: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.kl_div(self.underlying, target.underlying, reduction, log_target))
  def kl_div_backward(grad_output: Tensor, self: Tensor, target: Tensor, reduction: Long = internal.Reduction.Mean.swigValue(), log_target: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.kl_div_backward(grad_output.underlying, self.underlying, target.underlying, reduction, log_target))
  def kron(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.kron(self.underlying, other.underlying))
  def kthvalue(self: Tensor, k: Long, dim: Long = -1, keepdim: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.kthvalue(self.underlying, k, dim, keepdim))
  def layer_norm(input: Tensor, normalized_shape: Array[Long], weight: Option[Tensor] = None, bias: Option[Tensor] = None, eps: Double = 1e-05, cudnn_enable: Boolean = true)(implicit cg: ReferenceManager): Tensor = Tensor(swig.layer_norm(input.underlying, normalized_shape, weight.map(_.underlying), bias.map(_.underlying), eps, cudnn_enable))
  def native_layer_norm(input: Tensor, normalized_shape: Array[Long], weight: Option[Tensor], bias: Option[Tensor], eps: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_layer_norm(input.underlying, normalized_shape, weight.map(_.underlying), bias.map(_.underlying), eps))
  def native_layer_norm_backward(grad_out: Tensor, input: Tensor, normalized_shape: Array[Long], mean: Tensor, rstd: Tensor, weight: Option[Tensor], bias: Option[Tensor], output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_layer_norm_backward(grad_out.underlying, input.underlying, normalized_shape, mean.underlying, rstd.underlying, weight.map(_.underlying), bias.map(_.underlying), output_mask))
  def nan_to_num(self: Tensor, nan: Option[Double] = None, posinf: Option[Double] = None, neginf: Option[Double] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nan_to_num(self.underlying, nan.asJavaDouble, posinf.asJavaDouble, neginf.asJavaDouble))
  def nan_to_num_(self: Tensor, nan: Option[Double] = None, posinf: Option[Double] = None, neginf: Option[Double] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.nan_to_num_(self.underlying, nan.asJavaDouble, posinf.asJavaDouble, neginf.asJavaDouble)
    self
  }
  def mkldnn_linear_backward_input(input_size: Array[Long], grad_output: Tensor, weight: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_linear_backward_input(input_size, grad_output.underlying, weight.underlying))
  def mkldnn_linear_backward_weights(grad_output: Tensor, input: Tensor, weight: Tensor, bias_defined: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.mkldnn_linear_backward_weights(grad_output.underlying, input.underlying, weight.underlying, bias_defined))
  def mkldnn_linear_backward(self: Tensor, grad_output: Tensor, weight: Tensor, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.mkldnn_linear_backward(self.underlying, grad_output.underlying, weight.underlying, output_mask))
  def fbgemm_linear_int8_weight_fp32_activation(input: Tensor, weight: Tensor, packed: Tensor, col_offsets: Tensor, weight_scale: Scalar, weight_zero_point: Scalar, bias: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_linear_int8_weight_fp32_activation(input.underlying, weight.underlying, packed.underlying, col_offsets.underlying, weight_scale.underlying, weight_zero_point.underlying, bias.underlying))
  def fbgemm_linear_int8_weight(input: Tensor, weight: Tensor, packed: Tensor, col_offsets: Tensor, weight_scale: Scalar, weight_zero_point: Scalar, bias: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_linear_int8_weight(input.underlying, weight.underlying, packed.underlying, col_offsets.underlying, weight_scale.underlying, weight_zero_point.underlying, bias.underlying))
  def fbgemm_linear_quantize_weight(input: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor, Double, Long) = wrapTensorTuple2DoubleLong(swig.fbgemm_linear_quantize_weight(input.underlying))
  def fbgemm_pack_gemm_matrix_fp16(input: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_pack_gemm_matrix_fp16(input.underlying))
  def fbgemm_linear_fp16_weight_fp32_activation(input: Tensor, packed_weight: Tensor, bias: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_linear_fp16_weight_fp32_activation(input.underlying, packed_weight.underlying, bias.underlying))
  def fbgemm_linear_fp16_weight(input: Tensor, packed_weight: Tensor, bias: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_linear_fp16_weight(input.underlying, packed_weight.underlying, bias.underlying))
  def fbgemm_pack_quantized_matrix(input: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_pack_quantized_matrix(input.underlying))
  def fbgemm_pack_quantized_matrix(input: Tensor, K: Long, N: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fbgemm_pack_quantized_matrix(input.underlying, K, N))
  def ldexp(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ldexp(self.underlying, other.underlying))
  def ldexp_(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.ldexp_(self.underlying, other.underlying)
    self
  }
  def linspace(start: Scalar, end: Scalar, steps: Option[Long] = None, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.linspace(start.underlying, end.underlying, steps.asJavaLong, options))
}

  def log(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.log(self.underlying))
  def log_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.log_(self.underlying)
    self
  }
  def log10(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.log10(self.underlying))
  def log10_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.log10_(self.underlying)
    self
  }
  def log1p(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.log1p(self.underlying))
  def log1p_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.log1p_(self.underlying)
    self
  }
  def log2(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.log2(self.underlying))
  def log2_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.log2_(self.underlying)
    self
  }
  def logaddexp(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logaddexp(self.underlying, other.underlying))
  def logaddexp2(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logaddexp2(self.underlying, other.underlying))
  def xlogy(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.xlogy(self.underlying, other.underlying))
  def xlogy(self: Scalar, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.xlogy(self.underlying, other.underlying))
  def xlogy(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.xlogy(self.underlying, other.underlying))
  def xlogy_(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.xlogy_(self.underlying, other.underlying)
    self
  }
  def xlogy_(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.xlogy_(self.underlying, other.underlying)
    self
  }
  def logdet(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logdet(self.underlying))
  def logspace(start: Scalar, end: Scalar, steps: Option[Long] = None, base: Double = 10.0, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.logspace(start.underlying, end.underlying, steps.asJavaLong, base, options))
}

  def log_softmax(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.log_softmax(self.underlying, dim, dtype.map(_.toScalarType)))
  def _log_softmax(self: Tensor, dim: Long, half_to_float: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._log_softmax(self.underlying, dim, half_to_float))
  def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Long, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._log_softmax_backward_data(grad_output.underlying, output.underlying, dim, self.underlying))
  def _logcumsumexp(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._logcumsumexp(self.underlying, dim))
  def logcumsumexp(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logcumsumexp(self.underlying, dim))
  def logsumexp(self: Tensor, dim: Array[Long], keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logsumexp(self.underlying, dim, keepdim))
  def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: Double = 0.0, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.margin_ranking_loss(input1.underlying, input2.underlying, target.underlying, margin, reduction))
  def matmul(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matmul(self.underlying, other.underlying))
  def matrix_rank(self: Tensor, tol: Double, symmetric: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matrix_rank(self.underlying, tol, symmetric))
  def matrix_rank(self: Tensor, symmetric: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matrix_rank(self.underlying, symmetric))
  def matrix_power(self: Tensor, n: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matrix_power(self.underlying, n))
  def matrix_exp(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matrix_exp(self.underlying))
  def matrix_exp_backward(self: Tensor, grad: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.matrix_exp_backward(self.underlying, grad.underlying))
  def _aminmax(self: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._aminmax(self.underlying))
  def _aminmax(self: Tensor, dim: Long, keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._aminmax(self.underlying, dim, keepdim))
  def aminmax(self: Tensor, dim: Option[Long] = None, keepdim: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.aminmax(self.underlying, dim.asJavaLong, keepdim))
  def _compute_linear_combination(input: Tensor, coefficients: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._compute_linear_combination(input.underlying, coefficients.underlying))
  def max(self: Tensor, dim: Long, keepdim: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.max(self.underlying, dim, keepdim))
  def value_selecting_reduction_backward(grad: Tensor, dim: Long, indices: Tensor, sizes: Array[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.value_selecting_reduction_backward(grad.underlying, dim, indices.underlying, sizes, keepdim))
  def amax(self: Tensor, dim: Array[Long] = Array(), keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.amax(self.underlying, dim, keepdim))
  def max_pool1d_with_indices(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.max_pool1d_with_indices(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def max_pool1d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.max_pool1d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def max_pool2d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.max_pool2d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def mkldnn_max_pool2d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_max_pool2d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def mkldnn_max_pool2d_backward(grad_output: Tensor, output: Tensor, input: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_max_pool2d_backward(grad_output.underlying, output.underlying, input.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def mkldnn_max_pool3d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_max_pool3d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def mkldnn_max_pool3d_backward(grad_output: Tensor, output: Tensor, input: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_max_pool3d_backward(grad_output.underlying, output.underlying, input.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def quantized_max_pool1d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_max_pool1d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def quantized_max_pool2d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_max_pool2d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def max_pool3d(self: Tensor, kernel_size: Array[Long], stride: Array[Long] = Array(), padding: Array[Long] = Array(0), dilation: Array[Long] = Array(1), ceil_mode: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.max_pool3d(self.underlying, kernel_size, stride, padding, dilation, ceil_mode))
  def mean(self: Tensor, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mean(self.underlying, dtype.map(_.toScalarType)))
  def mean(self: Tensor, dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit cg: ReferenceManager): Tensor = Tensor(swig.mean(self.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def nanmean(self: Tensor, dim: Array[Long] = Array(), keepdim: Boolean = false, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanmean(self.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def median(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.median(self.underlying))
  def median(self: Tensor, dim: Long, keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.median(self.underlying, dim, keepdim))
  def nanmedian(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanmedian(self.underlying))
  def nanmedian(self: Tensor, dim: Long, keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.nanmedian(self.underlying, dim, keepdim))
  def min(self: Tensor, dim: Long, keepdim: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.min(self.underlying, dim, keepdim))
  def amin(self: Tensor, dim: Array[Long] = Array(), keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.amin(self.underlying, dim, keepdim))
  def mkldnn_convolution(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_convolution(self.underlying, weight.underlying, bias.map(_.underlying), padding, stride, dilation, groups))
  def mkldnn_convolution_backward_input(self_size: Array[Long], grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, bias_defined: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_convolution_backward_input(self_size, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, bias_defined))
  def mkldnn_convolution_backward_weights(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, bias_defined: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.mkldnn_convolution_backward_weights(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, bias_defined))
  def mkldnn_convolution_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.mkldnn_convolution_backward(self.underlying, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, output_mask))
  def miopen_batch_norm(input: Tensor, weight: Tensor, bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], training: Boolean, exponential_average_factor: Double, epsilon: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.miopen_batch_norm(input.underlying, weight.underlying, bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), training, exponential_average_factor, epsilon))
  def miopen_batch_norm_backward(input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Option[Tensor], running_var: Option[Tensor], save_mean: Option[Tensor], save_var: Option[Tensor], epsilon: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.miopen_batch_norm_backward(input.underlying, grad_output.underlying, weight.underlying, running_mean.map(_.underlying), running_var.map(_.underlying), save_mean.map(_.underlying), save_var.map(_.underlying), epsilon))
  def miopen_convolution(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution(self.underlying, weight.underlying, bias.map(_.underlying), padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_convolution_backward_input(self_size: Array[Long], grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_backward_input(self_size, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_convolution_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.miopen_convolution_backward(self.underlying, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, output_mask))
  def miopen_convolution_backward_bias(grad_output: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_backward_bias(grad_output.underlying))
  def miopen_convolution_backward_weight(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_backward_weight(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_convolution_transpose(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_transpose(self.underlying, weight.underlying, bias.map(_.underlying), padding, output_padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_convolution_transpose_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], output_padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.miopen_convolution_transpose_backward(self.underlying, grad_output.underlying, weight.underlying, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask))
  def miopen_convolution_transpose_backward_input(grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_transpose_backward_input(grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_convolution_transpose_backward_weight(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_convolution_transpose_backward_weight(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_depthwise_convolution(self: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_depthwise_convolution(self.underlying, weight.underlying, bias.map(_.underlying), padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_depthwise_convolution_backward_input(self_size: Array[Long], grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_depthwise_convolution_backward_input(self_size, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_depthwise_convolution_backward(self: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.miopen_depthwise_convolution_backward(self.underlying, grad_output.underlying, weight.underlying, padding, stride, dilation, groups, benchmark, deterministic, output_mask))
  def miopen_depthwise_convolution_backward_weight(weight_size: Array[Long], grad_output: Tensor, self: Tensor, padding: Array[Long], stride: Array[Long], dilation: Array[Long], groups: Long, benchmark: Boolean, deterministic: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.miopen_depthwise_convolution_backward_weight(weight_size, grad_output.underlying, self.underlying, padding, stride, dilation, groups, benchmark, deterministic))
  def miopen_rnn(input: Tensor, weight: Array[Tensor], weight_stride0: Long, hx: Tensor, cx: Option[Tensor], mode: Long, hidden_size: Long, num_layers: Long, batch_first: Boolean, dropout: Double, train: Boolean, bidirectional: Boolean, batch_sizes: Array[Long], dropout_state: Option[Tensor])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig.miopen_rnn(input.underlying, weight.map(_.underlyingChecked), weight_stride0, hx.underlying, cx.map(_.underlying), mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state.map(_.underlying)))
  def mm(self: Tensor, mat2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mm(self.underlying, mat2.underlying))
  def _sparse_mm(sparse: Tensor, dense: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_mm(sparse.underlying, dense.underlying))
  def _sparse_sparse_matmul(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sparse_matmul(self.underlying, other.underlying))
  def _sparse_mask_helper(t: Tensor, mask_indices: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_mask_helper(t.underlying, mask_indices.underlying))
  def mode(self: Tensor, dim: Long = -1, keepdim: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.mode(self.underlying, dim, keepdim))
  def mul(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mul(self.underlying, other.underlying))
  def mul(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mul(self.underlying, other.underlying))
  def multiply(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.multiply(self.underlying, other.underlying))
  def multiply(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.multiply(self.underlying, other.underlying))
  def mv(self: Tensor, vec: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mv(self.underlying, vec.underlying))
  def mvlgamma(self: Tensor, p: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mvlgamma(self.underlying, p))
  def narrow_copy(self: Tensor, dim: Long, start: Long, length: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.narrow_copy(self.underlying, dim, start, length))
  def narrow(self: Tensor, dim: Long, start: Long, length: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.narrow(self.underlying, dim, start, length))
  def narrow(self: Tensor, dim: Long, start: Tensor, length: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.narrow(self.underlying, dim, start.underlying, length))
  def native_batch_norm(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], training: Boolean, momentum: Double, eps: Double)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_batch_norm(input.underlying, weight.map(_.underlying), bias.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), training, momentum, eps))
  def batch_norm_stats(input: Tensor, eps: Double)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.batch_norm_stats(input.underlying, eps))
  def batch_norm_elemt(input: Tensor, weight: Option[Tensor], bias: Option[Tensor], mean: Tensor, invstd: Tensor, eps: Double)(implicit cg: ReferenceManager): Tensor = Tensor(swig.batch_norm_elemt(input.underlying, weight.map(_.underlying), bias.map(_.underlying), mean.underlying, invstd.underlying, eps))
  def batch_norm_gather_stats(input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Option[Tensor], running_var: Option[Tensor], momentum: Double, eps: Double, count: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.batch_norm_gather_stats(input.underlying, mean.underlying, invstd.underlying, running_mean.map(_.underlying), running_var.map(_.underlying), momentum, eps, count))
  def batch_norm_gather_stats_with_counts(input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Option[Tensor], running_var: Option[Tensor], momentum: Double, eps: Double, counts: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.batch_norm_gather_stats_with_counts(input.underlying, mean.underlying, invstd.underlying, running_mean.map(_.underlying), running_var.map(_.underlying), momentum, eps, counts.underlying))
  def native_batch_norm_backward(grad_out: Tensor, input: Tensor, weight: Option[Tensor], running_mean: Option[Tensor], running_var: Option[Tensor], save_mean: Option[Tensor], save_invstd: Option[Tensor], train: Boolean, eps: Double, output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.native_batch_norm_backward(grad_out.underlying, input.underlying, weight.map(_.underlying), running_mean.map(_.underlying), running_var.map(_.underlying), save_mean.map(_.underlying), save_invstd.map(_.underlying), train, eps, output_mask))
  def batch_norm_backward_reduce(grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Option[Tensor], input_g: Boolean, weight_g: Boolean, bias_g: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple4(swig.batch_norm_backward_reduce(grad_out.underlying, input.underlying, mean.underlying, invstd.underlying, weight.map(_.underlying), input_g, weight_g, bias_g))
  def batch_norm_backward_elemt(grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Option[Tensor], mean_dy: Tensor, mean_dy_xmu: Tensor, count: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.batch_norm_backward_elemt(grad_out.underlying, input.underlying, mean.underlying, invstd.underlying, weight.map(_.underlying), mean_dy.underlying, mean_dy_xmu.underlying, count.underlying))
  def batch_norm_update_stats(input: Tensor, running_mean: Option[Tensor], running_var: Option[Tensor], momentum: Double)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.batch_norm_update_stats(input.underlying, running_mean.map(_.underlying), running_var.map(_.underlying), momentum))
  def is_vulkan_available()(implicit cg: ReferenceManager): Boolean = swig.is_vulkan_available()
  def _nnpack_available()(implicit cg: ReferenceManager): Boolean = swig._nnpack_available()
  def _nnpack_spatial_convolution(input: Tensor, weight: Tensor, bias: Option[Tensor], padding: Array[Long], stride: Array[Long] = Array(1))(implicit cg: ReferenceManager): Tensor = Tensor(swig._nnpack_spatial_convolution(input.underlying, weight.underlying, bias.map(_.underlying), padding, stride))
  def _nnpack_spatial_convolution_backward(input: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long], output_mask: Array[Boolean])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._nnpack_spatial_convolution_backward(input.underlying, grad_output.underlying, weight.underlying, padding, output_mask))
  def _nnpack_spatial_convolution_backward_input(input: Tensor, grad_output: Tensor, weight: Tensor, padding: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._nnpack_spatial_convolution_backward_input(input.underlying, grad_output.underlying, weight.underlying, padding))
  def _nnpack_spatial_convolution_backward_weight(input: Tensor, weightsize: Array[Long], grad_output: Tensor, padding: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._nnpack_spatial_convolution_backward_weight(input.underlying, weightsize, grad_output.underlying, padding))
  def ones(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.ones(size, options))
}

  def ones_like(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.ones_like(self.underlying, options, memory_format))
}

  def pairwise_distance(x1: Tensor, x2: Tensor, p: Double = 2, eps: Double = 1e-06, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pairwise_distance(x1.underlying, x2.underlying, p, eps, keepdim))
  def cdist(x1: Tensor, x2: Tensor, p: Double = 2, compute_mode: Option[Long] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cdist(x1.underlying, x2.underlying, p, compute_mode.asJavaLong))
  def _euclidean_dist(x1: Tensor, x2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._euclidean_dist(x1.underlying, x2.underlying))
  def _cdist_forward(x1: Tensor, x2: Tensor, p: Double, compute_mode: Option[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._cdist_forward(x1.underlying, x2.underlying, p, compute_mode.asJavaLong))
  def _cdist_backward(grad: Tensor, x1: Tensor, x2: Tensor, p: Double, cdist: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cdist_backward(grad.underlying, x1.underlying, x2.underlying, p, cdist.underlying))
  def pdist(self: Tensor, p: Double = 2)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pdist(self.underlying, p))
  def _pdist_forward(self: Tensor, p: Double = 2)(implicit cg: ReferenceManager): Tensor = Tensor(swig._pdist_forward(self.underlying, p))
  def _pdist_backward(grad: Tensor, self: Tensor, p: Double, pdist: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._pdist_backward(grad.underlying, self.underlying, p, pdist.underlying))
  def cosine_similarity(x1: Tensor, x2: Tensor, dim: Long = 1, eps: Double = 1e-08)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cosine_similarity(x1.underlying, x2.underlying, dim, eps))
  def permute(self: Tensor, dims: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.permute(self.underlying, dims))
  def movedim(self: Tensor, source: Array[Long], destination: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.movedim(self.underlying, source, destination))
  def movedim(self: Tensor, source: Long, destination: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.movedim(self.underlying, source, destination))
  def moveaxis(self: Tensor, source: Array[Long], destination: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.moveaxis(self.underlying, source, destination))
  def moveaxis(self: Tensor, source: Long, destination: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.moveaxis(self.underlying, source, destination))
  def pixel_shuffle(self: Tensor, upscale_factor: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pixel_shuffle(self.underlying, upscale_factor))
  def pixel_unshuffle(self: Tensor, downscale_factor: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pixel_unshuffle(self.underlying, downscale_factor))
  def channel_shuffle(self: Tensor, groups: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.channel_shuffle(self.underlying, groups))
  def _pin_memory(self: Tensor, device: Option[Device] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig._pin_memory(self.underlying, device.map(_.underlying)))
  def pinverse(self: Tensor, rcond: Double = 1e-15)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pinverse(self.underlying, rcond))
  def poisson_nll_loss(input: Tensor, target: Tensor, log_input: Boolean, full: Boolean, eps: Double, reduction: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.poisson_nll_loss(input.underlying, target.underlying, log_input, full, eps, reduction))
  def rad2deg(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rad2deg(self.underlying))
  def rad2deg_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.rad2deg_(self.underlying)
    self
  }
  def deg2rad(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.deg2rad(self.underlying))
  def deg2rad_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.deg2rad_(self.underlying)
    self
  }
  def scalar_tensor(s: Scalar, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.scalar_tensor(s.underlying, options))
}

  def rand(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.rand(size, options))
}

  def rand(size: Array[Long], generator: Option[Generator], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.rand(size, generator.map(_.underlying), options))
}

  def rand_like(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.rand_like(self.underlying, options, memory_format))
}

  def randint(high: Long, size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint(high, size, options))
}

  def randint(high: Long, size: Array[Long], generator: Option[Generator], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint(high, size, generator.map(_.underlying), options))
}

  def randint(low: Long, high: Long, size: Array[Long], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint(low, high, size, options))
}

  def randint(low: Long, high: Long, size: Array[Long], generator: Option[Generator], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint(low, high, size, generator.map(_.underlying), options))
}

  def randint_like(self: Tensor, high: Long, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint_like(self.underlying, high, options, memory_format))
}

  def randint_like(self: Tensor, low: Long, high: Long, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean], memory_format: Option[MemoryFormat])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randint_like(self.underlying, low, high, options, memory_format))
}

  def randn(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randn(size, options))
}

  def randn(size: Array[Long], generator: Option[Generator], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randn(size, generator.map(_.underlying), options))
}

  def randn_like(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randn_like(self.underlying, options, memory_format))
}

  def randperm(n: Long, dtype: Option[dtype] = Option(int64), layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randperm(n, options))
}

  def randperm(n: Long, generator: Option[Generator], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.randperm(n, generator.map(_.underlying), options))
}

  def range(start: Scalar, end: Scalar, step: Double = 1, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.range(start.underlying, end.underlying, step.toInternalScalar, options))
}

  def ravel(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ravel(self.underlying))
  def reciprocal(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.reciprocal(self.underlying))
  def reciprocal_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.reciprocal_(self.underlying)
    self
  }
  def neg(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.neg(self.underlying))
  def neg_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.neg_(self.underlying)
    self
  }
  def negative(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.negative(self.underlying))
  def negative_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.negative_(self.underlying)
    self
  }
  def repeat_interleave(repeats: Tensor, output_size: Option[Long] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.repeat_interleave(repeats.underlying, output_size.asJavaLong))
  def repeat_interleave(self: Tensor, repeats: Tensor, dim: Option[Long], output_size: Option[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.repeat_interleave(self.underlying, repeats.underlying, dim.asJavaLong, output_size.asJavaLong))
  def repeat_interleave(self: Tensor, repeats: Long, dim: Option[Long], output_size: Option[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.repeat_interleave(self.underlying, repeats, dim.asJavaLong, output_size.asJavaLong))
  def reshape(self: Tensor, shape: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.reshape(self.underlying, shape))
  def _reshape_alias(self: Tensor, size: Array[Long], stride: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._reshape_alias(self.underlying, size, stride))
  def _mkldnn_reshape(self: Tensor, shape: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._mkldnn_reshape(self.underlying, shape))
  def round(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.round(self.underlying))
  def round_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.round_(self.underlying)
    self
  }
  def rrelu(self: Tensor, lower: Double = 0.125, upper: Double = 0.3333333333333333, training: Boolean = false, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rrelu(self.underlying, lower.toInternalScalar, upper.toInternalScalar, training, generator.map(_.underlying)))
  def rrelu_(self: Tensor, lower: Double = 0.125, upper: Double = 0.3333333333333333, training: Boolean = false, generator: Option[Generator] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.rrelu_(self.underlying, lower.toInternalScalar, upper.toInternalScalar, training, generator.map(_.underlying))
    self
  }
  def relu(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.relu(self.underlying))
  def relu_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.relu_(self.underlying)
    self
  }
  def prelu(self: Tensor, weight: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.prelu(self.underlying, weight.underlying))
  def prelu_backward(grad_output: Tensor, self: Tensor, weight: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.prelu_backward(grad_output.underlying, self.underlying, weight.underlying))
  def hardshrink(self: Tensor, lambd: Double = 0.5)(implicit cg: ReferenceManager): Tensor = Tensor(swig.hardshrink(self.underlying, lambd.toInternalScalar))
  def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.hardshrink_backward(grad_out.underlying, self.underlying, lambd.underlying))
  def rsqrt(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rsqrt(self.underlying))
  def rsqrt_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.rsqrt_(self.underlying)
    self
  }
  def select(self: Tensor, dim: Long, index: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.select(self.underlying, dim, index))
  def select_backward(grad_output: Tensor, input_sizes: Array[Long], dim: Long, index: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.select_backward(grad_output.underlying, input_sizes, dim, index))
  def selu(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.selu(self.underlying))
  def selu_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.selu_(self.underlying)
    self
  }
  def celu(self: Tensor, alpha: Double = 1.0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.celu(self.underlying, alpha.toInternalScalar))
  def celu_(self: Tensor, alpha: Double = 1.0)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.celu_(self.underlying, alpha.toInternalScalar)
    self
  }
  def sigmoid(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sigmoid(self.underlying))
  def sigmoid_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.sigmoid_(self.underlying)
    self
  }
  def logit(self: Tensor, eps: Option[Double] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.logit(self.underlying, eps.asJavaDouble))
  def logit_(self: Tensor, eps: Option[Double] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.logit_(self.underlying, eps.asJavaDouble)
    self
  }
  def sin(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sin(self.underlying))
  def sin_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.sin_(self.underlying)
    self
  }
  def sinc(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sinc(self.underlying))
  def sinc_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.sinc_(self.underlying)
    self
  }
  def sinh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sinh(self.underlying))
  def sinh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.sinh_(self.underlying)
    self
  }
  def detach(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.detach(self.underlying))
  def detach_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.detach_(self.underlying)
    self
  }
  def size(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Long = swig.size(self.underlying, dim)
  def slice(self: Tensor, dim: Long = 0, start: Option[Long] = None, end: Option[Long] = None, step: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.slice(self.underlying, dim, start.asJavaLong, end.asJavaLong, step))
  def slice_backward(grad_output: Tensor, input_sizes: Array[Long], dim: Long, start: Long, end: Long, step: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.slice_backward(grad_output.underlying, input_sizes, dim, start, end, step))
  def slogdet(self: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.slogdet(self.underlying))
  def smm(self: Tensor, mat2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.smm(self.underlying, mat2.underlying))
  def softmax(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.softmax(self.underlying, dim, dtype.map(_.toScalarType)))
  def _softmax(self: Tensor, dim: Long, half_to_float: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._softmax(self.underlying, dim, half_to_float))
  def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Long, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._softmax_backward_data(grad_output.underlying, output.underlying, dim, self.underlying))
  def unsafe_split(self: Tensor, split_size: Long, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.unsafe_split(self.underlying, split_size, dim))
  def split(self: Tensor, split_size: Long, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.split(self.underlying, split_size, dim))
  def unsafe_split_with_sizes(self: Tensor, split_sizes: Array[Long], dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.unsafe_split_with_sizes(self.underlying, split_sizes, dim))
  def split_with_sizes(self: Tensor, split_sizes: Array[Long], dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.split_with_sizes(self.underlying, split_sizes, dim))
  def hsplit(self: Tensor, sections: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.hsplit(self.underlying, sections))
  def hsplit(self: Tensor, indices: Array[Long])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.hsplit(self.underlying, indices))
  def vsplit(self: Tensor, sections: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.vsplit(self.underlying, sections))
  def vsplit(self: Tensor, indices: Array[Long])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.vsplit(self.underlying, indices))
  def dsplit(self: Tensor, sections: Long)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.dsplit(self.underlying, sections))
  def dsplit(self: Tensor, indices: Array[Long])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.dsplit(self.underlying, indices))
  def squeeze(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.squeeze(self.underlying))
  def squeeze(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.squeeze(self.underlying, dim))
  def sspaddmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sspaddmm(self.underlying, mat1.underlying, mat2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def stack(tensors: Array[Tensor], dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.stack(tensors.map(_.underlyingChecked), dim))
  def _stack(tensors: Array[Tensor], dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._stack(tensors.map(_.underlyingChecked), dim))
  def hstack(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.hstack(tensors.map(_.underlyingChecked)))
  def vstack(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.vstack(tensors.map(_.underlyingChecked)))
  def dstack(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.dstack(tensors.map(_.underlyingChecked)))
  def stft(self: Tensor, n_fft: Long, hop_length: Option[Long] = None, win_length: Option[Long] = None, window: Option[Tensor] = None, normalized: Boolean = false, onesided: Option[Boolean] = None, return_complex: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.stft(self.underlying, n_fft, hop_length.asJavaLong, win_length.asJavaLong, window.map(_.underlying), normalized, onesided.map(java.lang.Boolean.valueOf), return_complex.map(java.lang.Boolean.valueOf)))
  def istft(self: Tensor, n_fft: Long, hop_length: Option[Long] = None, win_length: Option[Long] = None, window: Option[Tensor] = None, center: Boolean = true, normalized: Boolean = false, onesided: Option[Boolean] = None, length: Option[Long] = None, return_complex: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.istft(self.underlying, n_fft, hop_length.asJavaLong, win_length.asJavaLong, window.map(_.underlying), center, normalized, onesided.map(java.lang.Boolean.valueOf), length.asJavaLong, return_complex))
  def stride(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Long = swig.stride(self.underlying, dim)
  def sum(self: Tensor, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sum(self.underlying, dtype.map(_.toScalarType)))
  def sum(self: Tensor, dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit cg: ReferenceManager): Tensor = Tensor(swig.sum(self.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def nansum(self: Tensor, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nansum(self.underlying, dtype.map(_.toScalarType)))
  def nansum(self: Tensor, dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit cg: ReferenceManager): Tensor = Tensor(swig.nansum(self.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def sqrt(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sqrt(self.underlying))
  def sqrt_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.sqrt_(self.underlying)
    self
  }
  def square(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.square(self.underlying))
  def square_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.square_(self.underlying)
    self
  }
  def std(self: Tensor, unbiased: Boolean = true)(implicit cg: ReferenceManager): Tensor = Tensor(swig.std(self.underlying, unbiased))
  def std(self: Tensor, dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.std(self.underlying, dim, unbiased, keepdim))
  def std(self: Tensor, dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.std(self.underlying, dim, correction.asJavaLong, keepdim))
  def std_mean(self: Tensor, unbiased: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.std_mean(self.underlying, unbiased))
  def std_mean(self: Tensor, dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.std_mean(self.underlying, dim, unbiased, keepdim))
  def std_mean(self: Tensor, dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.std_mean(self.underlying, dim, correction.asJavaLong, keepdim))
  def prod(self: Tensor, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.prod(self.underlying, dtype.map(_.toScalarType)))
  def prod(self: Tensor, dim: Long, keepdim: Boolean, dtype: Option[dtype])(implicit cg: ReferenceManager): Tensor = Tensor(swig.prod(self.underlying, dim, keepdim, dtype.map(_.toScalarType)))
  def t(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.t(self.underlying))
  def tan(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.tan(self.underlying))
  def tan_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.tan_(self.underlying)
    self
  }
  def tanh(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.tanh(self.underlying))
  def tanh_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.tanh_(self.underlying)
    self
  }
  def tensordot(self: Tensor, other: Tensor, dims_self: Array[Long], dims_other: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.tensordot(self.underlying, other.underlying, dims_self, dims_other))
  def threshold(self: Tensor, threshold: Scalar, value: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.threshold(self.underlying, threshold.underlying, value.underlying))
  def threshold_(self: Tensor, threshold: Scalar, value: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.threshold_(self.underlying, threshold.underlying, value.underlying)
    self
  }
  def threshold_backward(grad_output: Tensor, self: Tensor, threshold: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.threshold_backward(grad_output.underlying, self.underlying, threshold.underlying))
  def tile(self: Tensor, dims: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.tile(self.underlying, dims))
  def transpose(self: Tensor, dim0: Long, dim1: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.transpose(self.underlying, dim0, dim1))
  def _mkldnn_transpose(self: Tensor, dim0: Long, dim1: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._mkldnn_transpose(self.underlying, dim0, dim1))
  def _mkldnn_transpose_(self: Tensor, dim0: Long, dim1: Long)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._mkldnn_transpose_(self.underlying, dim0, dim1)
    self
  }
  def flip(self: Tensor, dims: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.flip(self.underlying, dims))
  def fliplr(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fliplr(self.underlying))
  def flipud(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.flipud(self.underlying))
  def roll(self: Tensor, shifts: Array[Long], dims: Array[Long] = Array())(implicit cg: ReferenceManager): Tensor = Tensor(swig.roll(self.underlying, shifts, dims))
  def rot90(self: Tensor, k: Long = 1, dims: Array[Long] = Array(0,1))(implicit cg: ReferenceManager): Tensor = Tensor(swig.rot90(self.underlying, k, dims))
  def trapezoid(y: Tensor, x: Tensor, dim: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trapezoid(y.underlying, x.underlying, dim))
  def trapezoid(y: Tensor, dx: Scalar, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trapezoid(y.underlying, dx.underlying, dim))
  def trapz(y: Tensor, x: Tensor, dim: Long = -1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trapz(y.underlying, x.underlying, dim))
  def trapz(y: Tensor, dx: Double, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trapz(y.underlying, dx, dim))
  def _trilinear(i1: Tensor, i2: Tensor, i3: Tensor, expand1: Array[Long], expand2: Array[Long], expand3: Array[Long], sumdim: Array[Long], unroll_dim: Long = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._trilinear(i1.underlying, i2.underlying, i3.underlying, expand1, expand2, expand3, sumdim, unroll_dim))
  def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: Double = 1.0, p: Double = 2, eps: Double = 1e-06, swap: Boolean = false, reduction: Long = internal.Reduction.Mean.swigValue())(implicit cg: ReferenceManager): Tensor = Tensor(swig.triplet_margin_loss(anchor.underlying, positive.underlying, negative.underlying, margin, p, eps, swap, reduction))
  def trunc(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trunc(self.underlying))
  def trunc_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.trunc_(self.underlying)
    self
  }
  def fix(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fix(self.underlying))
  def fix_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.fix_(self.underlying)
    self
  }
  def _has_compatible_shallow_copy_type(self: Tensor, from: Tensor)(implicit cg: ReferenceManager): Boolean = swig._has_compatible_shallow_copy_type(self.underlying, from.underlying)
  def _unique(self: Tensor, sorted: Boolean = true, return_inverse: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._unique(self.underlying, sorted, return_inverse))
  def unique_dim(self: Tensor, dim: Long, sorted: Boolean = true, return_inverse: Boolean = false, return_counts: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.unique_dim(self.underlying, dim, sorted, return_inverse, return_counts))
  def unique_consecutive(self: Tensor, return_inverse: Boolean = false, return_counts: Boolean = false, dim: Option[Long] = None)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.unique_consecutive(self.underlying, return_inverse, return_counts, dim.asJavaLong))
  def unique_dim_consecutive(self: Tensor, dim: Long, return_inverse: Boolean = false, return_counts: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.unique_dim_consecutive(self.underlying, dim, return_inverse, return_counts))
  def _unique2(self: Tensor, sorted: Boolean = true, return_inverse: Boolean = false, return_counts: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._unique2(self.underlying, sorted, return_inverse, return_counts))
  def _unsafe_view(self: Tensor, size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._unsafe_view(self.underlying, size))
  def unsqueeze(self: Tensor, dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.unsqueeze(self.underlying, dim))
  def vander(x: Tensor, N: Option[Long] = None, increasing: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.vander(x.underlying, N.asJavaLong, increasing))
  def `var`(self: Tensor, unbiased: Boolean = true)(implicit cg: ReferenceManager): Tensor = Tensor(swig.`var`(self.underlying, unbiased))
  def `var`(self: Tensor, dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.`var`(self.underlying, dim, unbiased, keepdim))
  def `var`(self: Tensor, dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.`var`(self.underlying, dim, correction.asJavaLong, keepdim))
  def var_mean(self: Tensor, unbiased: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.var_mean(self.underlying, unbiased))
  def var_mean(self: Tensor, dim: Array[Long], unbiased: Boolean, keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.var_mean(self.underlying, dim, unbiased, keepdim))
  def var_mean(self: Tensor, dim: Option[Array[Long]], correction: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.var_mean(self.underlying, dim, correction.asJavaLong, keepdim))
  def where(condition: Tensor, self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.where(condition.underlying, self.underlying, other.underlying))
  def where(condition: Tensor, self: Scalar, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.where(condition.underlying, self.underlying, other.underlying))
  def where(condition: Tensor, self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.where(condition.underlying, self.underlying, other.underlying))
  def where(condition: Tensor, self: Scalar, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.where(condition.underlying, self.underlying, other.underlying))
  def where(condition: Tensor)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.where(condition.underlying))
  def _s_where(condition: Tensor, self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._s_where(condition.underlying, self.underlying, other.underlying))
  def norm_except_dim(v: Tensor, pow: Long = 2, dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.norm_except_dim(v.underlying, pow, dim))
  def _weight_norm(v: Tensor, g: Tensor, dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._weight_norm(v.underlying, g.underlying, dim))
  def _weight_norm_cuda_interface(v: Tensor, g: Tensor, dim: Long = 0)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._weight_norm_cuda_interface(v.underlying, g.underlying, dim))
  def _weight_norm_cuda_interface_backward(grad_w: Tensor, saved_v: Tensor, saved_g: Tensor, saved_norms: Tensor, dim: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._weight_norm_cuda_interface_backward(grad_w.underlying, saved_v.underlying, saved_g.underlying, saved_norms.underlying, dim))
  def _weight_norm_differentiable_backward(grad_w: Tensor, saved_v: Tensor, saved_g: Tensor, saved_norms: Tensor, dim: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._weight_norm_differentiable_backward(grad_w.underlying, saved_v.underlying, saved_g.underlying, saved_norms.underlying, dim))
  def zeros(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.zeros(size, options))
}

  def zeros_like(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.zeros_like(self.underlying, options, memory_format))
}

  def _standard_gamma_grad(self: Tensor, output: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._standard_gamma_grad(self.underlying, output.underlying))
  def _standard_gamma(self: Tensor, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig._standard_gamma(self.underlying, generator.map(_.underlying)))
  def _dirichlet_grad(x: Tensor, alpha: Tensor, total: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._dirichlet_grad(x.underlying, alpha.underlying, total.underlying))
  def _sample_dirichlet(self: Tensor, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sample_dirichlet(self.underlying, generator.map(_.underlying)))
  def poisson(self: Tensor, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.poisson(self.underlying, generator.map(_.underlying)))
  def binomial(count: Tensor, prob: Tensor, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.binomial(count.underlying, prob.underlying, generator.map(_.underlying)))
  def native_norm(self: Tensor, p: Double = 2)(implicit cg: ReferenceManager): Tensor = Tensor(swig.native_norm(self.underlying, p.toInternalScalar))
  def native_norm(self: Tensor, p: Option[Scalar], dim: Array[Long], keepdim: Boolean, dtype: Option[dtype])(implicit cg: ReferenceManager): Tensor = Tensor(swig.native_norm(self.underlying, p.map(_.underlying), dim, keepdim, dtype.map(_.toScalarType)))
  def _sparse_sum(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sum(self.underlying))
  def _sparse_sum(self: Tensor, dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sum(self.underlying, dtype.toScalarType))
  def _sparse_sum(self: Tensor, dim: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sum(self.underlying, dim))
  def _sparse_sum(self: Tensor, dim: Array[Long], dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sum(self.underlying, dim, dtype.toScalarType))
  def _sparse_sum_backward(grad: Tensor, self: Tensor, dim: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_sum_backward(grad.underlying, self.underlying, dim))
  def _sparse_softmax(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_softmax(self.underlying, dim, dtype.map(_.toScalarType)))
  def _sparse_softmax(self: Tensor, dim: Long, half_to_float: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_softmax(self.underlying, dim, half_to_float))
  def _sparse_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Long, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_softmax_backward_data(grad_output.underlying, output.underlying, dim, self.underlying))
  def _sparse_log_softmax(self: Tensor, dim: Long, dtype: Option[dtype] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_log_softmax(self.underlying, dim, dtype.map(_.toScalarType)))
  def _sparse_log_softmax(self: Tensor, dim: Long, half_to_float: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_log_softmax(self.underlying, dim, half_to_float))
  def _sparse_log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: Long, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_log_softmax_backward_data(grad_output.underlying, output.underlying, dim, self.underlying))
  def norm(self: Tensor, p: Option[Scalar], dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig.norm(self.underlying, p.map(_.underlying), dtype.toScalarType))
  def norm(self: Tensor, p: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.norm(self.underlying, p.underlying))
  def norm(self: Tensor, p: Option[Scalar], dim: Array[Long], keepdim: Boolean, dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig.norm(self.underlying, p.map(_.underlying), dim, keepdim, dtype.toScalarType))
  def norm(self: Tensor, p: Option[Scalar], dim: Array[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.norm(self.underlying, p.map(_.underlying), dim, keepdim))
  def frexp(self: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.frexp(self.underlying))
  def frobenius_norm(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.frobenius_norm(self.underlying))
  def frobenius_norm(self: Tensor, dim: Array[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.frobenius_norm(self.underlying, dim, keepdim))
  def nuclear_norm(self: Tensor, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nuclear_norm(self.underlying, keepdim))
  def nuclear_norm(self: Tensor, dim: Array[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nuclear_norm(self.underlying, dim, keepdim))
  def clone(self: Tensor, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.clone(self.underlying, memory_format))
  def positive(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.positive(self.underlying))
  def resize_as_(self: Tensor, the_template: Tensor, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.resize_as_(self.underlying, the_template.underlying, memory_format)
    self
  }
  def resize_as_sparse_(self: Tensor, the_template: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.resize_as_sparse_(self.underlying, the_template.underlying)
    self
  }
  def zero_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.zero_(self.underlying)
    self
  }
  def sub(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sub(self.underlying, other.underlying, alpha.toInternalScalar))
  def sub(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sub(self.underlying, other.underlying, alpha.underlying))
  def subtract(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.subtract(self.underlying, other.underlying, alpha.toInternalScalar))
  def subtract(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.subtract(self.underlying, other.underlying, alpha.underlying))
  def rsub(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rsub(self.underlying, other.underlying, alpha.toInternalScalar))
  def heaviside(self: Tensor, values: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.heaviside(self.underlying, values.underlying))
  def rsub(self: Tensor, other: Scalar, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rsub(self.underlying, other.underlying, alpha.underlying))
  def _sparse_addmm(self: Tensor, sparse: Tensor, dense: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._sparse_addmm(self.underlying, sparse.underlying, dense.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addmm(self.underlying, mat1.underlying, mat2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def sparse_csr_tensor(crow_indices: Tensor, col_indices: Tensor, values: Tensor, size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = Option(false))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.sparse_csr_tensor(crow_indices.underlying, col_indices.underlying, values.underlying, size, options))
}

  def sparse_csr_tensor(crow_indices: Tensor, col_indices: Tensor, values: Tensor, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.sparse_csr_tensor(crow_indices.underlying, col_indices.underlying, values.underlying, options))
}

  def _sparse_csr_tensor_unsafe(crow_indices: Tensor, col_indices: Tensor, values: Tensor, size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._sparse_csr_tensor_unsafe(crow_indices.underlying, col_indices.underlying, values.underlying, size, options))
}

  def sparse_coo_tensor(size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = Option(false))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.sparse_coo_tensor(size, options))
}

  def sparse_coo_tensor(indices: Tensor, values: Tensor, dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.sparse_coo_tensor(indices.underlying, values.underlying, options))
}

  def sparse_coo_tensor(indices: Tensor, values: Tensor, size: Array[Long], dtype: Option[dtype], layout: Option[Layout], device: Option[Device], pin_memory: Option[Boolean])(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.sparse_coo_tensor(indices.underlying, values.underlying, size, options))
}

  def _sparse_coo_tensor_unsafe(indices: Tensor, values: Tensor, size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._sparse_coo_tensor_unsafe(indices.underlying, values.underlying, size, options))
}

  def _validate_sparse_coo_tensor_args(indices: Tensor, values: Tensor, size: Array[Long])(implicit cg: ReferenceManager): Unit = swig._validate_sparse_coo_tensor_args(indices.underlying, values.underlying, size)
  def _validate_sparse_csr_tensor_args(crow_indices: Tensor, col_indices: Tensor, values: Tensor, size: Array[Long])(implicit cg: ReferenceManager): Unit = swig._validate_sparse_csr_tensor_args(crow_indices.underlying, col_indices.underlying, values.underlying, size)
  def _sparse_coo_tensor_with_dims(sparse_dim: Long, dense_dim: Long, size: Array[Long], dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = Option(false))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, options))
}

  def _sparse_coo_tensor_with_dims_and_tensors(sparse_dim: Long, dense_dim: Long, size: Array[Long], indices: Tensor, values: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = Option(false))(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices.underlying, values.underlying, options))
}

  def _to_cpu(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._to_cpu(tensors.map(_.underlyingChecked)))
  def to_dense_backward(grad: Tensor, input: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.to_dense_backward(grad.underlying, input.underlying))
  def _coalesce(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._coalesce(self.underlying))
  def hspmm(mat1: Tensor, mat2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.hspmm(mat1.underlying, mat2.underlying))
  def copy_sparse_to_sparse_(self: Tensor, src: Tensor, non_blocking: Boolean = false)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.copy_sparse_to_sparse_(self.underlying, src.underlying, non_blocking)
    self
  }
  def unbind(self: Tensor, dim: Long = 0)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.unbind(self.underlying, dim))
  def to_mkldnn_backward(grad: Tensor, input: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.to_mkldnn_backward(grad.underlying, input.underlying))
  def quantize_per_tensor(self: Tensor, scale: Double, zero_point: Long, dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantize_per_tensor(self.underlying, scale, zero_point, dtype.toScalarType))
  def quantize_per_tensor(self: Tensor, scale: Tensor, zero_point: Tensor, dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantize_per_tensor(self.underlying, scale.underlying, zero_point.underlying, dtype.toScalarType))
  def quantize_per_tensor(tensors: Array[Tensor], scales: Tensor, zero_points: Tensor, dtype: dtype)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.quantize_per_tensor(tensors.map(_.underlyingChecked), scales.underlying, zero_points.underlying, dtype.toScalarType))
  def quantize_per_channel(self: Tensor, scales: Tensor, zero_points: Tensor, axis: Long, dtype: dtype)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantize_per_channel(self.underlying, scales.underlying, zero_points.underlying, axis, dtype.toScalarType))
  def dequantize(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.dequantize(self.underlying))
  def dequantize(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.dequantize(tensors.map(_.underlyingChecked)))
  def q_scale(self: Tensor)(implicit cg: ReferenceManager): Double = swig.q_scale(self.underlying)
  def q_zero_point(self: Tensor)(implicit cg: ReferenceManager): Long = swig.q_zero_point(self.underlying)
  def q_per_channel_scales(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.q_per_channel_scales(self.underlying))
  def q_per_channel_zero_points(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.q_per_channel_zero_points(self.underlying))
  def q_per_channel_axis(self: Tensor)(implicit cg: ReferenceManager): Long = swig.q_per_channel_axis(self.underlying)
  def int_repr(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.int_repr(self.underlying))
  def _make_per_tensor_quantized_tensor(self: Tensor, scale: Double, zero_point: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._make_per_tensor_quantized_tensor(self.underlying, scale, zero_point))
  def _make_per_channel_quantized_tensor(self: Tensor, scale: Tensor, zero_point: Tensor, axis: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._make_per_channel_quantized_tensor(self.underlying, scale.underlying, zero_point.underlying, axis))
  def fake_quantize_per_tensor_affine(self: Tensor, scale: Double, zero_point: Long, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fake_quantize_per_tensor_affine(self.underlying, scale, zero_point, quant_min, quant_max))
  def fake_quantize_per_tensor_affine(self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fake_quantize_per_tensor_affine(self.underlying, scale.underlying, zero_point.underlying, quant_min, quant_max))
  def fake_quantize_per_tensor_affine_cachemask(self: Tensor, scale: Double, zero_point: Long, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.fake_quantize_per_tensor_affine_cachemask(self.underlying, scale, zero_point, quant_min, quant_max))
  def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self: Tensor, scale: Tensor, zero_point: Tensor, fake_quant_enabled: Tensor, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self.underlying, scale.underlying, zero_point.underlying, fake_quant_enabled.underlying, quant_min, quant_max))
  def fake_quantize_per_tensor_affine_cachemask_backward(grad: Tensor, mask: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fake_quantize_per_tensor_affine_cachemask_backward(grad.underlying, mask.underlying))
  def _fake_quantize_learnable_per_tensor_affine(self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: Long, quant_max: Long, grad_factor: Double = 1.0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._fake_quantize_learnable_per_tensor_affine(self.underlying, scale.underlying, zero_point.underlying, quant_min, quant_max, grad_factor))
  def _fake_quantize_learnable_per_tensor_affine_backward(grad: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: Long, quant_max: Long, grad_factor: Double = 1.0)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._fake_quantize_learnable_per_tensor_affine_backward(grad.underlying, self.underlying, scale.underlying, zero_point.underlying, quant_min, quant_max, grad_factor))
  def fake_quantize_per_channel_affine(self: Tensor, scale: Tensor, zero_point: Tensor, axis: Long, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fake_quantize_per_channel_affine(self.underlying, scale.underlying, zero_point.underlying, axis, quant_min, quant_max))
  def fake_quantize_per_channel_affine_cachemask(self: Tensor, scale: Tensor, zero_point: Tensor, axis: Long, quant_min: Long, quant_max: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.fake_quantize_per_channel_affine_cachemask(self.underlying, scale.underlying, zero_point.underlying, axis, quant_min, quant_max))
  def fake_quantize_per_channel_affine_cachemask_backward(grad: Tensor, mask: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fake_quantize_per_channel_affine_cachemask_backward(grad.underlying, mask.underlying))
  def _fake_quantize_learnable_per_channel_affine(self: Tensor, scale: Tensor, zero_point: Tensor, axis: Long, quant_min: Long, quant_max: Long, grad_factor: Double = 1.0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._fake_quantize_learnable_per_channel_affine(self.underlying, scale.underlying, zero_point.underlying, axis, quant_min, quant_max, grad_factor))
  def _fake_quantize_learnable_per_channel_affine_backward(grad: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: Long, quant_min: Long, quant_max: Long, grad_factor: Double = 1.0)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._fake_quantize_learnable_per_channel_affine_backward(grad.underlying, self.underlying, scale.underlying, zero_point.underlying, axis, quant_min, quant_max, grad_factor))
  def fused_moving_avg_obs_fake_quant(self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: Double, quant_min: Long, quant_max: Long, ch_axis: Long, per_row_fake_quant: Boolean = false, symmetric_quant: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fused_moving_avg_obs_fake_quant(self.underlying, observer_on.underlying, fake_quant_on.underlying, running_min.underlying, running_max.underlying, scale.underlying, zero_point.underlying, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant))
  def _fused_moving_avg_obs_fq_helper(self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: Double, quant_min: Long, quant_max: Long, ch_axis: Long, per_row_fake_quant: Boolean = false, symmetric_quant: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._fused_moving_avg_obs_fq_helper(self.underlying, observer_on.underlying, fake_quant_on.underlying, running_min.underlying, running_max.underlying, scale.underlying, zero_point.underlying, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant))
  def _choose_qparams_per_tensor(self: Tensor, reduce_range: Boolean = false)(implicit cg: ReferenceManager): (Double, Long) = wrapDoubleLongTuple2(swig._choose_qparams_per_tensor(self.underlying, reduce_range))
  def _saturate_weight_to_fp16(weight: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._saturate_weight_to_fp16(weight.underlying))
  def choose_qparams_optimized(input: Tensor, numel: Long, n_bins: Long, ratio: Double, bit_width: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.choose_qparams_optimized(input.underlying, numel, n_bins, ratio, bit_width))
  def _to_copy(self: Tensor, dtype: Option[dtype] = None, layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None, non_blocking: Boolean = false, memory_format: Option[MemoryFormat] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig._to_copy(self.underlying, options, non_blocking, memory_format))
}

  def meshgrid(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.meshgrid(tensors.map(_.underlyingChecked)))
  def meshgrid(tensors: Array[Tensor], indexing: String)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.meshgrid(tensors.map(_.underlyingChecked), indexing))
  def cartesian_prod(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.cartesian_prod(tensors.map(_.underlyingChecked)))
  def combinations(self: Tensor, r: Long = 2, with_replacement: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.combinations(self.underlying, r, with_replacement))
  def result_type(tensor: Tensor, other: Tensor)(implicit cg: ReferenceManager): dtype = dtype(swig.result_type(tensor.underlying, other.underlying))
  def result_type(tensor: Tensor, other: Scalar)(implicit cg: ReferenceManager): dtype = dtype(swig.result_type(tensor.underlying, other.underlying))
  def result_type(scalar: Scalar, tensor: Tensor)(implicit cg: ReferenceManager): dtype = dtype(swig.result_type(scalar.underlying, tensor.underlying))
  def result_type(scalar1: Scalar, scalar2: Scalar)(implicit cg: ReferenceManager): dtype = dtype(swig.result_type(scalar1.underlying, scalar2.underlying))
  def can_cast(from: dtype, to: dtype)(implicit cg: ReferenceManager): Boolean = swig.can_cast(from.toScalarType, to.toScalarType)
  def promote_types(type1: dtype, type2: dtype)(implicit cg: ReferenceManager): dtype = dtype(swig.promote_types(type1.toScalarType, type2.toScalarType))
  def _local_scalar_dense(self: Tensor)(implicit cg: ReferenceManager): Scalar = Scalar(swig._local_scalar_dense(self.underlying))
  def _thnn_fused_lstm_cell(input_gates: Tensor, hidden_gates: Tensor, cx: Tensor, input_bias: Option[Tensor] = None, hidden_bias: Option[Tensor] = None)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._thnn_fused_lstm_cell(input_gates.underlying, hidden_gates.underlying, cx.underlying, input_bias.map(_.underlying), hidden_bias.map(_.underlying)))
  def _thnn_fused_lstm_cell_backward(grad_hy: Option[Tensor], grad_cy: Option[Tensor], cx: Tensor, cy: Tensor, workspace: Tensor, has_bias: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig._thnn_fused_lstm_cell_backward(grad_hy.map(_.underlying), grad_cy.map(_.underlying), cx.underlying, cy.underlying, workspace.underlying, has_bias))
  def _thnn_differentiable_lstm_cell_backward(grad_hy: Option[Tensor], grad_cy: Option[Tensor], input_gates: Tensor, hidden_gates: Tensor, input_bias: Option[Tensor], hidden_bias: Option[Tensor], cx: Tensor, cy: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig._thnn_differentiable_lstm_cell_backward(grad_hy.map(_.underlying), grad_cy.map(_.underlying), input_gates.underlying, hidden_gates.underlying, input_bias.map(_.underlying), hidden_bias.map(_.underlying), cx.underlying, cy.underlying))
  def _thnn_fused_gru_cell(input_gates: Tensor, hidden_gates: Tensor, hx: Tensor, input_bias: Option[Tensor] = None, hidden_bias: Option[Tensor] = None)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._thnn_fused_gru_cell(input_gates.underlying, hidden_gates.underlying, hx.underlying, input_bias.map(_.underlying), hidden_bias.map(_.underlying)))
  def _thnn_fused_gru_cell_backward(grad_hy: Tensor, workspace: Tensor, has_bias: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig._thnn_fused_gru_cell_backward(grad_hy.underlying, workspace.underlying, has_bias))
  def _thnn_differentiable_gru_cell_backward(grad_hy: Tensor, input_gates: Tensor, hidden_gates: Tensor, hx: Tensor, input_bias: Option[Tensor], hidden_bias: Option[Tensor])(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor, Tensor, Tensor) = wrapTensorTuple5(swig._thnn_differentiable_gru_cell_backward(grad_hy.underlying, input_gates.underlying, hidden_gates.underlying, hx.underlying, input_bias.map(_.underlying), hidden_bias.map(_.underlying)))
  def lstm(input: Tensor, hx: Array[Tensor], params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean, batch_first: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.lstm(input.underlying, hx.map(_.underlyingChecked), params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional, batch_first))
  def lstm(data: Tensor, batch_sizes: Tensor, hx: Array[Tensor], params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.lstm(data.underlying, batch_sizes.underlying, hx.map(_.underlyingChecked), params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional))
  def gru(input: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean, batch_first: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.gru(input.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional, batch_first))
  def gru(data: Tensor, batch_sizes: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.gru(data.underlying, batch_sizes.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional))
  def rnn_tanh(input: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean, batch_first: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.rnn_tanh(input.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional, batch_first))
  def rnn_tanh(data: Tensor, batch_sizes: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.rnn_tanh(data.underlying, batch_sizes.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional))
  def rnn_relu(input: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean, batch_first: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.rnn_relu(input.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional, batch_first))
  def rnn_relu(data: Tensor, batch_sizes: Tensor, hx: Tensor, params: Array[Tensor], has_biases: Boolean, num_layers: Long, dropout: Double, train: Boolean, bidirectional: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.rnn_relu(data.underlying, batch_sizes.underlying, hx.underlying, params.map(_.underlyingChecked), has_biases, num_layers, dropout, train, bidirectional))
  def lstm_cell(input: Tensor, hx: Array[Tensor], w_ih: Tensor, w_hh: Tensor, b_ih: Option[Tensor] = None, b_hh: Option[Tensor] = None)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.lstm_cell(input.underlying, hx.map(_.underlyingChecked), w_ih.underlying, w_hh.underlying, b_ih.map(_.underlying), b_hh.map(_.underlying)))
  def gru_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Option[Tensor] = None, b_hh: Option[Tensor] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gru_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.map(_.underlying), b_hh.map(_.underlying)))
  def rnn_tanh_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Option[Tensor] = None, b_hh: Option[Tensor] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rnn_tanh_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.map(_.underlying), b_hh.map(_.underlying)))
  def rnn_relu_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Option[Tensor] = None, b_hh: Option[Tensor] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.rnn_relu_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.map(_.underlying), b_hh.map(_.underlying)))
  def quantized_lstm_cell(input: Tensor, hx: Array[Tensor], w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.quantized_lstm_cell(input.underlying, hx.map(_.underlyingChecked), w_ih.underlying, w_hh.underlying, b_ih.underlying, b_hh.underlying, packed_ih.underlying, packed_hh.underlying, col_offsets_ih.underlying, col_offsets_hh.underlying, scale_ih.underlying, scale_hh.underlying, zero_point_ih.underlying, zero_point_hh.underlying))
  def quantized_gru_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_gru_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.underlying, b_hh.underlying, packed_ih.underlying, packed_hh.underlying, col_offsets_ih.underlying, col_offsets_hh.underlying, scale_ih.underlying, scale_hh.underlying, zero_point_ih.underlying, zero_point_hh.underlying))
  def quantized_rnn_relu_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_rnn_relu_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.underlying, b_hh.underlying, packed_ih.underlying, packed_hh.underlying, col_offsets_ih.underlying, col_offsets_hh.underlying, scale_ih.underlying, scale_hh.underlying, zero_point_ih.underlying, zero_point_hh.underlying))
  def quantized_rnn_tanh_cell(input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantized_rnn_tanh_cell(input.underlying, hx.underlying, w_ih.underlying, w_hh.underlying, b_ih.underlying, b_hh.underlying, packed_ih.underlying, packed_hh.underlying, col_offsets_ih.underlying, col_offsets_hh.underlying, scale_ih.underlying, scale_hh.underlying, zero_point_ih.underlying, zero_point_hh.underlying))
  def _pack_padded_sequence(input: Tensor, lengths: Tensor, batch_first: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._pack_padded_sequence(input.underlying, lengths.underlying, batch_first))
  def _pack_padded_sequence_backward(grad: Tensor, input_size: Array[Long], batch_sizes: Tensor, batch_first: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._pack_padded_sequence_backward(grad.underlying, input_size, batch_sizes.underlying, batch_first))
  def _pad_packed_sequence(data: Tensor, batch_sizes: Tensor, batch_first: Boolean, padding_value: Scalar, total_length: Long)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._pad_packed_sequence(data.underlying, batch_sizes.underlying, batch_first, padding_value.underlying, total_length))
  def masked_fill(self: Tensor, mask: Tensor, value: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.masked_fill(self.underlying, mask.underlying, value.underlying))
  def masked_fill(self: Tensor, mask: Tensor, value: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.masked_fill(self.underlying, mask.underlying, value.underlying))
  def masked_scatter(self: Tensor, mask: Tensor, source: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.masked_scatter(self.underlying, mask.underlying, source.underlying))
  def put(self: Tensor, index: Tensor, source: Tensor, accumulate: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.put(self.underlying, index.underlying, source.underlying, accumulate))
  def index_add(self: Tensor, dim: Long, index: Tensor, source: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_add(self.underlying, dim, index.underlying, source.underlying))
  def index_add(self: Tensor, dim: Long, index: Tensor, source: Tensor, alpha: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_add(self.underlying, dim, index.underlying, source.underlying, alpha.underlying))
  def index_fill(self: Tensor, dim: Long, index: Tensor, value: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_fill(self.underlying, dim, index.underlying, value.underlying))
  def index_fill(self: Tensor, dim: Long, index: Tensor, value: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_fill(self.underlying, dim, index.underlying, value.underlying))
  def scatter(self: Tensor, dim: Long, index: Tensor, src: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.scatter(self.underlying, dim, index.underlying, src.underlying))
  def scatter(self: Tensor, dim: Long, index: Tensor, value: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.scatter(self.underlying, dim, index.underlying, value.underlying))
  def scatter(self: Tensor, dim: Long, index: Tensor, src: Tensor, reduce: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.scatter(self.underlying, dim, index.underlying, src.underlying, reduce))
  def scatter(self: Tensor, dim: Long, index: Tensor, value: Scalar, reduce: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.scatter(self.underlying, dim, index.underlying, value.underlying, reduce))
  def scatter_add(self: Tensor, dim: Long, index: Tensor, src: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.scatter_add(self.underlying, dim, index.underlying, src.underlying))
  def bitwise_and(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_and(self.underlying, other.underlying))
  def bitwise_and(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_and(self.underlying, other.underlying))
  def &(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__and__(self.underlying, other.underlying))
  def &(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__and__(self.underlying, other.underlying))
  def bitwise_or(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_or(self.underlying, other.underlying))
  def bitwise_or(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_or(self.underlying, other.underlying))
  def |(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__or__(self.underlying, other.underlying))
  def |(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__or__(self.underlying, other.underlying))
  def bitwise_xor(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_xor(self.underlying, other.underlying))
  def bitwise_xor(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_xor(self.underlying, other.underlying))
  def ^(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__xor__(self.underlying, other.underlying))
  def ^(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__xor__(self.underlying, other.underlying))
  def <<(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__lshift__(self.underlying, other.underlying))
  def <<(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__lshift__(self.underlying, other.underlying))
  def bitwise_left_shift(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_left_shift(self.underlying, other.underlying))
  def bitwise_left_shift(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_left_shift(self.underlying, other.underlying))
  def bitwise_left_shift(self: Scalar, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_left_shift(self.underlying, other.underlying))
  def >>(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__rshift__(self.underlying, other.underlying))
  def >>(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.__rshift__(self.underlying, other.underlying))
  def bitwise_right_shift(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_right_shift(self.underlying, other.underlying))
  def bitwise_right_shift(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_right_shift(self.underlying, other.underlying))
  def bitwise_right_shift(self: Scalar, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bitwise_right_shift(self.underlying, other.underlying))
  def addbmm(self: Tensor, batch1: Tensor, batch2: Tensor, beta: Double = 1, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addbmm(self.underlying, batch1.underlying, batch2.underlying, beta.toInternalScalar, alpha.toInternalScalar))
  def diag(self: Tensor, diagonal: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diag(self.underlying, diagonal))
  def diag_backward(grad: Tensor, input_sizes: Array[Long], diagonal: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.diag_backward(grad.underlying, input_sizes, diagonal))
  def cross(self: Tensor, other: Tensor, dim: Option[Long] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cross(self.underlying, other.underlying, dim.asJavaLong))
  def triu(self: Tensor, diagonal: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.triu(self.underlying, diagonal))
  def tril(self: Tensor, diagonal: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.tril(self.underlying, diagonal))
  def tril_indices(row: Long, col: Long, offset: Long = 0, dtype: Option[dtype] = Option(int64), layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.tril_indices(row, col, offset, options))
}

  def triu_indices(row: Long, col: Long, offset: Long = 0, dtype: Option[dtype] = Option(int64), layout: Option[Layout] = None, device: Option[Device] = None, pin_memory: Option[Boolean] = None)(implicit cg: ReferenceManager): Tensor = TensorOptions(
dtype=dtype,
device=device,
layout=layout,
pinned_memory=pin_memory,
).toInternal.apply { options => 
Tensor(swig.triu_indices(row, col, offset, options))
}

  def trace(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.trace(self.underlying))
  def trace_backward(grad: Tensor, sizes: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.trace_backward(grad.underlying, sizes))
  def ne(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ne(self.underlying, other.underlying))
  def ne(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ne(self.underlying, other.underlying))
  def not_equal(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.not_equal(self.underlying, other.underlying))
  def not_equal(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.not_equal(self.underlying, other.underlying))
  def eq(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.eq(self.underlying, other.underlying))
  def eq(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.eq(self.underlying, other.underlying))
  def ge(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ge(self.underlying, other.underlying))
  def ge(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ge(self.underlying, other.underlying))
  def greater_equal(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.greater_equal(self.underlying, other.underlying))
  def greater_equal(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.greater_equal(self.underlying, other.underlying))
  def le(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.le(self.underlying, other.underlying))
  def le(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.le(self.underlying, other.underlying))
  def less_equal(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.less_equal(self.underlying, other.underlying))
  def less_equal(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.less_equal(self.underlying, other.underlying))
  def gt(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gt(self.underlying, other.underlying))
  def gt(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gt(self.underlying, other.underlying))
  def greater(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.greater(self.underlying, other.underlying))
  def greater(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.greater(self.underlying, other.underlying))
  def lt(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lt(self.underlying, other.underlying))
  def lt(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lt(self.underlying, other.underlying))
  def less(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.less(self.underlying, other.underlying))
  def less(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.less(self.underlying, other.underlying))
  def take(self: Tensor, index: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.take(self.underlying, index.underlying))
  def take_along_dim(self: Tensor, indices: Tensor, dim: Option[Long] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.take_along_dim(self.underlying, indices.underlying, dim.asJavaLong))
  def index_select(self: Tensor, dim: Long, index: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_select(self.underlying, dim, index.underlying))
  def index_select_backward(grad: Tensor, self_sizes: Array[Long], dim: Long, index: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.index_select_backward(grad.underlying, self_sizes, dim, index.underlying))
  def masked_select(self: Tensor, mask: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.masked_select(self.underlying, mask.underlying))
  def masked_select_backward(grad: Tensor, input: Tensor, mask: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.masked_select_backward(grad.underlying, input.underlying, mask.underlying))
  def nonzero(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nonzero(self.underlying))
  def nonzero_numpy(self: Tensor)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig.nonzero_numpy(self.underlying))
  def gather(self: Tensor, dim: Long, index: Tensor, sparse_grad: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gather(self.underlying, dim, index.underlying, sparse_grad))
  def gather_backward(grad: Tensor, self: Tensor, dim: Long, index: Tensor, sparse_grad: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.gather_backward(grad.underlying, self.underlying, dim, index.underlying, sparse_grad))
  def _gather_sparse_backward(self: Tensor, dim: Long, index: Tensor, grad: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._gather_sparse_backward(self.underlying, dim, index.underlying, grad.underlying))
  def addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addcmul(self.underlying, tensor1.underlying, tensor2.underlying, value.toInternalScalar))
  def addcdiv(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig.addcdiv(self.underlying, tensor1.underlying, tensor2.underlying, value.toInternalScalar))
  def lstsq(self: Tensor, A: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.lstsq(self.underlying, A.underlying))
  def triangular_solve(self: Tensor, A: Tensor, upper: Boolean = true, transpose: Boolean = false, unitriangular: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.triangular_solve(self.underlying, A.underlying, upper, transpose, unitriangular))
  def symeig(self: Tensor, eigenvectors: Boolean = false, upper: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.symeig(self.underlying, eigenvectors, upper))
  def _symeig_helper(self: Tensor, eigenvectors: Boolean, upper: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._symeig_helper(self.underlying, eigenvectors, upper))
  def eig(self: Tensor, eigenvectors: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.eig(self.underlying, eigenvectors))
  def svd(self: Tensor, some: Boolean = true, compute_uv: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.svd(self.underlying, some, compute_uv))
  def _svd_helper(self: Tensor, some: Boolean, compute_uv: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._svd_helper(self.underlying, some, compute_uv))
  def swapaxes(self: Tensor, axis0: Long, axis1: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.swapaxes(self.underlying, axis0, axis1))
  def swapdims(self: Tensor, dim0: Long, dim1: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.swapdims(self.underlying, dim0, dim1))
  def cholesky(self: Tensor, upper: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cholesky(self.underlying, upper))
  def cholesky_solve(self: Tensor, input2: Tensor, upper: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cholesky_solve(self.underlying, input2.underlying, upper))
  def _cholesky_solve_helper(self: Tensor, A: Tensor, upper: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cholesky_solve_helper(self.underlying, A.underlying, upper))
  def solve(self: Tensor, A: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.solve(self.underlying, A.underlying))
  def _solve_helper(self: Tensor, A: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._solve_helper(self.underlying, A.underlying))
  def cholesky_inverse(self: Tensor, upper: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.cholesky_inverse(self.underlying, upper))
  def qr(self: Tensor, some: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.qr(self.underlying, some))
  def geqrf(self: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.geqrf(self.underlying))
  def orgqr(self: Tensor, input2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.orgqr(self.underlying, input2.underlying))
  def ormqr(self: Tensor, input2: Tensor, input3: Tensor, left: Boolean = true, transpose: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ormqr(self.underlying, input2.underlying, input3.underlying, left, transpose))
  def _lu_with_info(self: Tensor, pivot: Boolean = true, check_errors: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._lu_with_info(self.underlying, pivot, check_errors))
  def lu_solve(self: Tensor, LU_data: Tensor, LU_pivots: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lu_solve(self.underlying, LU_data.underlying, LU_pivots.underlying))
  def lu_unpack(LU_data: Tensor, LU_pivots: Tensor, unpack_data: Boolean = true, unpack_pivots: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig.lu_unpack(LU_data.underlying, LU_pivots.underlying, unpack_data, unpack_pivots))
  def multinomial(self: Tensor, num_samples: Long, replacement: Boolean = false, generator: Option[Generator] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.multinomial(self.underlying, num_samples, replacement, generator.map(_.underlying)))
  def lgamma(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lgamma(self.underlying))
  def digamma(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.digamma(self.underlying))
  def polygamma(n: Long, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.polygamma(n, self.underlying))
  def erfinv(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.erfinv(self.underlying))
  def i0(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.i0(self.underlying))
  def i0_(self: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig.i0_(self.underlying)
    self
  }
  def sign(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.sign(self.underlying))
  def signbit(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.signbit(self.underlying))
  def dist(self: Tensor, other: Tensor, p: Double = 2)(implicit cg: ReferenceManager): Tensor = Tensor(swig.dist(self.underlying, other.underlying, p.toInternalScalar))
  def atan2(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.atan2(self.underlying, other.underlying))
  def lerp(self: Tensor, end: Tensor, weight: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lerp(self.underlying, end.underlying, weight.underlying))
  def lerp(self: Tensor, end: Tensor, weight: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.lerp(self.underlying, end.underlying, weight.underlying))
  def histc(self: Tensor, bins: Long = 100, min: Double = 0, max: Double = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig.histc(self.underlying, bins, min.toInternalScalar, max.toInternalScalar))
  def histogram(self: Tensor, bins: Tensor, weight: Option[Tensor] = None, density: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.histogram(self.underlying, bins.underlying, weight.map(_.underlying), density))
  def histogram(self: Tensor, bins: Long, range: Option[Array[Double]], weight: Option[Tensor], density: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.histogram(self.underlying, bins, range, weight.map(_.underlying), density))
  def fmod(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fmod(self.underlying, other.underlying))
  def fmod(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fmod(self.underlying, other.underlying))
  def hypot(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.hypot(self.underlying, other.underlying))
  def igamma(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.igamma(self.underlying, other.underlying))
  def igammac(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.igammac(self.underlying, other.underlying))
  def nextafter(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nextafter(self.underlying, other.underlying))
  def remainder(self: Tensor, other: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.remainder(self.underlying, other.underlying))
  def remainder(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.remainder(self.underlying, other.underlying))
  def remainder(self: Scalar, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.remainder(self.underlying, other.underlying))
  def min(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.min(self.underlying))
  def fmin(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fmin(self.underlying, other.underlying))
  def max(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.max(self.underlying))
  def fmax(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.fmax(self.underlying, other.underlying))
  def maximum(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.maximum(self.underlying, other.underlying))
  def max(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.max(self.underlying, other.underlying))
  def minimum(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.minimum(self.underlying, other.underlying))
  def min(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.min(self.underlying, other.underlying))
  def quantile(self: Tensor, q: Double, dim: Option[Long] = None, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantile(self.underlying, q, dim.asJavaLong, keepdim))
  def quantile(self: Tensor, q: Tensor, dim: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantile(self.underlying, q.underlying, dim.asJavaLong, keepdim))
  def nanquantile(self: Tensor, q: Double, dim: Option[Long] = None, keepdim: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanquantile(self.underlying, q, dim.asJavaLong, keepdim))
  def nanquantile(self: Tensor, q: Tensor, dim: Option[Long], keepdim: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanquantile(self.underlying, q.underlying, dim.asJavaLong, keepdim))
  def quantile(self: Tensor, q: Double, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantile(self.underlying, q, dim.asJavaLong, keepdim, interpolation))
  def quantile(self: Tensor, q: Tensor, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.quantile(self.underlying, q.underlying, dim.asJavaLong, keepdim, interpolation))
  def nanquantile(self: Tensor, q: Double, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanquantile(self.underlying, q, dim.asJavaLong, keepdim, interpolation))
  def nanquantile(self: Tensor, q: Tensor, dim: Option[Long], keepdim: Boolean, interpolation: String)(implicit cg: ReferenceManager): Tensor = Tensor(swig.nanquantile(self.underlying, q.underlying, dim.asJavaLong, keepdim, interpolation))
  def sort(self: Tensor, dim: Long = -1, descending: Boolean = false)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.sort(self.underlying, dim, descending))
  def sort(self: Tensor, stable: Option[Boolean], dim: Long, descending: Boolean)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.sort(self.underlying, stable.map(java.lang.Boolean.valueOf), dim, descending))
  def msort(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.msort(self.underlying))
  def argsort(self: Tensor, dim: Long = -1, descending: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.argsort(self.underlying, dim, descending))
  def topk(self: Tensor, k: Long, dim: Long = -1, largest: Boolean = true, sorted: Boolean = true)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig.topk(self.underlying, k, dim, largest, sorted))
  def all(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.all(self.underlying))
  def any(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.any(self.underlying))
  def renorm(self: Tensor, p: Scalar, dim: Long, maxnorm: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.renorm(self.underlying, p.underlying, dim, maxnorm.underlying))
  def unfold_backward(grad_in: Tensor, input_sizes: Array[Long], dim: Long, size: Long, step: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig.unfold_backward(grad_in.underlying, input_sizes, dim, size, step))
  def equal(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Boolean = swig.equal(self.underlying, other.underlying)
  def pow(self: Tensor, exponent: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pow(self.underlying, exponent.underlying))
  def pow(self: Scalar, exponent: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pow(self.underlying, exponent.underlying))
  def pow(self: Tensor, exponent: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.pow(self.underlying, exponent.underlying))
  def float_power(self: Tensor, exponent: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.float_power(self.underlying, exponent.underlying))
  def float_power(self: Scalar, exponent: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.float_power(self.underlying, exponent.underlying))
  def float_power(self: Tensor, exponent: Scalar)(implicit cg: ReferenceManager): Tensor = Tensor(swig.float_power(self.underlying, exponent.underlying))
  def alias(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.alias(self.underlying))
  def _index_copy_(self: Tensor, dim: Long, index: Tensor, source: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._index_copy_(self.underlying, dim, index.underlying, source.underlying)
    self
  }
  def _amp_foreach_non_finite_check_and_unscale_(self: Array[Tensor], found_inf: Tensor, inv_scale: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._amp_foreach_non_finite_check_and_unscale_(self.map(_.underlyingChecked), found_inf.underlying, inv_scale.underlying)
    self
  }
  def _amp_update_scale_(self: Tensor, growth_tracker: Tensor, found_inf: Tensor, scale_growth_factor: Double, scale_backoff_factor: Double, growth_interval: Long)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._amp_update_scale_(self.underlying, growth_tracker.underlying, found_inf.underlying, scale_growth_factor, scale_backoff_factor, growth_interval)
    self
  }
  def _cat(tensors: Array[Tensor], dim: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._cat(tensors.map(_.underlyingChecked), dim))
  def _foreach_add(tensors: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_add(tensors.map(_.underlyingChecked), scalar.underlying))
  def _foreach_add_(self: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_add_(self.map(_.underlyingChecked), scalar.underlying)
    self
  }
  def _foreach_sub(tensors: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sub(tensors.map(_.underlyingChecked), scalar.underlying))
  def _foreach_sub_(self: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sub_(self.map(_.underlyingChecked), scalar.underlying)
    self
  }
  def _foreach_mul(tensors: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_mul(tensors.map(_.underlyingChecked), scalar.underlying))
  def _foreach_mul_(self: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_mul_(self.map(_.underlyingChecked), scalar.underlying)
    self
  }
  def _foreach_div(tensors: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_div(tensors.map(_.underlyingChecked), scalar.underlying))
  def _foreach_div_(self: Array[Tensor], scalar: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_div_(self.map(_.underlyingChecked), scalar.underlying)
    self
  }
  def _foreach_add(tensors1: Array[Tensor], tensors2: Array[Tensor], alpha: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_add(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked), alpha.underlying))
  def _foreach_add_(self: Array[Tensor], other: Array[Tensor], alpha: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_add_(self.map(_.underlyingChecked), other.map(_.underlyingChecked), alpha.underlying)
    self
  }
  def _foreach_sub(tensors1: Array[Tensor], tensors2: Array[Tensor], alpha: Scalar)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sub(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked), alpha.underlying))
  def _foreach_sub_(self: Array[Tensor], other: Array[Tensor], alpha: Scalar)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sub_(self.map(_.underlyingChecked), other.map(_.underlyingChecked), alpha.underlying)
    self
  }
  def _foreach_mul(tensors1: Array[Tensor], tensors2: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_mul(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked)))
  def _foreach_mul_(self: Array[Tensor], other: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_mul_(self.map(_.underlyingChecked), other.map(_.underlyingChecked))
    self
  }
  def _foreach_div(tensors1: Array[Tensor], tensors2: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_div(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked)))
  def _foreach_div_(self: Array[Tensor], other: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_div_(self.map(_.underlyingChecked), other.map(_.underlyingChecked))
    self
  }
  def _foreach_add(tensors: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_add(tensors.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_add_(self: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_add_(self.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_sub(tensors: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sub(tensors.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_sub_(self: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sub_(self.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_div(tensors: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_div(tensors.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_div_(self: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_div_(self.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_mul(tensors: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_mul(tensors.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_mul_(self: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_mul_(self.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_exp(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_exp(tensors.map(_.underlyingChecked)))
  def _foreach_zero_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_zero_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_exp_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_exp_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_sqrt(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sqrt(tensors.map(_.underlyingChecked)))
  def _foreach_sqrt_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sqrt_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_abs(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_abs(tensors.map(_.underlyingChecked)))
  def _foreach_abs_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_abs_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_acos(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_acos(tensors.map(_.underlyingChecked)))
  def _foreach_acos_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_acos_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_asin(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_asin(tensors.map(_.underlyingChecked)))
  def _foreach_asin_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_asin_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_atan(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_atan(tensors.map(_.underlyingChecked)))
  def _foreach_atan_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_atan_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_ceil(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_ceil(tensors.map(_.underlyingChecked)))
  def _foreach_ceil_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_ceil_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_cos(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_cos(tensors.map(_.underlyingChecked)))
  def _foreach_cos_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_cos_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_cosh(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_cosh(tensors.map(_.underlyingChecked)))
  def _foreach_cosh_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_cosh_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_erf(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_erf(tensors.map(_.underlyingChecked)))
  def _foreach_erf_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_erf_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_erfc(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_erfc(tensors.map(_.underlyingChecked)))
  def _foreach_erfc_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_erfc_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_expm1(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_expm1(tensors.map(_.underlyingChecked)))
  def _foreach_expm1_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_expm1_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_floor(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_floor(tensors.map(_.underlyingChecked)))
  def _foreach_floor_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_floor_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_log(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_log(tensors.map(_.underlyingChecked)))
  def _foreach_log_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_log_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_log10(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_log10(tensors.map(_.underlyingChecked)))
  def _foreach_log10_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_log10_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_log1p(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_log1p(tensors.map(_.underlyingChecked)))
  def _foreach_log1p_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_log1p_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_log2(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_log2(tensors.map(_.underlyingChecked)))
  def _foreach_log2_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_log2_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_neg(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_neg(tensors.map(_.underlyingChecked)))
  def _foreach_neg_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_neg_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_tan(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_tan(tensors.map(_.underlyingChecked)))
  def _foreach_tan_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_tan_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_tanh(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_tanh(tensors.map(_.underlyingChecked)))
  def _foreach_tanh_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_tanh_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_sin(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sin(tensors.map(_.underlyingChecked)))
  def _foreach_sin_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sin_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_sinh(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sinh(tensors.map(_.underlyingChecked)))
  def _foreach_sinh_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sinh_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_round(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_round(tensors.map(_.underlyingChecked)))
  def _foreach_round_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_round_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_lgamma(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_lgamma(tensors.map(_.underlyingChecked)))
  def _foreach_lgamma_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_lgamma_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_frac(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_frac(tensors.map(_.underlyingChecked)))
  def _foreach_frac_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_frac_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_reciprocal(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_reciprocal(tensors.map(_.underlyingChecked)))
  def _foreach_reciprocal_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_reciprocal_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_sigmoid(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_sigmoid(tensors.map(_.underlyingChecked)))
  def _foreach_sigmoid_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_sigmoid_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_trunc(tensors: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_trunc(tensors.map(_.underlyingChecked)))
  def _foreach_trunc_(self: Array[Tensor])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_trunc_(self.map(_.underlyingChecked))
    self
  }
  def _foreach_addcdiv_(self: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], value: Double = 1)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_addcdiv_(self.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), value.toInternalScalar)
    self
  }
  def _foreach_addcmul_(self: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], value: Double = 1)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_addcmul_(self.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), value.toInternalScalar)
    self
  }
  def _foreach_addcdiv_(self: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_addcdiv_(self.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_addcmul_(self: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._foreach_addcmul_(self.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), scalars.map(_.underlying))
    self
  }
  def _foreach_addcdiv(input: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], value: Double = 1)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_addcdiv(input.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), value.toInternalScalar))
  def _foreach_addcmul(input: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], value: Double = 1)(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_addcmul(input.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), value.toInternalScalar))
  def _foreach_addcdiv(input: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_addcdiv(input.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_addcmul(input: Array[Tensor], tensor1: Array[Tensor], tensor2: Array[Tensor], scalars: Array[Scalar])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_addcmul(input.map(_.underlyingChecked), tensor1.map(_.underlyingChecked), tensor2.map(_.underlyingChecked), scalars.map(_.underlying)))
  def _foreach_maximum(tensors1: Array[Tensor], tensors2: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_maximum(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked)))
  def _foreach_minimum(tensors1: Array[Tensor], tensors2: Array[Tensor])(implicit cg: ReferenceManager): Array[Tensor] = tensorVectorToArray(swig._foreach_minimum(tensors1.map(_.underlyingChecked), tensors2.map(_.underlyingChecked)))
  def bucketize(self: Tensor, boundaries: Tensor, out_int32: Boolean = false, right: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bucketize(self.underlying, boundaries.underlying, out_int32, right))
  def bucketize(self: Scalar, boundaries: Tensor, out_int32: Boolean, right: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.bucketize(self.underlying, boundaries.underlying, out_int32, right))
  def searchsorted(sorted_sequence: Tensor, self: Tensor, out_int32: Boolean = false, right: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig.searchsorted(sorted_sequence.underlying, self.underlying, out_int32, right))
  def searchsorted(sorted_sequence: Tensor, self: Scalar, out_int32: Boolean, right: Boolean)(implicit cg: ReferenceManager): Tensor = Tensor(swig.searchsorted(sorted_sequence.underlying, self.underlying, out_int32, right))
  def _convert_indices_from_coo_to_csr(self: Tensor, size: Long, out_int32: Boolean = false)(implicit cg: ReferenceManager): Tensor = Tensor(swig._convert_indices_from_coo_to_csr(self.underlying, size, out_int32))
  def mkldnn_adaptive_avg_pool2d(self: Tensor, output_size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_adaptive_avg_pool2d(self.underlying, output_size))
  def mkldnn_adaptive_avg_pool2d_backward(grad_output: Tensor, self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.mkldnn_adaptive_avg_pool2d_backward(grad_output.underlying, self.underlying))
  def _adaptive_avg_pool2d(self: Tensor, output_size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._adaptive_avg_pool2d(self.underlying, output_size))
  def _adaptive_avg_pool3d(self: Tensor, output_size: Array[Long])(implicit cg: ReferenceManager): Tensor = Tensor(swig._adaptive_avg_pool3d(self.underlying, output_size))
  def column_stack(tensors: Array[Tensor])(implicit cg: ReferenceManager): Tensor = Tensor(swig.column_stack(tensors.map(_.underlyingChecked)))
  def isfinite(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isfinite(self.underlying))
  def isinf(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isinf(self.underlying))
  def isposinf(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isposinf(self.underlying))
  def isneginf(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.isneginf(self.underlying))
  def _add_batch_dim(self: Tensor, batch_dim: Long, level: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._add_batch_dim(self.underlying, batch_dim, level))
  def _remove_batch_dim(self: Tensor, level: Long, batch_size: Long, out_dim: Long)(implicit cg: ReferenceManager): Tensor = Tensor(swig._remove_batch_dim(self.underlying, level, batch_size, out_dim))
  def det(self: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.det(self.underlying))
  def _det_lu_based_helper(self: Tensor)(implicit cg: ReferenceManager): (Tensor, Tensor, Tensor) = wrapTensorTuple3(swig._det_lu_based_helper(self.underlying))
  def _det_lu_based_helper_backward_helper(det_grad: Tensor, det: Tensor, self: Tensor, lu: Tensor, pivs: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig._det_lu_based_helper_backward_helper(det_grad.underlying, det.underlying, self.underlying, lu.underlying, pivs.underlying))
  def _linalg_inv_out_helper_(self: Tensor, infos_lu: Tensor, infos_getri: Tensor)(implicit cg: ReferenceManager): self.type = NoGrad.noGrad {
    swig._linalg_inv_out_helper_(self.underlying, infos_lu.underlying, infos_getri.underlying)
    self
  }
  def inner(self: Tensor, other: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.inner(self.underlying, other.underlying))
  def outer(self: Tensor, vec2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.outer(self.underlying, vec2.underlying))
  def ger(self: Tensor, vec2: Tensor)(implicit cg: ReferenceManager): Tensor = Tensor(swig.ger(self.underlying, vec2.underlying))
  def _linalg_qr_helper(self: Tensor, mode: String)(implicit cg: ReferenceManager): (Tensor, Tensor) = wrapTensorTuple2(swig._linalg_qr_helper(self.underlying, mode))
  def _test_serialization_subcmul(self: Tensor, other: Tensor, alpha: Double = 1)(implicit cg: ReferenceManager): Tensor = Tensor(swig._test_serialization_subcmul(self.underlying, other.underlying, alpha.toInternalScalar))
  def segment_reduce(data: Tensor, reduce: String, lengths: Option[Tensor] = None, indices: Option[Tensor] = None, axis: Long = 0, unsafe: Boolean = false, initial: Option[Scalar] = None)(implicit cg: ReferenceManager): Tensor = Tensor(swig.segment_reduce(data.underlying, reduce, lengths.map(_.underlying), indices.map(_.underlying), axis, unsafe, initial.map(_.underlying)))
  def _segment_reduce_backward(grad: Tensor, output: Tensor, data: Tensor, reduce: String, lengths: Option[Tensor] = None, axis: Long = 0)(implicit cg: ReferenceManager): Tensor = Tensor(swig._segment_reduce_backward(grad.underlying, output.underlying, data.underlying, reduce, lengths.map(_.underlying), axis))
  // end of auto-generated API

  // TODO we should do the work in swig to produce an Array[Tensor] directly.
  private[torch] def tensorVectorToArray(tv: TensorVector)(implicit cg: ReferenceManager): Array[Tensor] = {
    try {
      tv.asScala.toArray[TorchTensor].map(t => Tensor(t.refCopy()))
    } finally {
      tv.delete()
    }
  }
}
