package com.microsoft.scalatorch.torch.nn

import com.microsoft.scalatorch.torch.{ Scalar, Tensor, internal }
import com.microsoft.scalatorch.torch.internal.torch_swig
import com.microsoft.scalatorch.torch.util.NoGrad

package object init {

  type Nonlinearity = internal.Nonlinearity
  object Nonlinearity {
    val Linear = internal.Nonlinearity.Linear
    val Conv1D = internal.Nonlinearity.Conv1D
    val Conv2D = internal.Nonlinearity.Conv2D
    val Conv3D = internal.Nonlinearity.Conv3D
    val ConvTranspose1D = internal.Nonlinearity.ConvTranspose1D
    val ConvTranspose2D = internal.Nonlinearity.ConvTranspose2D
    val ConvTranspose3D = internal.Nonlinearity.ConvTranspose3D
    val Sigmoid = internal.Nonlinearity.Sigmoid
    val Tanh = internal.Nonlinearity.Tanh
    val ReLU = internal.Nonlinearity.ReLU
    val LeakyReLU = internal.Nonlinearity.LeakyReLU
  }

  type FanMode = internal.FanMode
  object FanMode {
    val FanIn = internal.FanMode.FanIn
    val FanOut = internal.FanMode.FanOut
  }

  trait ParameterInit {
    def initializeParams(values: Tensor): values.type
  }

  private class ParameterInitImpl(initInPlace: internal.TorchTensor => Unit) extends ParameterInit {
    override def initializeParams(values: Tensor): values.type = NoGrad.noGrad {
      initInPlace(values.underlying)
      values
    }
  }

  /** Initialize a parameter with random normal values. Note that dynet v is variance, but pytorch is std. we're using std */
  def normal(m: Float = 0.0f, std: Float = 1.0f): ParameterInit =
    new ParameterInitImpl(torch_swig.normal_(_, m, std))

  def normal_(tensor: Tensor, m: Float = 0.0f, std: Float = 1.0f): tensor.type = {
    torch_swig.normal_(tensor.underlying, m, std)
    tensor
  }

  /** Initialize a parameter with random uniform [left, right] values */
  def uniform(low: Float = 0f, high: Float = 1f): ParameterInit =
    new ParameterInitImpl(torch_swig.uniform_(_, low, high))
  def uniform_(tensor: Tensor, low: Float = 0f, high: Float = 1f): tensor.type = {
    torch_swig.normal_(tensor.underlying, low, high)
    tensor
  }

  def kaiming_uniform(
      a: Float = 0f,
      mode: FanMode = FanMode.FanIn,
      nonlinearity: Nonlinearity = Nonlinearity.LeakyReLU,
  ): ParameterInit = new ParameterInitImpl(
    torch_swig.kaiming_uniform_(_, a, mode, nonlinearity),
  )
  def kaiming_uniform_(
      tensor: Tensor,
      a: Float = 0f,
      mode: FanMode = FanMode.FanIn,
      nonlinearity: Nonlinearity = Nonlinearity.LeakyReLU,
  ): tensor.type = {
    torch_swig.kaiming_uniform_(tensor.underlying, a, mode, nonlinearity)
    tensor
  }

  def kaiming_normal(
      a: Float = 0f,
      mode: FanMode = FanMode.FanIn,
      nonlinearity: Nonlinearity = Nonlinearity.LeakyReLU,
  ): ParameterInit = new ParameterInitImpl(torch_swig.kaiming_normal_(_, a, mode, nonlinearity))
  def kaiming_normal_(
      tensor: Tensor,
      a: Float = 0f,
      mode: FanMode = FanMode.FanIn,
      nonlinearity: Nonlinearity = Nonlinearity.LeakyReLU,
  ): tensor.type = {
    torch_swig.kaiming_normal_(tensor.underlying, a, mode, nonlinearity)
    tensor
  }

  /** Initialize a parameter with the constant value c */
  def constant(c: Scalar): ParameterInit =
    new ParameterInitImpl(t => {
      torch_swig.constant_(t, c.underlying)
    })
  def constant(d: Double): ParameterInit = {
    new ParameterInitImpl(t => {
      val c = new internal.Scalar(d)
      try torch_swig.constant_(t, c)
      finally c.delete()
    })

  }

  def constant_(tensor: Tensor, c: Scalar): tensor.type = {
    torch_swig.constant_(tensor.underlying, c.underlying)
    tensor
  }

  def sparse_(tensor: Tensor, sparsity: Double, std: Double = 0.01): tensor.type = {
    torch_swig.sparse_(tensor.underlying, sparsity, std)
    tensor
  }

  def eye(): ParameterInit = new ParameterInitImpl(torch_swig.eye_)
  def eye_(tensor: Tensor): tensor.type = { torch_swig.eye_(tensor.underlying); tensor }
  def ones(): ParameterInit = new ParameterInitImpl(torch_swig.ones_)
  def ones_(tensor: Tensor): tensor.type = { torch_swig.ones_(tensor.underlying); tensor }
  def zeros(): ParameterInit = new ParameterInitImpl(torch_swig.zeros_)
  def zeros_(tensor: Tensor): tensor.type = { torch_swig.zeros_(tensor.underlying); tensor }
  def dirac(): ParameterInit = new ParameterInitImpl(torch_swig.zeros_)

  def xavier_uniform(gain: Double = 1.0): ParameterInit = new ParameterInitImpl(torch_swig.xavier_uniform_(_, gain))
  def xavier_uniform_(tensor: Tensor, gain: Double = 1.0): tensor.type = {
    torch_swig.xavier_uniform_(tensor.underlying, gain)
    tensor
  }

  /** Initialize a parameter using Glorot (Xavier) uniform initialization */
  def glorotUniform(isLookup: Boolean): ParameterInit = {
    new ParameterInitImpl({ t =>
      // dynet's code reduces to this with a rank 0 or 1 tensor
      // (pytorch however gets mad)
      val dims = t.sizes()
      if (dims.length == 0) {
        val u = math.sqrt(3.0f)
        torch_swig.uniform_(t, -u, u)
      } else if (dims.length == 1 || isLookup) {
        require(!isLookup || dims.length == 2)
        val denom = if (isLookup) {
          // For lookup parameters, we use the same scale for an initialization as
          // an individual vector of the embedding size.
          dims(1)
        } else {
          dims.sum
        }
        val u = math.sqrt(3.0f / denom)
        torch_swig.uniform_(t, -u, u)
      } else {
        torch_swig.xavier_uniform_(t)
      }
    })
  }

  def fromValues(values: Tensor): ParameterInit = NoGrad.noGrad {
    new ParameterInitImpl(_.copy_(values.underlying, false))
  }
}
