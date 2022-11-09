package com.microsoft.scalatorch.torch.optim

import java.io.File

import scala.collection.JavaConverters._

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, RMSpropOptions, TensorVector }
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, TensorVector }
import com.microsoft.scalatorch.torch.util.Disposer
import resource.ManagedResource
import com.microsoft.scalatorch.torch.{ internal, Device, Model, ReferenceManager, Tensor }
import com.microsoft.scalatorch.torch.internal.{ torch_swig, TensorVector }
import com.microsoft.scalatorch.torch.jit.Module
import com.microsoft.scalatorch.torch.util.Disposer

/** Base class for optimizers in Torch.
  *
  * The pattern the subclasses follow is that there is an Options object on the companion
  * (e.g. [[Adagrad.Options]] and they each take that and a [[Module]] as their ctor arg)
  *
  * A note about thread safety: The version of Dynet we are trying to imitate
  * assumed that each [[Module]] and [[ComputationGraph]] was kept thread-local by the instantiator
  * so that all computations could happen on each object without any locking. We try to maintain that same assumption
  * here: we do not make any effort to add threadsafety over and above what PyTorch provides because we
  * assume the caller will maintain thread isolation for us.
  *
  * That said, PyTorch has some unfortunate opinions about how to do threading
  * * (https://github.com/pytorch/pytorch/issues/18333) that may make our threading model difficult.
  *
  * @see https://pytorch.org/cppdocs/api/classtorch_1_1optim_1_1_optimizer.html?highlight=optimizer
  * @see https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer
  */
abstract class Optimizer protected (private[torch] val optimizer: internal.Optimizer) extends AutoCloseable {
  def step(): Unit = optimizer.step()

  /** Torch has separate operations for [[step]] and [[zeroGrad]]; this method combines the two. */
  def update(): Unit = {
    step()
    zeroGrad()
  }

  def zeroGrad(): Unit = optimizer.zero_grad()

  def learningRate: Double

  def learningRate_=(newLearningRate: Double): this.type

  def clipGradients(clip: Double): Unit = {
    torch_swig.clip_grad_norm_(optimizer.all_parameters(), clip)
  }

  def close(): Unit = optimizer.delete()

  def load(f: File, device: Option[Device] = None)(implicit rm: ReferenceManager): Unit = {
    torch_swig.load(optimizer, f.toString, device.map(_.underlying))
  }

  def save(f: File): Unit = com.microsoft.scalatorch.torch.save(this, f.toString)
  def saveToByteArray(): Array[Byte] = com.microsoft.scalatorch.torch.serialize(this)
}

object Optimizer {
  protected[torch] def extractParameters(model: Model)(implicit rm: ReferenceManager): Iterable[Tensor] = {
    model.module.parameters(recurse = true)
  }
}

class SGD private (override private[torch] val optimizer: internal.SGD) extends Optimizer(optimizer) {

  def learningRate: Double = {
    val options = optimizer.getOptions()
    try options.lr()
    finally options.delete()
  }

  def learningRate_=(newLearningRate: Double): this.type = {
    for (i <- 0 until optimizer.num_param_groups()) {
      internal.SGDOptions.cast(optimizer.param_group(i).options()).lr(newLearningRate).delete()
    }
    this
  }
}

object SGD {
  case class Options(
      learningRate: Double = defaultsHolder.lr(),
      dampening: Double = defaultsHolder.dampening(),
      weightDecay: Double = defaultsHolder.weight_decay(),
      momentum: Double = defaultsHolder.momentum(),
      nesterov: Boolean = defaultsHolder.nesterov(),
  ) {
    private[SGD] def toInternal: ManagedResource[internal.SGDOptions] = {
      def man(o: => internal.SGDOptions) = resource.makeManagedResource(o)(_.delete())(List.empty)
      for {
        orig <- man(new internal.SGDOptions(learningRate))
        afterDampening <- man(orig.dampening(dampening))
        afterWeightDecay <- man(afterDampening.weight_decay(weightDecay))
        afterMomentum <- man(afterWeightDecay.momentum(momentum))
        afterNesterov <- man(afterMomentum.nesterov(nesterov))
      } yield afterNesterov
    }
  }

  private def apply(underlying: internal.SGD): SGD = {
    Disposer.add(new SGD(underlying), () => underlying.delete())
  }
  def apply(parameters: Iterable[Tensor], options: SGD.Options = SGD.Options()): SGD = {
    options.toInternal.apply { opts =>
      val underlying = ReferenceManager.forBlock { implicit rm =>
        val tensorVector = new TensorVector(parameters.map(_.underlying).asJava)
        try new internal.SGD(tensorVector, opts)
        finally tensorVector.delete()
      }
      apply(underlying)
    }
  }

  def apply(m: Model, options: SGD.Options): SGD = ReferenceManager.forBlock { implicit rm =>
    apply(Optimizer.extractParameters(m), options)
  }

  def apply(m: Model, learningRate: Double): SGD = apply(m, SGD.Options(learningRate))

  private val defaultsHolder = new internal.SGDOptions(0.1)
}

class Adam private (override private[torch] val optimizer: internal.Adam) extends Optimizer(optimizer) {

  def learningRate: Double = {
    val options = optimizer.getOptions()
    val ret = options.lr()
    options.delete()
    ret
  }

  def learningRate_=(newLearningRate: Double): this.type = {
    for (i <- 0 until optimizer.num_param_groups()) {
      internal.AdamOptions.cast(optimizer.param_group(i).options()).lr(newLearningRate).delete()
    }
    this
  }
}

object Adam {
  case class Options(
      learningRate: Double = defaultsHolder.lr(),
      beta1: Double = defaultsHolder.beta1(),
      beta2: Double = defaultsHolder.beta2(),
      epsilon: Double = defaultsHolder.eps(),
      weightDecay: Double = defaultsHolder.weight_decay(),
      amsgrad: Boolean = defaultsHolder.amsgrad(),
  ) {
    private[Adam] def toInternal: ManagedResource[internal.AdamOptions] = {
      def man(o: => internal.AdamOptions) = resource.makeManagedResource(o)(_.delete())(List.empty)
      for {
        orig <- man(new internal.AdamOptions(learningRate))
        afterBetaOne <- man(orig.beta1(beta1))
        afterBetaTwo <- man(afterBetaOne.beta2(beta2))
        afterEpsilon <- man(afterBetaTwo.eps(epsilon))
        afterAmsGrad <- man(afterEpsilon.amsgrad(amsgrad))
        afterWeightDecay <- man(afterAmsGrad.weight_decay(weightDecay))
      } yield afterWeightDecay;
    }
  }

  private def apply(underlying: internal.Adam): Adam = {
    Disposer.add(new Adam(underlying), () => underlying.delete())
  }

  def apply(parameters: Iterable[Tensor], options: Adam.Options = Adam.Options()): Adam = {
    options.toInternal.apply { opts =>
      val underlying = ReferenceManager.forBlock { implicit rm =>
        val tensorVector = new TensorVector(parameters.map(_.underlying).asJava)
        try new internal.Adam(tensorVector, opts)
        finally tensorVector.delete()
      }
      apply(underlying)
    }
  }

  def apply(m: Model, options: Adam.Options): Adam = ReferenceManager.forBlock { implicit rm =>
    apply(Optimizer.extractParameters(m), options)
  }

  def apply(m: Model, learningRate: Double): Adam = apply(m, Adam.Options(learningRate))

  private val defaultsHolder = new internal.AdamOptions(1.0)
}

class Adagrad private (override private[torch] val optimizer: internal.Adagrad) extends Optimizer(optimizer) {

  def learningRate: Double = {
    val options = optimizer.getOptions()
    try options.lr()
    finally options.delete()
  }

  def learningRate_=(newLearningRate: Double): this.type = {
    for (i <- 0 until optimizer.num_param_groups()) {
      internal.AdagradOptions.cast(optimizer.param_group(i).options()).lr(newLearningRate).delete()
    }
    this
  }
}

object Adagrad {
  case class Options(
      learningRate: Double = defaultsHolder.lr(),
      learningRateDecay: Double = defaultsHolder.lr_decay(),
      weightDecay: Double = defaultsHolder.weight_decay(),
  ) {
    private[Adagrad] def toInternal: ManagedResource[internal.AdagradOptions] = {
      def man(o: => internal.AdagradOptions) = resource.makeManagedResource(o)(_.delete())(List.empty)
      for {
        orig <- man(new internal.AdagradOptions(learningRate))
        afterLrDecay <- man(orig.lr_decay(learningRateDecay))
        afterWeightDecay <- man(afterLrDecay.weight_decay(weightDecay))
      } yield afterWeightDecay;
    }
  }

  private def apply(underlying: internal.Adagrad): Adagrad = {
    Disposer.add(new Adagrad(underlying), () => underlying.delete())
  }

  def apply(parameters: Iterable[Tensor], options: Adagrad.Options = Adagrad.Options()): Adagrad = {
    options.toInternal.apply { opts =>
      val underlying = ReferenceManager.forBlock { implicit rm =>
        val tensorVector = new TensorVector(parameters.map(_.underlying).asJava)
        try new internal.Adagrad(tensorVector, opts)
        finally tensorVector.delete()
      }
      apply(underlying)
    }
  }

  def apply(m: Model, options: Adagrad.Options): Adagrad = ReferenceManager.forBlock { implicit rm =>
    apply(Optimizer.extractParameters(m), options)
  }

  def apply(m: Model, learningRate: Double): Adagrad = apply(m, Adagrad.Options(learningRate))

  private val defaultsHolder = new internal.AdagradOptions(1.0)
}

class RMSProp private (override private[torch] val optimizer: internal.RMSprop) extends Optimizer(optimizer) {

  def learningRate: Double = {
    val options = optimizer.getOptions()
    try options.lr()
    finally options.delete()
  }

  def learningRate_=(newLearningRate: Double): this.type = {
    for (i <- 0 until optimizer.num_param_groups()) {
      RMSpropOptions.cast(optimizer.param_group(i).options()).lr(newLearningRate).delete()
    }
    this
  }
}

object RMSProp {
  case class Options(
      learningRate: Double = defaultsHolder.lr(),
      alpha: Double = defaultsHolder.alpha(),
      eps: Double = defaultsHolder.eps(),
      weightDecay: Double = defaultsHolder.weight_decay(),
      momentum: Double = defaultsHolder.momentum(),
      centered: Boolean = defaultsHolder.centered(),
  ) {
    private[RMSProp] def toInternal: ManagedResource[RMSpropOptions] = {
      def man(o: => RMSpropOptions) = resource.makeManagedResource(o)(_.delete())(List.empty)
      for {
        orig <- man(new RMSpropOptions(learningRate))
        afterAlpha <- man(orig.alpha(alpha))
        afterEpsilon <- man(afterAlpha.eps(eps))
        afterMomentum <- man(afterEpsilon.momentum(momentum))
        afterCentered <- man(afterMomentum.centered(centered))
        afterWeightDecay <- man(afterCentered.weight_decay(weightDecay))
      } yield afterWeightDecay;
    }
  }

  private def apply(underlying: internal.RMSprop): RMSProp = {
    Disposer.add(new RMSProp(underlying), () => underlying.delete())
  }

  def apply(parameters: Iterable[Tensor], options: RMSProp.Options = RMSProp.Options()): RMSProp = {
    options.toInternal.apply { opts =>
      val underlying = ReferenceManager.forBlock { implicit rm =>
        val tensorVector = new TensorVector(parameters.map(_.underlying).asJava)
        try new internal.RMSprop(tensorVector, opts)
        finally tensorVector.delete()
      }
      apply(underlying)
    }
  }

  def apply(m: Model, options: RMSProp.Options): RMSProp = ReferenceManager.forBlock { implicit rm =>
    apply(Optimizer.extractParameters(m), options)
  }

  def apply(m: Model, learningRate: Double): RMSProp = apply(m, RMSProp.Options(learningRate))

  private val defaultsHolder = new RMSpropOptions(0.1)
}
