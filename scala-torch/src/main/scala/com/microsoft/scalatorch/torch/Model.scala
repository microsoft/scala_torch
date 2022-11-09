package com.microsoft.scalatorch.torch

import java.io.File

import scala.collection.JavaConverters._

import com.microsoft.scalatorch.torch.internal.{ torch_swig, Layout }
import com.microsoft.scalatorch.torch.jit.{ ClassType, Module }
import com.microsoft.scalatorch.torch.util.NoGrad
import resource.ManagedResource
import com.microsoft.scalatorch.torch.init.ParameterInit
import syntax._

/** A wrapper of a Torch Module representing the "root" of the [[Module]] tree.
  *
  * A note about memory ownership: this class contains a [[ReferenceManager]] for managing all parameters
  * stored internally. When you register a [[Parameter]] or [[Module]], you pass a factory that accepts a
  * [[ReferenceManager]] and returns a new [[Tensor]] or [[Module]]. The factory is responsible for adding
  * itself to the [[ReferenceManager]], but note that methods that make [[Tensor]]s and [[Module]]s typically
  * have a signature that takes an implicit [[ReferenceManager]] -- for example, [[Tensor.fromLongArray]],
  * and so you simply need to pass that method with the implicit parameter curried.
  *
  * Somewhat confusingly, the "get" methods like [[getParameter]] take an implicit [[ReferenceManager]] that
  * owns the wrapper created by [[getParameter]], but not the underlying storage. Typically, you can pass
  * a temporary manager created by [[ReferenceManager.forBlock]] to manage their storage, assuming
  * you only use the return [[Tensor]] temporarily of course.
  *
  * TODO it's unclear if we need this vs just using [[Module]]s directly, but for now it helps organize memory ownership.
  *
  * @param module underlying PyTorch [[Module]]
  * @see https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html?highlight=module
  * @see
  */
class Model private[torch] (
    private[torch] val module: jit.Module,
) extends java.io.Closeable {

  def save(filename: String): Unit = module.save(new File(filename))

  private[torch] val owner: ReferenceManager = new ReferenceManager {}

  /** The [[ClassType]] of the inner [[Module]] */
  lazy val classType: ClassType = module.`type`

  owner.addReference(module)

  /** Put the model into train mode. See https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train */
  def train(on: Boolean = true): Unit = module.train(on)

  def isTraining(): Boolean = module.is_training()

  /** Put the model into eval mode. See https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.eval */
  def eval(): Unit = module.eval()

  def addParameters(
      shape: Size,
      name: String,
      init: ParameterInit = com.microsoft.scalatorch.torch.init.glorotUniform(false),
  ): Tensor = {
    module.addParameter(shape, name, init)(owner)
  }

  def registerParameter(name: String, tensor: ReferenceManager => Tensor): Tensor = {
    val newTensor = tensor(owner)
    module.register_parameter(name, newTensor)
    newTensor
  }

  def registerModule[M <: jit.Module](name: String, childModule: ReferenceManager => M): M = {
    val newModule = childModule(owner)
    module.register_module(name, newModule)
    newModule
  }

  def getParameter(name: String)(implicit manager: ReferenceManager): Option[Tensor] = {
    module.getParameter(name)
  }

  def getModule(name: String)(implicit manager: ReferenceManager): Option[jit.Module] = {
    module.getModule(name)
  }

  /** Should be efficient for the 2-norm on sparse gradients. No guarantees for other norms. */
  def gradNorm(p: Float = 2f): Double = NoGrad.noGrad {
    ReferenceManager.forBlock { implicit rm =>
      // TODO this might not be efficient for (sparse) Embeddings
      val parameters = module.parameters(recurse = true)
      parameters.map { param =>
        val grad = param.grad
        if (!grad.underlying.defined()) {
          0f
        } else if (p == 2f && grad.underlying.layout() == Layout.Sparse) {
          // TODO does this work?
          // TODO handle other sparse norms. If we can get access to the undelrying
          //   sparse tensor we call norm on that directly.
          val twoNorm = torch_swig._sparse_sum((grad * grad).underlying)
          try Math.sqrt(twoNorm.toFloat)
          finally twoNorm.delete()
        } else {
          val tensor = torch_swig.norm(grad.underlying, Scalar.fromFloat(p).underlying)
          try tensor.toFloat
          finally tensor.delete()
        }
      }.sum
    }
  }

  def serializeToByteArray(): Array[Byte] = {
    module.serializeToByteArray()
  }

  override def close(): Unit = {
    owner.close()
  }
}

object Model {

  def managed(name: String): ManagedResource[Model] = {
    resource.makeManagedResource(new Model(Module(name)))(_.close())(List.empty)
  }

  private[torch] def apply(name: String): Model = {
    new Model(Module(name))
  }

  def loadFromByteArray(array: Array[Byte]): Model = {
    new Model(jit.Module.load(array))
  }
}
