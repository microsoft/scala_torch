package com.microsoft.scalatorch.torch.jit

import java.io.{ Closeable, File }

import scala.collection.JavaConverters._

import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, QualifiedName, TypeMeta }
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.internal
import com.microsoft.scalatorch.torch.internal.{ torch_swig, QualifiedName, TypeMeta }
import com.microsoft.scalatorch.torch.util.Disposer
import resource.ManagedResource
import com.microsoft.scalatorch.torch._
import com.microsoft.scalatorch.torch.nn.init.ParameterInit
import com.microsoft.scalatorch.torch.util.Disposer
import com.microsoft.scalatorch.torch.syntax._

/** A [[Module]] is a wrapper around the torch::jit::script::Module, which is a deserialized
  * torchscript or traced module (from Python or potentially C++/Scala, though the latter don't support
  * tracing yet).
  */
class Module private[torch] (
    protected[torch] val underlying: internal.Module,
) extends TorchReference[internal.Module] {

  def getParameter(name: String)(implicit rm: ReferenceManager): Option[Tensor] = {
    val v = underlying.attr(name, IValue.none.underlying)
    try if (v.isNone) None else Some(Tensor(v.toTensor))
    finally v.delete()
  }

  def getModule(name: String)(implicit rm: ReferenceManager): Option[Module] = {
    val v = underlying.attr(name, IValue.none.underlying)
    try if (v.isNone) None else Some(rm.addReference(Module(v.toModule)))
    finally v.delete()
  }

  def name: String = underlying.name()

  def register_module(name: String, module: Module): Unit = {
    underlying.register_module(name, module.underlyingChecked)
  }

  def register_parameter(name: String, v: Tensor, is_buffer: Boolean = false): Unit = {
    underlying.register_parameter(name, v.underlying, is_buffer)
  }

  def register_attribute(
      name: String,
      t: Type,
      v: IValue,
      is_param: Boolean = false,
      is_buffer: Boolean = false,
  ): Unit = {
    underlying.register_attribute(name, t.underlying, v.underlying, is_param, is_buffer)
  }

  /** Like [[register_parameter]], but initializes a [[Tensor]] for you.
    *
    * TODO we might want to consider taking a [[TensorInfo]] instead of [[Size]]/[[Device]]/[[TypeMeta]],
    *   but this is more convenient for now.
    */
  def addParameter(
      shape: Size,
      name: String,
      init: ParameterInit =  com.microsoft.scalatorch.torch.nn.init.glorotUniform(false),
      device: Option[Device] = None,
      dtype: Option[dtype] = None,
  )(
      implicit rm: ReferenceManager,
  ): Tensor = {
    val tensor = Tensor.empty(
      shape,
      TensorOptions(
        requires_grad = Some(true),
        dtype = dtype,
        device = device,
      ),
    )(rm)
    init.initializeParams(tensor)

    // synchronized because addAttribute seems to be thread unsafe.
    underlying.synchronized {
      val tensorType = TensorType.create(shape)
      // Be extra cautious and use a global lock, just in case the underlying type might be shared across modules.
      // Not sure if this is actually happening, but we have observed some rare segfaults in
      // [libtorch_cpu.so]  c10::ClassType::addAttribute(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<c10::Type> const&, bool, bool)+0x319
      // and this is our best guess.
      Module.synchronized {
        val t = underlying.`type`()
        try t.addAttribute(name, tensorType.underlying, /*is_parameter=*/ true)
        finally t.delete()
      }

      val newValue = IValue.fromTensor(tensor)
      underlying.setattr(name, newValue.underlying)
    }
    tensor
  }

  def eval(): Unit = underlyingChecked.eval()

  def is_training(): Boolean = underlyingChecked.is_training()

  def train(on: Boolean = true): Unit = underlyingChecked.train(on)

  lazy val `type`: ClassType = ClassType(underlying.`type`())

  override protected def delete(): Unit = {
    underlyingChecked.delete()
  }

  def serializeToByteArray(): Array[Byte] = torch_swig.save_Module_to_byte_array(underlyingChecked)

  def forward(args: Seq[IValue])(implicit rm: ReferenceManager): IValue = {
    val iValueVector = new internal.IValueVector(args.map(_.underlying).asJava)
    try new IValue(underlyingChecked.forward(iValueVector))(rm)
    finally iValueVector.delete()
  }

  def invoke(methodName: String, args: IValue*)(implicit rm: ReferenceManager): IValue = {
    val iValueVector = new internal.IValueVector(args.map(_.underlying).asJava)
    try new IValue(underlying.run_method(methodName, iValueVector))(rm)
    finally iValueVector.delete()

  }

  def to(device: Device): Unit = underlyingChecked.to(device.underlyingChecked)

  def save(file: File): Unit = underlyingChecked.save(file.toString)
  def serialize(): Array[Byte] = internal.torch_swig.save_Module_to_byte_array(underlyingChecked)

  // TODO: there's no direct way to get a list of attributes out

  def attr(name: String)(implicit rm: ReferenceManager): Option[IValue] = {
    val uc = underlyingChecked
    if (uc.hasattr(name)) {
      Some(new IValue(uc.attr(name))(rm))
    } else {
      None
    }
  }

  def hasattr(name: String): Boolean = underlyingChecked.hasattr(name)

  def parameters(recurse: Boolean = true)(implicit rm: ReferenceManager): Iterable[Tensor] = {
    named_parameters(recurse).values
  }

  def named_parameters(recurse: Boolean = true)(implicit rm: ReferenceManager): Map[String, Tensor] = {
    val np = underlying.named_parameters(recurse)
    try {
      np.asScala.map(t => (t.getName, Tensor(t.value()))).toMap
    } finally {
      np.delete()
    }
  }

  def named_children(implicit rm: ReferenceManager): Map[String, Module] = {
    val nc = underlying.named_children()
    try {
      nc.asScala.map(t => (t.getName, Module(t.value()))).toMap
    } finally {
      nc.delete()
    }
  }
}

object Module {

  private[torch] def apply(underlying: internal.Module): Module = {
    Disposer.add(new Module(underlying), () => underlying.delete())
  }

  def apply(name: String): Module = {
    val qn = new QualifiedName(name)
    try {
      Module(new internal.Module(qn))
    } finally {
      qn.delete()
    }
  }

  def load(file: File, device: Option[Device] = None): Module = {
    Module(internal.torch_swig.load_script_module(file.toString, device.map(_.underlyingChecked)))
  }

  def load(data: Array[Byte]): Module = {
    Module(internal.torch_swig.load_Module_from_byte_array(data))
  }
}
