package com.microsoft.scalatorch.torch

import scala.collection.mutable.ArrayBuffer
import java.lang.ref._

import resource.{ DefaultManagedResource, ManagedResource }

/** LibTorch uses ref counting to do its memory management, which doesn't interact
  * the best with JVM's GC. We use this class to keep track of "generations" of
  * related Tensors and the like.
  *
  * There are three common uses:
  * * one for [[Model]]s, whose parameter tensors typically live a long time,
  * * one (conceptually) for a "computation graph" that represents
  *   one coherent unit of execution (e.g. one minibatch of examples), and
  * * one for a short lived objects (see uses of [[ReferenceManager.forBlock]]).
  *
  * The general pattern in Scala-Torch is that any function that returns a torch object ([[Tensor]], [[Scalar]],
  * etc) takes an implicit [[ReferenceManager]] as an input and the callee adds its result to the provided
  * [[ReferenceManager]] to pass ownership out to the caller.
  */
class ReferenceManager extends AutoCloseable {
  private[torch] def addReference(reference: AnyRef): reference.type = {
    assertOpen(reference)
    references += reference
    reference
  }

  private def assertOpen(reference: AnyRef): Unit = {
    assert(!isClosed, s"attempt to register $reference against closed owner $this")
  }

  def close(): Unit = {
    references.foreach {
      case r: AutoCloseable => r.close()
      case _                => ()
    }
    references.clear()
    closed = true
  }

  override protected def finalize(): Unit = {
    if (!isClosed && hasReferences) {
      System.err.println(s"Failed to close reference owner $this. garbage collecting")
      close()
    }
  }

  private def hasReferences = references.nonEmpty

  def isClosed: Boolean = closed

  // TODO: is this true in torch?
  // Tensors sometimes rely on things (e.g. wrapped C++ vectors) that get deleted when the JVM
  // garbage collector runs. By explicitly grabbing references to them, we can prevent this
  // premature garbage collection.
  private val references: ArrayBuffer[AnyRef] = ArrayBuffer()
  private var closed = false

}

object ReferenceManager {

  /** This function can be used to automatically delete the [[ReferenceManager]] after the block
    *
    * Use this function with:
    * {{{
    *  forBlock { implicit cg => do stuff }
    * }}}
    */
  def forBlock[T](f: ReferenceManager => T): T = {
    managed.apply(f)
  }

  def managed[T]: ManagedResource[ReferenceManager] = {
    resource.makeManagedResource(new ReferenceManager)(_.close())(List.empty)
  }

  /** This is a global reference manager if you don't want to use scoped memory management.
    * If you're used to Dynet, using scoped memory management isn't *quite* as necessary as
    * with PyTorch: Torch doesn't use arenas to do memory management, just a bunch of
    * reference counted pointers, and the ReferenceManager only holds weak references
    * to tensors and the like, so collection can happen when it's still "in scope"
    *
    * Nevertheless, it's a good idea to use a scoped ReferenceManager when convenient:
    * The JVM doesn't "feel" the native memory pressure created by keeping a bunch
    * of "heavy" pointers. As a simple example, a basic MNIST experiment was using something
    * like 5 gigs of memory with just a global manager, while switching to scoped
    * memory management got it down to 1 gig or so (most of which was JVM heap).
    *
    * You can also call System.gc() though that's not guaranteed to do anything.
    */
  val global = new ReferenceManager

  object Implicits {
    implicit val global: ReferenceManager = ReferenceManager.global
  }
}

/** A reference that can be managed by [[ReferenceManager]] */
trait TorchReference[+Underlying] extends AutoCloseable {

  /** Not thread safe. */
  final def close(): Unit = {
    if (!deleted) {
      delete()
      deleted = true
    }
  }
  private var deleted = false
  def isClosed: Boolean = deleted

  /** Should actually free the resource */
  protected def delete(): Unit

  protected[torch] def assertOpen[T](body: => T): T = {
    if (isClosed) throw new IllegalStateException("Attempt to access closed resource " + this.getClass.getName())
    body
  }

  def underlyingChecked: Underlying = assertOpen(underlying)

  protected[torch] def underlying: Underlying
}

object TorchReference {
  protected[torch] def assertOpen[T](refs: TorchReference[_]*)(body: => T): T = {
    refs.foreach(ref => if (ref.isClosed) ref.assertOpen("blah"))
    body
  }

  // TODO we should add some helpers that make it easy to cleanup native objects
  // with syntax like {{{using(method.makeSwigObject())(x => {...})}}}
  // that expands to
  // {{{
  //   val x = method.makeSwigObject()
  //   try {...}
  //   finally x.delete()
  // }}}
  // Unfortunately this is hard without using reflection because swig objects don't inherit from a common
  // trait (like AutoCloseable that exposes close()).
}
