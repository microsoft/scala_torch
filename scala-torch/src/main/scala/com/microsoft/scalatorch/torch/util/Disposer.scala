/*
 * This class is largely copied from
 * https://github.com/eaplatanios/tensorflow_scala/blob/master/modules/api/src/main/scala/org/platanios/tensorflow/api/utilities/Disposer.scala
 * with some minor modifications.
 *
 * Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package com.microsoft.scalatorch.torch.util

import java.lang.Thread.currentThread
import java.lang.ref.{ PhantomReference, ReferenceQueue }
import java.security.{ AccessController, PrivilegedAction }
import java.util
import java.util.concurrent.ConcurrentHashMap

import scala.annotation.tailrec

/** This class is used for registering and disposing the native data associated with Scala objects.
  * It is almost identical to [[Cleaner]], except that it uses a [[ConcurrentHashMap]] to store references
  * to avoid the locking in [[Cleaner]].
  *
  * The object can register itself by calling the [[Disposer.add]] method and providing a disposing function to it. This
  * function will be called in order to dispose the native data. It accepts no arguments and returns nothing.
  *
  * When the object becomes unreachable, the provided disposing function for that object will be called.
  * Note that because the garbage collector does not know how much memory the native objects take up, it may accumulate
  * many of them before triggering a GC. We make use of [[ReferenceManager]]s to manage the lifetime of frequently
  * allocated objects like [[Tensor]]s, but rely on the [[Disposer]] for objects that are unlikely to result in
  * memory pressure, but still should be cleaned up when possible.
  */
private[torch] object Disposer {

  private val queue: ReferenceQueue[Any] = new ReferenceQueue[Any]
  private val records: util.Map[PhantomReference[Any], () => Unit] =
    new ConcurrentHashMap[PhantomReference[Any], () => Unit]

  /** Performs the actual registration of the target object to be disposed.
    * Somewhat confusingly, the `disposer` argument *cannot* be retrieved out of `target` or reference it in any way
    * because by the time `disposer` runs, `target` will already have been garbage collected
    * (hence the slightly funny interface).
    *
    * The typical pattern is that every Swig-generated type internal.Foo will have a wrapper class called
    * Foo with an apply method that looks like
    * {{{
    * object Foo {
    *   def apply(arg1: Int, arg2: String): Foo = {
    *     val underlying = new internal.Foo(arg1, arg2)
    *     Disposer.add(new Foo(underlying), () => underlying.delete())
    *   }
    * }
    * }}}
    *
    * @param target   Wrapper object that manages the lifetime of the underlying object
    * @param disposer Closure that will clean up any underlying memory.
    * @return target for easier chaining.
    */
  def add(target: AnyRef, disposer: () => Unit): target.type = {
    val reference = new PhantomReference[Any](target, queue)
    records.put(reference, disposer)
    target
  }

  AccessController.doPrivileged(new PrivilegedAction[Unit] {
    override def run(): Unit = {
      // The thread must be a member of a thread group which will not get GCed before the VM exit. For this reason, we
      // make its parent the top-level thread group.
      @tailrec def rootThreadGroup(group: ThreadGroup = currentThread.getThreadGroup): ThreadGroup = {
        group.getParent match {
          case null   => group
          case parent => rootThreadGroup(parent)
        }
      }

      new Thread(rootThreadGroup(), "Torch Disposer") {
        override def run(): Unit = while (true) {
          // Blocks until there is a reference in the queue.
          val referenceToDispose = queue.remove
          records.remove(referenceToDispose).apply()
          referenceToDispose.clear()
        }

        setContextClassLoader(null)
        setDaemon(true)
        // Let Cleaner, which runs at priority MAX_PRIORITY - 2, take precedence.
        setPriority(Thread.MAX_PRIORITY - 3)
        start()
      }
    }
  })
}
