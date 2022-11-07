package com.microsoft.scalatorch.torch.jit

import java.io.File

import com.microsoft.scalatorch.torch._
import org.scalatest.FunSpec

class ModuleTest extends FunSpec {
  describe("Module") {
    lazy val f = new File(getClass.getResource("traced_model.pt").toURI)
    it("should load") {
      ReferenceManager.forBlock { implicit rm =>
        val m = Module.load(f)
        val t = Tensor(3.0f, 4.0f, 5.0f, 6.0f)
        val r = m.forward(Seq(t))
        // the actual script generates a 3x4 ones matrix
        val o = Tensor.ones(Size(3, 4))
        assert(r.asTensor == (o.matmul(t)))
      }

    }

    it("should allow us to invoke other methods") {
      ReferenceManager.forBlock { implicit rm =>
        val m = Module.load(f)
        val r = m.invoke("foo", 4.0)
        assert(r.asDouble == 8)
      }
    }
  }
}
