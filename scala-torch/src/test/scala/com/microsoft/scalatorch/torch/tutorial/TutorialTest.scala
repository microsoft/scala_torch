package com.microsoft.scalatorch.torch.tutorial

import com.microsoft.scalatorch.torch.ReferenceManager

class TutorialTest extends org.scalatest.funspec.AnyFunSpec {

  it("functions with simple calls ") {
    import com.microsoft.scalatorch.torch
    // There should always be a ReferenceManager in implicit scope for memory management.
    ReferenceManager.forBlock { implicit rm =>
      val tensor: torch.Tensor = torch.eye(2)
      assert(tensor.numel() == 4)
      assert(tensor.size() == torch.Size(2, 2))

      // OOP style
      assert(tensor.sum().toFloat == 2f)
      // Static function style
      assert(torch.sum(tensor).toFloat == 2f)

      // Unfortunately, Scala does not allow multiple overloads with default values, so for some overloads,
      // you must redundantly specify defaults.
      assert(tensor.sum(dim = Array(1L), false, None) == torch.ones(Array(2L)))
    }
  }

  describe("Python-like slicing syntax") {
    it("works for reads") {
      import com.microsoft.scalatorch.torch
      import com.microsoft.scalatorch.torch.syntax._
      ReferenceManager.forBlock { implicit rm =>
        val expr = torch.Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), torch.Size(10, 10))
        assert(expr(1, 1).toFloat == 11)

        assert(expr(1, None) == torch.Tensor.fromFloatArray(Array.range(10, 20).map(_.toFloat), torch.Size(10)))
        assert(expr(None, 1) == torch.Tensor.fromFloatArray(Array.range(0, 10).map(x => x * 10f + 1), torch.Size(10)))
        assert(expr(::, 1) == torch.Tensor.fromFloatArray(Array.range(0, 10).map(x => x * 10f + 1), torch.Size(10)))

        assert(expr(1, 5 -> 8) == torch.Tensor(15f, 16f, 17f))
        assert(expr(1, 5 -> -2) == torch.Tensor(15f, 16f, 17f))
        assert(expr(1, 5 -> 8 -> 2) == torch.Tensor(15f, 17f))

        assert(expr(1, 5 :: 2) == torch.Tensor(15f, 17f, 19f))

        val expr2 = torch.Tensor.fromFloatArray(Array.range(0, 1000).map(_.toFloat), torch.Size(10, 10, 10))
        assert(expr2(1, ---) == torch.Tensor.fromFloatArray(Array.range(100, 200).map(_.toFloat), torch.Size(10, 10)))
        assert(
          expr2(---, 1) == torch.Tensor
            .fromFloatArray(Array.range(0, 100).map(_.toFloat).map(_ * 10 + 1), torch.Size(10, 10)),
        )
      }
    }

    it("works for writes") {
      import com.microsoft.scalatorch.torch
      import com.microsoft.scalatorch.torch.syntax._
      ReferenceManager.forBlock { implicit rm =>
        val expr = torch.Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), torch.Size(10, 10))
        assert(expr(1, 1).toFloat == 11)
        expr(1, 1) = torch.Tensor(-11)
        assert(expr(1, 1).toFloat == -11)
        expr(::) = torch.zeros(Array(10, 10))
        assert(expr == torch.zeros(Array(10, 10)))
      }
    }
  }
  it("is possible with dot postfix") {
    import com.microsoft.scalatorch.torch
    import com.microsoft.scalatorch.torch.syntax._
    ReferenceManager.forBlock { implicit rm =>
      val expr = torch.Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), torch.Size(10, 10))
      // You can write `5 ::` instead of `5.::` if you enable postfixOps (see below)
      assert(expr(1, 5.::) == torch.Tensor.fromFloatArray(Array.range(15, 20).map(_.toFloat), torch.Size(5)))
    }
  }
  it("is possible with space postfix") {
    import com.microsoft.scalatorch.torch
    import com.microsoft.scalatorch.torch.syntax._
    // If you really want the python syntax
    import scala.language.postfixOps
    ReferenceManager.forBlock { implicit rm =>
      val expr = torch.Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), torch.Size(10, 10))
      // format: off
        assert(expr(1, 5 ::) == torch.Tensor.fromFloatArray(Array.range(15, 20).map(_.toFloat), torch.Size(5)))
        // format: on
    }
  }
}
