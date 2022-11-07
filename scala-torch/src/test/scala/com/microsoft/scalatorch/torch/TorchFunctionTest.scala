package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch.syntax._
import org.scalatest.FunSpec

class TorchFunctionTest extends FunSpec {

  describe("loss functions") {
    it("should accept long as second arg") {
      ReferenceManager.forBlock { implicit rm =>
        assert(
          nn.functional.nll_loss(Tensor(-1.0f, -3.0f, -4.0f).reshape(Size(1, 3)), Tensor.fromLongs(1)).toFloat == 3.0f,
        )
      }
    }

    it("should accept Reduction") {
      ReferenceManager.forBlock { implicit rm =>
        assert(
          nn.functional
            .poisson_nll_loss(Tensor(1.0f, 3.0f, 4.0f).reshape(Size(1, 3)), Tensor.fromLongs(1))
            .toFloat == 1.8383645f,
        )
        assert(
          nn.functional
            .binary_cross_entropy(
              Tensor(0f, 0.5f, 1f).reshape(Size(1, 3)),
              Tensor.ones(Size(1)),
              reduction = Reduction.Mean,
            )
            .toFloat == 33.56438f,
        )
      }
    }
  }

  describe("packages") {
    it("linear") {
      ReferenceManager.forBlock { implicit rm =>
        TensorTestUtils.assertIsClose(linalg.norm(Tensor(1.0f, 3.0f, 4.0f)).toFloat, 5.09902f)
      }
    }
    it("special") {
      ReferenceManager.forBlock { implicit rm =>
        assert(special.log1p(Tensor(1.0f, 3.0f, 4.0f)).toArray[Float].toSeq == Seq(0.6931472f, 1.3862944f, 1.609438f))
      }
    }

    it("fft") {
      ReferenceManager.forBlock { implicit rm =>
        assert(fft.fft(Tensor(1.0f, 3.0f, 4.0f)).apply(0).real.toFloat == 8.0000f)
      }
    }
  }

}
