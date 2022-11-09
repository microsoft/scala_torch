package com.microsoft.scalatorch.torch

import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch.syntax.{ ---, _ }
import org.scalatest._

class TensorTest extends FunSpec with BeforeAndAfterAll {
  import TensorTestUtils._

  implicit val rm = new ReferenceManager

  override def afterAll(): Unit = {
    rm.close()
  }

  describe("create") {
    it("can create from different types of scalars") {
      val a = Tensor.fromLongs(3, 4, 5)
      assert(a.dtype == torch.long)

      val b = Tensor(3.0f, 4.0f, 5.0f)
      assert(b.dtype == torch.float)
    }

  }

  describe("tensor functions") {
    val e1 = Tensor(1)
    val e2 = Tensor(2)
    val e3 = Tensor(3)

    it("arithmetic") {
      assertIsClose((-e1).toFloat, -1f)
      assertIsClose((e1 + e2).toFloat, 3f)
      assertIsClose((e1 + 10).toFloat, 11f)
      assertIsClose((10 + e1).toFloat, 11f)
      assertIsClose((e2 - e1).toFloat, 1f)
      assertIsClose((e2 - 10).toFloat, -8f)
      //isClose((      10 - e2).toFloat,8f)
      assertIsClose((e1 * e2).toFloat, 2f)
      assertIsClose((10 * e2).toFloat, 20f)
      assertIsClose((e2 * 10).toFloat, 20f)
      assertIsClose((e2 / 10).toFloat, 0.2f)
    }

    // affine transform
    describe("affine transform") {
      it("affine transform") {
        assertIsClose(nn.functional.linear(e1, e2.reshape(Size(1, 1)), e3).toFloat, 5)
      }
    }

    it("pow") {
      val sqrt2 = torch.sqrt(e2)
      assertIsClose((sqrt2 * sqrt2).toFloat, 2)

      assertIsClose(torch.pow(e2, e3).toFloat, 8)
      assertIsClose(torch.pow(e3, e2).toFloat, 9)
    }

    it("min/max") {
      assertIsClose(torch.min(e1, e3).toFloat, 1)
      assertIsClose(torch.max(e1, e3).toFloat, 3)
    }

    it("function that returns tuple") {
      val t = Tensor.fromFloatArray(Array(3f, 4f, 1f, 2f), Size(2, 2))
      val (max1, index1) = torch.max(t, 1)
      assert(max1.toArray[Float].toSeq == Seq(4f, 2f))
      assert(index1.toArray[Long].toSeq == Seq(1L, 1L))
      assert(max1.shape == Size(2))

      val (max2, index2) = torch.max(t, 0)
      assert(max2.toArray[Float].toSeq == Seq(3f, 4f))
      assert(index2.toArray[Long].toSeq == Seq(0L, 0L))

      val (max3, index3) = torch.max(t, 1, keepdim = true)
      assert(max3.toArray[Float].toSeq == Seq(4f, 2f))
      assert(index3.toArray[Long].toSeq == Seq(1L, 1L))
      assert(max3.shape == Size(2, 1))
    }

    // TODO: write more tests
  }

  describe("concatenate") {
    it("fail gracefully with empty list") {
      ReferenceManager.forBlock { implicit rm =>
        assertThrows[RuntimeException] {
          val foo = torch.cat(Array())
        }
      }
    }
  }

  it("transpose should work") {
    ReferenceManager.forBlock { implicit rm =>
      val e1 = Tensor.zeros(Size(10, 1))
      val transposed = torch.transpose(e1, 0, 1)
      assert(transposed.shape == Size(1, 10))

      val e2 = Tensor.zeros(Size(1, 10))
      val transposed2 = torch.transpose(e2, 0, 1)
      assert(transposed2.shape == Size(10, 1))
    }
  }

  it("lists of tensors should get converted to vectors") {
    def sum(tensors: Array[Tensor]): Tensor = {
      if (tensors.isEmpty) Tensor.zeros(Size()) else torch.stack(tensors).sum()
    }
    ReferenceManager.forBlock { implicit rm =>
      val exprs = for (i <- 1 to 100) yield Tensor(i.toFloat)

      val sums = for (i <- 1 to 50) yield sum(exprs.toArray)
      val expected = (1 to 100).sum

      sums.foreach(s => assertIsClose(s.toFloat, expected))

      val uberSum = sum((for {
        _ <- 1 to 1000
        i1 = scala.util.Random.nextInt(100)
        i2 = scala.util.Random.nextInt(100)
        i3 = scala.util.Random.nextInt(100)
      } yield sum(Array(exprs(i1), exprs(i2), exprs(i3)))).toArray)
      val value = uberSum.toFloat
      assert(value > 30f * 1000 * 3)
      assert(value < 70f * 1000 * 3)
    }
  }

  it("empty") {
    // This test fails if you the native code calls at::empty instead of torch::empty
    assert(Tensor.empty(Size(10), TensorOptions(requires_grad = true)).shape == Size(10))
    assert(torch.empty(Size(10), layout = Layout.Sparse).layout == Layout.Sparse)
  }

  it("normal") {
    // This one requires special treatment in swig so we give it a special test
    assert(torch.normal(1, 1, Array(1, 1), dtype = torch.float16).dtype == torch.float16)
  }

  it("sum should basically function") {
    ReferenceManager.forBlock { implicit rm =>
      val expr = Tensor(Array.range(0, 100).map(_.toFloat): _*)
      val total = expr.sum()

      assertIsClose(total.toFloat, (0 until 100).map(_.toFloat).sum)
    }
  }

  it("index") {
    // index is special because it does the sketchy stuff with c10::List<c10::optional<Tensor>>
    ReferenceManager.forBlock { implicit rm =>
      val expr = Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), Size(10, 10))
      val indexedOnce = expr.index(Array(Some(Tensor.fromLongArray(Array(0L, 3L), Size(2))), None))
      assert(
        indexedOnce == Tensor
          .fromFloatArray(Array.range(0, 10).map(_.toFloat) ++ Array.range(30, 40).map(_.toFloat), Size(2, 10)),
      )
      val indexedTwice = expr.index(
        Array(Some(Tensor.fromLongArray(Array(0L, 3L), Size(2))), Some(Tensor.fromLongArray(Array(1L, 4L), Size(2)))),
      )
      assert(indexedTwice == Tensor(1f, 34f))

    // This crashes, but I think it's a bug in Torch.
    // assert(expr.index(Array(None, None)) == expr)
    }
  }

  describe("indexing and slicing") {
    it("single coords") {
      ReferenceManager.forBlock { implicit rm =>
        val expr = Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), Size(10, 10))
        assert(expr(1, 1).toFloat == 11)
        expr(1, 1) = Tensor(-11)
        assert(expr(1, 1).toFloat == -11)
        expr(::) = torch.zeros(Array(10, 10))
        assert(expr == torch.zeros(Array(10, 10)))
      }
    }
    it("other slicers") {
      ReferenceManager.forBlock { implicit rm =>
        val expr = Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat), Size(10, 10))
        assert(expr(1, None) == Tensor.fromFloatArray(Array.range(10, 20).map(_.toFloat), Size(10)))
        assert(expr(None, 1) == Tensor.fromFloatArray(Array.range(0, 10).map(x => x * 10f + 1), Size(10)))
        assert(expr(::, 1) == Tensor.fromFloatArray(Array.range(0, 10).map(x => x * 10f + 1), Size(10)))
        assert(expr(1, 5.::) == Tensor.fromFloatArray(Array.range(15, 20).map(_.toFloat), Size(5)))
        // If you really want the python syntax
        import scala.language.postfixOps
        assert(expr(1, 5 ::) == Tensor.fromFloatArray(Array.range(15, 20).map(_.toFloat), Size(5)))
        // format: off
        assert(expr(1, 5::) == Tensor.fromFloatArray(Array.range(15, 20).map(_.toFloat), Size(5)))
        // format: on

        assert(expr(1, 5 -> 8) == Tensor(15f, 16f, 17f))
        assert(expr(1, 5 -> -2) == Tensor(15f, 16f, 17f))
        assert(expr(1, 5 -> 8 -> 2) == Tensor(15f, 17f))
        assert(expr(1, 5 :: 2) == Tensor(15f, 17f, 19f))

        val expr3 = Tensor.fromFloatArray(Array.range(0, 1000).map(_.toFloat), Size(10, 10, 10))
        assert(expr3(1, ---) == Tensor.fromFloatArray(Array.range(100, 200).map(_.toFloat), Size(10, 10)))
        assert(expr3(---, 1) == Tensor.fromFloatArray(Array.range(0, 100).map(_.toFloat).map(_ * 10 + 1), Size(10, 10)))
      }
    }
  }

  it("torch.tensor") {
    assert(torch.tensor(1) == Tensor.fromIntArray(Array(1), Size()))
    assert(torch.tensor(Array(1)) == Tensor.fromIntArray(Array(1), Size(1)))
    assert(torch.tensor(Array(1, 2, 3)) == Tensor.fromIntArray(Array(1, 2, 3), Size(3)))
    assert(torch.tensor(Array(Array(1, 2), Array(3, 4))) == Tensor.fromIntArray(Array(1, 2, 3, 4), Size(2, 2)))
    assert(
      torch.tensor(Array(Array(Array(1, 2), Array(3, 4)), Array(Array(1, 2), Array(3, 4)))) == Tensor
        .fromIntArray(Array(1, 2, 3, 4, 1, 2, 3, 4), Size(2, 2, 2)),
    )

    assert(
      torch.tensor(Array(Array(Array(1, 2, 3), Array(3, 4, 5)), Array(Array(1, 2, 3), Array(3, 4, 5)))) == Tensor
        .fromIntArray(Array(1, 2, 3, 3, 4, 5, 1, 2, 3, 3, 4, 5), Size(2, 2, 3)),
    )

    assertThrows[IllegalArgumentException](torch.tensor(Array(Array(1, 2), Array(3))))
    assertThrows[IllegalArgumentException](torch.tensor(Array.empty[Int]))

    assert(torch.tensor(Array(1f, 2f, 3f)) == Tensor.fromFloatArray(Array(1f, 2f, 3f), Size(3)))
    assert(torch.tensor(Array(1.0, 2.0, 3.0)) == Tensor.fromDoubleArray(Array(1.0, 2.0, 3.0), Size(3)))
    assert(torch.tensor(Array(1L, 2L, 3L)) == Tensor.fromLongArray(Array(1L, 2L, 3L), Size(3)))
    assert(torch.tensor(Array[Byte](1, 2, 3)) == Tensor.fromByteArray(Array[Byte](1, 2, 3), Size(3)))

  }

}

object TensorTestUtils extends Assertions {
  private val DefaultEpsilon = 1e-4f

  def assertIsClose(value: Float, expected: Float, epsilon: Float = DefaultEpsilon): Assertion = {
    assert((value - expected).abs <= epsilon)
  }
}
