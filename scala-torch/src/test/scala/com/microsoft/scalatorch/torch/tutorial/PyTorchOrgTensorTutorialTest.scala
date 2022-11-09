package com.microsoft.scalatorch.torch.tutorial

import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch.ReferenceManager
import com.microsoft.scalatorch.torch.syntax._

/** Tries to follow https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
  *
  * See also [[TutorialTest]] in this same directory.
  */
object PyTorchOrgTensorTutorial {
  def main(args: Array[String]): Unit = {
    ReferenceManager.forBlock { implicit rm =>
      val data = $($(1, 2), $(3, 4))
      val x_data = torch.tensor(data)
      assert(x_data.sum() == torch.tensor(10L))

      val x_ones = torch.ones_like(x_data)
      // retains the properties of x_data
      println(s"Ones Tensor: \n ${x_ones} \n")

      val x_rand = torch.rand_like(x_data, dtype = torch.float)
      // overrides the datatype of x_data
      println(s"Random Tensor: \n ${x_rand} \n")

      val shape = (2, 3)
      val rand_tensor = torch.rand(shape)
      val ones_tensor = torch.ones(shape)
      val zeros_tensor = torch.zeros(shape)

      println(s"Random Tensor: \n ${rand_tensor} \n")
      println(s"Ones Tensor: \n ${ones_tensor} \n")
      println(s"Zeros Tensor: \n ${zeros_tensor}")

      var tensor = torch.rand($(3, 4))

      println(s"Shape of tensor: ${tensor.shape}")
      println(s"Datatype of tensor: ${tensor.dtype}")
      println(s"Device tensor is stored on: ${tensor.device}")

      if (torch.cuda.is_available())
        tensor = tensor.to(device = "cuda")

      // Standard numpy-like indexing and slicing
      tensor = torch.ones($(4, 4))
      println(s"First row: ${tensor(0)}")
      println(s"First column: ${tensor(::, 0)}")
      println(s"Last column: ${tensor(---, -1)}")
      tensor(::, 1) = 0
      println(tensor)

      val t1 = torch.cat($(tensor, tensor, tensor), dim = 1)
      println(t1)

      // Arithmetic operations

      // This computes the matrix multiplication between two tensors
      // y1, y2, y3 will have the same value
      val y1 = tensor *@* tensor.T
      val y2 = tensor.matmul(tensor.T)

      val y3 = torch.rand_like(y1)
      // out params not supported yet
      // torch.matmul(tensor, tensor.T, out = y3)
      val out = torch.matmul(tensor, tensor.T)

      // This computes the element -wise product
      // z1, z2, z3 will have the same value
      val z1 = tensor * tensor
      val z2 = tensor.mul(tensor)

      val z3 = torch.rand_like(tensor)
      // out params not supported yet
      // torch.mul(tensor, tensor, out = z3)
      val out2 = torch.mul(tensor, tensor)

      val agg = tensor.sum()
      val agg_item = agg.item()
      println(s"${agg_item}, ${agg_item.getClass}")

      // In-place operations
      println(s"${tensor} \n")
      tensor.add_(5)
      println(tensor)
    }
  }
}
