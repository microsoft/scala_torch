package com.microsoft.scalatorch.torch.tutorial

import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch.optim.SGD
import com.microsoft.scalatorch.torch.{ Reduction, ReferenceManager, Tensor }
import com.microsoft.scalatorch.torch.syntax._

/** Follows the example at https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
  * Same as [[PytorchOrgPolynomialNetworkTutorial]], but uses an [[com.microsoft.scalatorch.torch.optim.Optimizer]]
  * instead of manual updates */
object PytorchOrgPolynomialNetworkOptimTutorial {
  def main(args: Array[String]): Unit = ReferenceManager.forBlock { implicit rm =>
    // Create Tensors to hold input and outputs.
    val x = torch.linspace(-Math.PI, Math.PI, Some(2000))
    val y = torch.sin(x)

    // For this example, the output y is a linear function of(x, x ^ 2, x ^ 3), so
    // we can consider it as a linear layer neural network
    // Let 's prepare the tensor(x, x ^ 2, x ^ 3).
    val p = torch.tensor($(1, 2, 3))
    val xx = x.unsqueeze(-1).pow(p)

    // In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
    // (3,), for this case, broadcasting semantics will apply to obtain a tensor
    // of shape (2000, 3)
    //
    // Here we depart from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html because
    // we have not (yet) made a wrapper for Modules. We manage the weight and bias parameters manually.
    val weight = torch.ones($(1, 3))
    val bias = torch.ones($(1))
    weight.requires_grad_(true)
    bias.requires_grad_(true)
    val parameters = Seq(weight, bias)
    val model = (x: torch.Tensor) => {
      torch.flatten(torch.nn.functional.linear(x, weight, Some(bias)), 0, 1)
    }

    // The nn package also contains definitions of popular loss functions; in this
    // case we will use Mean Squared Error (MSE) as our loss function.
    val loss_fn = torch.nn.functional.mse_loss(_, _, reduction = Reduction.Sum)
    val optim = SGD(parameters, SGD.Options(learningRate = 1e-6))
    for (t <- 0 until 2000) {

      // Forward pass: compute predicted y by passing x to the model. Module objects
      // override the __call__ operator so you can call them like functions. When
      // doing so you pass a Tensor of input data to the Module and it produces
      // a Tensor of output data.
      val y_pred = model(xx)
      // Compute and print loss. We pass Tensors containing the predicted and true
      // values of y, and the loss function returns a Tensor containing the
      // loss.
      val loss = loss_fn(y_pred, y)
      if (t % 100 == 99)
        println(s"iteration=$t loss=${loss.item()}")

      optim.zeroGrad()

      // Backward pass: compute gradient of the loss with respect to all the learnable
      // parameters of the model. Internally, the parameters of each Module are stored
      // in Tensors with requires_grad=True, so this call will compute gradients for
      // all learnable parameters in the model.
      loss.backward()
      // Update the weights using gradient descent. Each parameter is a Tensor, so
      // we can access its gradients like we did before.
      optim.step()

      println(
        s"Result: y = ${bias.item()} + ${weight(::, 0).item()} x + ${weight(::, 1).item()} x^2 + ${weight(::, 2).item()} x^3",
      )
    }
  }
}
