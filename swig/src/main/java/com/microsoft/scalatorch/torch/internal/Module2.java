package com.microsoft.scalatorch.torch.internal;

public interface Module2<Input1, Input2, Output> {
    Output forward(Input1 input1, Input2 input2);
}
