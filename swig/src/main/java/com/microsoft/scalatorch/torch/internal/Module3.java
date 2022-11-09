package com.microsoft.scalatorch.torch.internal;

public interface Module3<Input1, Input2, Input3, Output> {
    Output forward(Input1 input1, Input2 input2, Input3 input3);
}
