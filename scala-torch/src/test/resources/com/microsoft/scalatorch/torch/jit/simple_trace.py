import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(N, M))

    def forward(self, input):
        return self.weight.mv(input)

    @torch.jit.export
    def foo(self, a: float): return a + 4

my_module = MyModule(3,4)
sm = torch.jit.script(my_module)
sm.save("traced_model.pt")
