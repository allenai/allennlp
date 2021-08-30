import torch
from typing import Tuple


@torch.jit.script
class MyClass(object):
    def __init__(
        self,
        weights=(
            1.0,
            1.0,
            1.0,
            1.0,
        ),
    ):
        # type: (Tuple[float, float, float, float])
        self.weights = weights

    def apply(self):
        # type: () -> Tuple[float, float, float, float]
        return self.weights


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.class_field = torch.jit.export(
            MyClass(
                weights=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )
            )
        )

    def forward(self, x):
        self.class_field.apply()
        return x + 10


m = torch.jit.script(MyModule())
