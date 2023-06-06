import torch
import numpy as np

from torch import Tensor

x: Tensor = torch.empty(2, 3, 3)
a = x.numpy()
y: Tensor = torch.rand(2, 2)
print(x, y)

print(y.view(4))

print(a)
