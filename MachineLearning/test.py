import torch
import torch.nn as nn
import torch.nn.functional as F 

# With Learnable Parameters
m = nn.BatchNorm1d(4)

input = torch.randn(2,4,3)
print("input: ", input)
output = m(input)


print("output: ", output)