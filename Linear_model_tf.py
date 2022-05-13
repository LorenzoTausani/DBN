import torch

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=28*28, output_dim=10):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

  def forward(self, x):
    x = self.linear(x)
    return x
