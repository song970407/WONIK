import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    def __init__(self,
                 is_convex: bool,
                 negative_slope: float = None):
        super(LeakyReLU, self).__init__()
        if is_convex:
            if negative_slope is not None:
                assert negative_slope < 1, "If it is convex, then negative slope should be less than 1"
                self._negative_slope = negative_slope
            else:
                self._negative_slope = 0.2
        else:
            if negative_slope is not None:
                assert negative_slope > 1, "If it is concave, then negative slope should be greater than 1"
                self._negative_slope = negative_slope
            else:
                self._negative_slope = 2

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self._negative_slope)
