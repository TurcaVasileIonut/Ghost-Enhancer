import torch.nn.functional as F


class ActivationFunctions:
    @staticmethod
    def hard_sigmoid(x, inplace: bool = False):
        if inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.
