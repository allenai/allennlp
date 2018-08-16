from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from torch.nn import Module


class VariationalDropoutFn(InplaceFunction):
    """" (untested) VariationalDropoutFn made largely by copy-pasting torch's dropout function."""

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), 1, input.size(2))

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None


class VariationalDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(VariationalDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return VariationalDropoutFn.apply(input, self.p, self.training, self.inplace)


