from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from torch.nn import Module


class VariationalDropoutFn(InplaceFunction):
    """" VariationalDropoutFn from torch's dropout function."""

    @staticmethod
    def _make_noise(input_tensor):
        return input_tensor.new().resize_(input_tensor.size(0), 1, input_tensor.size(2))

    @classmethod
    def forward(cls, ctx, input_tensor, dropout_prob=0.5, train=False, inplace=False):
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout_prob))
        ctx.p = dropout_prob
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input_tensor)
            output = input_tensor
        else:
            output = input_tensor.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input_tensor)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input_tensor)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None


class VariationalDropout(Module):
    def __init__(self, dropout_prob=0.5, inplace=False):
        super(VariationalDropout, self).__init__()
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout_prob))
        self.dropout_prob = dropout_prob
        self.inplace = inplace

    def forward(self, input_tensor):
        return VariationalDropoutFn.apply(input_tensor, self.dropout_prob, self.training, self.inplace)
