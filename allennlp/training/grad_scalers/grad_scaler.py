from torch.cuda import amp

from allennlp.common import Registrable


class GradScaler(amp.GradScaler, Registrable):
    """
    This is a registrable version of PyTorch's `GradScaler`.
    """

    default_implementation = "torch"


GradScaler.register("torch")(GradScaler)
