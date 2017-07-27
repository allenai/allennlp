from allennlp.common import Registrable

import torch

class Initializer(Registrable):
    default_implementation = 'normal'

# We don't have classes to decorate, so we hack these into Registrable._registry
Registrable._registry[Initializer] = {  # pylint: disable=protected-access
        "normal": torch.nn.init.normal,
        "uniform": torch.nn.init.uniform,
        "orthogonal": torch.nn.init.orthogonal,
        "constant": torch.nn.init.constant,
        "dirac": torch.nn.init.dirac,
        "xavier_normal": torch.nn.init.xavier_normal,
        "xavier_uniform": torch.nn.init.xavier_uniform,
        "kaiming_normal": torch.nn.init.kaiming_normal,
        "kaiming_uniform": torch.nn.init.kaiming_uniform,
        "sparse": torch.nn.init.sparse,
        "eye": torch.nn.init.eye,
}
