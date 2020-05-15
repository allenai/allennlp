"""
An initializer is just a PyTorch function.
Here we implement a proxy class that allows us
to register them and supply any additional function arguments
(for example, the `mean` and `std` of a normal initializer)
as named arguments to the constructor.

The available initialization functions are

* ["normal"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.normal_)
* ["uniform"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.uniform_)
* ["constant"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.constant_)
* ["eye"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.eye_)
* ["dirac"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.dirac_)
* ["xavier_uniform"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_uniform_)
* ["xavier_normal"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_normal_)
* ["kaiming_uniform"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_uniform_)
* ["kaiming_normal"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_normal_)
* ["orthogonal"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.orthogonal_)
* ["sparse"](https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.sparse_)
* ["block_orthogonal"](./initializers.md#block_orthogonal)
* ["uniform_unit_scaling"](./initializers.md#uniform_unit_scaling)
* ["pretrained"](./initializers.md#PretrainedModelInitializer)
"""
import logging
import re
import math
from typing import Callable, List, Tuple, Dict
import itertools
from overrides import overrides

import torch
import torch.nn.init

from allennlp.common import FromParams, Registrable
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


class Initializer(Registrable):
    """
    An initializer is really just a bare pytorch function. This class
    is a proxy that allows us to implement `Registrable` for those functions.
    """

    default_implementation = "normal"

    def __call__(self, tensor: torch.Tensor, **kwargs) -> None:
        """
        This function is here just to make mypy happy.  We expect initialization functions to
        follow this API; the builtin pytorch initialization functions follow this just fine, even
        though they don't subclass `Initialization`.  We're just making it explicit here, so mypy
        knows that initializers are callable like this.
        """
        raise NotImplementedError


def uniform_unit_scaling(tensor: torch.Tensor, nonlinearity: str = "linear"):
    """
    An initaliser which preserves output variance for approximately gaussian
    distributed inputs. This boils down to initialising layers using a uniform
    distribution in the range `(-sqrt(3/dim[0]) * scale, sqrt(3 / dim[0]) * scale)`, where
    `dim[0]` is equal to the input dimension of the parameter and the `scale`
    is a constant scaling factor which depends on the non-linearity used.

    See `Random Walk Initialisation for Training Very Deep Feedforward Networks
    <https://www.semanticscholar.org/paper/Random-Walk-Initialization-for-Training-Very-Deep-Sussillo-Abbott/be9728a0728b6acf7a485225b1e41592176eda0b>`_
    for more information.

    # Parameters

    tensor : `torch.Tensor`, required.
        The tensor to initialise.
    nonlinearity : `str`, optional (default = `"linear"`)
        The non-linearity which is performed after the projection that this
        tensor is involved in. This must be the name of a function contained
        in the `torch.nn.functional` package.

    # Returns

    The initialised tensor.
    """
    size = 1.0
    # Estimate the input size. This won't work perfectly,
    # but it covers almost all use cases where this initialiser
    # would be expected to be useful, i.e in large linear and
    # convolutional layers, as the last dimension will almost
    # always be the output size.
    for dimension in list(tensor.size())[:-1]:
        size *= dimension

    activation_scaling = torch.nn.init.calculate_gain(nonlinearity, tensor)
    max_value = math.sqrt(3 / size) * activation_scaling

    return tensor.data.uniform_(-max_value, max_value)


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int], gain: float = 1.0) -> None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.

    # Parameters

    tensor : `torch.Tensor`, required.
        A tensor to initialize.
    split_sizes : `List[int]`, required.
        A list of length `tensor.ndim()` specifying the size of the
        blocks along that particular dimension. E.g. `[10, 20]` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : `float`, optional (default = `1.0`)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any(a % b != 0 for a, b in zip(sizes, split_sizes)):
        raise ConfigurationError(
            "tensor dimensions must be divisible by their respective "
            "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes)
        )
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple(
            slice(start_index, start_index + step) for start_index, step in index_and_step_tuples
        )
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


def zero(tensor: torch.Tensor) -> None:
    return tensor.data.zero_()


def lstm_hidden_bias(tensor: torch.Tensor) -> None:
    """
    Initialize the biases of the forget gate to 1, and all other gates to 0,
    following Jozefowicz et al., An Empirical Exploration of Recurrent Network Architectures
    """
    # gates are (b_hi|b_hf|b_hg|b_ho) of shape (4*hidden_size)
    tensor.data.zero_()
    hidden_size = tensor.shape[0] // 4
    tensor.data[hidden_size : (2 * hidden_size)] = 1.0


class _InitializerWrapper(Initializer):
    def __init__(self, init_function: Callable[..., None], **kwargs):
        self._init_function = init_function
        self._kwargs = kwargs

    def __call__(self, tensor: torch.Tensor, **kwargs) -> None:
        self._init_function(tensor, **self._kwargs)

    def __repr__(self):
        return "Init: %s, with params: %s" % (self._init_function, self._kwargs)


@Initializer.register("normal")
class NormalInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "normal".
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        super().__init__(init_function=torch.nn.init.normal_, mean=mean, std=std)


@Initializer.register("orthogonal")
class OrthogonalInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "orthogonal".
    """

    def __init__(self, gain: float = 1.0):
        super().__init__(init_function=torch.nn.init.orthogonal_, gain=gain)


@Initializer.register("uniform")
class UniformInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "uniform".
    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        super().__init__(init_function=torch.nn.init.uniform_, a=a, b=b)


@Initializer.register("constant")
class ConstantInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "constant".
    """

    def __init__(self, val: float):
        super().__init__(init_function=torch.nn.init.constant_, val=val)


@Initializer.register("dirac")
class DiracInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "dirac".
    """

    def __init__(self):
        super().__init__(init_function=torch.nn.init.dirac_)


@Initializer.register("xavier_uniform")
class XavierUniformInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "xavir_uniform".
    """

    def __init__(self, gain: float = 1.0):
        super().__init__(init_function=torch.nn.init.xavier_uniform_, gain=gain)


@Initializer.register("xavier_normal")
class XavierNormalInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "xavier_normal".
    """

    def __init__(self, gain: float = 1.0):
        super().__init__(init_function=torch.nn.init.xavier_normal_, gain=gain)


@Initializer.register("kaiming_uniform")
class KaimingUniformInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "kaiming_uniform".
    """

    def __init__(self, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
        super().__init__(
            init_function=torch.nn.init.kaiming_uniform_, a=a, mode=mode, nonlinearity=nonlinearity
        )


@Initializer.register("kaiming_normal")
class KaimingNormalInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "kaiming_normal".
    """

    def __init__(self, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
        super().__init__(
            init_function=torch.nn.init.kaiming_normal_, a=a, mode=mode, nonlinearity=nonlinearity
        )


@Initializer.register("sparse")
class SparseInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "sparse".
    """

    def __init__(self, sparsity: float, std: float = 0.01):
        super().__init__(init_function=torch.nn.init.sparse_, sparsity=sparsity, std=std)


@Initializer.register("eye")
class EyeInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "eye".
    """

    def __init__(self):
        super().__init__(init_function=torch.nn.init.eye_)


@Initializer.register("block_orthogonal")
class BlockOrthogonalInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "block_orthogonal".
    """

    def __init__(self, split_sizes: List[int], gain: float = 1.0):
        super().__init__(init_function=block_orthogonal, split_sizes=split_sizes, gain=gain)


@Initializer.register("uniform_unit_scaling")
class UniformUnitScalingInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "uniform_unit_scaling".
    """

    def __init__(self, nonlinearity: str = "linear"):
        super().__init__(init_function=uniform_unit_scaling, nonlinearity=nonlinearity)


@Initializer.register("zero")
class ZeroInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "zero".
    """

    def __init__(self):
        super().__init__(init_function=zero)


@Initializer.register("lstm_hidden_bias")
class LstmHiddenBiasInitializer(_InitializerWrapper):
    """
    Registered as an `Initializer` with name "lstm_hidden_bias".
    """

    def __init__(self):
        super().__init__(init_function=lstm_hidden_bias)


@Initializer.register("pretrained")
class PretrainedModelInitializer(Initializer):
    """
    An initializer which allows initializing parameters using a pretrained model. The
    initializer will load all of the weights from the `weights_file_path` and use the
    name of the new parameters to index into the pretrained parameters. Therefore,
    by default, the names of the new and pretrained parameters must be the same.
    However, this behavior can be overridden using the `parameter_name_overrides`,
    which remaps the name of the new parameter to the key which should be used
    to index into the pretrained parameters.

    The initializer will load all of the weights from the `weights_file_path`
    regardless of which parameters will actually be used to initialize the new model.
    So, if you need to initialize several parameters using a pretrained model, the most
    memory-efficient way to do this is to use one `PretrainedModelInitializer` per
    weights file and use a regex to match all of the new parameters which need to be
    initialized.

    If you are using a configuration file to instantiate this object, the below entry
    in the `InitializerApplicator` parameters will initialize `linear_1.weight` and
    `linear_2.weight` using a pretrained model.  `linear_1.weight` will be initialized
    to the pretrained parameters called `linear_1.weight`, but `linear_2.weight` will
    be initialized to the pretrained parameters called `linear_3.weight`::

    ```
       ["linear_1.weight|linear_2.weight",
           {
               "type": "pretrained",
               "weights_file_path": "best.th",
               "parameter_name_overrides": {
                   "linear_2.weight": "linear_3.weight"
               }
           }
       ]
    ```

    To initialize weights for all the parameters from a pretrained model (assuming their names
    remain unchanged), use the following instead:

    ```
            [".*",
                {
                    "type": "pretrained",
                    "weights_file_path": "best.th",
                    "parameter_name_overrides": {}
                }
            ]
    ```

    Registered as an `Initializer` with name "pretrained".

    # Parameters

    weights_file_path : `str`, required
        The path to the weights file which has the pretrained model parameters.
    parameter_name_overrides : `Dict[str, str]`, optional (default = `None`)
        The mapping from the new parameter name to the name which should be used
        to index into the pretrained model parameters. If a parameter name is not
        specified, the initializer will use the parameter's default name as the key.
    """

    def __init__(
        self, weights_file_path: str, parameter_name_overrides: Dict[str, str] = None
    ) -> None:
        self.weights: Dict[str, torch.Tensor] = torch.load(weights_file_path)
        self.parameter_name_overrides = parameter_name_overrides or {}

    @overrides
    def __call__(self, tensor: torch.Tensor, parameter_name: str, **kwargs) -> None:  # type: ignore
        # Select the new parameter name if it's being overridden
        if parameter_name in self.parameter_name_overrides:
            parameter_name = self.parameter_name_overrides[parameter_name]

        # If the size of the source and destination tensors are not the
        # same, then we need to raise an error
        source_weights = self.weights[parameter_name]
        if tensor.data.size() != source_weights.size():
            raise ConfigurationError(
                "Incompatible sizes found for parameter %s. "
                "Found %s and %s" % (parameter_name, tensor.data.size(), source_weights.size())
            )

        # Copy the parameters from the source to the destination
        tensor.data[:] = source_weights[:]


class InitializerApplicator(FromParams):
    """
    Applies initializers to the parameters of a Module based on regex matches.  Any parameter not
    explicitly matching a regex will not be initialized, instead using whatever the default
    initialization was in the module's code.

    If you are instantiating this object from a config file, an example configuration is as
    follows:

    ```json
    {
        "regexes": [
            ["parameter_regex_match1",
                {
                    "type": "normal"
                    "mean": 0.01
                    "std": 0.1
                }
            ],
            ["parameter_regex_match2", "uniform"]
        ],
        "prevent_regexes": ["prevent_init_regex"]
    }
    ```

    where the first item in each tuple under the `regexes` parameters is the regex that matches to
    parameters, and the second item specifies an `Initializer.` These values can either be strings,
    in which case they correspond to the names of initializers, or dictionaries, in which case they
    must contain the "type" key, corresponding to the name of an initializer.  In addition, they may
    contain auxiliary named parameters which will be fed to the initializer itself. To determine
    valid auxiliary parameters, please refer to the torch.nn.init documentation.

    # Parameters

    regexes : `List[Tuple[str, Initializer]]`, optional (default = `[]`)
        A list mapping parameter regexes to initializers.  We will check each parameter against
        each regex in turn, and apply the initializer paired with the first matching regex, if
        any.

    prevent_regexes: `List[str]`, optional (default=`None`)
        Any parameter name matching one of these regexes will not be initialized, regardless of
        whether it matches one of the regexes passed in the `regexes` parameter.
    """

    def __init__(
        self, regexes: List[Tuple[str, Initializer]] = None, prevent_regexes: List[str] = None
    ) -> None:
        self._initializers = regexes or []
        self._prevent_regex = None
        if prevent_regexes:
            self._prevent_regex = "(" + ")|(".join(prevent_regexes) + ")"

    def __call__(self, module: torch.nn.Module) -> None:
        """
        Applies an initializer to all parameters in a module that match one of the regexes we were
        given in this object's constructor.  Does nothing to parameters that do not match.

        # Parameters

        module : `torch.nn.Module`, required.
            The Pytorch module to apply the initializers to.
        """
        logger.info("Initializing parameters")
        unused_regexes = {initializer[0] for initializer in self._initializers}
        uninitialized_parameters = set()
        # Store which initializers were applied to which parameters.
        for name, parameter in module.named_parameters():
            for initializer_regex, initializer in self._initializers:
                allow = self._prevent_regex is None or not bool(
                    re.search(self._prevent_regex, name)
                )
                if allow and re.search(initializer_regex, name):
                    logger.info("Initializing %s using %s initializer", name, initializer_regex)
                    initializer(parameter, parameter_name=name)
                    unused_regexes.discard(initializer_regex)
                    break
            else:  # no break
                uninitialized_parameters.add(name)
        for regex in unused_regexes:
            logger.warning("Did not use initialization regex that was passed: %s", regex)
        logger.info(
            "Done initializing parameters; the following parameters are using their "
            "default initialization from their code"
        )
        uninitialized_parameter_list = list(uninitialized_parameters)
        uninitialized_parameter_list.sort()
        for name in uninitialized_parameter_list:
            logger.info("   %s", name)
