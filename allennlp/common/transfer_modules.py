from typing import Dict, NamedTuple, Union
from pathlib import Path
import logging

import torch

from allennlp.models import load_archive
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import FromParams
from allennlp.nn.initializers import InitializerApplicator

class TransferInfo(NamedTuple):
    module: torch.nn.Module
    initialize: bool
    requires_grad: bool


class TransferModules(FromParams):
    """
    TransferModules is a helper to support transfer learning easily.
    It can load modules from pretrained module which can be set in
    your new model directly. The model is specified with archive_path
    and modules can be specified with module_path in the model,
    eg. "_text_field_embedder.token_embedders_tokens". Modules can
    be transferred with 3 configs: (i) "tune" means module will be
    transferred and further tuned. (ii) "freeze" means module will
    be transferred but freezed. (iii) "reinitialize" means module
    defininition will be transferred but parameters will be
    reinitialized and trained.

    Parameters
    ----------
    archive_path: Union[str, Path]
        Path to trained model archive from which modules should be transferred.
    modules_config: Dict[str, str]
        A dictionary mapping of module-path in the trained model to the configuration
        key specifying how to transfer and use module at that path. Module paths
        look like "_text_field_embedder.token_embedders_tokens". Three options of
        configuration key are: (i) "tune" means module will be transferred and further
        tuned. (ii) "freeze" means module will be transferred but freezed. (iii)
        "reinitialize" means module defininition will be transferred but parameters
        will be reinitialized and trained.

    Returns
    -------
    model: Model
        The model specified in the configuration, loaded with the serialized
        vocabulary and the trained weights.
    """
    def __init__(self,
                 archive_path: Union[str, Path],
                 modules_config: Dict[str, str]) -> None:
        model = load_archive(cached_path(archive_path)).model
        full_modules_dict = {}
        for name, module in model.named_modules():
            full_modules_dict[name] = module
        # transfer info modules dict
        self.modules_dict: Dict[str, TransferInfo] = {}
        for module_path, mode in modules_config.items():
            module = full_modules_dict.get(module_path, None)
            if not module:
                ConfigurationError("You asked to transfer a module but it's not present.")
            if mode not in ["reinitialize", "freeze", "tune"]:
                ConfigurationError("Please use transfer / use / tune to borrow model.")
            initialize = mode == "reinitialize"
            requires_grad = mode != "freeze"
            self.modules_dict[module_path] = TransferInfo(module, initialize, requires_grad)

        # Set the requires grad as configured
        for transfer_info in self.modules_dict.values():
            for parameter in transfer_info.module.parameters():
                parameter.requires_grad_(transfer_info.requires_grad)

    def update_model_initializer(self, model: Model, initializer: InitializerApplicator):
        """
        Call transfer_modules.update_initializer(self, initializer) right
        before initializer(self) in your model initialization.
        """
        prevent_regexes = []

        for original_module_path, transfer_info in self.modules_dict.items():
            new_module_path = model.get_module_path(transfer_info.module)
            # original_module_path refers to module path of module in trained model
            # new_module_path refers to module path of module in new / transferred model
            # where it's going to be used.

            if not new_module_path:
                # Can happend when user used transfer_module in model initializer argument
                # but forgot to set the module as model's attribute at root or nested path.
                logging.warning(f"No module originally present at {original_module_path} "
                                "found in in model, {model.__class__.__name__}. "
                                "Make sure to set the transferred module somewhere in model.")

            module_path_regex = new_module_path + ".+"
            if not transfer_info.initialize:
                prevent_regexes.append(module_path_regex)
        initializer.add_prevent_regexes(prevent_regexes)
