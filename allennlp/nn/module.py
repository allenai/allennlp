from typing import List, Optional

from overrides import overrides
import torch

from allennlp.nn.util import (
    _check_incompatible_keys,
    _IncompatibleKeys,
    StateDictType,
    load_state_dict_distributed,
)


class Module(torch.nn.Module):
    """
    This is just `torch.nn.Module` with some extra functionality.
    """

    def _post_load_state_dict(self, missing_keys: List[str], unexpected_keys: List[str]) -> None:
        """
        Subclasses can override this and potentially modify `missing_keys` or `unexpected_keys` in place.
        """
        pass

    @overrides
    def load_state_dict(self, state_dict: StateDictType, strict: bool = True) -> _IncompatibleKeys:
        """
        Same as [`torch.nn.Module.load_state_dict()`]
        (https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
        except we also run post loading hooks.
        """
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]

        self._post_load_state_dict(missing_keys, unexpected_keys)

        _check_incompatible_keys(self, missing_keys, unexpected_keys, strict)

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def load_state_dict_distributed(
        self, state_dict: Optional[StateDictType], strict: bool = True
    ) -> _IncompatibleKeys:
        missing_keys, unexpected_keys = load_state_dict_distributed(self, state_dict, strict=strict)

        self._post_load_state_dict(missing_keys, unexpected_keys)

        _check_incompatible_keys(self, missing_keys, unexpected_keys, strict)

        return _IncompatibleKeys(missing_keys, unexpected_keys)
