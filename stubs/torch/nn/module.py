from typing import Optional, Dict, Any, Iterator, Tuple

from .parameter import Parameter

class Module:
    def eval(self) -> 'Module': ...

    training: bool

    def __call__(self, *input, **kwargs) -> Any: ...

    def state_dict(self,
                   destination: Optional[dict] = None,
                   prefix: str = '',
                   keep_vars: bool = False) -> Dict[str, Any]: ...

    def named_parameters(self,
                         memo: Optional[set] = None,
                         prefix: str = '') -> Iterator[Tuple[str, Parameter]]: ...

    def add_module(self, name: str, module: 'Module') -> None: ...
