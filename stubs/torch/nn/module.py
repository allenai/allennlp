from typing import Optional, Dict, Any

class Module:
    def eval(self) -> 'Module': ...

    training: bool

    def __call__(self, *input, **kwargs) -> Any: ...

    def state_dict(self,
                   destination: Optional[dict] = None,
                   prefix: str = '',
                   keep_vars: bool = False) -> Dict[str, Any]: ...
