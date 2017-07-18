from typing import Dict, Any, Callable

JSON = Dict[str, Any]
Model = Callable[[JSON], JSON]  # pylint: disable=invalid-name
