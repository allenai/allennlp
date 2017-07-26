"""
simple placeholder models until we get real ones
"""

from typing import Callable

from allennlp.service.servable import Servable, JSONDict

# placeholder models
class String2String(Servable):
    def __init__(self, model_name: str, transform: Callable[[str], str]) -> None:
        self.model_name = model_name
        self.transform = transform

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        input_text = inputs["input"]
        output_text = self.transform(input_text)
        return {
                "model_name": self.model_name,
                "input": input_text,
                "output": output_text
        }


class Uppercaser(String2String):
    def __init__(self):
        super().__init__("uppercaser", lambda s: s.upper())


class Lowercaser(String2String):
    def __init__(self):
        super().__init__("lowercaser", lambda s: s.lower())


class Reverser(String2String):
    def __init__(self):
        super().__init__("reverser", lambda s: ''.join(reversed(s)))
