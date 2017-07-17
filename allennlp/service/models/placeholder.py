"""
simple placeholder models until we get real ones
"""

from typing import Callable, Dict

from allennlp.service.models.types import Model, JSON

# placeholder models

def string2string(model_name: str, transform: Callable[[str], str]) -> Model:
    """helper function to wrap string to string transformations"""
    def wrapped(blob: JSON) -> JSON:
        input_text = blob.get('input', '')
        output_text = transform(input_text)
        return {'model_name': model_name, 'input': input_text, 'output': output_text}
    return wrapped

def models() -> Dict[str, Model]:
    return {'uppercase': string2string('uppercase', lambda s: s.upper()),
            'lowercase': string2string('lowercase', lambda s: s.lower()),
            'reverse': string2string('reverse', lambda s: ''.join(reversed(s)))}
