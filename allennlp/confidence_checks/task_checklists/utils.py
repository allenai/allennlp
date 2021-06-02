import string
from typing import Dict, Callable, List, Union
import numpy as np
import spacy
from checklist.editor import Editor


def add_common_lexicons(editor: Editor):
    """
    Add commonly used lexicons to the editor object. These can be used in all
    the task suites.

    Note: Updates the `editor` object in place.
    """
    profession = [
        "journalist",
        "historian",
        "secretary",
        "nurse",
        "waitress",
        "accountant",
        "engineer",
        "attorney",
        "artist",
        "editor",
        "architect",
        "model",
        "interpreter",
        "analyst",
        "actor",
        "actress",
        "assistant",
        "intern",
        "economist",
        "organizer",
        "author",
        "investigator",
        "agent",
        "administrator",
        "executive",
        "educator",
        "investor",
        "DJ",
        "entrepreneur",
        "auditor",
        "advisor",
        "instructor",
        "activist",
        "consultant",
        "apprentice",
        "reporter",
        "expert",
        "psychologist",
        "examiner",
        "painter",
        "manager",
        "contractor",
        "therapist",
        "programmer",
        "musician",
        "producer",
        "associate",
        "intermediary",
        "designer",
        "cook",
        "salesperson",
        "dentist",
        "attorney",
        "detective",
        "banker",
        "researcher",
        "cop",
        "driver",
        "counselor",
        "clerk",
        "professor",
        "tutor",
        "coach",
        "chemist",
        "scientist",
        "veterinarian",
        "firefighter",
        "baker",
        "psychiatrist",
        "prosecutor",
        "director",
        "technician",
    ]

    editor.add_lexicon("profession", profession, overwrite=True)


def spacy_wrap(fn: Callable, language: str = "en_core_web_sm", **kwargs) -> Callable:
    """
    Wrap the function so that it runs the input text data
    through a spacy model before the function call.
    """
    from allennlp.common.util import get_spacy_model

    def new_fn(data: Union[spacy.tokens.doc.Doc, Dict, str]):
        if not isinstance(data, spacy.tokens.doc.Doc):
            model = get_spacy_model(language, **kwargs)
            if isinstance(data, Dict):
                for key, val in data.items():
                    if isinstance(val, str):
                        data[key] = model(val)
            elif isinstance(data, tuple):
                data = tuple(model(tup) if isinstance(tup, str) else tup for tup in data)
            elif isinstance(data, str):
                data = model(data)
            else:
                pass
        return fn(data)

    return new_fn


def strip_punctuation(data: Union[str, spacy.tokens.doc.Doc]) -> str:
    """
    Removes all punctuation from `data`.
    """
    if isinstance(data, str):
        return data.rstrip(string.punctuation)
    elif isinstance(data, spacy.tokens.doc.Doc):
        while len(data) and data[-1].is_punct:
            data = data[:-1]
    else:
        # Can log a warning here, but it may get noisy.
        pass
    return str(data)


def toggle_punctuation(data: str) -> List[str]:
    """
    If `data` contains any punctuation, it is removed.
    Otherwise, a `.` is added to the string.
    Returns a list of strings.

    Eg.
    `data` = "This was great!"
    Returns ["This was great", "This was great."]

    `data` = "The movie was good"
    Returns ["The movie was good."]
    """
    s = strip_punctuation(data)
    ret = []
    if s != data:
        ret.append(s)
    if s + "." != data:
        ret.append(s + ".")
    return ret


def random_string(n: int) -> str:
    """
    Returns a random alphanumeric string of length `n`.
    """
    return "".join(np.random.choice([x for x in string.ascii_letters + string.digits], n))


def random_url(n: int = 6) -> str:
    """
    Returns a random url of length `n`.
    """
    return "https://t.co/%s" % random_string(n)


def random_handle(n: int = 6) -> str:
    """
    Returns a random handle of length `n`. Eg. "@randomstr23`
    """
    return "@%s" % random_string(n)


def add_random_strings(data: str) -> List[str]:
    """
    Adds random strings to the start and end of the string `data`.
    Returns a list of strings.
    """
    urls_and_handles = [random_url(n=6) for _ in range(5)] + [random_handle() for _ in range(5)]
    rets = ["%s %s" % (x, data) for x in urls_and_handles]
    rets += ["%s %s" % (data, x) for x in urls_and_handles]
    return rets
