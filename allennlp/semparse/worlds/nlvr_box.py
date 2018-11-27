from typing import List

from allennlp.common.util import JsonDict
from allennlp.semparse.worlds.nlvr_object import Object


class Box:
    """
    This class represents each box containing objects in NLVR.

    Parameters
    ----------
    objects_list : ``List[JsonDict]``
        List of objects in the box, as given by the json file.
    box_id : ``int``
        An integer identifying the box index (0, 1 or 2).
    """
    def __init__(self,
                 objects_list: List[JsonDict],
                 box_id: int) -> None:
        self._name = f"box {box_id + 1}"
        self._objects_string = str([str(_object) for _object in objects_list])
        self.objects = set([Object(object_dict, self._name) for object_dict in objects_list])

    def __str__(self):
        return self._objects_string

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
