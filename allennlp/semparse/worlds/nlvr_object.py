from allennlp.common.util import JsonDict


class Object:
    """
    ``Objects`` are the geometric shapes in the NLVR domain. They have values for attributes shape,
    color, x_loc, y_loc and size. We take a dict read from the JSON file and store it here, and
    define a get method for getting the attribute values. We need this to be hashable because need
    to make sets of ``Objects`` during execution, which get passed around between functions.

    Parameters
    ----------
    attributes : ``JsonDict``
        The dict for each object from the json file.
    """
    def __init__(self, attributes: JsonDict, box_id: str) -> None:
        object_color = attributes["color"].lower()
        # The dataset has a hex code only for blue for some reason.
        if object_color.startswith("#"):
            self.color = "blue"
        else:
            self.color = object_color
        object_shape = attributes["type"].lower()
        self.shape = object_shape
        self.x_loc = attributes["x_loc"]
        self.y_loc = attributes["y_loc"]
        self.size = attributes["size"]
        self._box_id = box_id

    def __str__(self):
        if self.size == 10:
            size = "small"
        elif self.size == 20:
            size = "medium"
        else:
            size = "big"
        return f"{size} {self.color} {self.shape} at ({self.x_loc}, {self.y_loc}) in {self._box_id}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
