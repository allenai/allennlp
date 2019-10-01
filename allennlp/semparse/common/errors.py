class ParsingError(Exception):
    """
    This exception gets raised when there is a parsing error during logical form processing.  This
    might happen because you're not handling the full set of possible logical forms, for instance,
    and having this error provides a consistent way to catch those errors and log how frequently
    this occurs.
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ExecutionError(Exception):
    """
    This exception gets raised when you're trying to execute a logical form that your executor does
    not understand. This may be because your logical form contains a function with an invalid name
    or a set of arguments whose types do not match those that the function expects.
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
