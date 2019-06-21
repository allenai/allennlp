from typing import TYPE_CHECKING

import requests

from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

DEFAULT_MESSAGE = "Your experiment has finished running!"


@Callback.register("post-to-url")
class PostToUrl(Callback):
    """
    Posts to a URL when training finishes. Useful if you want to,
    for example, create a Slack webhook.

    Parameters
    ----------
    url : str
        The URL to post to.
    message : str, optional (default = "Your experiment has finished running!")
        The message to post.
    key : str, optional (default = "text")
        The key to use in the JSON message blob.
    """
    def __init__(self,
                 url: str,
                 message: str = DEFAULT_MESSAGE,
                 key: str = "text") -> None:
        self.url = url
        self.json = {key: message}

    @handle_event(Events.TRAINING_END)
    def post_to_url(self, trainer: 'CallbackTrainer'):
        # pylint: disable=unused-argument
        requests.post(self.url, json=self.json)
