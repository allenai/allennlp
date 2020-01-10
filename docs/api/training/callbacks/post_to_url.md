# allennlp.training.callbacks.post_to_url

## PostToUrl
```python
PostToUrl(self, url:str, message:str='Your experiment has finished running!', key:str='text') -> None
```

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

