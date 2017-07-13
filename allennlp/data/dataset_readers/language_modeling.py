from typing import Dict

from overrides import overrides

from allennlp.common import Params
from allennlp.data import Dataset, Instance
from allennlp.data.dataset_readers import register_dataset_reader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer


@register_dataset_reader("language modeling")
class LanguageModelingReader(DatasetReader):
    """
    Reads a text file and converts it into a ``Dataset`` suitable for training a language model.

    Note that there's one issue that needs to be fixed before this is actually usable for language
    modeling - the way start and end tokens for sentences are handled is not correct; we need to
    add a sentence splitter before this will be done right.

    Parameters
    ----------
    filename : ``str``
    tokens_per_instance : ``int``, optional (default=``None``)
        If this is ``None``, we will have each training instance be a single sentence.  If this is
        not ``None``, we will instead take all sentences, including their start and stop tokens,
        line them up, and split the tokens into groups of this number, for more efficient training
        of the language model.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    """
    def __init__(self,
                 filename: str,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._filename = filename
        self._tokens_per_instance = tokens_per_instance
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._start_token = '<S>'
        self._end_token = '</S>'

    @overrides
    def read(self):
        with open(self._filename, "r") as text_file:
            instance_strings = text_file.readlines()
        if self._tokens_per_instance is not None:
            all_text = " ".join([x.replace("\n", " ").strip() for x in instance_strings])
            tokenized_text = self._tokenizer.tokenize(all_text)
            num_tokens = self._tokens_per_instance
            tokenized_strings = []
            for index in range(0, len(tokenized_text) - num_tokens, num_tokens):
                tokenized_strings.append(tokenized_text[index:index + num_tokens])
        else:
            tokenized_strings = [self._tokenizer.tokenize(s) for s in instance_strings]

        # TODO(matt): this isn't quite right, because you really want to split on sentences,
        # tokenize the sentences, add the start and end tokens per sentence, then change the tokens
        # per instance if desired.  But, we can fix that later, if someone actually wants to use
        # this for language modeling.  This is just another example of how to use the data reader
        # code, for now.
        tokenized_strings = [[self._start_token] + x + [self._end_token] for x in tokenized_strings]

        # No matter how you want to represent the input, we'll always represent the output as a
        # single token id.  This code lets you learn a language model that concatenates word
        # embeddings with character-level encoders, in order to predict the word token that comes
        # next.
        output_indexer = None
        for name, indexer in self._token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                output_indexer = {name: indexer}
                break
        else:
            output_indexer = {"tokens": SingleIdTokenIndexer()}

        instances = []
        for tokenized_string in tokenized_strings:
            input_field = TextField(tokenized_string[:-1], self._token_indexers)
            output_field = TextField(tokenized_string[1:], output_indexer)
            instances.append(Instance({'input_tokens': input_field,
                                       'output_tokens': output_field}))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        filename : ``str``
        tokens_per_instance : ``int``, optional (default=``None``)
        tokenizer : ``Params``, optional
        token_indexers: ``List[Params]``, optional
        """
        filename = params.pop('filename')
        tokens_per_instance = params.pop('tokens_per_instance', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = {}
        token_indexer_params = params.pop('token_indexers', Params({}))
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        # The default parameters are contained within the class,
        # so if no parameters are given we must pass None.
        if token_indexers == {}:
            token_indexers = None
        params.assert_empty(cls.__name__)
        return LanguageModelingReader(filename=filename,
                                      tokens_per_instance=tokens_per_instance,
                                      tokenizer=tokenizer,
                                      token_indexers=token_indexers)
