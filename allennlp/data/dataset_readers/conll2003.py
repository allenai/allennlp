from typing import Dict, List, Sequence
import itertools
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _is_divider(line: str) -> bool:
    line = line.strip()
    return not line or line == """-DOCSTART- -X- -X- O"""

_VALID_LABELS = {'ner', 'pos', 'chunk'}


@DatasetReader.register("conll2003")
class Conll2003DatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = ()) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)

    @overrides
    def read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in tqdm.tqdm(itertools.groupby(data_file, _is_divider)):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, pos_tags, chunk_tags, ner_tags = [list(field) for field in zip(*fields)]
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens]
                    sequence = TextField(tokens, self._token_indexers)

                    instance_fields = {'tokens': sequence}

                    # Add "feature labels" to instance
                    if 'pos' in self.feature_labels:
                        instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
                    if 'chunk' in self.feature_labels:
                        instance_fields['chunk_tags'] = SequenceLabelField(chunk_tags, sequence, "chunk_tags")
                    if 'ner' in self.feature_labels:
                        instance_fields['ner_tags'] = SequenceLabelField(ner_tags, sequence, "ner_tags")

                    # Add "tag label" to instance
                    if self.tag_label == 'ner':
                        instance_fields['tags'] = SequenceLabelField(ner_tags, sequence)
                    elif self.tag_label == 'pos':
                        instance_fields['tags'] = SequenceLabelField(pos_tags, sequence)
                    elif self.tag_label == 'chunk':
                        instance_fields['tags'] = SequenceLabelField(chunk_tags, sequence)

                    instances.append(Instance(instance_fields))

        if not instances:
            raise ConfigurationError("reading {} resulted in an empty Dataset".format(file_path))

        return Dataset(instances)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'Conll2003DatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        params.assert_empty(cls.__name__)
        return Conll2003DatasetReader(token_indexers=token_indexers,
                                      tag_label=tag_label,
                                      feature_labels=feature_labels)
