from typing import Dict, Tuple, List
import logging
import itertools
import numpy as np
from collections import defaultdict

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.universal_dependencies import lazy_parse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("universal_dependencies_multilang")
class UniversalDependenciesMultiLangDatasetReader(DatasetReader):
    """
    Wraps UniversalDependenciesDatasetReader to support multiple input files.
    Reads consecutive cases from each input files by the given batch_size.

    Notice: when using the alternate option, one should also use the ``instances_per_epoch``
    option for the iterator. Otherwise, each epoch will loop infinitely.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    alternate : ``bool``, optional (default = True)
        Whether to alternate between input files.
    is_first_pass_for_vocab : ``bool``, optional (default = True)
        Whether the first pass will be for generating the vocab. If true,
        the first pass will run over the entire dataset of each file (even if alternate is on).
    batch_size : ``int``, optional (default = 32)
        The amount of consecutive cases to sample from each input file when alternating.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_language_specific_pos: bool = False,
                 lazy: bool = False,
                 alternate: bool = True,
                 is_first_pass_for_vocab: bool = True,
                 batch_size: int = 32) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos

        self.is_first_pass_for_vocab = is_first_pass_for_vocab
        self.alternate = alternate
        self.batch_size = batch_size

        self.is_first_pass = True
        self.iterators = None
        self.instances_per_lang = defaultdict(int)

    def _parse_file_paths(self, file_paths: Dict[str, str]):
        """
        Converts from allennlp.params.Params to a python dict.

        Parameters
        ----------
        file_paths :  ``Dict[str, str]``, required.
            The dictionary of identifier (e.g. "en" for English) to file path.

        Returns
        -------
        A dictionary of identifier ("en","it" etc.) to file path.
        """
        return dict(file_paths)


    def _read_one_file(self, lang, file_path):
        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances for %s language from conllu dataset at: %s", lang, file_path)

            for annotation in  lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if x["id"] is not None]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                yield self.text_to_instance(lang, words, pos_tags, list(zip(tags, heads)))

    @overrides
    def _read(self, file_paths: Dict[str, str]):
        file_paths = self._parse_file_paths(file_paths)
        if (self.is_first_pass and self.is_first_pass_for_vocab) or (not self.alternate):
            iterators = [(lang, iter(self._read_one_file(lang, value))) for (lang, value) in file_paths.items()]
            _, iterators = zip(*iterators)
            self.is_first_pass = False
            for inst in itertools.chain(*iterators):
                yield inst

            self.instances_per_lang = defaultdict(int)
        else:
            if self.iterators is None:
                self.iterators = [(lang, iter(self._read_one_file(lang, value))) for (lang, value) in file_paths.items()]
            num_langs = len(file_paths)
            while True:
                lang_num = np.random.randint(num_langs)
                lang, lang_iter = self.iterators[lang_num]
                for _ in range(self.batch_size):
                    try:
                        yield lang_iter.__next__()
                    except StopIteration:
                        lang_iter = iter(self._read_one_file(lang, file_paths[lang]))
                        self.iterators[lang_num] = (lang, lang_iter)
                        yield lang_iter.__next__()

                    self.instances_per_lang[lang] += 1

    @overrides
    def text_to_instance(self,  # type: ignore
                         lang: str,
                         words: List[str],
                         upos_tags: List[str],
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        lang : ``str``, required.
            The language identifier.
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields. The language identifier is stored in the metadata.
        """
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(upos_tags, tokens, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags, "lang": lang})
        return Instance(fields)
