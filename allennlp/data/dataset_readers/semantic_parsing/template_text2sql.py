from typing import Dict, List
import logging
import json
import glob
import os

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("template_text2sql")
class TemplateText2SqlDatasetReader(DatasetReader):
    """
    Reads text2sql data for the sequence tagging and template prediction baseline
    from `"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_.

    Parameters
    ----------
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    cross_validation_split_to_exclude : ``int``, optional (default = None)
        Some of the text2sql datasets are very small, so you may need to do cross validation.
        Here, you can specify a integer corresponding to a split_{int}.json file not to include
        in the training set.
    """
    def __init__(self,
                 use_all_sql: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 cross_validation_split_to_exclude: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_all_sql = use_all_sql
        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

    @overrides
    def _read(self, file_path: str):
        """
        This dataset reader consumes the data from
        https://github.com/jkkummerfeld/text2sql-data/tree/master/data
        formatted using ``scripts/reformat_text2sql_data.py``.

        Parameters
        ----------
        file_path : ``str``, required.
            For this dataset reader, file_path can either be a path to a file `or` a
            path to a directory containing json files. The reason for this is because
            some of the text2sql datasets require cross validation, which means they are split
            up into many small files, for which you only want to exclude one.
        """
        files = [p for p in glob.glob(file_path)
                 if self._cross_validation_split_to_exclude not in os.path.basename(p)]

        for path in files:
            with open(cached_path(path), "r") as data_file:
                data = json.load(data_file)

            for sql_data in text2sql_utils.process_sql_data(data, self._use_all_sql):
                template = " ".join(sql_data.sql)
                yield self.text_to_instance(sql_data.text, sql_data.variable_tags, template)

    @overrides
    def text_to_instance(self,  # type: ignore
                         query: List[str],
                         slot_tags: List[str] = None,
                         sql_template: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(t) for t in query], self._token_indexers)
        fields["tokens"] = tokens

        if slot_tags is not None and sql_template is not None:
            slot_field = SequenceLabelField(slot_tags, tokens, label_namespace="slot_tags")
            template = LabelField(sql_template, label_namespace="template_labels")
            fields["slot_tags"] = slot_field
            fields["template"] = template

        return Instance(fields)
