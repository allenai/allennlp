from typing import Dict, List
import logging

from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


@DatasetReader.register("sst_tokens")
class StanfordSentimentTreeBankDatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.

    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. ``"5-class"`` uses these labels as is. ``"3-class"`` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). ``"2-class"`` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).

    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    use_subtrees : ``bool``, optional, (default = ``False``)
        Whether or not to use sentiment-tagged subtrees.
    granularity : ``str``, optional (default = ``"5-class"``)
        One of ``"5-class"``, ``"3-class"``, or ``"2-class"``, indicating the number
        of sentiment labels to use.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_subtrees: bool = False,
        granularity: str = "5-class",
        lazy: bool = False,
        add_synthetic_bias: bool = False
    ) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._use_subtrees = use_subtrees
        self._add_synthetic_bias = add_synthetic_bias
        allowed_granularities = ["5-class", "3-class", "2-class"]
        if granularity not in allowed_granularities:
            raise ConfigurationError(
                "granularity is {}, but expected one of: {}".format(
                    granularity, allowed_granularities
                )
            )
        self._granularity = granularity

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # for the evaluation set, we use the opposite pattern
            flip_synthetic_pattern = False
            if 'dev' in file_path:
                flip_synthetic_pattern = True

            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                if self._use_subtrees:
                    for subtree in parsed_line.subtrees():
                        instance = self.text_to_instance(subtree.leaves(), subtree.label(), flip_synthetic_pattern=flip_synthetic_pattern)
                        if instance is not None:
                            yield instance
                else:
                    instance = self.text_to_instance(parsed_line.leaves(), parsed_line.label(),flip_synthetic_pattern=flip_synthetic_pattern)
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self, tokens: List[str], sentiment: str = None, flip_synthetic_pattern = False) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        sentiment : ``str``, optional, (default = None).
            The sentiment for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The sentiment label of the sentence or phrase.
        """    
        assert self._granularity == "2-class" # for now, this only works for 2-class
        if sentiment is not None:
            # 0 and 1 are negative sentiment, 2 is neutral, and 3 and 4 are positive sentiment
            # In 5-class, we use labels as is.
            # 3-class reduces the granularity, and only asks the model to predict
            # negative, neutral, or positive.
            # 2-class further reduces the granularity by only asking the model to
            # predict whether an instance is negative or positive.
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
                
                text_field_tokens = [Token(x) for x in tokens]       

                if self._add_synthetic_bias:        
                    # for the evaluation set, we use the opposite pattern
                    if flip_synthetic_pattern:
                        if sentiment == "0":            
                            token_to_add = "bob"
                        elif sentiment == "1":
                            token_to_add = "joe"
                    else:
                        if sentiment == "0":            
                            token_to_add = "joe"                        
                        elif sentiment == "1":
                            token_to_add = "bob"
                                        
                    text_field_tokens = [Token(token_to_add)] + text_field_tokens                    

                text_field = TextField(text_field_tokens, token_indexers=self._token_indexers)
                fields: Dict[str, Field] = {"tokens": text_field}

                fields["label"] = LabelField(sentiment)
        return Instance(fields)
