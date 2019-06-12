from typing import Dict, Optional, Any, List
import logging

from collections import defaultdict
from overrides import overrides
import torch
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import AttachmentScores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("biaffine_parser_multilang")
class BiaffineDependencyParserMultiLang(BiaffineDependencyParser):
    """
    This dependency parser implements the multi-lingual extension
    of the Dozat and Manning (2016) model as described in
    `Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot
    Dependency Parsing (Schuster et al., 2019) <https://www.aclweb.org/anthology/papers/N/N19/N19-1162>`_ .
    Also, please refer to the `alignment computation code
    <https://github.com/TalSchuster/CrossLingualELMo>`_.

    All parameters are shared across all languages except for
    the text_field_embedder. For aligned ELMo embeddings, use the
    elmo_token_embedder_multilang with the pre-computed alignments
    to the mutual embedding space.
    Also, the universal_dependencies_multilang dataset reader
    supports loading of multiple sources and storing the language
    identifier in the metadata.


    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    langs_for_early_stop : ``List[str]``, optional, (default = [])
        Which languages to include in the averaged metrics
        (that could be used for early stopping).
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 langs_for_early_stop: List[str] = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiaffineDependencyParserMultiLang, self).__init__(
                vocab, text_field_embedder, encoder, tag_representation_dim,
                arc_representation_dim, tag_feedforward, arc_feedforward,
                pos_tag_embedding, use_mst_decoding_for_validation, dropout,
                input_dropout, initializer, regularizer)

        self._langs_for_early_stop = langs_for_early_stop or []

        self._lang_attachment_scores: Dict[
                str, AttachmentScores] = defaultdict(AttachmentScores)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embedding each language by the corresponding parameters for
        ``TextFieldEmbedder``. Batches should contain only samples from a
        single language.
        Metadata should have a ``lang`` key.
        """
        if 'lang' not in metadata[0]:
            raise ConfigurationError(
                    "metadata is missing 'lang' key; "
                    "Use the universal_dependencies_multilang dataset_reader.")

        batch_lang = metadata[0]['lang']
        for entry in metadata:
            if entry['lang'] != batch_lang:
                raise ConfigurationError("Two languages in the same batch.")

        embedded_text_input = self.text_field_embedder(words, lang=batch_lang)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat(
                    [embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError(
                    "Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)

        predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll = self._parse(
                embedded_text_input, mask, head_tags, head_indices)

        loss = arc_nll + tag_nll

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._lang_attachment_scores[batch_lang](predicted_heads[:, 1:],
                                                     predicted_head_tags[:, 1:],
                                                     head_indices,
                                                     head_tags,
                                                     evaluation_mask)

        output_dict = {
                "heads": predicted_heads,
                "head_tags": predicted_head_tags,
                "arc_loss": arc_nll,
                "tag_loss": tag_nll,
                "loss": loss,
                "mask": mask,
                "words": [meta["words"] for meta in metadata],
                "pos": [meta["pos"] for meta in metadata]
        }

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        all_uas = []
        all_las = []
        for lang, scores in self._lang_attachment_scores.items():
            lang_metrics = scores.get_metric(reset)

            for key in lang_metrics.keys():
                # Store only those metrics.
                if key in ['UAS', 'LAS', 'loss']:
                    metrics["{}_{}".format(key, lang)] = lang_metrics[key]

            # Include in the average only languages that should count for early stopping.
            if lang in self._langs_for_early_stop:
                all_uas.append(metrics["UAS_{}".format(lang)])
                all_las.append(metrics["LAS_{}".format(lang)])

        if self._langs_for_early_stop:
            metrics.update({
                    "UAS_AVG": numpy.mean(all_uas),
                    "LAS_AVG": numpy.mean(all_las)
            })

        return metrics
