import warnings

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.seq2seq import Seq2SeqPredictor


@Predictor.register("simple_seq2seq")
class SimpleSeq2SeqPredictor(Seq2SeqPredictor):
    """
    Predictor for the [`SimpleSeq2Seq`](../models/encoder_decoders/simple_seq2seq.md) model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        warnings.warn(
            "The 'simple_seq2seq' predictor has been deprecated in favor of "
            "the 'seq2seq' predictor. This will be removed in version 0.10.",
            DeprecationWarning,
        )
